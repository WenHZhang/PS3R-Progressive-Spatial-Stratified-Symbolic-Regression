"""Fixed-split PS3R simulation recovery experiment.

This is not cross-validation. The fixed 80%/20% split stored
in ``simulation_data.csv`` is reused for model
seeds 0--4.  The
outer test rows never enter ``PS3RModel.fit``. Within the outer training rows,
PS3R reserves complete 10 x 10 spatial-grid blocks for its fixed
internal validation holdout.

Only x1--x9 and noisy y are used for fitting.  ``coord_x``/``coord_y`` are
used only to construct spatial blocks.  ``y_clean`` and
``true_stratum`` are used after fitting to measure function and stratum
recovery. The DSO setup draws 50,000 expressions per node.
"""

import argparse
import datetime
import gc
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
)
from sklearn.tree import export_text


SCRIPT_DIR = Path(__file__).resolve().parent

from ps3r.utils import get_logger, resolve_feature_roles


logger = get_logger(__name__)
DATA_PATH = SCRIPT_DIR / "simulation_data.csv"
CONFIG_PATH = SCRIPT_DIR / "config.json"
FEATURE_NAMES = ["x{}".format(index) for index in range(1, 10)]
TRUE_STRATA = ("S1", "S2", "S3", "S4")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
N_SPATIAL_BINS = 10

ROUTER_PARAMETERS = {
    "max_depth": 10,
    "min_samples_split": 200,
    "proposal_tree_max_depth": 3,
    "proposal_min_samples_leaf": 100,
    "min_path_fraction": 0.1,
    "min_path_q": 0.0,
    "path_stability_top_k": 5,
    "enable_residual_restratification": True,
    "refit_final_leaves": False,
    "path_selection_mode": "all_subpaths",
    "fuzzy_membership_alpha": 5.0,
    "validation_hard_routing": True,
    "hard_pred": True,
}
TRUE_VARIABLES = {
    "S1": {"x1"},
    "S2": {"x1", "x2"},
    "S3": {"x1", "x2", "x3"},
    "S4": {"x1", "x2", "x3", "x4"},
}
TRUE_THRESHOLDS = {"x8": (0.20,), "x9": (-0.10, 0.10)}


def _cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _sanitize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return np.nan_to_num(array, nan=0.0, posinf=1e10, neginf=-1e10)


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=float).ravel()
    prediction = _sanitize(y_pred)
    if truth.size != prediction.size:
        raise ValueError("Metric input lengths differ")
    if truth.size == 0:
        return {"n_samples": 0, "mae": np.nan, "rmse": np.nan,
                "nrmse_sd": np.nan, "r2": np.nan}
    rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
    target_sd = float(np.std(truth, ddof=0))
    r2 = (float(r2_score(truth, prediction))
          if truth.size >= 2 and np.unique(truth).size >= 2 else np.nan)
    return {
        "n_samples": int(truth.size),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": rmse,
        "nrmse_sd": rmse / target_sd if target_sd > 0 else np.nan,
        "r2": r2,
    }


def _prefixed(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {"{}_{}".format(prefix, key): value for key, value in values.items()}


def _write_csv(data: Any, path: Path) -> None:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved %s rows to %s", len(frame), path)


def _spatial_blocks(frame: pd.DataFrame) -> np.ndarray:
    """Return deterministic 10 x 10 blocks from the known [-1, 1] domain."""
    coordinates = frame[["coord_x", "coord_y"]].to_numpy(dtype=float)
    if np.any(coordinates < -1.0) or np.any(coordinates > 1.0):
        raise ValueError("coord_x/coord_y must lie within the simulated [-1, 1] domain")
    bins = np.floor((coordinates + 1.0) * N_SPATIAL_BINS / 2.0).astype(int)
    bins = np.clip(bins, 0, N_SPATIAL_BINS - 1)
    return (bins[:, 1] * N_SPATIAL_BINS + bins[:, 0] + 1).astype(int)


def _prepare_data(path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError("Simulation CSV not found: {}".format(path))
    frame = pd.read_csv(path, na_values=["NA", ""])
    required = set(FEATURE_NAMES + [
        "coord_x", "coord_y", "y", "y_clean", "true_stratum", "dataset_split"
    ])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError("Missing simulation columns: {}".format(missing))
    if frame[list(required)].isna().any().any():
        raise ValueError("Required simulation columns contain missing values")
    frame = frame.copy()
    frame["dataset_split"] = frame["dataset_split"].astype(str).str.lower()
    if set(frame["dataset_split"].unique()) != {"train", "test"}:
        raise ValueError("dataset_split must contain train and test")
    if set(frame["true_stratum"].astype(str).unique()) != set(TRUE_STRATA):
        raise ValueError("Expected true strata S1--S4")
    frame["simulation_block"] = _spatial_blocks(frame)
    X = frame[FEATURE_NAMES].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    y_clean = frame["y_clean"].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(y_clean).all():
        raise ValueError("Non-finite simulation values remain")
    train = frame["dataset_split"].to_numpy() == "train"
    test = ~train
    if set(frame.loc[train, "true_stratum"]) != set(TRUE_STRATA):
        raise ValueError("Training split does not contain every true stratum")
    if set(frame.loc[test, "true_stratum"]) != set(TRUE_STRATA):
        raise ValueError("Test split does not contain every true stratum")
    logger.info(
        "Fixed split verified: train=%s, test=%s, training blocks=%s, test blocks=%s",
        int(train.sum()), int(test.sum()),
        int(frame.loc[train, "simulation_block"].nunique()),
        int(frame.loc[test, "simulation_block"].nunique()),
    )
    return frame, X, y, y_clean


def _iter_nodes(root: Any) -> List[Any]:
    result, stack = [], [root]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(sorted(node.children, key=lambda item: item.node_id, reverse=True))
    return sorted(result, key=lambda item: (item.depth, item.node_id))


def _formula(node: Any) -> str:
    try:
        return str(node.solver.program_.pretty())
    except Exception as exc:
        return "Formula unavailable: {}".format(exc)


def _formula_variables(formula: str) -> set:
    import re

    indices = {int(value) for value in re.findall(r"\b[xX]_?(\d+)\b", formula)}
    if 0 in indices:
        indices = {index + 1 for index in indices}
    return {"x{}".format(index) for index in indices if 1 <= index <= 9}


def _match_strata(true_strata: np.ndarray, leaf_ids: np.ndarray) -> Tuple[Dict[int, str], np.ndarray, Dict[str, Any]]:
    true_strata = np.asarray(true_strata, dtype=str)
    leaf_ids = np.asarray(leaf_ids, dtype=int)
    leaves = sorted(np.unique(leaf_ids).tolist())
    matrix = np.asarray([
        [np.sum((true_strata == stratum) & (leaf_ids == leaf)) for leaf in leaves]
        for stratum in TRUE_STRATA
    ], dtype=int)
    mapping = {}
    if matrix.size:
        rows, columns = linear_sum_assignment(-matrix)
        mapping = {int(leaves[column]): TRUE_STRATA[row]
                   for row, column in zip(rows, columns)}
    matched = np.asarray([mapping.get(int(leaf), "unmatched") for leaf in leaf_ids])
    summary = {
        "strata_ari": float(adjusted_rand_score(true_strata, leaf_ids)),
        "strata_nmi": float(normalized_mutual_info_score(true_strata, leaf_ids)),
        "strata_matched_accuracy": float(np.mean(matched == true_strata)),
        "leaf_to_true_mapping": json.dumps(mapping, ensure_ascii=False, sort_keys=True),
        "true_leaf_contingency": json.dumps(matrix.tolist()),
    }
    return mapping, matched, summary


def _formula_rows(router: Any, mapping: Dict[int, str], seed: int) -> List[Dict[str, Any]]:
    rows = []
    for node in sorted(router._get_all_leaves(), key=lambda item: item.node_id):
        formula = _formula(node)
        learned = _formula_variables(formula)
        stratum = mapping.get(int(node.node_id), "unmatched")
        expected = TRUE_VARIABLES.get(stratum, set())
        common, union = learned & expected, learned | expected
        precision = len(common) / float(len(learned)) if learned else np.nan
        recall = len(common) / float(len(expected)) if expected else np.nan
        rows.append({
            "model_seed": seed, "leaf_node_id": int(node.node_id),
            "leaf_depth": int(node.depth), "matched_true_stratum": stratum,
            "formula": formula, "learned_variables": json.dumps(sorted(learned)),
            "expected_variables": json.dumps(sorted(expected)),
            "variable_precision": precision, "variable_recall": recall,
            "variable_jaccard": len(common) / float(len(union)) if union else np.nan,
            "exact_active_variable_recovery": learned == expected,
        })
    return rows


def _threshold_rows(router: Any, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for node in _iter_nodes(router.root):
        classifier = node.discriminator
        if classifier is None or not node.children or not hasattr(classifier, "tree_"):
            continue
        for tree_node, feature_index in enumerate(classifier.tree_.feature):
            if feature_index < 0:
                continue
            feature = FEATURE_NAMES[int(feature_index)]
            threshold = float(classifier.tree_.threshold[tree_node])
            candidates = TRUE_THRESHOLDS.get(feature, ())
            nearest = min(candidates, key=lambda value: abs(threshold - value)) if candidates else np.nan
            rows.append({
                "model_seed": seed, "ps3r_node_id": int(node.node_id),
                "ps3r_node_depth": int(node.depth),
                "classifier_tree_node_id": int(tree_node),
                "learned_feature": feature, "learned_threshold": threshold,
                "nearest_true_threshold": nearest,
                "absolute_threshold_error": abs(threshold - nearest) if candidates else np.nan,
                "is_true_stratification_feature": bool(candidates),
            })
    return rows


def _subgroup_metric_rows(test_frame: pd.DataFrame, y_test: np.ndarray,
                          y_clean_test: np.ndarray, prediction: np.ndarray,
                          seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    block_rows, stratum_rows = [], []
    for block in sorted(test_frame["simulation_block"].unique().tolist()):
        mask = test_frame["simulation_block"].to_numpy(dtype=int) == int(block)
        row = {"model_seed": seed, "simulation_block": int(block)}
        row.update(_prefixed("noisy", _metrics(y_test[mask], prediction[mask])))
        row.update(_prefixed("clean_recovery", _metrics(y_clean_test[mask], prediction[mask])))
        block_rows.append(row)
    true_values = test_frame["true_stratum"].to_numpy(dtype=str)
    for stratum in TRUE_STRATA:
        mask = true_values == stratum
        row = {"model_seed": seed, "true_stratum": stratum}
        row.update(_prefixed("noisy", _metrics(y_test[mask], prediction[mask])))
        row.update(_prefixed("clean_recovery", _metrics(y_clean_test[mask], prediction[mask])))
        stratum_rows.append(row)
    return block_rows, stratum_rows


def _accepted_split_rows(router: Any, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for record in router.sbcv_split_records:
        validation = record["whole_tree_validation"]
        rows.append({
            "model_seed": seed, "node_id": int(record["node_id"]),
            "node_depth": int(record["node_depth"]),
            "depth_iteration": int(record["depth_iteration"]),
            "split_order": int(record["split_order"]),
            "selected_router_kind": str(record["selected_router_kind"]),
            "selected_q_path_rule": str(record["selected_q_path_rule"]),
            "selected_q_path_kind": str(record["selected_q_path_kind"]),
            "selected_q_path_depth": int(record["selected_q_path_depth"]),
            "selected_q_path_train_q": float(record["selected_q_path_train_q"]),
            "selected_q_path_validation_q": float(record["selected_q_path_validation_q"]),
            "classification_train_balanced_accuracy": record.get("classification_train_balanced_accuracy", np.nan),
            "classification_validation_balanced_accuracy": record.get("classification_validation_balanced_accuracy", np.nan),
            "classification_reliability_passed": record.get("classification_reliability_passed", False),
            "validation_before_rmse": validation["before_rmse"],
            "validation_after_rmse": validation["after_rmse"],
            "validation_rmse_improvement": validation["rmse_improvement"],
            "validation_wilcoxon_p_value": validation["wilcoxon_p_value"],
        })
    return rows


def _record_rows(records: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rows = []
    for source in records:
        row = {"model_seed": seed}
        for key, value in source.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (tuple, list, dict, np.ndarray)):
                value = json.dumps(value, ensure_ascii=False, default=lambda item: item.item() if isinstance(item, np.generic) else str(item))
            row[key] = value
        rows.append(row)
    return rows


def _growth_rows(router: Any, y_train: np.ndarray, y_test: np.ndarray,
                 y_clean_test: np.ndarray, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for snapshot in router.sbcv_growth_history:
        train_prediction = snapshot["predictions"].get("train")
        test_prediction = snapshot["predictions"].get("test")
        if train_prediction is None or test_prediction is None:
            raise RuntimeError("Growth monitoring predictions are incomplete")
        row = {
            "model_seed": seed, "growth_step": int(snapshot["growth_step"]),
            "event_type": str(snapshot["event_type"]),
            "depth_iteration": int(snapshot["depth_iteration"]),
            "n_effective_strata": int(snapshot["n_effective_strata"]),
            "accepted_node_ids": json.dumps(snapshot["accepted_node_ids"]),
        }
        row.update(_prefixed("train_noisy", _metrics(y_train, train_prediction)))
        row.update(_prefixed("test_noisy", _metrics(y_test, test_prediction)))
        row.update(_prefixed("test_clean", _metrics(y_clean_test, test_prediction)))
        rows.append(row)
    return rows


def _model_text(router: Any, seed: int) -> str:
    lines = ["Fixed 80%/20% split | model seed {}".format(seed), ""]
    split_map = {int(record["node_id"]): record for record in router.sbcv_split_records}
    for node in _iter_nodes(router.root):
        lines.append("Node {} | depth={} | type={} | samples={}".format(
            node.node_id, node.depth, "leaf" if node.is_leaf else "internal",
            len(node.sample_indices)))
        lines.append("Regression equation: {}".format(_formula(node)))
        if node.discriminator is not None and node.children:
            record = split_map[int(node.node_id)]
            lines.append("Selected router: {}".format(record["selected_router_kind"]))
            lines.append("Maximum-q path: q_train={:.8f}, q_validation={:.8f}, rule={}".format(
                record["selected_q_path_train_q"], record["selected_q_path_validation_q"],
                record["selected_q_path_rule"]))
            if hasattr(node.discriminator, "rule_text"):
                lines.append("Final rule: {}".format(node.discriminator.rule_text(FEATURE_NAMES)))
            elif hasattr(node.discriminator, "tree_"):
                lines.extend(["Final classification tree:", export_text(node.discriminator, feature_names=FEATURE_NAMES)])
        lines.append("")
    lines.append("Final leaf refit records: {} (expected 0)".format(len(router.sbcv_leaf_refit_records)))
    return "\n".join(lines)


def _build_router(input_dim: int, seed: int) -> Any:
    from ps3r.model import PS3RModel

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("experiment", {})["logdir"] = None
    _, continuous = resolve_feature_roles(FEATURE_NAMES, [])
    return PS3RModel(
        input_dim=input_dim, continuous_feature_indices=continuous,
        solver_kwargs={"config": config}, random_state=seed,
        **ROUTER_PARAMETERS
    )


def _run_seed(frame: pd.DataFrame, X: np.ndarray, y: np.ndarray,
              y_clean: np.ndarray, seed: int) -> Dict[str, Any]:
    train_mask = frame["dataset_split"].to_numpy() == "train"
    test_mask = ~train_mask
    train_indices, test_indices = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    y_clean_test = y_clean[test_indices]
    train_blocks = frame.iloc[train_indices]["simulation_block"].to_numpy(dtype=int)
    test_frame = frame.iloc[test_indices].copy()
    router = _build_router(X_train.shape[1], seed)
    router.set_sbcv_monitoring_data(train=(X_train, y_train), test=(X_test, y_test))
    started = time.time()
    router.fit(X_train, y_train, spatial_block_ids=train_blocks)
    runtime = time.time() - started
    internal_train_blocks = np.asarray(router._fixed_root_train_blocks)
    internal_validation_blocks = np.asarray(router._fixed_root_validation_blocks)
    if np.intersect1d(internal_train_blocks, internal_validation_blocks).size:
        raise RuntimeError("Internal training and validation blocks overlap")
    if not set(np.concatenate([internal_train_blocks, internal_validation_blocks])).issubset(
            set(np.unique(train_blocks))):
        raise RuntimeError("Internal validation contains a block absent from outer training")

    train_prediction = _sanitize(router.predict(X_train))
    test_prediction = _sanitize(router.predict(X_test))
    leaf_ids = np.asarray(router.predict_leaf_ids(X_test), dtype=int)
    mapping, matched, recovery = _match_strata(
        test_frame["true_stratum"].to_numpy(dtype=str), leaf_ids)
    metric = {
        "model_seed": seed, "runtime_seconds": runtime,
        "n_train": int(len(train_indices)), "n_test": int(len(test_indices)),
        "n_train_blocks": int(np.unique(train_blocks).size),
        "n_internal_train_blocks": int(len(internal_train_blocks)),
        "n_internal_validation_blocks": int(len(internal_validation_blocks)),
        "internal_block_overlap": int(np.intersect1d(
            internal_train_blocks, internal_validation_blocks).size),
        "final_tree_nodes": int(len(_iter_nodes(router.root))),
        "final_strata": int(len(router._get_all_leaves())),
        "final_leaf_refits": int(len(router.sbcv_leaf_refit_records)),
    }
    metric.update(recovery)
    metric.update(_prefixed("train_noisy", _metrics(y_train, train_prediction)))
    metric.update(_prefixed("test_noisy", _metrics(y_test, test_prediction)))
    metric.update(_prefixed("test_clean_recovery", _metrics(y_clean_test, test_prediction)))

    predictions = test_frame.copy()
    predictions["model_seed"] = seed
    predictions["ps3r_prediction"] = test_prediction
    predictions["noisy_residual"] = y_test - test_prediction
    predictions["clean_recovery_residual"] = y_clean_test - test_prediction
    predictions["routed_leaf_node_hard"] = leaf_ids
    predictions["matched_predicted_stratum"] = matched
    predictions["stratum_match_correct"] = matched == test_frame["true_stratum"].to_numpy(dtype=str)
    predictions["source_row_index"] = test_indices

    block_rows, stratum_rows = _subgroup_metric_rows(
        test_frame, y_test, y_clean_test, test_prediction, seed)

    result = {
        "metric": metric, "predictions": predictions,
        "block_rows": block_rows, "stratum_rows": stratum_rows,
        "formula_rows": _formula_rows(router, mapping, seed),
        "threshold_rows": _threshold_rows(router, seed),
        "accepted_split_rows": _accepted_split_rows(router, seed),
        "attempt_rows": _record_rows(router.sbcv_split_attempt_records, seed),
        "path_rows": _record_rows(router.sbcv_path_candidate_records, seed),
        "growth_rows": _growth_rows(router, y_train, y_test, y_clean_test, seed),
        "model_text": _model_text(router, seed),
    }
    del router
    _cleanup_memory()
    return result


def _save_results(output_dir: Path, run_id: str, results: Dict[int, Dict[str, Any]],
                  statuses: Dict[int, Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    collections = {key: [] for key in (
        "metric", "block_rows", "stratum_rows", "formula_rows", "threshold_rows", "accepted_split_rows",
        "attempt_rows", "path_rows", "growth_rows")}
    predictions = []
    for seed in sorted(results):
        result = results[seed]
        collections["metric"].append(result["metric"])
        for key in collections:
            if key != "metric":
                collections[key].extend(result[key])
        predictions.append(result["predictions"])
        _write_csv(result["predictions"], output_dir / "seed_{}_test_predictions_{}.csv".format(seed, run_id))
        (output_dir / "seed_{}_model_details_{}.txt".format(seed, run_id)).write_text(
            result["model_text"], encoding="utf-8")
    stems = {
        "metric": "fixed_split_seed_metrics", "formula_rows": "fixed_split_leaf_formula_recovery",
        "block_rows": "fixed_split_test_block_metrics",
        "stratum_rows": "fixed_split_test_true_stratum_metrics",
        "threshold_rows": "fixed_split_learned_thresholds",
        "accepted_split_rows": "fixed_split_accepted_splits",
        "attempt_rows": "fixed_split_split_attempts",
        "path_rows": "fixed_split_path_candidate_stability",
        "growth_rows": "fixed_split_growth_metrics",
    }
    for key, stem in stems.items():
        if collections[key]:
            _write_csv(collections[key], output_dir / "{}_{}.csv".format(stem, run_id))
    if predictions:
        _write_csv(pd.concat(predictions, ignore_index=True),
                   output_dir / "fixed_split_test_predictions_all_seeds_{}.csv".format(run_id))
    if collections["metric"]:
        metric_frame = pd.DataFrame(collections["metric"])
        summary = []
        for column in metric_frame.select_dtypes(include=[np.number, "bool"]).columns:
            if column == "model_seed":
                continue
            values = pd.to_numeric(metric_frame[column], errors="coerce").dropna()
            summary.append({
                "metric": column, "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "minimum": float(values.min()) if len(values) else np.nan,
                "maximum": float(values.max()) if len(values) else np.nan,
            })
        _write_csv(summary, output_dir / "fixed_split_seed_sensitivity_summary_{}.csv".format(run_id))
    _write_csv([statuses[seed] for seed in sorted(statuses)],
               output_dir / "fixed_split_seed_status_{}.csv".format(run_id))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PS3R on the fixed simulation split (no cross-validation).")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--validate-data-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds or len(seeds) > 5:
        raise ValueError("Supply between one and five distinct model seeds")
    if not 1 <= args.n_jobs <= 5:
        raise ValueError("n-jobs must be between 1 and 5")
    frame, X, y, y_clean = _prepare_data(args.data.resolve())
    if args.validate_data_only:
        logger.info("Fixed-split data and spatial-block validation completed; training skipped.")
        return

    from ps3r.runtime import require_dso
    require_dso()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir.resolve() if args.output_dir else
                  Path.cwd() / "simulation_results_{}".format(run_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "design": "fixed_80_20_split_five_seed_sensitivity_not_cross_validation",
        "dataset_design": "controlled_four_strata_with_spatial_covariates",
        "generator": str(SCRIPT_DIR / "generate_simulation_data.ipynb"),
        "data": str(args.data.resolve()), "config": str(CONFIG_PATH),
        "feature_names": FEATURE_NAMES, "spatial_grid_bins_per_axis": N_SPATIAL_BINS,
        "coordinates_used_as_model_features": False,
        "seeds": seeds, "router_parameters": ROUTER_PARAMETERS,
        "outer_test_passed_to_fit": False,
    }
    (output_dir / "run_configuration_{}.json".format(run_id)).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    results, statuses = {}, {}

    def retain(seed: int, result: Dict[str, Any]) -> None:
        results[seed] = result
        statuses[seed] = {"model_seed": seed, "status": "success", "error": ""}

    def fail(seed: int, exc: BaseException) -> None:
        logger.error("Seed %s failed: %s", seed, exc)
        logger.error(traceback.format_exc())
        statuses[seed] = {"model_seed": seed, "status": "failed",
                          "error": "{}: {}".format(type(exc).__name__, exc)}

    if args.n_jobs == 1:
        for position, seed in enumerate(seeds, 1):
            logger.info("Starting fixed-split seed %s (%s/%s)", seed, position, len(seeds))
            try:
                retain(seed, _run_seed(frame, X, y, y_clean, seed))
            except Exception as exc:
                fail(seed, exc)
            finally:
                _cleanup_memory()
    else:
        with ProcessPoolExecutor(max_workers=min(args.n_jobs, len(seeds)),
                                 mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(_run_seed, frame, X, y, y_clean, seed): seed for seed in seeds}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    retain(seed, future.result())
                except Exception as exc:
                    fail(seed, exc)
                finally:
                    _cleanup_memory()

    _save_results(output_dir, run_id, results, statuses)
    if len(results) != len(seeds):
        raise RuntimeError("Only {}/{} seeds succeeded; see status CSV".format(len(results), len(seeds)))
    logger.info("All fixed-split seeds completed: %s", output_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()
