"""Five-seed fixed-split PS3R experiment for the single-stratum SOCD function.

The train/test CSV files were split by soil profile before this script
is run.  The matching ``*_full.csv`` files are used only to recover the
``Profile`` group identifier for PS3R's internal grouped validation split.
No columns from the full files are passed to DSO or to the routing trees.

This is a fixed 80%/20% train/test experiment repeated with model seeds 0--4;
it is not five-fold cross-validation.  The outer test set never enters fit.
"""

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import argparse
import datetime
import gc
import hashlib
import json
import multiprocessing as mp
import pickle
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import export_text


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils_dso import get_logger, resolve_feature_roles


logger = get_logger(__name__)

TRAIN_DATA_PATH = SCRIPT_DIR / "train_datasetv10_profile80_cf1.csv"
TEST_DATA_PATH = SCRIPT_DIR / "test_datasetv10_profile20_cf1.csv"
TRAIN_PROFILE_SOURCE_PATH = SCRIPT_DIR / "train_datasetv10_profile80_full.csv"
TEST_PROFILE_SOURCE_PATH = SCRIPT_DIR / "test_datasetv10_profile20_full.csv"
CONFIG_PATH = SCRIPT_DIR / "config.json"

TARGET_COLUMN = "socd_calculate"
FEATURE_NAMES = ["BD_g_cm3", "Depth_cm", "SOC_g_kg", "cf"]
PROFILE_COLUMN = "Profile"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

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

TRUE_FORMULA = (
    "socd_calculate = BD_g_cm3 * Depth_cm * SOC_g_kg "
    "* (1 - cf / 100) / 100"
)


def _cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return np.nan_to_num(array, nan=0.0, posinf=1e10, neginf=-1e10)


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=float).ravel()
    prediction = _sanitize(y_pred)
    if truth.size != prediction.size:
        raise ValueError("Metric input lengths differ")
    if truth.size == 0:
        return {
            "n_samples": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "nrmse_sd": np.nan,
            "mape_percent": np.nan,
            "r2": np.nan,
        }
    rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
    target_sd = float(np.std(truth, ddof=0))
    nonzero = truth != 0.0
    mape = (
        float(np.mean(np.abs((truth[nonzero] - prediction[nonzero]) / truth[nonzero])) * 100.0)
        if np.any(nonzero)
        else np.nan
    )
    return {
        "n_samples": int(truth.size),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": rmse,
        "nrmse_sd": rmse / target_sd if target_sd > 0.0 else np.nan,
        "mape_percent": mape,
        "r2": (
            float(r2_score(truth, prediction))
            if truth.size >= 2 and np.unique(truth).size >= 2
            else np.nan
        ),
    }


def _prefixed(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {"{}_{}".format(prefix, key): value for key, value in values.items()}


def _write_csv(data: Any, path: Path) -> None:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved %s rows to %s", len(frame), path)


def _read_primary_and_profiles(
    data_path: Path, profile_source_path: Path, dataset_name: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError("{} data not found: {}".format(dataset_name, data_path))
    if not profile_source_path.exists():
        raise FileNotFoundError(
            "{} profile source not found: {}".format(dataset_name, profile_source_path)
        )

    frame = pd.read_csv(data_path, na_values=["NA", ""])
    profile_source = pd.read_csv(
        profile_source_path,
        usecols=[PROFILE_COLUMN] + FEATURE_NAMES[:-1],
        na_values=["NA", ""],
    )
    required = [TARGET_COLUMN] + FEATURE_NAMES
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError("{} data missing columns: {}".format(dataset_name, missing))
    if len(frame) != len(profile_source):
        raise ValueError(
            "{} primary/profile row counts differ: {} versus {}".format(
                dataset_name, len(frame), len(profile_source)
            )
        )
    if frame[required].isna().any().any():
        raise ValueError("{} primary data contain missing values".format(dataset_name))
    if profile_source.isna().any().any():
        raise ValueError("{} profile source contains missing values".format(dataset_name))

    for feature in FEATURE_NAMES[:-1]:
        primary_values = frame[feature].to_numpy(dtype=float)
        source_values = profile_source[feature].to_numpy(dtype=float)
        if not np.allclose(primary_values, source_values, rtol=0.0, atol=1e-12):
            raise ValueError(
                "{} row alignment failed for feature {}".format(dataset_name, feature)
            )

    profiles = profile_source[PROFILE_COLUMN].to_numpy()
    if np.unique(profiles).size < 2:
        raise ValueError("{} requires at least two profile groups".format(dataset_name))
    return frame, profiles


def _prepare_data(
    train_data_path: Path,
    test_data_path: Path,
    train_profile_source_path: Path,
    test_profile_source_path: Path,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    train_frame, train_profiles = _read_primary_and_profiles(
        train_data_path, train_profile_source_path, "training"
    )
    test_frame, test_profiles = _read_primary_and_profiles(
        test_data_path, test_profile_source_path, "test"
    )

    overlap = np.intersect1d(np.unique(train_profiles), np.unique(test_profiles))
    if overlap.size:
        raise ValueError(
            "Outer train/test profile leakage detected: {} overlapping profiles".format(
                overlap.size
            )
        )

    combined = pd.concat([train_frame, test_frame], ignore_index=True)
    analytical_target = (
        combined["BD_g_cm3"].to_numpy(dtype=float)
        * combined["Depth_cm"].to_numpy(dtype=float)
        * combined["SOC_g_kg"].to_numpy(dtype=float)
        * (1.0 - combined["cf"].to_numpy(dtype=float) / 100.0)
        / 100.0
    )
    recorded_target = combined[TARGET_COLUMN].to_numpy(dtype=float)
    maximum_formula_error = float(np.max(np.abs(recorded_target - analytical_target)))
    if maximum_formula_error > 1e-6:
        raise ValueError(
            "SOCD target does not match the documented analytical formula; "
            "maximum absolute error={:.9g}".format(maximum_formula_error)
        )

    X_train = train_frame[FEATURE_NAMES].to_numpy(dtype=float)
    y_train = train_frame[TARGET_COLUMN].to_numpy(dtype=float)
    X_test = test_frame[FEATURE_NAMES].to_numpy(dtype=float)
    y_test = test_frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not all(np.isfinite(values).all() for values in (X_train, y_train, X_test, y_test)):
        raise ValueError("Non-finite SOCD values remain")

    logger.info(
        "SOCD fixed split verified: train=%s rows/%s profiles, "
        "test=%s rows/%s profiles, profile overlap=0, formula max error=%.3g",
        len(train_frame),
        np.unique(train_profiles).size,
        len(test_frame),
        np.unique(test_profiles).size,
        maximum_formula_error,
    )
    return train_frame, test_frame, train_profiles, test_profiles, X_train, y_train, X_test, y_test


def _iter_nodes(root: Any) -> List[Any]:
    result = []
    stack = [root]
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


def _record_rows(records: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rows = []
    for source in records:
        row = {"model_seed": seed}
        for key, value in source.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (tuple, list, dict, np.ndarray)):
                value = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=lambda item: item.item()
                    if isinstance(item, np.generic)
                    else str(item),
                )
            row[key] = value
        rows.append(row)
    return rows


def _node_rows(router: Any, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for node in _iter_nodes(router.root):
        rows.append(
            {
                "model_seed": seed,
                "node_id": int(node.node_id),
                "node_depth": int(node.depth),
                "node_type": "leaf" if node.is_leaf else "internal",
                "node_sample_count": int(len(node.sample_indices)),
                "formula": _formula(node),
            }
        )
    return rows


def _accepted_split_rows(router: Any, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for record in router.sbcv_split_records:
        validation = record["whole_tree_validation"]
        rows.append(
            {
                "model_seed": seed,
                "node_id": int(record["node_id"]),
                "node_depth": int(record["node_depth"]),
                "depth_iteration": int(record["depth_iteration"]),
                "split_order": int(record["split_order"]),
                "selected_router_kind": str(record["selected_router_kind"]),
                "selected_q_path_rule": str(record["selected_q_path_rule"]),
                "selected_q_path_kind": str(record["selected_q_path_kind"]),
                "selected_q_path_depth": int(record["selected_q_path_depth"]),
                "selected_q_path_train_q": float(record["selected_q_path_train_q"]),
                "selected_q_path_validation_q": float(
                    record["selected_q_path_validation_q"]
                ),
                "classification_train_balanced_accuracy": record.get(
                    "classification_train_balanced_accuracy", np.nan
                ),
                "classification_validation_balanced_accuracy": record.get(
                    "classification_validation_balanced_accuracy", np.nan
                ),
                "classification_reliability_passed": record.get(
                    "classification_reliability_passed", False
                ),
                "validation_before_rmse": validation["before_rmse"],
                "validation_after_rmse": validation["after_rmse"],
                "validation_rmse_improvement": validation["rmse_improvement"],
                "validation_wilcoxon_p_value": validation["wilcoxon_p_value"],
            }
        )
    return rows


def _growth_rows(router: Any, y_train: np.ndarray, y_test: np.ndarray, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for snapshot in router.sbcv_growth_history:
        train_prediction = snapshot["predictions"].get("train")
        test_prediction = snapshot["predictions"].get("test")
        if train_prediction is None or test_prediction is None:
            raise RuntimeError("Growth monitoring predictions are incomplete")
        row = {
            "model_seed": seed,
            "growth_step": int(snapshot["growth_step"]),
            "event_type": str(snapshot["event_type"]),
            "depth_iteration": int(snapshot["depth_iteration"]),
            "n_effective_strata": int(snapshot["n_effective_strata"]),
            "accepted_node_ids": json.dumps(snapshot["accepted_node_ids"]),
        }
        row.update(_prefixed("train", _metrics(y_train, train_prediction)))
        row.update(_prefixed("test", _metrics(y_test, test_prediction)))
        rows.append(row)
    return rows


def _membership_rows(
    router: Any,
    train_leaf_ids: np.ndarray,
    test_leaf_ids: np.ndarray,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = []
    for node in sorted(router._get_all_leaves(), key=lambda item: item.node_id):
        node_id = int(node.node_id)
        rows.append(
            {
                "model_seed": seed,
                "leaf_node_id": node_id,
                "leaf_depth": int(node.depth),
                "hard_train_sample_count": int(np.sum(train_leaf_ids == node_id)),
                "hard_test_sample_count": int(np.sum(test_leaf_ids == node_id)),
                "formula": _formula(node),
            }
        )
    return rows


def _model_text(router: Any, seed: int) -> str:
    lines = [
        "SOCD fixed profile split | model seed {}".format(seed),
        "True single-stratum formula: {}".format(TRUE_FORMULA),
        "Model features only: {}".format(", ".join(FEATURE_NAMES)),
        "Internal validation groups: Profile identifiers from outer training only",
        "",
    ]
    early_stop_records = list(
        getattr(router, "dso_early_stop_records", [])
    )
    if early_stop_records:
        lines.append("DSO success-based early-stop actions:")
        for record in early_stop_records:
            lines.append(
                "- event={} | node={} | child={} | nevals={} | budget={}".format(
                    record.get("event", ""),
                    record.get("node_id", ""),
                    record.get("child_label", ""),
                    record.get("nevals", ""),
                    record.get("n_samples_budget", ""),
                )
            )
        lines.append("")
    split_map = {int(record["node_id"]): record for record in router.sbcv_split_records}
    for node in _iter_nodes(router.root):
        lines.append(
            "Node {} | depth={} | type={} | samples={}".format(
                node.node_id,
                node.depth,
                "leaf" if node.is_leaf else "internal",
                len(node.sample_indices),
            )
        )
        lines.append("Regression equation: {}".format(_formula(node)))
        if node.discriminator is not None and node.children:
            record = split_map[int(node.node_id)]
            lines.append("Selected router: {}".format(record["selected_router_kind"]))
            lines.append(
                "Maximum-q path: q_train={:.8f}, q_validation={:.8f}, rule={}".format(
                    record["selected_q_path_train_q"],
                    record["selected_q_path_validation_q"],
                    record["selected_q_path_rule"],
                )
            )
            if hasattr(node.discriminator, "rule_text"):
                lines.append(
                    "Final rule: {}".format(node.discriminator.rule_text(FEATURE_NAMES))
                )
            elif hasattr(node.discriminator, "tree_"):
                lines.extend(
                    [
                        "Final classification tree:",
                        export_text(node.discriminator, feature_names=FEATURE_NAMES),
                    ]
                )
        lines.append("")
    lines.append(
        "Final leaf refit records: {} (expected 0)".format(
            len(router.sbcv_leaf_refit_records)
        )
    )
    return "\n".join(lines)


def _build_router(input_dim: int, seed: int) -> Any:
    from model_ps3r import PS3RModel

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("experiment", {})["logdir"] = None
    _, continuous = resolve_feature_roles(FEATURE_NAMES, [])
    return PS3RModel(
        input_dim=input_dim,
        continuous_feature_indices=continuous,
        solver_kwargs={"config": config},
        random_state=seed,
        **ROUTER_PARAMETERS
    )


def _run_seed(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_profiles: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    _cleanup_memory()
    router = _build_router(X_train.shape[1], seed)
    router.set_sbcv_monitoring_data(train=(X_train, y_train), test=(X_test, y_test))
    started = time.time()
    router.fit(X_train, y_train, spatial_block_ids=train_profiles)
    runtime = time.time() - started
    logger.info(
        "Seed %s: router.fit returned; starting final train/test predictions.", seed
    )

    internal_train_profiles = np.asarray(router._fixed_root_train_blocks)
    internal_validation_profiles = np.asarray(router._fixed_root_validation_blocks)
    if np.intersect1d(internal_train_profiles, internal_validation_profiles).size:
        raise RuntimeError("Internal training and validation profiles overlap")
    if not set(
        np.concatenate([internal_train_profiles, internal_validation_profiles]).tolist()
    ).issubset(set(np.unique(train_profiles).tolist())):
        raise RuntimeError("Internal validation contains a profile absent from outer training")

    train_prediction = _sanitize(router.predict(X_train))
    logger.info("Seed %s: final training prediction completed.", seed)
    test_prediction = _sanitize(router.predict(X_test))
    logger.info("Seed %s: final test prediction completed.", seed)
    train_leaf_ids = np.asarray(router.predict_leaf_ids(X_train), dtype=int)
    test_leaf_ids = np.asarray(router.predict_leaf_ids(X_test), dtype=int)
    logger.info("Seed %s: hard leaf routing completed; assembling diagnostics.", seed)

    metric = {
        "model_seed": seed,
        "runtime_seconds": runtime,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_outer_train_profiles": int(np.unique(train_profiles).size),
        "n_internal_train_profiles": int(len(internal_train_profiles)),
        "n_internal_validation_profiles": int(len(internal_validation_profiles)),
        "internal_profile_overlap": int(
            np.intersect1d(internal_train_profiles, internal_validation_profiles).size
        ),
        "final_tree_nodes": int(len(_iter_nodes(router.root))),
        "final_strata": int(len(router._get_all_leaves())),
        "remained_single_stratum": bool(len(router._get_all_leaves()) == 1),
        "final_leaf_refits": int(len(router.sbcv_leaf_refit_records)),
        "dso_early_stop_events": int(
            len(getattr(router, "dso_early_stop_records", []))
        ),
        "root_dso_early_stopped": bool(
            any(
                record.get("event") == "root_frozen"
                for record in getattr(router, "dso_early_stop_records", [])
            )
        ),
        "child_dso_early_stop_promotions": int(
            sum(
                record.get("event") == "child_promoted_to_parent"
                for record in getattr(router, "dso_early_stop_records", [])
            )
        ),
    }
    metric.update(_prefixed("train", _metrics(y_train, train_prediction)))
    metric.update(_prefixed("test", _metrics(y_test, test_prediction)))

    train_predictions = train_frame.copy()
    train_predictions["model_seed"] = seed
    train_predictions["ps3r_prediction"] = train_prediction
    train_predictions["residual"] = y_train - train_prediction
    train_predictions["routed_leaf_node_hard"] = train_leaf_ids

    test_predictions = test_frame.copy()
    test_predictions["model_seed"] = seed
    test_predictions["ps3r_prediction"] = test_prediction
    test_predictions["residual"] = y_test - test_prediction
    test_predictions["routed_leaf_node_hard"] = test_leaf_ids

    result = {
        "metric": metric,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "node_rows": _node_rows(router, seed),
        "membership_rows": _membership_rows(
            router, train_leaf_ids, test_leaf_ids, seed
        ),
        "accepted_split_rows": _accepted_split_rows(router, seed),
        "attempt_rows": _record_rows(router.sbcv_split_attempt_records, seed),
        "path_rows": _record_rows(router.sbcv_path_candidate_records, seed),
        "early_stop_rows": _record_rows(
            getattr(router, "dso_early_stop_records", []), seed
        ),
        "growth_rows": _growth_rows(router, y_train, y_test, seed),
        "model_text": _model_text(router, seed),
    }
    del router
    logger.info(
        "Seed %s: diagnostics assembled; returning result before child exit.", seed
    )
    return result


def _run_seed_safe(payload: Tuple[Any, ...]) -> Dict[str, Any]:
    seed = int(payload[-1])
    try:
        return {"seed": seed, "status": "success", "result": _run_seed(*payload)}
    except Exception as exc:
        return {
            "seed": seed,
            "status": "failed",
            "error": "{}: {}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }


def _ignore_keyboard_interrupt_in_worker() -> None:
    """Let only the parent process handle Ctrl+C on Windows."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _write_seed_outcome_file(result_path: str, payload: Tuple[Any, ...]) -> None:
    """Write one seed outcome outside the synchronized project directory."""
    _ignore_keyboard_interrupt_in_worker()
    path = Path(result_path)
    temporary_path = path.with_suffix(".tmp")
    outcome = _run_seed_safe(payload)
    with temporary_path.open("wb") as handle:
        pickle.dump(outcome, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary_path), str(path))


def _write_seed_outcome_task(task: Tuple[str, Tuple[Any, ...]]) -> Dict[str, Any]:
    """Pool task that transfers only a small file reference to the parent."""
    result_path, payload = task
    seed = int(payload[-1])
    try:
        _write_seed_outcome_file(result_path, payload)
        return {"seed": seed, "status": "result_file", "result_path": result_path}
    except Exception as exc:
        return {
            "seed": seed,
            "status": "failed",
            "error": "{}: {}".format(type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }


def _load_seed_outcome_file(reference: Dict[str, Any]) -> Dict[str, Any]:
    if reference.get("status") != "result_file":
        return reference
    result_path = Path(reference["result_path"])
    if not result_path.exists():
        return {
            "seed": int(reference["seed"]),
            "status": "failed",
            "error": "FileNotFoundError: completed worker result file is missing",
            "traceback": "",
        }
    with result_path.open("rb") as handle:
        return pickle.load(handle)


def _seed_checkpoint_path(output_dir: Path, run_id: str, seed: int) -> Path:
    return output_dir / "seed_{}_result_checkpoint_{}.pkl".format(seed, run_id)


def _write_seed_checkpoint(
    output_dir: Path, run_id: str, seed: int, result: Dict[str, Any]
) -> Path:
    """Persist one completed seed atomically before releasing parent memory."""
    path = _seed_checkpoint_path(output_dir, run_id, seed)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary_path), str(path))
    return path


def _load_completed_checkpoints(
    output_dir: Path, run_id: str, seeds: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    results = {}
    for seed in sorted(seeds):
        path = _seed_checkpoint_path(output_dir, run_id, seed)
        if not path.exists():
            raise FileNotFoundError(
                "Completed seed checkpoint is missing: {}".format(path)
            )
        with path.open("rb") as handle:
            results[int(seed)] = pickle.load(handle)
    return results


def _terminate_process(process: Any) -> None:
    """Stop one child process without leaving an orphan behind."""
    if process.is_alive():
        process.terminate()
    process.join(timeout=10.0)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if kill is not None:
            kill()
            process.join(timeout=5.0)


def _run_one_seed_in_fresh_process(
    context: Any, payload: Tuple[Any, ...]
) -> Dict[str, Any]:
    """Use a non-respawning child and a temporary file instead of a Pipe."""
    seed = int(payload[-1])
    with tempfile.TemporaryDirectory(prefix="ps3r_socd_seed_{}__".format(seed)) as temp_dir:
        result_path = Path(temp_dir) / "seed_outcome.pkl"
        process = context.Process(
            target=_write_seed_outcome_file,
            args=(str(result_path), payload),
        )
        process.start()
        try:
            while process.is_alive():
                process.join(timeout=0.5)
            process.join()
            if process.exitcode != 0 or not result_path.exists():
                return {
                    "seed": seed,
                    "status": "failed",
                    "error": (
                        "ChildProcessError: seed worker exited with code {} "
                        "without a complete result file"
                    ).format(process.exitcode),
                    "traceback": "",
                }
            with result_path.open("rb") as handle:
                return pickle.load(handle)
        except KeyboardInterrupt:
            _terminate_process(process)
            raise


def _save_results(
    output_dir: Path,
    run_id: str,
    results: Dict[int, Dict[str, Any]],
    statuses: Dict[int, Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_keys = (
        "metric",
        "node_rows",
        "membership_rows",
        "accepted_split_rows",
        "attempt_rows",
        "path_rows",
        "early_stop_rows",
        "growth_rows",
    )
    collections = {key: [] for key in collection_keys}
    all_train_predictions = []
    all_test_predictions = []

    for seed in sorted(results):
        result = results[seed]
        collections["metric"].append(result["metric"])
        for key in collection_keys:
            if key != "metric":
                collections[key].extend(result.get(key, []))
        all_train_predictions.append(result["train_predictions"])
        all_test_predictions.append(result["test_predictions"])
        _write_csv(
            result["train_predictions"],
            output_dir / "seed_{}_train_predictions_{}.csv".format(seed, run_id),
        )
        _write_csv(
            result["test_predictions"],
            output_dir / "seed_{}_test_predictions_{}.csv".format(seed, run_id),
        )
        (output_dir / "seed_{}_model_details_{}.txt".format(seed, run_id)).write_text(
            result["model_text"], encoding="utf-8"
        )

    stems = {
        "metric": "fixed_split_seed_metrics",
        "node_rows": "fixed_split_node_models",
        "membership_rows": "fixed_split_leaf_membership_counts",
        "accepted_split_rows": "fixed_split_accepted_splits",
        "attempt_rows": "fixed_split_split_attempts",
        "path_rows": "fixed_split_path_candidate_stability",
        "early_stop_rows": "fixed_split_dso_early_stops",
        "growth_rows": "fixed_split_growth_metrics",
    }
    for key, stem in stems.items():
        if collections[key]:
            _write_csv(
                collections[key], output_dir / "{}_{}.csv".format(stem, run_id)
            )

    if all_train_predictions:
        _write_csv(
            pd.concat(all_train_predictions, ignore_index=True),
            output_dir / "fixed_split_train_predictions_all_seeds_{}.csv".format(run_id),
        )
    if all_test_predictions:
        _write_csv(
            pd.concat(all_test_predictions, ignore_index=True),
            output_dir / "fixed_split_test_predictions_all_seeds_{}.csv".format(run_id),
        )

    if collections["metric"]:
        metric_frame = pd.DataFrame(collections["metric"])
        summary = []
        for column in metric_frame.select_dtypes(include=[np.number, "bool"]).columns:
            if column == "model_seed":
                continue
            values = pd.to_numeric(metric_frame[column], errors="coerce").dropna()
            summary.append(
                {
                    "metric": column,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "minimum": float(values.min()) if len(values) else np.nan,
                    "maximum": float(values.max()) if len(values) else np.nan,
                }
            )
        _write_csv(
            summary,
            output_dir / "fixed_split_seed_sensitivity_summary_{}.csv".format(run_id),
        )

    _write_csv(
        [statuses[seed] for seed in sorted(statuses)],
        output_dir / "fixed_split_seed_status_{}.csv".format(run_id),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PS3R on the fixed-profile SOCD function-recovery split "
            "for one to five model seeds."
        )
    )
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA_PATH)
    parser.add_argument(
        "--train-profile-source", type=Path, default=TRAIN_PROFILE_SOURCE_PATH
    )
    parser.add_argument(
        "--test-profile-source", type=Path, default=TEST_PROFILE_SOURCE_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--validate-data-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds or len(seeds) > 5:
        raise ValueError("Supply between one and five distinct model seeds")
    if not 1 <= args.n_jobs <= 5:
        raise ValueError("n-jobs must be between 1 and 5")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Configuration file not found: {}".format(CONFIG_PATH))

    prepared = _prepare_data(
        args.train_data.resolve(),
        args.test_data.resolve(),
        args.train_profile_source.resolve(),
        args.test_profile_source.resolve(),
    )
    (
        train_frame,
        test_frame,
        train_profiles,
        _test_profiles,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepared
    if args.validate_data_only:
        logger.info("SOCD data, formula, row alignment, and profile isolation validated; training skipped.")
        return

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else SCRIPT_DIR / "socd_results_{}".format(run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "train_data": args.train_data.resolve(),
        "test_data": args.test_data.resolve(),
        "train_profile_source": args.train_profile_source.resolve(),
        "test_profile_source": args.test_profile_source.resolve(),
        "config": CONFIG_PATH.resolve(),
    }
    metadata = {
        "design": "fixed_profile_80_20_split_five_seed_sensitivity_not_cross_validation",
        "purpose": "single_stratum_socd_function_recovery",
        "true_formula": TRUE_FORMULA,
        "feature_names": FEATURE_NAMES,
        "target_column": TARGET_COLUMN,
        "profile_column_used_only_for_internal_group_validation": PROFILE_COLUMN,
        "full_file_columns_used_as_model_features": [],
        "outer_test_passed_to_fit": False,
        "seeds": seeds,
        "n_jobs": args.n_jobs,
        "router_parameters": ROUTER_PARAMETERS,
        "inputs": {name: str(path) for name, path in input_paths.items()},
        "input_sha256": {name: _sha256(path) for name, path in input_paths.items()},
    }
    (output_dir / "run_configuration_{}.json".format(run_id)).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    common_payload = (
        train_frame,
        test_frame,
        train_profiles,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    tasks = [common_payload + (seed,) for seed in seeds]
    successful_seeds = set()
    statuses = {
        seed: {
            "model_seed": seed,
            "status": "pending",
            "error": "",
            "traceback": "",
        }
        for seed in seeds
    }
    context = mp.get_context("spawn")

    def retain_outcome(outcome: Dict[str, Any], completed: int) -> None:
        seed = int(outcome["seed"])
        if outcome["status"] == "success":
            result = outcome.pop("result")
            checkpoint_path = _write_seed_checkpoint(
                output_dir, run_id, seed, result
            )
            statuses[seed] = {
                "model_seed": seed,
                "status": "success",
                "error": "",
                "traceback": "",
            }
            _save_results(output_dir, run_id, {seed: result}, statuses)
            successful_seeds.add(seed)
            logger.info(
                "Seed %s completed (%s/%s); checkpoint=%s",
                seed,
                completed,
                len(seeds),
                checkpoint_path,
            )
            del result
        else:
            statuses[seed] = {
                "model_seed": seed,
                "status": "failed",
                "error": outcome["error"],
                "traceback": outcome["traceback"],
            }
            logger.error("Seed %s failed: %s", seed, outcome["error"])
            logger.error("%s", outcome["traceback"])
            _save_results(output_dir, run_id, {}, statuses)

    active_seed = None
    pool = None
    try:
        if args.n_jobs == 1:
            for completed, payload in enumerate(tasks, 1):
                active_seed = int(payload[-1])
                statuses[active_seed]["status"] = "running"
                logger.info(
                    "Starting SOCD seed %s (%s/%s) in a fresh process.",
                    active_seed,
                    completed,
                    len(seeds),
                )
                outcome = _run_one_seed_in_fresh_process(context, payload)
                retain_outcome(outcome, completed)
                active_seed = None
        else:
            for seed in seeds:
                statuses[seed]["status"] = "running"
            with tempfile.TemporaryDirectory(prefix="ps3r_socd_parallel_") as temp_dir:
                file_tasks = [
                    (
                        str(Path(temp_dir) / "seed_{}_outcome.pkl".format(payload[-1])),
                        payload,
                    )
                    for payload in tasks
                ]
                pool = context.Pool(
                    processes=min(args.n_jobs, len(seeds)),
                    initializer=_ignore_keyboard_interrupt_in_worker,
                    maxtasksperchild=1,
                )
                completed = 0
                try:
                    for reference in pool.imap_unordered(
                        _write_seed_outcome_task, file_tasks, chunksize=1
                    ):
                        completed += 1
                        retain_outcome(
                            _load_seed_outcome_file(reference), completed
                        )
                except BaseException:
                    pool.terminate()
                    pool.join()
                    pool = None
                    raise
                else:
                    pool.close()
                    pool.join()
                    pool = None
    except KeyboardInterrupt:
        if pool is not None:
            pool.terminate()
            pool.join()
        for seed in seeds:
            if statuses[seed]["status"] == "running":
                statuses[seed] = {
                    "model_seed": seed,
                    "status": "interrupted",
                    "error": "KeyboardInterrupt",
                    "traceback": "",
                }
        if active_seed is not None and statuses[active_seed]["status"] != "success":
            statuses[active_seed]["status"] = "interrupted"
            statuses[active_seed]["error"] = "KeyboardInterrupt"
        _save_results(output_dir, run_id, {}, statuses)
        logger.warning(
            "Interrupted by user. Child processes were terminated and completed "
            "seed checkpoints were retained in %s",
            output_dir,
        )
        raise SystemExit(130)

    if len(successful_seeds) != len(seeds):
        raise RuntimeError(
            "Only {}/{} seeds succeeded; see status CSV".format(
                len(successful_seeds), len(seeds)
            )
        )
    logger.info("All seed training completed; assembling aggregate CSV files.")
    aggregate_results = _load_completed_checkpoints(
        output_dir, run_id, successful_seeds
    )
    _save_results(output_dir, run_id, aggregate_results, statuses)
    del aggregate_results
    logger.info("All SOCD fixed-split seeds completed: %s", output_dir)


if __name__ == "__main__":
    mp.freeze_support()
    main()
