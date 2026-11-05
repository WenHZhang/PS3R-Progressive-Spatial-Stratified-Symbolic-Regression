"""Five-fold spatial block cross-validation for the PS3R BD model.

This script uses ``SOCS_V10_spatial_block_5fold_354571m.csv`` from the
same ``bd`` folder. In each run, one spatial fold is held out for testing and the
other four folds are used for training. A single fixed model seed is used;
the five runs correspond to spatial folds, not five random-seed repetitions.

Outputs include sample predictions, fold and block metrics, the residual
regression tree, maximum-q binary path, final routing diagnostics for each
accepted node split, and performance after every split in its actual training
order, after every completed depth iteration, and for the final model.
Rejected split attempts and the top training-q paths' validation-q stability
and sample/block coverage are also written to separate CSV files.

The shared ``model_ps3r`` module supplies the PS3R model. DSO settings are
specified independently for each outer fold near the top of this file. Final
leaf refitting is disabled.
"""

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ["KMP_WARNINGS"] = "0"

import argparse
import datetime
import gc
import json
import sys
import multiprocessing as mp
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.tree import export_text

SCRIPT_DIR = Path(__file__).resolve().parent

from ps3r.utils import get_logger, resolve_feature_roles

logger = get_logger(__name__)
DATA_PATH = SCRIPT_DIR / "SOCS_V10_spatial_block_5fold_354571m.csv"
CONFIG_PATH = SCRIPT_DIR / "config.json"

TARGET_COLUMN = "BD_g_cm3"
FOLD_COLUMN = "fold"
BLOCK_COLUMN = "block"
MODEL_SEED = 0

ROUTER_PARAMETERS = {
    "max_depth": 10,
    "min_samples_split": 50,
    "proposal_tree_max_depth": 3,
    "proposal_min_samples_leaf": 50,
    "min_path_fraction": 0.1,
    "min_path_q": 0.0,
    "path_stability_top_k": 5,
    "enable_residual_restratification": True,
    "refit_final_leaves": False,
    "path_selection_mode": "all_subpaths",
    "fuzzy_membership_alpha": 5.0,
    "validation_hard_routing": False,
    "hard_pred": False,
}

DSO_N_SAMPLES = 50000
FOLD_DSO_PARAMETERS = {
    1: {
        "num_layers": 1,
        "num_units": 64,
        "learning_rate": 0.0003,
        "batch_size": 250,
        "entropy_weight": 0.01,
        "epsilon": 0.05,
    },
    2: {
        "num_layers": 2,
        "num_units": 32,
        "learning_rate": 0.001,
        "batch_size": 500,
        "entropy_weight": 0.01,
        "epsilon": 0.10,
    },
    3: {
        "num_layers": 1,
        "num_units": 64,
        "learning_rate": 0.0003,
        "batch_size": 100,
        "entropy_weight": 0.01,
        "epsilon": 0.10,
    },
    4: {
        "num_layers": 1,
        "num_units": 64,
        "learning_rate": 0.0005,
        "batch_size": 500,
        "entropy_weight": 0.01,
        "epsilon": 0.15,
    },
    5: {
        "num_layers": 1,
        "num_units": 128,
        "learning_rate": 0.0003,
        "batch_size": 100,
        "entropy_weight": 0.15,
        "epsilon": 0.05,
    },
}

FEATURE_NAMES = [
    "Depth_cm",
    "SOC_g_kg",
    "pH",
    "Sand_%",
    "Silt_%",
    "Clay_%",
    "DEM_m",
    "NDVI",
    "MAT",
    "MAP_mm",
    "PET_mm",
    "AI",
    "CLCD_Forest",
    "CLCD_Grassland",
    "CLCD_Other",
]
CATEGORICAL_FEATURE_NAMES = [
    "CLCD_Forest",
    "CLCD_Grassland",
    "CLCD_Other",
]


def _configure_worker_threads():
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass


def _cleanup_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _sanitize_predictions(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return np.nan_to_num(array, nan=0.0, posinf=1e10, neginf=-1e10)


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float).ravel()
    y_pred_array = _sanitize_predictions(y_pred)
    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            f"Metric input lengths differ: y_true={len(y_true_array)}, "
            f"y_pred={len(y_pred_array)}"
        )
    if len(y_true_array) == 0:
        return {
            "n_samples": 0,
            "mae": np.nan,
            "mape_percent": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    nonzero_mask = y_true_array != 0
    mape = (
        float(
            np.mean(
                np.abs(
                    (y_true_array[nonzero_mask] - y_pred_array[nonzero_mask])
                    / y_true_array[nonzero_mask]
                )
            )
            * 100.0
        )
        if np.any(nonzero_mask)
        else np.nan
    )
    r2 = (
        float(r2_score(y_true_array, y_pred_array))
        if len(y_true_array) >= 2 and np.unique(y_true_array).size >= 2
        else np.nan
    )
    return {
        "n_samples": int(len(y_true_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "mape_percent": mape,
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": r2,
    }


def _prefix_metrics(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _write_csv(data: Any, path: Path):
    if isinstance(data, pd.DataFrame):
        frame = data
    else:
        frame = pd.DataFrame(data)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(frame)} rows to {path}")


def _prepare_data(data_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not data_path.exists():
        raise FileNotFoundError(f"Spatial CV data not found: {data_path}")

    frame = pd.read_csv(data_path, na_values=["NA", ""])
    required_columns = {
        TARGET_COLUMN,
        FOLD_COLUMN,
        BLOCK_COLUMN,
        "CLCD",
        *FEATURE_NAMES[:12],
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    if frame.isna().any().any():
        missing = frame.isna().sum()
        raise ValueError(
            "Input still contains missing values: "
            + str(missing[missing > 0].to_dict())
        )
    if len(frame) != 19978:
        raise ValueError(f"Expected 19978 rows, found {len(frame)}")

    frame[FOLD_COLUMN] = frame[FOLD_COLUMN].astype(int)
    frame[BLOCK_COLUMN] = frame[BLOCK_COLUMN].astype(int)
    folds = sorted(frame[FOLD_COLUMN].unique().tolist())
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected folds [1, 2, 3, 4, 5], found {folds}")
    if (frame.groupby(BLOCK_COLUMN)[FOLD_COLUMN].nunique() != 1).any():
        raise ValueError("At least one spatial block belongs to multiple folds")

    clcd = frame["CLCD"].astype(str)
    allowed_clcd = {"Cropland", "Forest", "Grassland", "Other"}
    unexpected_clcd = sorted(set(clcd.unique()) - allowed_clcd)
    if unexpected_clcd:
        raise ValueError(f"Unexpected CLCD categories: {unexpected_clcd}")
    frame = frame.copy()
    frame["CLCD_Forest"] = (clcd == "Forest").astype(int)
    frame["CLCD_Grassland"] = (clcd == "Grassland").astype(int)
    frame["CLCD_Other"] = (clcd == "Other").astype(int)

    X = frame[FEATURE_NAMES].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite model inputs remain after preprocessing")

    logger.info(
        f"Loaded {len(frame)} samples, {frame[BLOCK_COLUMN].nunique()} blocks, "
        f"fold counts={frame[FOLD_COLUMN].value_counts().sort_index().to_dict()}"
    )
    return frame, X, y


def _solver_config_for_fold(fold: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the template and apply the DSO overrides for one outer fold."""
    if fold not in FOLD_DSO_PARAMETERS:
        raise KeyError(f"No DSO parameters configured for fold {fold}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"DSO configuration not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        solver_config = json.load(config_file)
    parameters = dict(FOLD_DSO_PARAMETERS[fold])

    policy = solver_config.setdefault("policy", {})
    policy["num_layers"] = int(parameters["num_layers"])
    policy["num_units"] = int(parameters["num_units"])

    optimizer = solver_config.setdefault("policy_optimizer", {})
    optimizer["learning_rate"] = float(parameters["learning_rate"])
    optimizer["entropy_weight"] = float(parameters["entropy_weight"])

    training = solver_config.setdefault("training", {})
    training["n_samples"] = int(DSO_N_SAMPLES)
    training["batch_size"] = int(parameters["batch_size"])
    training["epsilon"] = float(parameters["epsilon"])

    solver_config.setdefault("experiment", {})["logdir"] = None
    return solver_config, parameters


def _iter_nodes(root: Any) -> List[Any]:
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        children = sorted(
            node.children,
            key=lambda child: (
                child.split_label if child.split_label is not None else -1,
                child.node_id,
            ),
            reverse=True,
        )
        stack.extend(children)
    return sorted(nodes, key=lambda node: (node.depth, node.node_id))


def _child_map(node: Any) -> Dict[int, Any]:
    mapping = {}
    for index, child in enumerate(node.children):
        label = index if child.split_label is None else int(child.split_label)
        mapping[label] = child
    return mapping


def _discriminator_labels(discriminator: Any, X: np.ndarray) -> np.ndarray:
    labels = np.asarray(discriminator.predict(X))
    if np.issubdtype(labels.dtype, np.floating):
        labels = np.rint(labels)
    return labels.astype(int).ravel()


def _route_indices_by_node(router: Any, X: np.ndarray) -> Dict[int, np.ndarray]:
    routed: Dict[int, List[int]] = {}
    stack = [(router.root, np.arange(len(X), dtype=int))]
    while stack:
        node, indices = stack.pop()
        routed.setdefault(int(node.node_id), []).extend(indices.tolist())
        if (
            indices.size == 0
            or node.is_leaf
            or node.discriminator is None
            or not node.children
        ):
            continue
        predicted_labels = _discriminator_labels(node.discriminator, X[indices])
        mapping = _child_map(node)
        for label in np.unique(predicted_labels):
            child = mapping.get(int(label))
            if child is None:
                continue
            child_indices = indices[predicted_labels == label]
            if child_indices.size:
                stack.append((child, child_indices))
    return {
        node_id: np.asarray(indices, dtype=int)
        for node_id, indices in routed.items()
    }


def _node_formula(node: Any) -> str:
    try:
        return str(node.solver.program_.pretty())
    except Exception as exc:
        return f"Formula unavailable: {exc}"


def _proposal_tree_regression_metrics(
    router: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
) -> List[Dict[str, Any]]:
    """Evaluate each proposal tree against its signed parent-DSO residual."""
    test_routes = _route_indices_by_node(router, X_test)
    nodes_by_id = {int(node.node_id): node for node in _iter_nodes(router.root)}
    rows = []
    for record in router.sbcv_split_records:
        node_id = int(record["node_id"])
        fit_indices = np.asarray(record["proposal_train_indices"], dtype=int)
        validation_indices = np.asarray(
            record["proposal_validation_indices"], dtype=int
        )
        test_indices = test_routes.get(node_id, np.array([], dtype=int))
        proposal_tree = record["proposal_tree"]
        parent_node = nodes_by_id[node_id]

        fit_targets = np.asarray(
            record["proposal_parent_residuals_train"], dtype=float
        ).ravel()
        validation_parent_predictions = _sanitize_predictions(
            parent_node.solver.predict(
                router._select_continuous_features(
                    X_train[validation_indices]
                )
            )
        )
        validation_targets = (
            np.asarray(y_train[validation_indices], dtype=float).ravel()
            - validation_parent_predictions
        )
        if test_indices.size:
            test_parent_predictions = _sanitize_predictions(
                parent_node.solver.predict(
                    router._select_continuous_features(X_test[test_indices])
                )
            )
            test_targets = (
                np.asarray(y_test[test_indices], dtype=float).ravel()
                - test_parent_predictions
            )
        else:
            test_targets = np.array([], dtype=float)

        fit_predictions = _sanitize_predictions(
            proposal_tree.predict(X_train[fit_indices])
        )
        validation_predictions = _sanitize_predictions(
            proposal_tree.predict(X_train[validation_indices])
        )
        if test_indices.size:
            test_predictions = _sanitize_predictions(
                proposal_tree.predict(X_test[test_indices])
            )
        else:
            test_predictions = np.array([], dtype=float)

        try:
            rules = export_text(proposal_tree, feature_names=FEATURE_NAMES)
        except Exception as exc:
            rules = f"Rules unavailable: {exc}"

        row = {
            "fold": fold,
            "node_id": node_id,
            "depth": int(record["node_depth"]),
            "depth_iteration": int(record["depth_iteration"]),
            "split_order": int(record["split_order"]),
            "tree_kind": str(record["proposal_tree_kind"]),
            "proposal_target": str(record["proposal_target"]),
            "tree_rules": rules,
            "eligible_q_path_count": int(len(record["q_path_candidates"])),
            "selected_q_path_leaf_id": int(
                record["selected_q_path_leaf_id"]
            ),
            "selected_q_path_depth": int(record["selected_q_path_depth"]),
            "selected_q_path_rule": str(record["selected_q_path_rule"]),
            "selected_q_path_train_q": float(
                record["selected_q_path_train_q"]
            ),
            "selected_q_path_validation_q": float(
                record["selected_q_path_validation_q"]
            ),
            "selected_router_kind": str(record["selected_router_kind"]),
        }
        row.update(
            _prefix_metrics(
                "fit",
                _metrics(fit_targets, fit_predictions),
            )
        )
        row.update(
            _prefix_metrics(
                "validation",
                _metrics(
                    validation_targets,
                    validation_predictions,
                ),
            )
        )
        row.update(
            _prefix_metrics(
                "test_routed",
                _metrics(test_targets, test_predictions),
            )
        )
        rows.append(row)
    return rows


def _candidate_solver_labels(
    router: Any,
    candidate_solvers: Sequence[Any],
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Assign labels using the candidate solvers retained during training."""
    errors = []
    for solver in candidate_solvers:
        prediction = _sanitize_predictions(
            solver.predict(router._select_continuous_features(X))
        )
        errors.append(np.square(prediction - np.asarray(y).ravel()))
    error_matrix = np.column_stack(errors)
    return np.argmin(error_matrix, axis=1).astype(int)


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "n_samples": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "confusion_matrix": "[]",
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": json.dumps(
            confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            ensure_ascii=False,
        ),
    }


def _node_classification_metrics(
    router: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
) -> List[Dict[str, Any]]:
    test_routes = _route_indices_by_node(router, X_test)
    rows = []
    for record in router.sbcv_split_records:
        node_id = int(record["node_id"])
        train_indices = np.asarray(record["proposal_train_indices"], dtype=int)
        validation_indices = np.asarray(
            record["proposal_validation_indices"], dtype=int
        )
        test_indices = test_routes.get(node_id, np.array([], dtype=int))
        classifier = record["classification_tree"]
        candidate_solvers = record["candidate_child_solvers"]
        child_labels = list(range(len(candidate_solvers)))

        train_true = np.asarray(
            record["classification_train_true_labels"], dtype=int
        )
        train_predicted = _discriminator_labels(
            classifier,
            X_train[train_indices],
        )
        validation_true = np.asarray(
            record["classification_validation_true_labels"], dtype=int
        )
        validation_predicted = _discriminator_labels(
            classifier,
            X_train[validation_indices],
        )
        if test_indices.size:
            test_true = _candidate_solver_labels(
                router,
                candidate_solvers,
                X_test[test_indices],
                y_test[test_indices],
            )
            test_predicted = _discriminator_labels(
                classifier,
                X_test[test_indices],
            )
        else:
            test_true = np.array([], dtype=int)
            test_predicted = np.array([], dtype=int)

        try:
            rules = export_text(
                classifier,
                feature_names=FEATURE_NAMES,
            )
        except Exception as exc:
            rules = f"Rules unavailable: {exc}"
        whole_tree_validation = record["whole_tree_validation"]
        weight_summary = record.get("classification_sample_weight_summary", {})
        row = {
            "fold": fold,
            "node_id": node_id,
            "depth": int(record["node_depth"]),
            "depth_iteration": int(record["depth_iteration"]),
            "split_order": int(record["split_order"]),
            "candidate_child_node_ids": json.dumps(
                list(record["candidate_child_node_ids"])
            ),
            "child_labels": json.dumps(child_labels),
            "classification_tree_rules": rules,
            "training_label_definition": (
                "exact minimum-error child-solver labels used during fitting"
            ),
            "split_acceptance_scope": record["split_acceptance_scope"],
            "split_acceptance_inference": record[
                "split_acceptance_inference"
            ],
            "whole_tree_validation_n_samples": whole_tree_validation[
                "n_validation_samples"
            ],
            "whole_tree_before_rmse": whole_tree_validation["before_rmse"],
            "whole_tree_after_rmse": whole_tree_validation["after_rmse"],
            "whole_tree_rmse_improvement": whole_tree_validation[
                "rmse_improvement"
            ],
            "whole_tree_wilcoxon_p_value": whole_tree_validation[
                "wilcoxon_p_value"
            ],
            "classification_sample_weight_enabled": bool(
                record.get("classification_sample_weight_enabled", False)
            ),
            "classification_sample_weight_formula": str(
                record.get(
                    "classification_sample_weight_formula",
                    "uniform_weight_1",
                )
            ),
            "classification_sample_weight_n": weight_summary.get("n", np.nan),
            "classification_sample_weight_n_positive": weight_summary.get(
                "n_positive", np.nan
            ),
            "classification_sample_weight_minimum": weight_summary.get(
                "minimum", np.nan
            ),
            "classification_sample_weight_median": weight_summary.get(
                "median", np.nan
            ),
            "classification_sample_weight_mean": weight_summary.get(
                "mean", np.nan
            ),
            "classification_sample_weight_maximum": weight_summary.get(
                "maximum", np.nan
            ),
        }
        row.update(
            _prefix_metrics(
                "train",
                _classification_metrics(
                    train_true,
                    train_predicted,
                    child_labels,
                ),
            )
        )
        row.update(
            _prefix_metrics(
                "validation",
                _classification_metrics(
                    validation_true,
                    validation_predicted,
                    child_labels,
                ),
            )
        )
        row.update(
            _prefix_metrics(
                "test_routed",
                _classification_metrics(
                    test_true,
                    test_predicted,
                    child_labels,
                ),
            )
        )
        rows.append(row)
    return rows


def _split_attempt_metrics(router: Any, fold: int) -> List[Dict[str, Any]]:
    """Return one flat diagnostic row for every accepted/rejected split try."""
    rows = []
    for record in router.sbcv_split_attempt_records:
        row = {"fold": int(fold)}
        row.update(record)
        rows.append(row)
    return rows


def _path_candidate_stability_metrics(
    router: Any,
    fold: int,
) -> List[Dict[str, Any]]:
    """Return the top training-q paths with spatial validation diagnostics."""
    rows = []
    for record in router.sbcv_path_candidate_records:
        row = {"fold": int(fold)}
        row.update(record)
        row["source_leaf_ids"] = json.dumps(
            list(record.get("source_leaf_ids", ())),
            ensure_ascii=False,
        )
        rows.append(row)
    return rows


def _growth_metrics(
    router: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
) -> List[Dict[str, Any]]:
    rows = []
    for snapshot in router.sbcv_growth_history:
        train_predictions = snapshot["predictions"].get("train")
        test_predictions = snapshot["predictions"].get("test")
        if train_predictions is None or test_predictions is None:
            raise RuntimeError(
                "Growth history is missing train/test monitoring predictions."
            )
        row = {
            "fold": fold,
            "growth_step": int(snapshot["growth_step"]),
            "event_type": str(snapshot["event_type"]),
            "depth_iteration": int(snapshot["depth_iteration"]),
            "activated_node_id": (
                np.nan
                if snapshot["activated_node_id"] is None
                else int(snapshot["activated_node_id"])
            ),
            "activated_node_depth": (
                np.nan
                if snapshot["activated_node_depth"] is None
                else int(snapshot["activated_node_depth"])
            ),
            "n_active_splits": int(snapshot["growth_step"]),
            "n_effective_strata": int(snapshot["n_effective_strata"]),
            "accepted_node_ids": json.dumps(snapshot["accepted_node_ids"]),
            "layer_attempted_node_ids": json.dumps(
                snapshot.get("layer_attempted_node_ids", ())
            ),
            "layer_accepted_node_ids": json.dumps(
                snapshot.get("layer_accepted_node_ids", ())
            ),
            "layer_growth_occurred": bool(
                snapshot.get("layer_growth_occurred", False)
            ),
            "is_layer_complete": snapshot["event_type"] == "layer_complete",
            "is_final_model": str(snapshot["event_type"]).startswith(
                "final_model"
            ),
        }
        row.update(_prefix_metrics("train", _metrics(y_train, train_predictions)))
        row.update(_prefix_metrics("test", _metrics(y_test, test_predictions)))
        rows.append(row)

    if not rows or not str(rows[-1]["event_type"]).startswith("final_model"):
        raise RuntimeError("Growth history must end with a final_model record.")

    final_train_predictions = _sanitize_predictions(router.predict(X_train))
    final_predictions = _sanitize_predictions(router.predict(X_test))
    last_train_predictions = np.asarray(
        router.sbcv_growth_history[-1]["predictions"]["train"]
    )
    last_predictions = (
        np.asarray(router.sbcv_growth_history[-1]["predictions"]["test"])
    )
    train_maximum_difference = float(
        np.max(np.abs(last_train_predictions - final_train_predictions))
    )
    test_maximum_difference = float(
        np.max(np.abs(last_predictions - final_predictions))
    )
    if train_maximum_difference > 1e-8 or test_maximum_difference > 1e-8:
        raise RuntimeError(
            f"Fold {fold}: final_model growth snapshot differs from router.predict; "
            f"train max difference={train_maximum_difference:.6g}, "
            f"test max difference={test_maximum_difference:.6g}"
        )
    rows[-1]["final_train_max_abs_difference"] = train_maximum_difference
    rows[-1]["final_test_max_abs_difference"] = test_maximum_difference
    return rows


def _model_details_text(router: Any, fold: int) -> str:
    dso_parameters = {
        "n_samples": DSO_N_SAMPLES,
        **FOLD_DSO_PARAMETERS[fold],
    }
    lines = [
        f"Spatial CV fold {fold}",
        "",
        "Run parameters",
        f"DSO: {json.dumps(dso_parameters, sort_keys=True)}",
        f"Router: {json.dumps(ROUTER_PARAMETERS, sort_keys=True)}",
        "",
        "Model nodes",
    ]
    split_records = {
        int(record["node_id"]): record for record in router.sbcv_split_records
    }
    for node in _iter_nodes(router.root):
        lines.append(
            f"Node {node.node_id} | depth={node.depth} | "
            f"type={'leaf' if node.is_leaf else 'internal'} | "
            f"train_samples={len(node.sample_indices)}"
        )
        lines.append(f"Regression equation: {_node_formula(node)}")
        if node.discriminator is not None and node.children:
            if hasattr(node.discriminator, "rule_text"):
                rules = (
                    "IF "
                    f"{node.discriminator.rule_text(FEATURE_NAMES)} "
                    "THEN child 1 ELSE child 0"
                )
            else:
                try:
                    rules = export_text(
                        node.discriminator,
                        feature_names=FEATURE_NAMES,
                    )
                except Exception as exc:
                    rules = f"Rules unavailable: {exc}"
            split_record = split_records[int(node.node_id)]
            lines.append(
                f"Selected router: {split_record['selected_router_kind']}"
            )
            lines.append(
                "Maximum-q proposal path: "
                f"q_train={split_record['selected_q_path_train_q']:.6f}, "
                f"q_validation="
                f"{split_record['selected_q_path_validation_q']:.6f}, "
                f"rule={split_record['selected_q_path_rule']}"
            )
            lines.append("Final routing rule:")
            lines.append(rules)
            validation = split_record["whole_tree_validation"]
            validation_routing = (
                "hard"
                if ROUTER_PARAMETERS["validation_hard_routing"]
                else "soft"
            )
            lines.append(
                f"Whole-tree {validation_routing} validation: "
                f"before_RMSE={validation['before_rmse']:.6f}, "
                f"after_RMSE={validation['after_rmse']:.6f}, "
                f"Wilcoxon_p={validation['wilcoxon_p_value']:.6g}"
            )
        lines.append("")
    lines.extend(["Final leaf full-sample refit", ""])
    if not router.sbcv_leaf_refit_records:
        lines.extend(["Disabled for this run.", ""])
    else:
        for record in router.sbcv_leaf_refit_records:
            lines.append(
                f"Node {record['node_id']} | outer_train_samples="
                f"{record['n_outer_training_samples']} | fixed_train_samples="
                f"{record['n_fixed_training_samples']} | "
                f"fixed_validation_samples="
                f"{record['n_fixed_validation_samples']}"
            )
            lines.append(f"Before refit: {record['old_formula']}")
            lines.append(f"After refit: {record['refit_formula']}")
            lines.append("")
    return "\n".join(lines)


def _block_metrics(
    test_frame: pd.DataFrame,
    y_test: np.ndarray,
    predictions: np.ndarray,
    fold: int,
) -> List[Dict[str, Any]]:
    rows = []
    blocks = test_frame[BLOCK_COLUMN].to_numpy(dtype=int)
    for block in sorted(np.unique(blocks)):
        mask = blocks == block
        row = {"fold": fold, "block": int(block)}
        row.update(_metrics(y_test[mask], predictions[mask]))
        rows.append(row)
    return rows


def _run_fold(
    frame: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    fold: int,
) -> Dict[str, Any]:
    _configure_worker_threads()
    from ps3r.model import PS3RModel

    train_mask = frame[FOLD_COLUMN].to_numpy() != fold
    test_mask = ~train_mask
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    train_blocks = frame.iloc[train_indices][BLOCK_COLUMN].to_numpy(dtype=int)
    test_frame = frame.iloc[test_indices].copy()

    _, continuous_feature_indices = resolve_feature_roles(
        FEATURE_NAMES,
        CATEGORICAL_FEATURE_NAMES,
    )
    logger.info(
        f"Fold {fold}: train={len(train_indices)}, test={len(test_indices)}, "
        f"train_blocks={frame.loc[train_mask, BLOCK_COLUMN].nunique()}, "
        f"test_blocks={test_frame[BLOCK_COLUMN].nunique()}"
    )

    solver_config, dso_parameters = _solver_config_for_fold(fold)
    logger.info(
        f"Fold {fold} parameters: DSO="
        f"{json.dumps({'n_samples': DSO_N_SAMPLES, **dso_parameters}, sort_keys=True)}, "
        f"router={json.dumps(ROUTER_PARAMETERS, sort_keys=True)}"
    )

    router = PS3RModel(
        input_dim=X_train.shape[1],
        continuous_feature_indices=continuous_feature_indices,
        solver_kwargs={"config": solver_config},
        random_state=MODEL_SEED,
        **ROUTER_PARAMETERS,
    )
    router.set_sbcv_monitoring_data(
        train=(X_train, y_train),
        test=(X_test, y_test),
    )
    start_time = time.time()
    router.fit(X_train, y_train, spatial_block_ids=train_blocks)
    runtime = time.time() - start_time

    train_predictions = _sanitize_predictions(router.predict(X_train))
    test_predictions = _sanitize_predictions(router.predict(X_test))
    test_leaf_ids = np.asarray(router.predict_leaf_ids(X_test), dtype=int)

    fold_row = {
        "fold": fold,
        "model_seed": MODEL_SEED,
        "fold_process_isolation": "fresh_spawn_process_per_fold",
        "blas_threads_per_fold": 1,
        "torch_threads_per_fold": 1,
        "dso_config_template": CONFIG_PATH.name,
        "dso_n_samples": int(DSO_N_SAMPLES),
        "dso_num_layers": int(dso_parameters["num_layers"]),
        "dso_num_units": int(dso_parameters["num_units"]),
        "dso_learning_rate": float(dso_parameters["learning_rate"]),
        "dso_batch_size": int(dso_parameters["batch_size"]),
        "dso_entropy_weight": float(dso_parameters["entropy_weight"]),
        "dso_epsilon": float(dso_parameters["epsilon"]),
        "router_parameters": json.dumps(ROUTER_PARAMETERS, sort_keys=True),
        "n_train_blocks": int(frame.loc[train_mask, BLOCK_COLUMN].nunique()),
        "n_test_blocks": int(test_frame[BLOCK_COLUMN].nunique()),
        "internal_validation_strategy": router.node_holdout_strategy,
        "n_internal_train_blocks": int(
            len(router._fixed_root_train_blocks)
        ),
        "n_internal_validation_blocks": int(
            len(router._fixed_root_validation_blocks)
        ),
        "n_internal_train_samples": int(
            len(router._fixed_root_train_indices)
        ),
        "n_internal_validation_samples": int(
            len(router._fixed_root_validation_indices)
        ),
        "n_internal_train_validation_overlapping_blocks": int(
            np.intersect1d(
                router._fixed_root_train_blocks,
                router._fixed_root_validation_blocks,
            ).size
        ),
        "runtime_seconds": runtime,
        "final_tree_nodes": int(len(_iter_nodes(router.root))),
        "final_strata": int(len(router._get_all_leaves())),
        "final_leaf_refits": int(len(router.sbcv_leaf_refit_records)),
        "final_leaf_refit_samples": int(
            sum(
                record["n_outer_training_samples"]
                for record in router.sbcv_leaf_refit_records
            )
        ),
        "split_acceptance": (
            "best_of_q_path_and_restratification_whole_tree_"
            f"{'hard' if ROUTER_PARAMETERS['validation_hard_routing'] else 'soft'}_"
            "fixed_validation"
        ),
        "final_prediction_routing": (
            "hard" if ROUTER_PARAMETERS["hard_pred"] else "fuzzy"
        ),
    }
    fold_row.update(_prefix_metrics("train", _metrics(y_train, train_predictions)))
    fold_row.update(_prefix_metrics("test", _metrics(y_test, test_predictions)))

    prediction_frame = test_frame.copy()
    prediction_frame["dso_sbcv_prediction"] = test_predictions
    prediction_frame["residual"] = y_test - test_predictions
    prediction_frame["absolute_error"] = np.abs(y_test - test_predictions)
    prediction_frame["squared_error"] = np.square(y_test - test_predictions)
    prediction_frame["routed_leaf_node_hard"] = test_leaf_ids
    prediction_frame["source_row_index"] = test_indices
    node_regression_rows = _proposal_tree_regression_metrics(
        router,
        X_train,
        y_train,
        X_test,
        y_test,
        fold,
    )
    node_classification_rows = _node_classification_metrics(
        router,
        X_train,
        y_train,
        X_test,
        y_test,
        fold,
    )
    split_attempt_rows = _split_attempt_metrics(router, fold)
    path_candidate_stability_rows = _path_candidate_stability_metrics(
        router,
        fold,
    )
    growth_rows = _growth_metrics(
        router,
        X_train,
        y_train,
        X_test,
        y_test,
        fold,
    )
    block_rows = _block_metrics(
        test_frame,
        y_test,
        test_predictions,
        fold,
    )
    model_details = _model_details_text(router, fold)

    result = {
        "fold_metric": fold_row,
        "block_metrics": block_rows,
        "node_regression_metrics": node_regression_rows,
        "node_classification_metrics": node_classification_rows,
        "split_attempt_metrics": split_attempt_rows,
        "path_candidate_stability_metrics": path_candidate_stability_rows,
        "growth_metrics": growth_rows,
        "predictions": prediction_frame,
        "model_details": model_details,
    }
    del router
    _cleanup_memory()
    return result


def _save_accumulated(
    output_dir: Path,
    run_id: str,
    fold_metrics: List[Dict[str, Any]],
    block_metrics: List[Dict[str, Any]],
    node_regression_metrics: List[Dict[str, Any]],
    node_classification_metrics: List[Dict[str, Any]],
    split_attempt_metrics: List[Dict[str, Any]],
    path_candidate_stability_metrics: List[Dict[str, Any]],
    growth_metrics: List[Dict[str, Any]],
    predictions: List[pd.DataFrame],
    statuses: List[Dict[str, Any]],
):
    datasets = [
        (fold_metrics, f"sbcv_fold_metrics_{run_id}.csv"),
        (block_metrics, f"sbcv_block_metrics_{run_id}.csv"),
        (
            node_regression_metrics,
            f"sbcv_node_initial_regression_tree_metrics_{run_id}.csv",
        ),
        (
            node_classification_metrics,
            f"sbcv_node_classification_metrics_{run_id}.csv",
        ),
        (
            split_attempt_metrics,
            f"sbcv_split_attempt_metrics_{run_id}.csv",
        ),
        (
            path_candidate_stability_metrics,
            f"sbcv_path_candidate_stability_metrics_{run_id}.csv",
        ),
        (growth_metrics, f"sbcv_growth_metrics_{run_id}.csv"),
        (statuses, f"sbcv_fold_status_{run_id}.csv"),
    ]
    for rows, filename in datasets:
        if rows:
            _write_csv(rows, output_dir / filename)
    if predictions:
        combined = pd.concat(predictions, axis=0, ignore_index=True)
        _write_csv(
            combined,
            output_dir / f"sbcv_predictions_all_folds_{run_id}.csv",
        )


def _save_final_results(
    output_dir: Path,
    run_id: str,
    results_by_fold: Dict[int, Dict[str, Any]],
    statuses_by_fold: Dict[int, Dict[str, Any]],
):
    """Write all retained results once, after model training has stopped."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics: List[Dict[str, Any]] = []
    block_metrics: List[Dict[str, Any]] = []
    node_regression_metrics: List[Dict[str, Any]] = []
    node_classification_metrics: List[Dict[str, Any]] = []
    split_attempt_metrics: List[Dict[str, Any]] = []
    path_candidate_stability_metrics: List[Dict[str, Any]] = []
    growth_metrics: List[Dict[str, Any]] = []
    predictions: List[pd.DataFrame] = []

    for fold in sorted(results_by_fold):
        result = results_by_fold[fold]
        fold_metrics.append(result["fold_metric"])
        block_metrics.extend(result["block_metrics"])
        node_regression_metrics.extend(result["node_regression_metrics"])
        node_classification_metrics.extend(
            result["node_classification_metrics"]
        )
        split_attempt_metrics.extend(result["split_attempt_metrics"])
        path_candidate_stability_metrics.extend(
            result["path_candidate_stability_metrics"]
        )
        growth_metrics.extend(result["growth_metrics"])
        predictions.append(result["predictions"])
        _write_csv(
            result["predictions"],
            output_dir / f"fold_{fold}_predictions_{run_id}.csv",
        )
        (output_dir / f"fold_{fold}_model_details_{run_id}.txt").write_text(
            result["model_details"],
            encoding="utf-8",
        )

    _save_accumulated(
        output_dir,
        run_id,
        fold_metrics,
        block_metrics,
        node_regression_metrics,
        node_classification_metrics,
        split_attempt_metrics,
        path_candidate_stability_metrics,
        growth_metrics,
        predictions,
        [statuses_by_fold[fold] for fold in sorted(statuses_by_fold)],
    )


def _n_jobs_value(value: str) -> int:
    try:
        n_jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("n-jobs must be an integer") from exc
    if not 1 <= n_jobs <= 5:
        raise argparse.ArgumentTypeError("n-jobs must be between 1 and 5")
    return n_jobs


def _is_array_memory_error(exc: BaseException) -> bool:
    """Recognize NumPy allocation failures returned by worker processes."""
    messages = []
    current: Any = exc
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = getattr(current, "__cause__", None)
    error_text = "\n".join(messages).lower()
    return (
        "arraymemoryerror" in error_text
        or "unable to allocate" in error_text
        or "out of memory" in error_text
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five-fold spatial block CV for the PS3R BD model."
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Validate input and feature construction without training models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; default is a timestamped directory in the working directory.",
    )
    parser.add_argument(
        "--n-jobs",
        type=_n_jobs_value,
        default=1,
        metavar="N",
        help=(
            "Number of spatial folds to train concurrently (1-5; default: 1). "
            "Use 1 on a single GPU unless concurrent processes fit in memory."
        ),
    )
    parser.add_argument(
        "--no-serial-memory-retry",
        action="store_false",
        dest="serial_memory_retry",
        help=(
            "Do not retry folds serially when their parallel workers fail "
            "with an array/out-of-memory error."
        ),
    )
    parser.set_defaults(serial_memory_retry=True)
    return parser.parse_args()


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    frame, X, y = _prepare_data(DATA_PATH)
    if args.validate_data_only:
        logger.info("Data validation completed successfully; model training skipped.")
        return

    from ps3r.runtime import require_dso
    require_dso()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path.cwd() / f"sbcv_bd_results_{run_id}"
    )
    logger.info(
        f"Results will remain in memory during training and be saved once to: "
        f"{output_dir}"
    )

    results_by_fold: Dict[int, Dict[str, Any]] = {}
    statuses_by_fold: Dict[int, Dict[str, Any]] = {}

    def record_fold_result(fold: int, result: Dict[str, Any]):
        results_by_fold[fold] = result
        statuses_by_fold[fold] = {
            "fold": fold,
            "status": "success",
            "error": "",
        }

    def record_fold_failure(fold: int, exc: BaseException):
        logger.error(f"Fold {fold} failed: {exc}")
        logger.error(traceback.format_exc())
        statuses_by_fold[fold] = {
            "fold": fold,
            "status": "failed",
            "error": str(exc),
        }

    def run_folds_in_fresh_processes(
        folds: Sequence[int],
        max_concurrent_folds: int,
    ) -> List[int]:
        """Run every fold in a dedicated spawn process that is never reused."""
        pending_folds = list(folds)
        active_folds: Dict[Any, Tuple[int, ProcessPoolExecutor]] = {}
        memory_failed_folds: List[int] = []
        spawn_context = mp.get_context("spawn")

        def launch_fold(fold: int):
            logger.info("=" * 80)
            logger.info(f"Starting spatial CV fold {fold}/5 in a fresh process")
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=spawn_context,
            )
            try:
                future = executor.submit(_run_fold, frame, X, y, fold)
            except Exception as exc:
                executor.shutdown(wait=True)
                record_fold_failure(fold, exc)
                if _is_array_memory_error(exc):
                    memory_failed_folds.append(fold)
                return
            active_folds[future] = (fold, executor)

        try:
            while pending_folds or active_folds:
                while (
                    pending_folds
                    and len(active_folds) < max_concurrent_folds
                ):
                    launch_fold(pending_folds.pop(0))
                if not active_folds:
                    continue

                completed, _ = wait(
                    set(active_folds),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    fold, executor = active_folds.pop(future)
                    try:
                        record_fold_result(fold, future.result())
                        logger.info(
                            f"Spatial CV fold {fold}/5 completed successfully."
                        )
                    except Exception as exc:
                        record_fold_failure(fold, exc)
                        if _is_array_memory_error(exc):
                            memory_failed_folds.append(fold)
                    finally:
                        executor.shutdown(wait=True)
                        _cleanup_memory()
        finally:
            for _, executor in active_folds.values():
                executor.shutdown(wait=True)
            active_folds.clear()
        return memory_failed_folds

    logger.info(
        f"Running five spatial folds with at most {args.n_jobs} concurrent "
        "processes. Every fold receives a new spawn process that is destroyed "
        "after that fold finishes."
    )
    if args.n_jobs > 1:
        logger.warning(
            "Concurrent DSO training can exhaust memory on a single GPU; "
            "use --n-jobs 1 if that occurs."
        )
    memory_failed_folds = run_folds_in_fresh_processes(
        range(1, 6),
        args.n_jobs,
    )

    if memory_failed_folds and args.serial_memory_retry:
        logger.warning(
            "Workers ran out of memory for folds %s. Retrying only these folds "
            "one at a time, still with a fresh process for every fold.",
            sorted(memory_failed_folds),
        )
        run_folds_in_fresh_processes(sorted(memory_failed_folds), 1)

    statuses = [statuses_by_fold[fold] for fold in sorted(statuses_by_fold)]
    successful_folds = sum(status["status"] == "success" for status in statuses)
    logger.info("Training has stopped; writing retained results now.")
    _save_final_results(
        output_dir,
        run_id,
        results_by_fold,
        statuses_by_fold,
    )
    logger.info(
        f"Spatial CV completed: {successful_folds}/5 folds succeeded. "
        f"Results: {output_dir}"
    )
    if successful_folds != 5:
        raise RuntimeError(
            f"Only {successful_folds}/5 folds succeeded; see fold status output."
        )


if __name__ == "__main__":
    main()
