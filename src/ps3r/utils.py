import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def get_logger(name: str):
    return logging.getLogger(name)


class ExperimentResult(NamedTuple):
    experiment_id: str
    train_rmse: float
    test_rmse: float
    test_mae: float
    test_mape: float
    test_r2: float
    runtime: float
    formulas_raw: Dict[str, str]
    formulas_pruned: Dict[str, str]
    model_structure_text: str


def resolve_feature_roles(
    feature_names: List[str],
    categorical_feature_names: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    categorical_names = list(categorical_feature_names or [])
    categorical_indices = [feature_names.index(name) for name in categorical_names]
    continuous_indices = [i for i in range(len(feature_names)) if i not in set(categorical_indices)]
    return categorical_indices, continuous_indices


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def resolve_data_path(filename: str) -> str:
    direct = PROJECT_ROOT / filename
    if direct.exists():
        return str(direct)
    local = Path(__file__).resolve().parent / filename
    if local.exists():
        return str(local)
    return str(direct)


def save_results(seed:int, results: List[ExperimentResult], output_dir: Optional[str] = None):
    logger = get_logger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(output_dir) if output_dir else PROJECT_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    for res in results:
        formula_str = "; ".join([f"{k}: {v}" for k, v in res.formulas_pruned.items()])
        summary_data.append(
            {
                "Experiment ID": res.experiment_id,
                "Test RMSE": res.test_rmse,
                "Test MAE": res.test_mae,
                "Test MAPE (%)": res.test_mape,
                "Test R2": res.test_r2,
                "Runtime (s)": res.runtime,
                "Formulas (Pruned)": formula_str,
            }
        )

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 80)
    print("FINAL EXPERIMENT SUMMARY REPORT")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")

    excel_path = target_dir / f"experiment_results_socd_{timestamp}.xlsx"
    csv_path = target_dir / f"experiment_results_socd_{timestamp}.csv"
    try:
        df_summary.to_csv(csv_path, index=False)
        logger.info(f"Saved summary results to {csv_path}")
        with pd.ExcelWriter(excel_path) as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            for res in results:
                sheet_name = res.experiment_id.replace(" ", "_")[:30]
                node_data = []
                for node_id, formula in res.formulas_pruned.items():
                    node_data.append(
                        {
                            "Node ID": node_id,
                            "Formula (Pruned)": formula,
                            "Formula (Raw)": res.formulas_raw.get(node_id, ""),
                        }
                    )
                if node_data:
                    pd.DataFrame(node_data).to_excel(writer, sheet_name=f"{sheet_name}_Formulas", index=False)
                pd.DataFrame({"Model Structure": [res.model_structure_text]}).to_excel(
                    writer,
                    sheet_name=f"{sheet_name}_Struct",
                    index=False,
                )
        logger.info(f"Saved detailed results to {excel_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
