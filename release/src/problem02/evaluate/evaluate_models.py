"""Rolling evaluation for the original and improved load/PV models.

Actual observations are used only after each prediction has been produced.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "result" / "problem02" / "evaluate" / "model"
APPENDIX_DIR = ROOT / "附件"

from src.problem02.evaluate.load import (
    RollingRidgeLoadV4,
    load_history as load_load_history,
    original_historical_prediction as load_original_prediction,
    select_lambda,
)
from src.problem02.evaluate.pv import (
    ExpandingRidgePVV4,
    load_history as load_pv_history,
    original_historical_prediction as pv_original_prediction,
    select_parameters,
)


LOAD_TIMES = ("00:10", "06:00", "09:00", "12:00", "14:00", "18:00", "20:00", "22:00")
PV_TIMES = ("06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "17:50")


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = actual - prediction
    if np.std(actual) == 0 or np.std(prediction) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(actual, prediction)[0, 1])
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "corr": correlation,
    }


def evaluate_dataset(
    time_labels: list[str],
    actual: np.ndarray,
    original: np.ndarray,
    improved: np.ndarray,
    start_index: int,
) -> list[dict]:
    valid_days = np.isfinite(improved[:, 0])
    valid_days[:start_index] = False
    rows = []
    for column, time_label in enumerate(time_labels):
        old_values = metrics(
            actual[valid_days, column], original[valid_days, column]
        )
        new_values = metrics(
            actual[valid_days, column], improved[valid_days, column]
        )
        rows.append(
            {
                "time": time_label,
                "samples": int(valid_days.sum()),
                "old_MAE": old_values["MAE"],
                "new_MAE": new_values["MAE"],
                "new_minus_old_MAE": new_values["MAE"] - old_values["MAE"],
                "old_RMSE": old_values["RMSE"],
                "new_RMSE": new_values["RMSE"],
                "old_bias": old_values["bias"],
                "new_bias": new_values["bias"],
                "old_corr": old_values["corr"],
                "new_corr": new_values["corr"],
            }
        )
    return rows


def plot_panels(
    dates,
    actual: np.ndarray,
    original: np.ndarray,
    improved: np.ndarray,
    time_labels: list[str],
    selected_times: tuple[str, ...],
    output: Path,
    title: str,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = (len(selected_times) + 1) // 2
    figure, axes = plt.subplots(
        rows, 2, figsize=(18, 3.2 * rows), sharex=True
    )
    axes = np.asarray(axes).reshape(-1)

    for axis, time_label in zip(axes, selected_times):
        column = time_labels.index(time_label)
        evaluation_period = np.asarray([day.month >= 2 for day in dates])
        valid = (
            np.isfinite(actual[:, column])
            & np.isfinite(original[:, column])
            & np.isfinite(improved[:, column])
            & evaluation_period
        )
        old_mae = np.mean(
            np.abs(
                actual[valid, column] - original[valid, column]
            )
        )
        new_mae = np.mean(
            np.abs(
                actual[valid, column] - improved[valid, column]
            )
        )
        axis.axvspan(dates[0], date(2025, 1, 31), color="#eceff1", alpha=0.75)
        axis.plot(dates, actual[:, column], color="#1f2933", linewidth=0.85, label="实际值")
        axis.plot(
            dates,
            original[:, column],
            color="#9aa0a6",
            linewidth=0.75,
            linestyle="--",
            label="原方法",
        )
        axis.plot(
            dates,
            improved[:, column],
            color="#d95f02",
            linewidth=0.85,
            label="新方法",
        )
        axis.set_title(
            f"{time_label}｜2月起 原 MAE={old_mae:.1f}｜新 MAE={new_mae:.1f}",
            fontsize=10,
        )
        axis.grid(color="#dddddd", linewidth=0.5, alpha=0.8)

    for axis in axes[len(selected_times):]:
        axis.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_metrics(path: Path, load_rows: list[dict], pv_rows: list[dict]) -> None:
    fields = [
        "series",
        "time",
        "samples",
        "old_MAE",
        "new_MAE",
        "new_minus_old_MAE",
        "old_RMSE",
        "new_RMSE",
        "old_bias",
        "new_bias",
        "old_corr",
        "new_corr",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for series, rows in (("load", load_rows), ("pv", pv_rows)):
            for row in rows:
                writer.writerow({"series": series, **row})


def main() -> None:
    attachment = APPENDIX_DIR / "附件2.xlsx"
    output_dir = RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dates, time_labels, load = load_load_history(attachment)
    pv_dates, _, pv = load_pv_history(attachment)
    if load_dates != pv_dates:
        raise ValueError("负载与光伏日期不一致")

    load_lambda, load_scores = select_lambda(load_dates, load)
    load_model = RollingRidgeLoadV4(load_dates, load, load_lambda)
    load_old = np.column_stack(
        [
            load_original_prediction(load_dates, load, column)
            for column in range(load.shape[1])
        ]
    )
    pv_old = pv_original_prediction(pv)
    pv_lambda, pv_blend, pv_records = select_parameters(
        pv_dates, pv, pv_old
    )
    pv_model = ExpandingRidgePVV4(pv_dates, pv, pv_lambda)
    pv_new = pv_blend * pv_old + (1 - pv_blend) * pv_model.historical_ridge

    load_rows = evaluate_dataset(
        time_labels,
        load,
        load_old,
        load_model.historical_prediction,
        31,
    )
    pv_rows = evaluate_dataset(
        time_labels,
        pv,
        pv_old,
        pv_new,
        31,
    )
    write_metrics(output_dir / "model_evaluation_metrics.csv", load_rows, pv_rows)
    plot_panels(
        load_dates,
        load,
        load_old,
        load_model.historical_prediction,
        time_labels,
        LOAD_TIMES,
        output_dir / "load_model_evaluation.svg",
        "第4版小区负载：2月起不同时段实际值与两种预测方法对比",
    )
    plot_panels(
        pv_dates,
        pv,
        pv_old,
        pv_new,
        time_labels,
        PV_TIMES,
        output_dir / "pv_model_evaluation.svg",
        "第4版光伏：2月起不同时段实际值与两种预测方法对比",
    )
    print(f"负载1月选定参数：lambda={load_lambda}, scores={load_scores}")
    print(f"光伏1月选定参数：lambda={pv_lambda}, blend={pv_blend}")
    print(f"指标表：{output_dir / 'model_evaluation_metrics.csv'}")
    print(f"负载图：{output_dir / 'load_model_evaluation.svg'}")
    print(f"光伏图：{output_dir / 'pv_model_evaluation.svg'}")


if __name__ == "__main__":
    main()
