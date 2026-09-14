"""比较减少光伏预报/调整节点对第三问费用和电量的影响。

实验保持第三问主模型不变，只改变允许进行购电调整的时刻：

    0+6          decision_hours=(6,)
    0+12         decision_hours=(12,)
    0+6+12       decision_hours=(6,12)
    0+6+18       decision_hours=(6,18)
    0+12+18      decision_hours=(12,18)
    0+6+12+18    decision_hours=(6,12,18)

其中每个组合都保留 0:00 的全天计划，数字表示额外允许的调整时刻。

运行（在项目根目录下）：
    python -m src.problem03.demonstrate
    python -m src.problem03.demonstrate --plot-only

输出（result/problem03/expend/）：
    question3_expend_summary.csv
    question3_expend_comparison.xlsx
    question3_expend_comparison.svg
    question3_expend_relative.svg
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
APPENDIX_DIR = ROOT / "附件"
RESULT_DIR = ROOT / "result" / "problem03" / "demonstrate"

from src.problem03 import optimize


QUANTILE_NIGHT = 0.60
QUANTILE_DAY = 0.60

SCENARIOS = (
    {
        "name": "0+6",
        "label": "0+6",
        "decision_hours": (6,),
        "description": "0:00 计划，仅 6:00 调整",
    },
    {
        "name": "0+12",
        "label": "0+12",
        "decision_hours": (12,),
        "description": "0:00 计划，仅 12:00 调整",
    },
    {
        "name": "0+6+12",
        "label": "0+6+12",
        "decision_hours": (6, 12),
        "description": "0:00 计划，6:00、12:00 调整",
    },
    {
        "name": "0+6+18",
        "label": "0+6+18",
        "decision_hours": (6, 18),
        "description": "0:00 计划，6:00、18:00 调整",
    },
    {
        "name": "0+12+18",
        "label": "0+12+18",
        "decision_hours": (12, 18),
        "description": "0:00 计划，12:00、18:00 调整",
    },
    {
        "name": "0+6+12+18",
        "label": "0+6+12+18",
        "decision_hours": (6, 12, 18),
        "description": "0:00 计划，6:00、12:00、18:00 调整",
    },
)


def _init_worker() -> None:
    optimize.load_inputs()


def run_scenario(scenario: dict) -> dict:
    """运行一个调整时刻组合，并返回统一口径的汇总结果。"""
    started = perf_counter()
    summary = optimize.run_simulation(
        quantile_overnight=QUANTILE_NIGHT,
        quantile_day=QUANTILE_DAY,
        use_correction=True,
        decision_hours=scenario["decision_hours"],
        collect_daily=False,
    )
    return {
        "scenario": scenario["name"],
        "description": scenario["description"],
        "decision_hours": ",".join(str(hour) for hour in scenario["decision_hours"]),
        "plan_kwh": float(summary["plan_kwh"]),
        "adjusted_kwh": float(summary["adjusted_kwh"]),
        "emergency_kwh": float(summary["emergency_kwh"]),
        "plan_cost": float(summary["plan_cost"]),
        "adjustment_cost": float(summary["adjustment_cost"]),
        "emergency_cost": float(summary["emergency_cost"]),
        "total_cost": float(summary["total_cost"]),
        "spill_kwh": float(summary["spill_kwh"]),
        "storage_end": float(summary["storage_end"]),
        "seconds": perf_counter() - started,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "scenario",
        "description",
        "decision_hours",
        "plan_kwh",
        "adjusted_kwh",
        "emergency_kwh",
        "plan_cost",
        "adjustment_cost",
        "emergency_cost",
        "total_cost",
        "spill_kwh",
        "storage_end",
        "seconds",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary_csv(path: Path) -> list[dict]:
    numeric_fields = {
        "plan_kwh",
        "adjusted_kwh",
        "emergency_kwh",
        "plan_cost",
        "adjustment_cost",
        "emergency_cost",
        "total_cost",
        "spill_kwh",
        "storage_end",
        "seconds",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def write_workbook(path: Path, rows: list[dict]) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "节点对比"
    headers = [
        "调整节点组合",
        "说明",
        "调整时刻",
        "计划购电量（kWh）",
        "调整后购电量（kWh）",
        "紧急购电量（kWh）",
        "计划购电费（元）",
        "调整费用（元）",
        "紧急购电费（元）",
        "总费用（元）",
        "弃电量（kWh）",
        "期末储电量（kWh）",
        "运行时间（秒）",
    ]
    sheet.append(headers)
    for row in rows:
        sheet.append(
            [
                row["scenario"],
                row["description"],
                row["decision_hours"],
                row["plan_kwh"],
                row["adjusted_kwh"],
                row["emergency_kwh"],
                row["plan_cost"],
                row["adjustment_cost"],
                row["emergency_cost"],
                row["total_cost"],
                row["spill_kwh"],
                row["storage_end"],
                row["seconds"],
            ]
        )

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(name="宋体", size=10, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in sheet.iter_rows(min_row=2):
        for cell in row[3:]:
            cell.number_format = "0.000000"
    sheet.freeze_panes = "A2"
    for column in range(1, sheet.max_column + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 20
    workbook.save(path)


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def draw_absolute_figure(path: Path, rows: list[dict]) -> None:
    labels = [row["scenario"] for row in rows]
    x = np.arange(len(rows))
    plan_gwh = np.asarray([row["plan_kwh"] / 1e6 for row in rows])
    emergency_wan_kwh = np.asarray([row["emergency_kwh"] / 1e4 for row in rows])
    total_wan_yuan = np.asarray([row["total_cost"] / 1e4 for row in rows])

    panels = (
        (plan_gwh, "计划购电量（GWh）", "#1f77b4"),
        (emergency_wan_kwh, "紧急购电量（万 kWh）", "#d62728"),
        (total_wan_yuan, "总费用（万元）", "#2ca02c"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(22, 5.5))
    for axis, (values, title, color) in zip(axes, panels):
        axis.plot(x, values, marker="o", markersize=7, linewidth=2.2, color=color)
        axis.set_xticks(x, labels)
        axis.set_xlabel("预报/调整节点组合")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.margins(x=0.08, y=0.18)
        axis.grid(alpha=0.3)
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.3f}",
                (index, value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
            )

    figure.suptitle("第三问减少光伏预报/调整节点的影响", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def draw_relative_figure(path: Path, rows: list[dict]) -> None:
    baseline = next(row for row in rows if row["scenario"] == "0+6+12+18")
    labels = [row["scenario"] for row in rows]
    x = np.arange(len(rows))

    series = (
        (
            "计划购电量变化",
            np.asarray(
                [
                    (row["plan_kwh"] / baseline["plan_kwh"] - 1.0) * 100
                    for row in rows
                ]
            ),
            "#1f77b4",
        ),
        (
            "紧急购电量变化",
            np.asarray(
                [
                    (row["emergency_kwh"] / baseline["emergency_kwh"] - 1.0)
                    * 100
                    for row in rows
                ]
            ),
            "#d62728",
        ),
        (
            "总费用变化",
            np.asarray(
                [
                    (row["total_cost"] / baseline["total_cost"] - 1.0) * 100
                    for row in rows
                ]
            ),
            "#2ca02c",
        ),
    )

    figure, axis = plt.subplots(figsize=(14, 6.0))
    for label, values, color in series:
        axis.plot(x, values, marker="o", linewidth=2.2, label=label, color=color)
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:+.2f}%",
                (index, value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
            )
    axis.axhline(0, color="black", linewidth=1, alpha=0.6)
    axis.set_xticks(x, labels)
    axis.set_xlabel("预报/调整节点组合")
    axis.set_ylabel("相对 0+6+12+18 的变化率（%）")
    axis.set_title("减少调整节点后的相对变化")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, len(SCENARIOS)),
        help="并行进程数，默认 6",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="只读取已有 CSV 重新生成 Excel 和图片",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers 必须为正整数")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "question3_expend_summary.csv"
    xlsx_path = RESULT_DIR / "question3_expend_comparison.xlsx"
    svg_path = RESULT_DIR / "question3_expend_comparison.svg"
    relative_svg_path = RESULT_DIR / "question3_expend_relative.svg"

    total_started = perf_counter()
    if args.plot_only:
        if not csv_path.is_file():
            raise FileNotFoundError(f"找不到汇总 CSV：{csv_path}")
        print("读取已有实验结果并重新绘图……")
        rows = load_summary_csv(csv_path)
    else:
        print("开始第三问调整节点对比实验……")
        results: dict[str, dict] = {}
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
        ) as executor:
            futures = {
                executor.submit(run_scenario, scenario): scenario["name"]
                for scenario in SCENARIOS
            }
            for future in as_completed(futures):
                name = futures[future]
                row = future.result()
                results[name] = row
                print(
                    f"完成 {name}: "
                    f"计划={row['plan_kwh'] / 1e6:.6f} GWh, "
                    f"紧急={row['emergency_kwh'] / 1e4:.6f} 万kWh, "
                    f"总费用={row['total_cost'] / 1e4:.4f} 万元"
                )
        rows = [results[scenario["name"]] for scenario in SCENARIOS]

    write_csv(csv_path, rows)
    write_workbook(xlsx_path, rows)
    configure_matplotlib()
    draw_absolute_figure(svg_path, rows)
    draw_relative_figure(relative_svg_path, rows)

    print(f"汇总 CSV：{csv_path}")
    print(f"汇总 Excel：{xlsx_path}")
    print(f"绝对指标折线图：{svg_path}")
    print(f"相对变化折线图：{relative_svg_path}")
    print(f"总耗时：{perf_counter() - total_started:.2f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
