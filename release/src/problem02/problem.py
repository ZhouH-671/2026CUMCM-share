"""第二问：基于日前滚动预测、风险裕量和储能的计划购电求解。

运行：
    cd release && python -m src.problem02.problem

依赖：
    src/problem02/evaluate/pv.py src/problem02/evaluate/load.py

输入：
    result/problem02/pred.xlsx   时间序列模型给出的 2025 年负载和光伏预测
    附件/附件1.xlsx              电价曲线
    附件/附件2.xlsx              实际负载和实际光伏

输出：
    result/problem02/result2.xlsx
    result/problem02/model_R2_evaluation.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result" / "problem02"
APPENDIX_DIR = ROOT / "附件"

from src.problem02.evaluate.load import (  # noqa: E402
    RollingRidgeLoadV4,
    load_history as load_load_history,
    select_lambda,
)
from src.problem02.evaluate.pv import (  # noqa: E402
    ExpandingRidgePVV4,
    load_history as load_pv_history,
    original_historical_prediction,
    select_parameters,
)


PERIODS_PER_DAY = 144
DT = 1 / 6
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_INITIAL = 6000.0
ETA_C = 0.9
ETA_D = 0.9
POWER_LIMIT = 5000.0 * DT
RISK_QUANTILE = 0.80
RESIDUAL_WINDOW_DAYS = 60
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)


@dataclass(frozen=True)
class DailyPlan:
    purchase_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    storage_end_kwh: np.ndarray


@dataclass(frozen=True)
class ActualResult:
    storage_initial_kwh: float
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    emergency_kwh: np.ndarray
    spill_kwh: np.ndarray
    storage_end_kwh: np.ndarray
    emergency_runs: list[tuple[str, float]]


def excel_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"日期格式错误：{value!r}")


def load_price(path: Path) -> np.ndarray:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.iter_rows(values_only=True))
    finally:
        workbook.close()
    if tuple(rows[0][:4]) != ("时间", "电价", "小区负载", "光伏发电预测功率"):
        raise ValueError("附件 1 表头不匹配")
    values = np.asarray([row[1] for row in rows[1:] if row[0] is not None], dtype=float)
    if values.shape != (PERIODS_PER_DAY,) or not np.isfinite(values).all():
        raise ValueError("附件 1 电价数据必须是 144 个有限数值")
    if (values < 0).any():
        raise ValueError("附件 1 电价不能为负")
    return values


def load_actual_series(path: Path) -> tuple[list[date], np.ndarray, np.ndarray]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames[:2] != ["小区负载", "光伏发电实际功率"]:
            raise ValueError("附件 2 工作表名称或顺序不匹配")
        matrices = []
        dates = None
        for worksheet in workbook.worksheets[:2]:
            rows = list(worksheet.iter_rows(values_only=True))
            current_dates = [excel_date(row[0]) for row in rows[1:] if row[0] is not None]
            values = np.asarray(
                [row[1:] for row in rows[1:] if row[0] is not None], dtype=float
            )
            if values.shape != (365, PERIODS_PER_DAY):
                raise ValueError(f"{worksheet.title} 数据形状错误：{values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{worksheet.title} 存在缺失值")
            if dates is None:
                dates = current_dates
            elif current_dates != dates:
                raise ValueError("附件 2 两个工作表的日期不一致")
            matrices.append(np.maximum(0.0, values))
    finally:
        workbook.close()
    return dates, matrices[0], matrices[1]


def load_prediction_series(
    path: Path,
) -> tuple[list[date], np.ndarray, np.ndarray]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames[:2] != ["小区负载", "光伏发电实际功率"]:
            raise ValueError("pred.xlsx 工作表名称或顺序不匹配")
        matrices = []
        dates = None
        for worksheet in workbook.worksheets[:2]:
            rows = list(worksheet.iter_rows(values_only=True))
            current_dates = [excel_date(row[0]) for row in rows[1:] if row[0] is not None]
            values = np.asarray(
                [row[1:] for row in rows[1:] if row[0] is not None], dtype=float
            )
            if values.shape != (334, PERIODS_PER_DAY):
                raise ValueError(f"pred.xlsx 的 {worksheet.title} 应为 334 x 144")
            if not np.isfinite(values).all():
                raise ValueError(f"pred.xlsx 的 {worksheet.title} 存在缺失值")
            if dates is None:
                dates = current_dates
            elif current_dates != dates:
                raise ValueError("pred.xlsx 两个工作表的日期不一致")
            matrices.append(np.maximum(0.0, values))
    finally:
        workbook.close()
    if dates[0] != OUTPUT_START or dates[-1] != OUTPUT_END:
        raise ValueError("pred.xlsx 的日期必须覆盖 2025-02-01 至 2025-12-31")
    return dates, matrices[0], matrices[1]


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """计算 R2、MAE 和 RMSE。"""
    actual_flat = np.asarray(actual, dtype=float).reshape(-1)
    prediction_flat = np.asarray(prediction, dtype=float).reshape(-1)
    if actual_flat.shape != prediction_flat.shape:
        raise ValueError("实际值和预测值形状不一致")
    if not np.isfinite(actual_flat).all() or not np.isfinite(prediction_flat).all():
        raise ValueError("评估数据中存在缺失或非有限值")

    error = actual_flat - prediction_flat
    residual_sum = float(np.sum(error**2))
    total_sum = float(np.sum((actual_flat - actual_flat.mean()) ** 2))
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return {
        "R2": float(r2),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
    }


def evaluate_predictions(
    actual_dates: list[date],
    actual_load: np.ndarray,
    actual_pv: np.ndarray,
    pred_dates: list[date],
    pred_load: np.ndarray,
    pred_pv: np.ndarray,
) -> list[dict]:
    """按照日期对齐，评估负载和光伏预测。"""
    actual_by_date = {
        target: index for index, target in enumerate(actual_dates)
    }
    missing = [target for target in pred_dates if target not in actual_by_date]
    if missing:
        raise ValueError(f"预测日期在附件 2 中不存在：{missing[:5]}")

    indices = [actual_by_date[target] for target in pred_dates]
    true_load = actual_load[indices]
    true_pv = actual_pv[indices]
    daylight = slice(36, 108)

    return [
        {"model": "小区负载", **regression_metrics(true_load, pred_load)},
        {"model": "光伏发电实际功率", **regression_metrics(true_pv, pred_pv)},
        {
            "model": "光伏白天时段(6:00-18:00)",
            **regression_metrics(true_pv[:, daylight], pred_pv[:, daylight]),
        },
    ]


def write_evaluation_csv(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "R2", "MAE", "RMSE"],
        )
        writer.writeheader()
        writer.writerows(results)


def print_evaluation(results: list[dict]) -> None:
    for result in results:
        print(
            "{model}：R2={R2:.6f}，MAE={MAE:.4f} kW，RMSE={RMSE:.4f} kW".format(
                **result
            )
        )


def fill_initial_missing(values: np.ndarray, name: str) -> np.ndarray:
    complete = np.isfinite(values).all(axis=1)
    valid = np.flatnonzero(complete)
    if not len(valid):
        raise ValueError(f"{name}没有完整预测")
    first = int(valid[0])
    if not complete[first:].all():
        raise ValueError(f"{name}存在中间缺失")
    result = values.copy()
    result[:first] = result[first]
    return result


def build_january_forecasts(
    attachment: Path,
) -> tuple[list[date], np.ndarray, np.ndarray]:
    load_dates, _, load = load_load_history(attachment)
    pv_dates, _, pv = load_pv_history(attachment)
    if load_dates != pv_dates:
        raise ValueError("附件 2 的负载和光伏日期不一致")

    load_lambda, _ = select_lambda(load_dates, load)
    load_prediction = RollingRidgeLoadV4(
        load_dates, load, load_lambda
    ).historical_prediction

    pv_original = original_historical_prediction(pv)
    pv_lambda, pv_blend, _ = select_parameters(pv_dates, pv, pv_original)
    pv_ridge = ExpandingRidgePVV4(pv_dates, pv, pv_lambda).historical_ridge
    pv_prediction = np.full_like(pv, np.nan)
    both = np.isfinite(pv_original) & np.isfinite(pv_ridge)
    pv_prediction[both] = (
        pv_blend * pv_original[both] + (1 - pv_blend) * pv_ridge[both]
    )
    original_only = np.isfinite(pv_original) & ~np.isfinite(pv_prediction)
    pv_prediction[original_only] = pv_original[original_only]
    ridge_only = np.isfinite(pv_ridge) & ~np.isfinite(pv_prediction)
    pv_prediction[ridge_only] = pv_ridge[ridge_only]

    return (
        load_dates,
        fill_initial_missing(load_prediction, "1 月负载预测"),
        fill_initial_missing(pv_prediction, "1 月光伏预测"),
    )


def interval_label(period: int) -> str:
    start = period * 10
    end = (period + 1) * 10

    def clock(minutes: int) -> str:
        day, minute_of_day = divmod(minutes, 24 * 60)
        hour, minute = divmod(minute_of_day, 60)
        suffix = "+1" if day else ""
        return f"{hour}:{minute:02d}{suffix}"

    return f"{clock(start)}-{clock(end)}"


def solve_plan_day(
    net_plan_kwh: np.ndarray,
    price: np.ndarray,
    storage_initial: float,
) -> DailyPlan:
    """固定计划净负荷下的日 MILP，保证日初日末储电量相同。"""
    n = PERIODS_PER_DAY
    blocks = {
        "g": slice(0, n),
        "c": slice(n, 2 * n),
        "d": slice(2 * n, 3 * n),
        "S": slice(3 * n, 4 * n + 1),
        "r": slice(4 * n + 1, 5 * n + 1),
        "z": slice(5 * n + 1, 6 * n + 1),
    }
    size = 6 * n + 1

    objective = np.zeros(size)
    objective[blocks["g"]] = price
    integrality = np.zeros(size, dtype=int)
    integrality[blocks["z"]] = 1

    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    upper[blocks["c"]] = POWER_LIMIT
    upper[blocks["d"]] = POWER_LIMIT
    upper[blocks["z"]] = 1
    lower[blocks["S"]] = SOC_MIN
    upper[blocks["S"]] = SOC_MAX
    lower[blocks["S"].start] = upper[blocks["S"].start] = storage_initial
    lower[blocks["S"].stop - 1] = upper[blocks["S"].stop - 1] = storage_initial

    matrix = lil_matrix((4 * n, size), dtype=float)
    row_lower = np.full(4 * n, -np.inf)
    row_upper = np.zeros(4 * n)
    for t in range(n):
        g = blocks["g"].start + t
        c = blocks["c"].start + t
        d = blocks["d"].start + t
        s = blocks["S"].start + t
        r = blocks["r"].start + t
        z = blocks["z"].start + t

        # g_t + d_t - c_t - spill_t = net_plan_t
        matrix[t, g], matrix[t, d] = 1, 1
        matrix[t, c], matrix[t, r] = -1, -1
        row_lower[t] = row_upper[t] = net_plan_kwh[t]

        row = n + t
        matrix[row, s + 1], matrix[row, s] = 1, -1
        matrix[row, c], matrix[row, d] = -ETA_C, 1 / ETA_D
        row_lower[row] = row_upper[row] = 0

        matrix[2 * n + t, c], matrix[2 * n + t, z] = 1, -POWER_LIMIT
        matrix[3 * n + t, d], matrix[3 * n + t, z] = 1, POWER_LIMIT
        row_upper[3 * n + t] = POWER_LIMIT

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsc(), row_lower, row_upper),
        options={"mip_rel_gap": 0.0},
    )
    if result.x is None or result.status != 0:
        raise RuntimeError(f"第二问单日 MILP 求解失败：{result.message}")

    return DailyPlan(
        purchase_kwh=result.x[blocks["g"]],
        charge_kwh=result.x[blocks["c"]],
        discharge_kwh=result.x[blocks["d"]],
        storage_end_kwh=result.x[blocks["S"]][1:],
    )


def simulate_actual_day(
    planned_purchase: np.ndarray,
    actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray,
    storage_initial: float,
) -> ActualResult:
    """按实际供需执行电池：富余优先充电，缺电优先放电。"""
    charge = np.zeros(PERIODS_PER_DAY)
    discharge = np.zeros(PERIODS_PER_DAY)
    emergency = np.zeros(PERIODS_PER_DAY)
    spill = np.zeros(PERIODS_PER_DAY)
    storage_end = np.zeros(PERIODS_PER_DAY)
    storage = storage_initial

    for t in range(PERIODS_PER_DAY):
        load = actual_load_kw[t] * DT
        pv = actual_pv_kw[t] * DT
        balance = planned_purchase[t] + pv - load

        if balance >= 0:
            charge[t] = min(
                balance,
                POWER_LIMIT,
                (SOC_MAX - storage) / ETA_C,
            )
            spill[t] = max(0.0, balance - charge[t])
        else:
            max_discharge = min(
                -balance,
                POWER_LIMIT,
                ETA_D * (storage - SOC_MIN),
            )
            discharge[t] = max(0.0, max_discharge)
            emergency[t] = max(0.0, -balance - discharge[t])

        storage += ETA_C * charge[t] - discharge[t] / ETA_D
        storage = min(SOC_MAX, max(SOC_MIN, storage))
        storage_end[t] = storage

    runs: list[tuple[str, float]] = []
    active = np.flatnonzero(emergency > 1e-6)
    if len(active):
        start = previous = int(active[0])
        for value in active[1:]:
            period = int(value)
            if period != previous + 1:
                runs.append(
                    (
                        f"{interval_label(start).split('-')[0]}-"
                        f"{interval_label(previous).split('-')[1]}",
                        float(emergency[start:previous + 1].sum()),
                    )
                )
                start = period
            previous = period
        runs.append(
            (
                f"{interval_label(start).split('-')[0]}-"
                f"{interval_label(previous).split('-')[1]}",
                float(emergency[start:previous + 1].sum()),
            )
        )

    return ActualResult(
        storage_initial_kwh=float(storage_initial),
        charge_kwh=charge,
        discharge_kwh=discharge,
        emergency_kwh=emergency,
        spill_kwh=spill,
        storage_end_kwh=storage_end,
        emergency_runs=runs,
    )


def style_sheet(worksheet) -> None:
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(name="宋体", size=10, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    worksheet.freeze_panes = "B2"


def write_result2(
    path: Path,
    output_dates: list[date],
    price: np.ndarray,
    plans: list[DailyPlan],
    actuals: list[ActualResult],
) -> None:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)

    plan_sheet = workbook.create_sheet("计划购电量")
    plan_sheet.append(
        ["日期\\时间", *[interval_label(t) for t in range(PERIODS_PER_DAY)],
         "全天购电量", "全天购电费"]
    )
    for target, plan in zip(output_dates, plans):
        plan_cost = float(price @ plan.purchase_kwh)
        plan_sheet.append(
            [datetime.combine(target, datetime.min.time()),
             *plan.purchase_kwh.tolist(), float(plan.purchase_kwh.sum()), plan_cost]
        )
    for row in plan_sheet.iter_rows(min_row=2):
        row[0].number_format = "yyyy-mm-dd"
        for cell in row[1:]:
            cell.number_format = "0.0000"

    battery_sheet = workbook.create_sheet("充放电量")
    battery_sheet.append(["日期", "时间段", "充电量", "放电量", "时刻", "储电量"])
    for target, actual in zip(output_dates, actuals):
        for block in range(6):
            lo, hi = block * 24, (block + 1) * 24
            battery_sheet.append(
                [
                    datetime.combine(target, datetime.min.time()),
                    f"{interval_label(lo).split('-')[0]}-"
                    f"{interval_label(hi - 1).split('-')[1]}",
                    float(actual.charge_kwh[lo:hi].sum()),
                    float(actual.discharge_kwh[lo:hi].sum()),
                    "0:00" if block == 0 else ("24:00" if block == 1 else None),
                    (
                        float(
                            actual.storage_initial_kwh
                            if block == 0
                            else actual.storage_end_kwh[-1]
                        )
                        if block in (0, 1)
                        else None
                    ),
                ]
            )

    emergency_sheet = workbook.create_sheet("紧急购电量")
    emergency_sheet.append(["日期", "购电时间段", "购电量"])
    for target, actual in zip(output_dates, actuals):
        if not actual.emergency_runs:
            emergency_sheet.append([datetime.combine(target, datetime.min.time()), None, None])
            continue
        for index, (interval, amount) in enumerate(actual.emergency_runs):
            emergency_sheet.append(
                [
                    datetime.combine(target, datetime.min.time()) if index == 0 else None,
                    interval,
                    amount,
                ]
            )

    for worksheet in workbook.worksheets:
        style_sheet(worksheet)
        worksheet.column_dimensions["A"].width = 13
        for column in range(2, worksheet.max_column + 1):
            worksheet.column_dimensions[get_column_letter(column)].width = 11

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=RESULT_DIR / "model_R2_evaluation.csv",
    )
    args = parser.parse_args()

    pred_path = args.pred or (RESULT_DIR / "prediction" / "prediction.xlsx")
    output_path = args.output or (RESULT_DIR / "result2.xlsx")
    price = load_price(APPENDIX_DIR / "附件1.xlsx")
    actual_dates, actual_load, actual_pv = load_actual_series(
        APPENDIX_DIR / "附件2.xlsx"
    )
    pred_dates, pred_load, pred_pv = load_prediction_series(pred_path)

    actual_by_date = {
        target: index for index, target in enumerate(actual_dates)
    }
    if any(target not in actual_by_date for target in pred_dates):
        raise ValueError("pred.xlsx 中存在附件 2 没有的日期")
    evaluation = evaluate_predictions(
        actual_dates,
        actual_load,
        actual_pv,
        pred_dates,
        pred_load,
        pred_pv,
    )
    write_evaluation_csv(args.evaluation_output, evaluation)

    start_time = perf_counter()
    jan_dates, jan_load_pred, jan_pv_pred = build_january_forecasts(
        APPENDIX_DIR / "附件2.xlsx"
    )
    jan_actual_indices = [actual_by_date[target] for target in jan_dates]
    jan_actual_net = (
        actual_load[jan_actual_indices] - actual_pv[jan_actual_indices]
    ) * DT
    jan_pred_net = (jan_load_pred - jan_pv_pred) * DT
    residual_history = list(jan_actual_net - jan_pred_net)

    storage = SOC_INITIAL
    output_dates: list[date] = []
    plans: list[DailyPlan] = []
    actuals: list[ActualResult] = []

    for target in actual_dates:
        actual_index = actual_by_date[target]
        if target in jan_dates:
            pred_index = jan_dates.index(target)
            forecast_net = (jan_load_pred[pred_index] - jan_pv_pred[pred_index]) * DT
        elif OUTPUT_START <= target <= OUTPUT_END:
            pred_index = pred_dates.index(target)
            forecast_net = (pred_load[pred_index] - pred_pv[pred_index]) * DT
        else:
            continue

        recent_residuals = np.asarray(
            residual_history[-RESIDUAL_WINDOW_DAYS:], dtype=float
        )
        risk_margin = np.quantile(
            recent_residuals, RISK_QUANTILE, axis=0, method="linear"
        )
        net_plan = forecast_net + risk_margin
        plan = solve_plan_day(net_plan, price, storage)
        actual = simulate_actual_day(
            plan.purchase_kwh,
            actual_load[actual_index],
            actual_pv[actual_index],
            storage,
        )
        storage = float(actual.storage_end_kwh[-1])

        actual_net_today = (
            actual_load[actual_index] - actual_pv[actual_index]
        ) * DT
        residual_history.append(actual_net_today - forecast_net)

        if OUTPUT_START <= target <= OUTPUT_END:
            output_dates.append(target)
            plans.append(plan)
            actuals.append(actual)

    if output_dates != pred_dates:
        raise ValueError("输出日期与 pred.xlsx 日期不一致")

    write_result2(output_path, output_dates, price, plans, actuals)
    total_plan_kwh = sum(float(plan.purchase_kwh.sum()) for plan in plans)
    total_plan_cost = sum(float(price @ plan.purchase_kwh) for plan in plans)
    total_emergency = sum(float(actual.emergency_kwh.sum()) for actual in actuals)
    total_charge = sum(float(actual.charge_kwh.sum()) for actual in actuals)
    total_discharge = sum(float(actual.discharge_kwh.sum()) for actual in actuals)
    total_emergency_cost = sum(
        float(price @ actual.emergency_kwh) * 5 for actual in actuals
    )
    print(f"求解日期：{output_dates[0]} 至 {output_dates[-1]}")
    print(f"计划购电量：{total_plan_kwh:.6f} kWh")
    print(f"计划购电费：{total_plan_cost:.6f} 元")
    print(f"实际充电量：{total_charge:.6f} kWh")
    print(f"实际放电量：{total_discharge:.6f} kWh")
    print(f"紧急购电量：{total_emergency:.6f} kWh")
    print(f"紧急购电费：{total_emergency_cost:.6f} 元")
    print(f"总费用：{total_plan_cost + total_emergency_cost:.6f} 元")
    print(f"期末储电量：{storage:.6f} kWh")
    print(f"总耗时：{perf_counter() - start_time:.3f} 秒")
    print(f"结果已保存：{output_path}")
    print(f"预测评估已保存：{args.evaluation_output}")
    print_evaluation(evaluation)


if __name__ == "__main__":
    main()
