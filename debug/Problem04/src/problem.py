"""问题四-2：波动电价下的计划购电策略。

运行（在项目根目录下）：
    python src/problem_zh.py

输入：
    data/appendix02.xlsx                第 1 页实际负载，第 2 页实际光伏
    data/appendix04.xlsx                实际电价，用于紧急购电结算
    data/pred.xlsx                      第 1 页预测负载，第 2 页预测光伏
    result/comparison/prediction_HoltWinters.xlsx  预测电价
    src/depend/comparison_load.py       1 月暖启动负载预测
    src/depend/comparison_pv.py         1 月暖启动光伏预测

输出：
    result/result4-2.xlsx

说明：
    与 problem.py（问题二）相比，仅修改数据来源与路径：
        - 计划阶段：用 prediction_HoltWinters.xlsx 的预测电价做 MILP；
        - 紧急购电：用 appendix04.xlsx 的实际电价的 5 倍结算；
    1 月只做暖启动，构造残差历史，不做 MILP 计划；
    算法逻辑、MILP 约束、模拟逻辑、风险余量、储能传递完全保持不变。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


# ============================================================
# 路径设置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../src
PROJECT_ROOT = SCRIPT_DIR.parent                      # 项目根

DEPEND_DIR = SCRIPT_DIR / "depend"
if str(DEPEND_DIR) not in sys.path:
    sys.path.insert(0, str(DEPEND_DIR))

from comparison_load import (  # noqa: E402
    RollingRidgeLoadV4,
    load_history as load_load_history,
    select_lambda,
)
from comparison_pv import (  # noqa: E402
    ExpandingRidgePVV4,
    load_history as load_pv_history,
    original_historical_prediction,
    select_parameters,
)


# ============================================================
# 常量
# ============================================================
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

# 路径常量
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "result"
COMPARISON_DIR = RESULT_DIR / "comparison"

DATA_PATH = DATA_DIR / "appendix02.xlsx"
ACTUAL_PRICE_PATH = DATA_DIR / "appendix04.xlsx"
PRED_PATH = DATA_DIR / "pred.xlsx"
PRICE_PRED_PATH = COMPARISON_DIR / "pred_price_rolling.xlsx"
OUTPUT_PATH = RESULT_DIR / "result4-2.xlsx"


# ============================================================
# 数据结构
# ============================================================
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


# ============================================================
# 数据读取
# ============================================================
def excel_date(value) -> date:
    """把 Excel 中的日期值统一转成 datetime.date。"""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise TypeError(f"日期格式错误：{value!r}")


def load_price_prediction(path: Path) -> dict[date, np.ndarray]:
    """读取 prediction_HoltWinters.xlsx，返回 {日期: 144 维预测电价}。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到预测电价文件：{path}")

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.worksheets[0]
        rows = list(worksheet.iter_rows(values_only=True))
    finally:
        workbook.close()

    if not rows:
        raise ValueError("预测电价文件为空")

    result: dict[date, np.ndarray] = {}
    for row in rows[1:]:
        if row[0] in (None, ""):
            continue
        target = excel_date(row[0])
        values = np.asarray(row[1:1 + PERIODS_PER_DAY], dtype=float)
        if values.shape != (PERIODS_PER_DAY,):
            raise ValueError(
                f"{target} 的预测电价列数不是 {PERIODS_PER_DAY}，"
                f"实际 {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{target} 的预测电价存在缺失值")
        result[target] = np.maximum(0.0, values)

    if not result:
        raise ValueError("预测电价文件中没有有效数据")
    return result


def load_actual_price_series(path: Path) -> dict[date, np.ndarray]:
    """读取 appendix04.xlsx，返回 {日期: 144 维实际电价}。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到实际电价文件：{path}")

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.worksheets[0]
        rows = list(worksheet.iter_rows(values_only=True))
    finally:
        workbook.close()

    if not rows:
        raise ValueError("实际电价文件为空")

    result: dict[date, np.ndarray] = {}
    for row in rows[1:]:
        if row[0] in (None, ""):
            continue
        target = excel_date(row[0])
        values = np.asarray(row[1:1 + PERIODS_PER_DAY], dtype=float)
        if values.shape != (PERIODS_PER_DAY,):
            raise ValueError(
                f"{target} 的实际电价列数不是 {PERIODS_PER_DAY}，"
                f"实际 {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{target} 的实际电价存在缺失值")
        result[target] = np.maximum(0.0, values)

    if not result:
        raise ValueError("实际电价文件中没有有效数据")
    return result


def load_actual_series(path: Path) -> tuple[list[date], np.ndarray, np.ndarray]:
    """读取 appendix02.xlsx 第 1 页实际负载和第 2 页实际光伏。"""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheets = workbook.worksheets
        if len(worksheets) < 2:
            raise ValueError("appendix02.xlsx 至少需要 2 个工作表")

        actual_sheets = [worksheets[0], worksheets[1]]

        matrices = []
        dates = None
        for worksheet in actual_sheets:
            rows = list(worksheet.iter_rows(values_only=True))
            current_dates = [
                excel_date(row[0]) for row in rows[1:] if row[0] is not None
            ]
            values = np.asarray(
                [row[1:] for row in rows[1:] if row[0] is not None],
                dtype=float,
            )
            if values.shape != (365, PERIODS_PER_DAY):
                raise ValueError(
                    f"{worksheet.title} 数据形状错误：{values.shape}，应为 (365, 144)"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{worksheet.title} 存在缺失值")
            if dates is None:
                dates = current_dates
            elif current_dates != dates:
                raise ValueError("实际负载和实际光伏的日期不一致")
            matrices.append(np.maximum(0.0, values))
    finally:
        workbook.close()

    return dates, matrices[0], matrices[1]


def load_prediction_series(path: Path) -> tuple[list[date], np.ndarray, np.ndarray]:
    """读取 pred.xlsx 第 1 页预测负载和第 2 页预测光伏。"""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheets = workbook.worksheets
        if len(worksheets) < 2:
            raise ValueError("pred.xlsx 至少需要 2 个工作表")

        matrices = []
        dates = None
        for worksheet in worksheets[:2]:
            rows = list(worksheet.iter_rows(values_only=True))
            current_dates = [
                excel_date(row[0]) for row in rows[1:] if row[0] is not None
            ]
            values = np.asarray(
                [row[1:] for row in rows[1:] if row[0] is not None],
                dtype=float,
            )
            if values.shape != (334, PERIODS_PER_DAY):
                raise ValueError(
                    f"pred.xlsx 的 {worksheet.title} 应为 334 x 144，"
                    f"实际为 {values.shape}"
                )
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

    if dates != sorted(dates):
        raise ValueError("pred.xlsx 的日期必须按升序排列")

    return dates, matrices[0], matrices[1]


# ============================================================
# 1 月暖启动预测
# ============================================================
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
    """用 appendix02.xlsx 的实际数据构造 1 月预测。"""
    load_dates, _, load = load_load_history(attachment)
    pv_dates, _, pv = load_pv_history(attachment)
    if load_dates != pv_dates:
        raise ValueError("appendix02.xlsx 的负载和光伏日期不一致")

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


# ============================================================
# 时间标签
# ============================================================
def interval_label(period: int) -> str:
    start = period * 10
    end = (period + 1) * 10

    def clock(minutes: int) -> str:
        day, minute_of_day = divmod(minutes, 24 * 60)
        hour, minute = divmod(minute_of_day, 60)
        suffix = "+1" if day else ""
        return f"{hour}:{minute:02d}{suffix}"

    return f"{clock(start)}-{clock(end)}"


# ============================================================
# 单日 MILP 计划模型（与问题二完全一致）
# ============================================================
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
        raise RuntimeError(f"问题四-2 单日 MILP 求解失败：{result.message}")

    return DailyPlan(
        purchase_kwh=result.x[blocks["g"]],
        charge_kwh=result.x[blocks["c"]],
        discharge_kwh=result.x[blocks["d"]],
        storage_end_kwh=result.x[blocks["S"]][1:],
    )


# ============================================================
# 实际运行（与问题二完全一致）
# ============================================================
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


# ============================================================
# 结果写入
# ============================================================
def style_sheet(worksheet) -> None:
    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(name="宋体", size=10, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    worksheet.freeze_panes = "B2"


def write_result4_2(
    path: Path,
    output_dates: list[date],
    predicted_price_by_date: dict[date, np.ndarray],
    actual_price_by_date: dict[date, np.ndarray],
    plans: list[DailyPlan],
    actuals: list[ActualResult],
) -> None:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)

    # 第一页：计划购电量
    plan_sheet = workbook.create_sheet("计划购电量")
    plan_sheet.append(
        ["日期\\时间", *[interval_label(t) for t in range(PERIODS_PER_DAY)],
         "全天购电量", "全天购电费"]
    )
    for target, plan in zip(output_dates, plans):
        predicted_price = predicted_price_by_date[target]
        plan_cost = float(predicted_price @ plan.purchase_kwh)
        plan_sheet.append(
            [
                datetime.combine(target, datetime.min.time()),
                *plan.purchase_kwh.tolist(),
                float(plan.purchase_kwh.sum()),
                plan_cost,
            ]
        )
    for row in plan_sheet.iter_rows(min_row=2):
        row[0].number_format = "yyyy-mm-dd"
        for cell in row[1:]:
            cell.number_format = "0.0000"

    # 第二页：充放电量
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
                            else actual.storage_end_kwh[hi - 1]
                        )
                        if block in (0, 1)
                        else None
                    ),
                ]
            )

    # 第三页：紧急购电量
    emergency_sheet = workbook.create_sheet("紧急购电量")
    emergency_sheet.append(["日期", "购电时间段", "购电量"])
    for target, actual in zip(output_dates, actuals):
        if not actual.emergency_runs:
            emergency_sheet.append(
                [datetime.combine(target, datetime.min.time()), None, None]
            )
            continue
        for index, (interval, amount) in enumerate(actual.emergency_runs):
            emergency_sheet.append(
                [
                    datetime.combine(target, datetime.min.time())
                    if index == 0
                    else None,
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


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, default=PRED_PATH)
    parser.add_argument(
        "--price-pred",
        type=Path,
        default=PRICE_PRED_PATH,
    )
    parser.add_argument(
        "--actual-price",
        type=Path,
        default=ACTUAL_PRICE_PATH,
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    data_path = DATA_PATH
    pred_path = args.pred
    price_pred_path = args.price_pred
    actual_price_path = args.actual_price
    output_path = args.output

    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到预测文件：{pred_path}")
    if not price_pred_path.exists():
        raise FileNotFoundError(f"找不到预测电价文件：{price_pred_path}")
    if not actual_price_path.exists():
        raise FileNotFoundError(f"找不到实际电价文件：{actual_price_path}")

    # 预测电价：prediction_HoltWinters.xlsx
    predicted_price_by_date = load_price_prediction(price_pred_path)
    print(f"[电价] 读取预测电价 {len(predicted_price_by_date)} 天，"
          f"{min(predicted_price_by_date)} 至 {max(predicted_price_by_date)}")

    # 实际电价：appendix04.xlsx
    actual_price_by_date = load_actual_price_series(actual_price_path)
    print(f"[电价] 读取实际电价 {len(actual_price_by_date)} 天，"
          f"{min(actual_price_by_date)} 至 {max(actual_price_by_date)}")

    # 实际负载和实际光伏：appendix02.xlsx 第 1、2 页
    actual_dates, actual_load, actual_pv = load_actual_series(data_path)

    # 预测数据：pred.xlsx
    pred_dates, pred_load, pred_pv = load_prediction_series(pred_path)

    actual_by_date = {target: index for index, target in enumerate(actual_dates)}
    if any(target not in actual_by_date for target in pred_dates):
        raise ValueError("pred.xlsx 中存在 appendix02.xlsx 没有的日期")

    missing_price = [d for d in pred_dates if d not in predicted_price_by_date]
    if missing_price:
        raise ValueError(
            f"预测电价缺少以下日期：{missing_price[:5]} ... "
            f"共 {len(missing_price)} 天"
        )

    missing_actual_price = [
        d for d in pred_dates if d not in actual_price_by_date
    ]
    if missing_actual_price:
        raise ValueError(
            f"实际电价缺少以下日期：{missing_actual_price[:5]} ... "
            f"共 {len(missing_actual_price)} 天"
        )

    start_time = perf_counter()

    # 1 月暖启动预测
    jan_dates, jan_load_pred, jan_pv_pred = build_january_forecasts(data_path)

    # ========================================================
    # 第一段：1 月暖启动，只构造残差历史，不做 MILP 计划
    # ========================================================
    residual_history: list[np.ndarray] = []
    for target in jan_dates:
        actual_index = actual_by_date[target]
        pred_index = jan_dates.index(target)
        forecast_net = (
            jan_load_pred[pred_index] - jan_pv_pred[pred_index]
        ) * DT
        actual_net_today = (
            actual_load[actual_index] - actual_pv[actual_index]
        ) * DT
        residual_history.append(actual_net_today - forecast_net)

    # ========================================================
    # 第二段：2 月到 12 月，按 pred_dates 顺序生成计划与结果
    # ========================================================
    storage = SOC_INITIAL
    output_dates: list[date] = []
    plans: list[DailyPlan] = []
    actuals: list[ActualResult] = []

    for target in pred_dates:
        actual_index = actual_by_date[target]
        pred_index = pred_dates.index(target)
        forecast_net = (pred_load[pred_index] - pred_pv[pred_index]) * DT

        # 当天预测电价，用于 MILP 决策
        price_today = predicted_price_by_date[target]

        recent_residuals = np.asarray(
            residual_history[-RESIDUAL_WINDOW_DAYS:], dtype=float
        )
        risk_margin = np.quantile(
            recent_residuals, RISK_QUANTILE, axis=0, method="linear"
        )
        net_plan = forecast_net + risk_margin

        plan = solve_plan_day(net_plan, price_today, storage)
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

        output_dates.append(target)
        plans.append(plan)
        actuals.append(actual)

    if output_dates != pred_dates:
        raise ValueError("输出日期与 pred.xlsx 日期不一致")

    write_result4_2(
        output_path,
        output_dates,
        predicted_price_by_date,
        actual_price_by_date,
        plans,
        actuals,
    )

    total_plan_kwh = sum(float(plan.purchase_kwh.sum()) for plan in plans)
    total_plan_cost = sum(
        float(predicted_price_by_date[target] @ plan.purchase_kwh)
        for target, plan in zip(output_dates, plans)
    )
    total_emergency = sum(float(actual.emergency_kwh.sum()) for actual in actuals)
    total_charge = sum(float(actual.charge_kwh.sum()) for actual in actuals)
    total_discharge = sum(float(actual.discharge_kwh.sum()) for actual in actuals)
    total_emergency_cost = sum(
        float(5 * actual_price_by_date[target] @ actual.emergency_kwh)
        for target, actual in zip(output_dates, actuals)
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


if __name__ == "__main__":
    main()