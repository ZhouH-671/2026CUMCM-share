"""第一问：光伏全部消纳、10 分钟调度、保留二元充放电模式的基础 MILP。

运行：python question1.py
输出：output/question1/solution.json、summary.md 与 result1_chatgpt.xlsx。
Excel 从官方 result1.xlsx 模板生成，输出采用当天 00:00--24:00 时间轴。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, time
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import openpyxl
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
APPENDIX_DIR = ROOT / "附件"
RESULT_DIR = ROOT / "result" / "problem01"

@dataclass(frozen=True)
class Battery:
    dt: float = 1 / 6  # 小时
    eta_c: float = 0.9
    eta_d: float = 0.9
    minimum: float = 1200.0  # kWh，电池内部口径
    maximum: float = 10800.0
    initial: float = 6000.0
    charge_kw: float = 5000.0  # 电池外部口径
    discharge_kw: float = 5000.0


@dataclass(frozen=True)
class Inputs:
    price: np.ndarray  # 元/kWh
    load_kw: np.ndarray
    pv_kw: np.ndarray

    @property
    def periods(self) -> int:
        return len(self.price)


@dataclass
class Model:
    objective: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    constraints: LinearConstraint
    blocks: dict[str, slice]


def time_minutes(value) -> int:
    """兼容附件混用的 Excel 时间、字符串和 0:00+1。"""
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        if value.second or value.microsecond:
            raise ValueError(f"时间包含非零秒：{value}")
        return value.hour * 60 + value.minute
    if isinstance(value, (float, int)):
        minutes = float(value) * 1440
        if not np.isfinite(minutes) or abs(minutes - round(minutes)) > 1e-6:
            raise ValueError(f"无法识别 Excel 时间：{value}")
        return round(minutes)
    text = str(value).strip()
    next_day = text.endswith("+1")
    if next_day:
        text = text[:-2]
    parts = text.split(":")
    if len(parts) not in (2, 3) or (len(parts) == 3 and int(parts[2]) != 0):
        raise ValueError(f"无法识别时间：{value}")
    hour, minute = map(int, parts[:2])
    if not 0 <= minute < 60 or not 0 <= hour <= 24 or (hour == 24 and minute):
        raise ValueError(f"时间超出范围：{value}")
    return hour * 60 + minute + 1440 * next_day


def load_inputs(path: Path) -> Inputs:
    """只读附件 1，并核验 144 条记录的时间顺序；不静默排序或填补缺失值。"""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.iter_rows(values_only=True))
    finally:
        workbook.close()
    expected = ("时间", "电价", "小区负载", "光伏发电预测功率")
    if tuple(rows[0][:4]) != expected:
        raise ValueError(f"附件 1 表头不匹配，应为：{expected}")
    records = [row for row in rows[1:] if any(v is not None for v in row)]
    if len(records) != 144:
        raise ValueError(f"需要 144 条数据，实际为 {len(records)} 条")
    labels = [time_minutes(row[0]) for row in records]
    if labels != list(range(10, 1441, 10)):
        raise ValueError("时间必须按 00:10、00:20、…、24:00 排列")
    values = np.asarray([row[1:4] for row in records], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("附件存在缺失或非有限数值")
    if (values < 0).any():
        raise ValueError("第一问输入的电价、负载和光伏功率应非负")
    return Inputs(values[:, 0], values[:, 1], values[:, 2])


def build_model(data: Inputs, battery: Battery = Battery()) -> Model:
    """变量顺序：[g(144), c(144), d(144), S(145), z(144)]。

    状态 S 含日初；其他变量按时段排列。所有充放电变量单位为 kWh。
    """
    t_count = data.periods
    for values in (data.price, data.load_kw, data.pv_kw):
        if np.shape(values) != (t_count,) or not np.isfinite(values).all():
            raise ValueError("输入向量必须等长且为有限数值")
        if np.any(values < 0):
            raise ValueError("第一问输入应非负")
    if t_count < 1:
        raise ValueError("至少需要一个时段")
    if not all(np.isfinite(v) for v in asdict(battery).values()):
        raise ValueError("电池参数必须为有限数值")
    if not (0 < battery.eta_c <= 1 and 0 < battery.eta_d <= 1):
        raise ValueError("效率应在 (0, 1] 内")
    if not (0 <= battery.minimum <= battery.initial <= battery.maximum):
        raise ValueError("初始储电量必须处于运行范围内")
    if battery.dt <= 0 or min(battery.charge_kw, battery.discharge_kw) < 0:
        raise ValueError("时段长度须为正，充放电功率上限须非负")

    n = t_count
    blocks = {
        "g": slice(0, n), "c": slice(n, 2 * n), "d": slice(2 * n, 3 * n),
        "S": slice(3 * n, 4 * n + 1), "z": slice(4 * n + 1, 5 * n + 1),
    }
    size = 5 * n + 1
    objective = np.zeros(size)
    objective[blocks["g"]] = data.price  # min sum(p_t * g_t)
    integrality = np.zeros(size, dtype=int)
    integrality[blocks["z"]] = 1

    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    c_max = battery.charge_kw * battery.dt
    d_max = battery.discharge_kw * battery.dt
    upper[blocks["c"]] = c_max
    upper[blocks["d"]] = d_max
    lower[blocks["S"]] = battery.minimum
    upper[blocks["S"]] = battery.maximum
    upper[blocks["z"]] = 1
    # S_0 = S_T = 6000，通过相等的上下界固定。
    for position in (blocks["S"].start, blocks["S"].stop - 1):
        lower[position] = upper[position] = battery.initial

    matrix = lil_matrix((4 * n, size), dtype=float)
    row_lower = np.full(4 * n, -np.inf)
    row_upper = np.zeros(4 * n)
    demand = (data.load_kw - data.pv_kw) * battery.dt
    for t in range(n):
        g, c, d, s, z = [blocks[key].start + t for key in ("g", "c", "d", "S", "z")]
        # g_t - c_t + d_t = (L_t - PV_t) * dt；光伏全部消纳。
        matrix[t, g], matrix[t, c], matrix[t, d] = 1, -1, 1
        row_lower[t] = row_upper[t] = demand[t]
        # S_{t+1} - S_t - eta_c*c_t + d_t/eta_d = 0。
        row = n + t
        matrix[row, s + 1], matrix[row, s] = 1, -1
        matrix[row, c], matrix[row, d] = -battery.eta_c, 1 / battery.eta_d
        row_lower[row] = row_upper[row] = 0
        # c_t <= C_max*z_t；d_t <= D_max*(1-z_t)。
        matrix[2 * n + t, c], matrix[2 * n + t, z] = 1, -c_max
        matrix[3 * n + t, d], matrix[3 * n + t, z] = 1, d_max
        row_upper[3 * n + t] = d_max

    return Model(objective, integrality, Bounds(lower, upper),
                 LinearConstraint(matrix.tocsc(), row_lower, row_upper), blocks)


def solve_model(model: Model, *, time_limit: float = 60.0,
                relative_gap: float = 0.0, relax: bool = False,
                fixed_z: np.ndarray | None = None):
    """精确 MILP；relax=True 仅用于 LP 下界诊断。

    fixed_z 给定完整 0/1 模式后，剩余问题是 LP。这是未来 GA+LP 的内层接口。
    该接口不使用罚函数，不自动放宽日末条件，也不修补不可行模式。
    """
    if time_limit <= 0 or relative_gap < 0:
        raise ValueError("求解时间须为正，最优性间隙须非负")
    lower, upper = model.bounds.lb.copy(), model.bounds.ub.copy()
    integer = model.integrality.copy()
    if relax:
        integer[:] = 0
    if fixed_z is not None:
        mode = np.asarray(fixed_z, dtype=float)
        block = model.blocks["z"]
        if mode.shape != (block.stop - block.start,) or not np.isin(mode, [0, 1]).all():
            raise ValueError("fixed_z 必须为与时段数相同的 0/1 向量")
        lower[block] = upper[block] = mode
        integer[:] = 0
    start = perf_counter()
    result = milp(model.objective, integrality=integer,
                  bounds=Bounds(lower, upper), constraints=model.constraints,
                  options={"time_limit": time_limit, "mip_rel_gap": relative_gap})
    return result, perf_counter() - start


def validate_solution(data: Inputs, battery: Battery, model: Model,
                      x: np.ndarray, tolerance: float = 1e-5) -> dict:
    """独立用物理关系核验解，不仅检查求解器的成功标志。"""
    if x is None or not np.isfinite(x).all():
        raise ValueError("没有有限的可行解")
    g, c, d, state, mode = [x[model.blocks[key]] for key in ("g", "c", "d", "S", "z")]
    load, pv = data.load_kw * battery.dt, data.pv_kw * battery.dt
    errors = {
        "balance_kwh": float(np.max(np.abs(g + pv + d - load - c))),
        "storage_transition_kwh": float(np.max(np.abs(
            np.diff(state) - battery.eta_c * c + d / battery.eta_d))),
        "storage_bounds_kwh": max(0.0, float(battery.minimum - state.min()),
                                   float(state.max() - battery.maximum)),
        "boundary_kwh": float(max(abs(state[0] - battery.initial), abs(state[-1] - battery.initial))),
        "nonnegative_kwh": max(0.0, float(-min(g.min(), c.min(), d.min()))),
        "charge_limit_kwh": max(0.0, float(np.max(c - battery.charge_kw * battery.dt * mode))),
        "discharge_limit_kwh": max(0.0, float(np.max(d - battery.discharge_kw * battery.dt * (1 - mode)))),
        "mode_bounds": max(0.0, float(-mode.min()), float(mode.max() - 1)),
        "mode_integrality": float(np.max(np.abs(mode - np.rint(mode)))),
        "simultaneous_flow_kwh": float(np.max(np.minimum(c, d))),
        "daily_cycle_kwh": float(abs(d.sum() - battery.eta_c * battery.eta_d * c.sum())),
        "daily_energy_kwh": float(abs(g.sum() - load.sum() + pv.sum()
                                       - (1 - battery.eta_c * battery.eta_d) * c.sum())),
    }
    failures = {key: value for key, value in errors.items() if value > tolerance}
    if failures:
        raise ValueError(f"解未通过可行性检查：{failures}")
    return {"passed": True, "absolute_tolerance": tolerance, "residuals": errors}


def clock_label(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def result_clock(minutes: int) -> str:
    """把绝对分钟数格式化为 result1 模板使用的时间标签。"""
    day, minute_of_day = divmod(minutes, 24 * 60)
    hour, minute = divmod(minute_of_day, 60)
    suffix = "+1" if day else ""
    return f"{hour}:{minute:02d}{suffix}"


def normalize_interval(label: str) -> str:
    """统一模板和内部区间的格式，例如 00:00-04:00 -> 0:00-4:00。"""
    try:
        start, end = str(label).strip().split("-", maxsplit=1)
        start_minutes = time_minutes(start)
        end_minutes = time_minutes(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法识别时间段：{label!r}") from exc
    return f"{result_clock(start_minutes)}-{result_clock(end_minutes)}"


def write_result_workbook(template: Path, output: Path, payload: dict) -> None:
    """按官方 result1 模板结构写入题目要求的计划购电与充放电结果。"""
    if not template.exists():
        raise FileNotFoundError(f"找不到 result1 模板：{template}")

    dispatch = payload["ten_minute_dispatch"]
    if len(dispatch) != 144:
        raise ValueError(f"计划购电量应有 144 条，实际为 {len(dispatch)} 条")

    workbook = openpyxl.load_workbook(template)
    try:
        plan = workbook["计划购电量"]
        if plan.max_row < 145:
            raise ValueError("“计划购电量”模板行数不足")
        for offset, item in enumerate(dispatch):
            row = offset + 2
            start_minutes = offset * 10
            end_minutes = start_minutes + 10
            plan.cell(row=row, column=1).value = (
                f"{result_clock(start_minutes)}-{result_clock(end_minutes)}"
            )
            plan.cell(row=row, column=2).value = float(item["purchase_kwh"])

        battery_sheet = workbook["充放电量"]
        four_hour_map = {
            normalize_interval(item["interval"]): item
            for item in payload["four_hour_summary"]
        }
        matched = set()
        for row in range(2, 8):
            label = battery_sheet.cell(row=row, column=1).value
            if label is None:
                continue
            key = normalize_interval(label)
            item = four_hour_map.get(key)
            if item is None:
                raise ValueError(f"充放电模板中存在未匹配区间：{label!r}")
            battery_sheet.cell(row=row, column=2).value = float(item["charge_kwh"])
            battery_sheet.cell(row=row, column=3).value = float(item["discharge_kwh"])
            matched.add(key)
        if matched != set(four_hour_map):
            missing = sorted(set(four_hour_map) - matched)
            raise ValueError(f"充放电模板缺少区间：{missing}")

        summary = payload["summary"]
        storage_values = {
            "0:00": float(summary["storage_initial_kwh"]),
            "24:00": float(summary["storage_final_kwh"]),
        }
        for row in range(2, 8):
            label = battery_sheet.cell(row=row, column=4).value
            if label is None:
                continue
            value = storage_values.get(str(label).strip())
            if value is not None:
                battery_sheet.cell(row=row, column=5).value = value

        output.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(output)
    finally:
        workbook.close()


def save_results(path: Path, source: Path, data: Inputs, battery: Battery,
                 model: Model, result, elapsed: float, checks: dict,
                 relaxation: dict | None) -> dict:
    g, c, d, state, mode = [result.x[model.blocks[key]] for key in ("g", "c", "d", "S", "z")]
    periods = []
    for t in range(data.periods):
        start, end = round(t * battery.dt * 60), round((t + 1) * battery.dt * 60)
        periods.append({"t": t + 1, "interval": f"{clock_label(start)}-{clock_label(end)}",
                        "source_end_time": clock_label(end), "price_yuan_per_kwh": float(data.price[t]),
                        "load_kw": float(data.load_kw[t]), "pv_kw": float(data.pv_kw[t]),
                        "purchase_kwh": float(g[t]), "charge_kwh": float(c[t]),
                        "discharge_kwh": float(d[t]), "storage_start_kwh": float(state[t]),
                        "storage_end_kwh": float(state[t + 1]), "z": int(round(mode[t])),
                        "cost_yuan": float(data.price[t] * g[t])})
    blocks = []
    for k in range(6):
        lo, hi = 24 * k, 24 * (k + 1)
        blocks.append({"interval": f"{clock_label(240*k)}-{clock_label(240*(k+1))}",
                       "charge_kwh": float(c[lo:hi].sum()), "discharge_kwh": float(d[lo:hi].sum())})
    summary = {
        "purchase_kwh": float(g.sum()), "cost_yuan": float(data.price @ g),
        "charge_kwh": float(c.sum()), "discharge_kwh": float(d.sum()),
        "storage_initial_kwh": float(state[0]), "storage_final_kwh": float(state[-1]),
        "storage_min_kwh": float(state.min()), "storage_max_kwh": float(state.max()),
        "load_kwh": float(data.load_kw.sum() * battery.dt),
        "pv_kwh": float(data.pv_kw.sum() * battery.dt),
        "storage_loss_kwh": float(c.sum() - d.sum()),
    }
    def optional_number(key):
        value = result.get(key)
        return float(value) if value is not None and np.isfinite(value) else None
    payload = {
        "input": str(source.resolve()), "battery": asdict(battery),
        "assumptions": ["输入时间为时段结束时刻，功率视为时段平均功率", "光伏全部消纳，不售电",
                        "充放电效率各90%，电量及功率上限按电池外部计量", "日初日末储电量均6000kWh"],
        "solver": {"name": "scipy.optimize.milp / HiGHS", "scipy_version": scipy.__version__,
                   "status": int(result.status), "message": result.message, "seconds": elapsed,
                   "optimal_within_solver_tolerances": bool(result.status == 0),
                   "mip_gap": optional_number("mip_gap"), "lower_bound_yuan": optional_number("mip_dual_bound")},
        "summary": summary, "validation": checks, "lp_relaxation_diagnostic": relaxation,
        "four_hour_summary": blocks, "ten_minute_dispatch": periods,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "solution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 第一问基础 MILP 求解结果", "", "采用输入标签为时段结束时刻的约定，光伏全部消纳。",
             "", f"- 求解状态：{result.message}", f"- 求解时间：{elapsed:.4f} 秒",
             f"- 最优性相对间隙：{optional_number('mip_gap')}",
             f"- 全天购电量：{summary['purchase_kwh']:.6f} kWh", f"- 全天购电费：{summary['cost_yuan']:.6f} 元",
             f"- 日初 / 日末储电量：{state[0]:.6f} / {state[-1]:.6f} kWh", "- 全部物理约束检查通过。", "",
             "## 表 1：指定时段购电量", "", "| 时段 | 购电量（kWh） |", "|---|---:|"]
    for t in (60, 72, 84, 96, 108, 120):
        lines.append(f"| {periods[t]['interval']} | {g[t]:.6f} |")
    lines += ["", "## 表 2：四小时充放电量", "", "| 时段 | 充电量（kWh） | 放电量（kWh） |", "|---|---:|---:|"]
    for block in blocks:
        lines.append(f"| {block['interval']} | {block['charge_kwh']:.6f} | {block['discharge_kwh']:.6f} |")
    lines += ["", "完整 144 时段结果及约束残差见 solution.json；题目要求的表格结果见 result1_chatgpt.xlsx。", ""]
    (path / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=APPENDIX_DIR / "附件1.xlsx")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--excel-template", type=Path,
                        default=APPENDIX_DIR / "附件5" / "result1.xlsx")
    parser.add_argument("--excel-output", type=Path,
                        default=RESULT_DIR / "result01.xlsx")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--check-lp-relaxation", action="store_true", help="额外求 LP 松弛下界，仅诊断，不替代 MILP 决策")
    args = parser.parse_args()
    data, battery = load_inputs(args.input), Battery()
    model = build_model(data, battery)
    result, elapsed = solve_model(model, time_limit=args.time_limit, relative_gap=args.mip_rel_gap)
    if result.x is None or result.status not in (0, 1):
        raise RuntimeError(f"求解失败，未输出调度结果：{result.message}")
    checks = validate_solution(data, battery, model, result.x)
    relaxation = None
    if args.check_lp_relaxation:
        lp, lp_elapsed = solve_model(model, time_limit=args.time_limit, relax=True)
        relaxation = {"status": int(lp.status), "message": lp.message, "seconds": lp_elapsed}
        if lp.status == 0:
            relaxation.update({"lower_bound_yuan": float(lp.fun),
                               "milp_minus_lp_yuan": float(result.fun - lp.fun),
                               "simultaneous_slots": int(np.count_nonzero(
                                   (lp.x[model.blocks['c']] > 1e-5) & (lp.x[model.blocks['d']] > 1e-5)))})
    payload = save_results(args.output_dir, args.input, data, battery, model, result, elapsed, checks, relaxation)
    write_result_workbook(args.excel_template, args.excel_output, payload)
    print(json.dumps({"solver": payload["solver"], "summary": payload["summary"],
                      "lp_relaxation": relaxation, "output_dir": str(args.output_dir),
                      "excel_output": str(args.excel_output)}, ensure_ascii=False, indent=2))
    # 达到时间限制但有可行解时保存结果，以非零退出码明确其未证明最优。
    return 0 if result.status == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
