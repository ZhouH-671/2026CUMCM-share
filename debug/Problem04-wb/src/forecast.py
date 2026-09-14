"""第四问之三：三种电价预测算法的对比（question4-3）。

目的很单纯：**只做电价预测**，不做购电优化，不引入多余参数。
对同一条目标序列（日均电价）用三种算法做**无泄露滚动 1 步预测**：

    1) 滚动均值    pred(d) = 最近 K 天日均价的均值（K 默认 7）
    2) SARIMA      SARIMA(p,d,q)(P,D,Q)_s，默认 (1,0,1)(1,0,1)_7（自实现，不依赖 statsmodels）
    3) Holt-Winters 加性趋势 + 加性周季节（m=7）

三者的预测对象、滚动方式、结算口径完全一致，差别只在算法本身，便于公平对比。

预测电价的重建（与 question4 / question4-2 一致）
    日均价预测值 -> 偏离水平 = 预测日均价 - mean(附件1)
    预测电价(slot) = max( 附件1(slot) + 偏离水平, 0 )
即"附件1 经验日内形状 + 模型预测的价格水平"。

输出（question4-3/result/）
    pred_price_rolling.xlsx        滚动均值预测电价（含 实际电价 表）
    pred_price_sarima.xlsx         SARIMA 预测电价（含 实际电价 表）
    pred_price_holtwinters.xlsx    Holt-Winters 预测电价（含 实际电价 表）
    price_forecast_metrics.csv     MAE / RMSE / R^2 指标
    fig_sample_day_compare.svg     某一天 0-24 的预测价 vs 实际价（三算法 + 实际）
    fig_daily_mean_compare.svg     全年日均价：实际 vs 三种预测
    fig_r2_compare.svg             三种算法的 R^2 / MAE 对比

运行（在 question4-3 目录下）：
    python src/price_forecast.py
    python src/price_forecast.py --ma-window 7 --sarima-order 1,0,1,1,0,1,7
    python src/price_forecast.py --sample-day 2025-06-23
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import openpyxl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
RESULT_DIR = PROJECT_ROOT / "result" / "forecast"

CANDIDATE_DATA_DIRS = (
    PROJECT_ROOT / "data",
    REPO_ROOT / "question4" / "data",
    REPO_ROOT / "question3-2" / "data",
    REPO_ROOT / "附件",
)
DATA_DIR = next(
    (path for path in CANDIDATE_DATA_DIRS if (path / "appendix04.xlsx").is_file()),
    CANDIDATE_DATA_DIRS[0],
)
CANDIDATE_DEPEND_DIRS = (
    SCRIPT_DIR / "depend",
    PROJECT_ROOT / "depend",
    REPO_ROOT / "question3-2" / "depend",
)
DEPEND_DIR = next(
    (path for path in CANDIDATE_DEPEND_DIRS if (path / "question_2nd.py").is_file()),
    CANDIDATE_DEPEND_DIRS[0],
)
if str(DEPEND_DIR) not in sys.path:
    sys.path.insert(0, str(DEPEND_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import question_2nd as q2  # noqa: E402
import sarima  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PERIODS_PER_DAY = 144
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
MA_WINDOW = 7
SARIMA_ORDER = (1, 0, 1, 1, 0, 1, 7)
SARIMA_WINDOW = 210
SARIMA_MAXITER = 120
# 建模所需的最少历史（天）：5 个完整周，保证周季节项可辨识。
# 低于该值才退化为"扩张均值"，避免早期用噪声拟合 SARIMA。
SARIMA_MIN_HISTORY = 35
HW_SEASON = 7
HW_WINDOW = 120
HW_MAXITER = 150
SAMPLE_SEED = 20250911

METHODS = ("rolling", "sarima", "holtwinters")
METHOD_LABELS = {
    "rolling": "滚动均值 (K=%d)",
    "sarima": "SARIMA%s",
    "holtwinters": "Holt-Winters (m=%d)",
}
METHOD_FILES = {
    "rolling": "pred_price_rolling.xlsx",
    "sarima": "pred_price_sarima.xlsx",
    "holtwinters": "pred_price_holtwinters.xlsx",
}
METHOD_COLORS = {"rolling": "#4c72b0", "sarima": "#dd8452", "holtwinters": "#55a868"}


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_actual_price(path: Path):
    workbook = openpyxl.load_workbook(path, data_only=True)
    sheet = workbook.worksheets[0]
    rows = list(sheet.iter_rows(values_only=True))
    header = [value for value in rows[0][1:1 + PERIODS_PER_DAY]]
    dates, prices = [], []
    for row in rows[1:]:
        if row[0] is None:
            continue
        dates.append(q2.excel_date(row[0]))
        prices.append([float(value) for value in row[1:1 + PERIODS_PER_DAY]])
    return dates, np.asarray(prices, dtype=float), header


# --------------------------------------------------------------------------
# 三种预测算法：都预测"日均电价"，都只看历史
# --------------------------------------------------------------------------
def rolling_levels(daily: np.ndarray, dates: list, window: int) -> dict:
    levels = {}
    for index, day in enumerate(dates):
        if index == 0:
            levels[day] = None
            continue
        history = daily[max(0, index - window):index]
        levels[day] = float(np.mean(history))
    return levels


def holtwinters_levels(daily: np.ndarray, dates: list, season: int,
                       window: int, maxiter: int) -> dict:
    """加性 Holt-Winters（水平 + 趋势 + 周季节），每天用历史重新拟合。"""
    levels = {}
    for index, day in enumerate(dates):
        history = daily[:index]
        if history.size < 2 * season:
            levels[day] = float(np.mean(history)) if history.size else None
            continue
        y = np.asarray(history[-window:], dtype=float)
        try:
            levels[day] = _hw_forecast_one(y, season, maxiter)
        except Exception:
            levels[day] = float(np.mean(y))
    return levels


def _hw_fit(y: np.ndarray, m: int, alpha: float, beta: float, gamma: float):
    n = y.size
    seasonal = np.zeros(n)
    level = float(y[:m].mean())
    seasonal[:m] = y[:m] - level
    seasonal[:m] -= seasonal[:m].mean()
    trend = float((y[m:2 * m].mean() - y[:m].mean()) / m) if n >= 2 * m else 0.0
    sse = 0.0
    for t in range(m, n):
        error = y[t] - (level + trend + seasonal[t - m])
        sse += error * error
        new_level = alpha * (y[t] - seasonal[t - m]) + (1.0 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1.0 - beta) * trend
        seasonal[t] = gamma * (y[t] - new_level) + (1.0 - gamma) * seasonal[t - m]
        level, trend = new_level, new_trend
    return sse, level, trend, seasonal


def _hw_forecast_one(y: np.ndarray, m: int, maxiter: int) -> float:
    def objective(params):
        return _hw_fit(
            y, m,
            float(np.clip(params[0], 0.01, 0.99)),
            float(np.clip(params[1], 0.0, 0.99)),
            float(np.clip(params[2], 0.01, 0.99)),
        )[0]

    best = minimize(objective, np.asarray([0.3, 0.1, 0.3]), method="Nelder-Mead",
                    options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4})
    alpha = float(np.clip(best.x[0], 0.01, 0.99))
    beta = float(np.clip(best.x[1], 0.0, 0.99))
    gamma = float(np.clip(best.x[2], 0.01, 0.99))
    _, level, trend, seasonal = _hw_fit(y, m, alpha, beta, gamma)
    return float(level + trend + seasonal[len(y) - m])


def sarima_levels(daily: np.ndarray, dates: list, order, window: int,
                  maxiter: int, fallback: dict,
                  min_history: int = SARIMA_MIN_HISTORY) -> dict:
    """SARIMA 每天用历史重新拟合；拟合失败时退化为最近 7 日均值。"""
    levels = {}
    for index, day in enumerate(dates):
        history = daily[:index]
        if history.size < min_history:
            levels[day] = float(np.mean(history)) if history.size else None
            fallback[day] = "样本不足"
            continue
        y = np.asarray(history[-window:], dtype=float)
        try:
            levels[day] = float(sarima.forecast_one(y, order, maxiter=maxiter))
        except Exception as exc:
            levels[day] = float(np.mean(y[-7:]))
            fallback[day] = type(exc).__name__
    return levels


# --------------------------------------------------------------------------
# 评估与输出
# --------------------------------------------------------------------------
def reconstruct(level: float, baseline: np.ndarray, baseline_mean: float) -> np.ndarray:
    return np.maximum(baseline + (level - baseline_mean), 0.0)


def evaluate(predicted: np.ndarray, actual: np.ndarray) -> dict:
    error = predicted - actual
    ss_res = float((error ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    pred_daily = predicted.mean(axis=1)
    actual_daily = actual.mean(axis=1)
    ss_res_daily = float(((pred_daily - actual_daily) ** 2).sum())
    ss_tot_daily = float(((actual_daily - actual_daily.mean()) ** 2).sum())
    return {
        "mae_slot": float(np.abs(error).mean()),
        "rmse_slot": float(np.sqrt((error ** 2).mean())),
        "r2_slot": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae_daily": float(np.abs(pred_daily - actual_daily).mean()),
        "rmse_daily": float(np.sqrt(((pred_daily - actual_daily) ** 2).mean())),
        "r2_daily": (1.0 - ss_res_daily / ss_tot_daily
                     if ss_tot_daily > 0 else float("nan")),
    }


def write_prediction_workbook(path: Path, header: list, dates: list,
                              predicted: np.ndarray, actual: np.ndarray) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "预测电价"
    sheet.append(["日期"] + list(header))
    for index, day in enumerate(dates):
        sheet.append([datetime.combine(day, datetime.min.time())]
                     + [float(value) for value in predicted[index]])
    sheet.freeze_panes = "B2"

    actual_sheet = workbook.create_sheet("实际电价")
    actual_sheet.append(["日期"] + list(header))
    for index, day in enumerate(dates):
        actual_sheet.append([datetime.combine(day, datetime.min.time())]
                            + [float(value) for value in actual[index]])
    actual_sheet.freeze_panes = "B2"

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def label_for(method: str, ma_window: int, order, season: int) -> str:
    if method == "rolling":
        return METHOD_LABELS["rolling"] % ma_window
    if method == "sarima":
        return METHOD_LABELS["sarima"] % (str(tuple(order)),)
    return METHOD_LABELS["holtwinters"] % season


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ma-window", type=int, default=MA_WINDOW)
    parser.add_argument("--sarima-order", type=str,
                        default=",".join(str(v) for v in SARIMA_ORDER))
    parser.add_argument("--sarima-window", type=int, default=SARIMA_WINDOW)
    parser.add_argument("--sarima-maxiter", type=int, default=SARIMA_MAXITER)
    parser.add_argument("--hw-season", type=int, default=HW_SEASON)
    parser.add_argument("--hw-window", type=int, default=HW_WINDOW)
    parser.add_argument("--sample-day", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--skip-sarima", action="store_true")
    args = parser.parse_args()

    order = tuple(int(part) for part in args.sarima_order.split(","))
    if len(order) != 7:
        raise SystemExit("--sarima-order 需要 7 个数，如 1,0,1,1,0,1,7")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = q2.load_price(DATA_DIR / "appendix01.xlsx")
    price_dates, price_matrix, header = load_actual_price(DATA_DIR / "appendix04.xlsx")
    baseline_mean = float(baseline.mean())
    print("数据目录：%s" % DATA_DIR)
    print("附件4：%d 天 x %d 时段，日均价 %.4f 元/kWh"
          % (price_matrix.shape[0], price_matrix.shape[1], price_matrix.mean()))
    print("附件1 基准形状：均值 %.4f 元/kWh" % baseline_mean)

    daily = price_matrix.mean(axis=1)
    eval_dates = [day for day in price_dates if OUTPUT_START <= day <= OUTPUT_END]
    print("评估区间：%s ~ %s（%d 天）"
          % (eval_dates[0], eval_dates[-1], len(eval_dates)))

    predictions, metrics = {}, {}
    for method in METHODS:
        if method == "sarima" and args.skip_sarima:
            continue
        started = perf_counter()
        fallback: dict = {}
        if method == "rolling":
            levels = rolling_levels(daily, price_dates, args.ma_window)
        elif method == "holtwinters":
            levels = holtwinters_levels(daily, price_dates, args.hw_season,
                                        args.hw_window, HW_MAXITER)
        else:
            levels = sarima_levels(daily, price_dates, order, args.sarima_window,
                                   args.sarima_maxiter, fallback)

        predicted = np.stack([
            reconstruct(levels[day], baseline, baseline_mean)
            if levels[day] is not None else baseline.copy()
            for day in eval_dates
        ])
        actual = np.stack([
            price_matrix[price_dates.index(day)] for day in eval_dates
        ])
        predictions[method] = predicted
        metrics[method] = evaluate(predicted, actual)
        metrics[method]["fallback_days"] = len(fallback)
        metrics[method]["seconds"] = perf_counter() - started
        print("  [done] %-12s MAE %.4f  RMSE %.4f  R^2(逐时段) %.4f  R^2(日均值) %.4f"
              "（%.0f 秒）"
              % (method, metrics[method]["mae_slot"], metrics[method]["rmse_slot"],
                 metrics[method]["r2_slot"], metrics[method]["r2_daily"],
                 metrics[method]["seconds"]), flush=True)

        write_prediction_workbook(
            RESULT_DIR / METHOD_FILES[method], header, eval_dates,
            predicted, actual,
        )

    # ---- 指标表 ----
    fields = ["method", "label", "mae_slot", "rmse_slot", "r2_slot",
              "mae_daily", "rmse_daily", "r2_daily", "fallback_days", "seconds"]
    with (RESULT_DIR / "price_forecast_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            if method not in predictions:
                continue
            row = dict(metrics[method])
            row["method"] = method
            row["label"] = label_for(method, args.ma_window, order, args.hw_season)
            writer.writerow(row)

    # ---- 图 1：某一天的预测 vs 实际 ----
    if args.sample_day:
        sample_day = datetime.strptime(args.sample_day, "%Y-%m-%d").date()
    else:
        rng = np.random.default_rng(args.seed)
        sample_day = eval_dates[int(rng.integers(len(eval_dates)))]
    position = eval_dates.index(sample_day)
    hours = np.arange(PERIODS_PER_DAY) / 6.0
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(hours, price_matrix[price_dates.index(sample_day)], color="#d62728",
            lw=2.0, label="实际电价（附件4）")
    for method in METHODS:
        if method not in predictions:
            continue
        ax.plot(hours, predictions[method][position], lw=1.6, ls="--",
                color=METHOD_COLORS[method],
                label=label_for(method, args.ma_window, order, args.hw_season))
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("时刻 / 小时")
    ax.set_ylabel("电价 / (元/kWh)")
    ax.set_title("三种算法的预测电价 vs 实际电价（%s）" % sample_day)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig_sample_day_compare.svg", dpi=150)
    plt.close(fig)

    # ---- 图 2：全年日均价 ----
    fig, ax = plt.subplots(figsize=(11, 4.3))
    actual_daily = np.array([price_matrix[price_dates.index(day)].mean()
                             for day in eval_dates])
    ax.plot(eval_dates, actual_daily, color="#d62728", lw=1.8, label="实际日均价")
    for method in METHODS:
        if method not in predictions:
            continue
        ax.plot(eval_dates, predictions[method].mean(axis=1), lw=1.4, ls="--",
                color=METHOD_COLORS[method],
                label=label_for(method, args.ma_window, order, args.hw_season))
    ax.set_xlabel("日期")
    ax.set_ylabel("日均价 / (元/kWh)")
    ax.set_title("全年日均价：实际 vs 三种算法的预测")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig_daily_mean_compare.svg", dpi=150)
    plt.close(fig)

    # ---- 图 3：R^2 / MAE 对比 ----
    active = [method for method in METHODS if method in predictions]
    labels = [label_for(m, args.ma_window, order, args.hw_season) for m in active]
    x = np.arange(len(active))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    width = 0.38
    axes[0].bar(x - width / 2, [metrics[m]["r2_slot"] for m in active], width,
                label="R²（逐时段）", color="#4c72b0")
    axes[0].bar(x + width / 2, [metrics[m]["r2_daily"] for m in active], width,
                label="R²（日均值）", color="#55a868")
    for index, method in enumerate(active):
        axes[0].text(index - width / 2, metrics[method]["r2_slot"], "%.3f"
                     % metrics[method]["r2_slot"], ha="center", va="bottom", fontsize=8)
        axes[0].text(index + width / 2, metrics[method]["r2_daily"], "%.3f"
                     % metrics[method]["r2_daily"], ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R²")
    axes[0].set_title("R² 对比（越大越好）")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width / 2, [metrics[m]["mae_slot"] for m in active], width,
                label="MAE（逐时段）", color="#4c72b0")
    axes[1].bar(x + width / 2, [metrics[m]["mae_daily"] for m in active], width,
                label="MAE（日均值）", color="#55a868")
    for index, method in enumerate(active):
        axes[1].text(index - width / 2, metrics[method]["mae_slot"], "%.4f"
                     % metrics[method]["mae_slot"], ha="center", va="bottom", fontsize=8)
        axes[1].text(index + width / 2, metrics[method]["mae_daily"], "%.4f"
                     % metrics[method]["mae_daily"], ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("MAE / (元/kWh)")
    axes[1].set_title("MAE 对比（越小越好）")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig_r2_compare.svg", dpi=150)
    plt.close(fig)

    print("=" * 74)
    for method in active:
        print("%-40s R²(逐时段)=%.4f  R²(日均值)=%.4f  MAE=%.4f"
              % (label_for(method, args.ma_window, order, args.hw_season),
                 metrics[method]["r2_slot"], metrics[method]["r2_daily"],
                 metrics[method]["mae_slot"]))
    print("预测电价 Excel：%s" % "、".join(METHOD_FILES[m] for m in active))
    print("指标表：%s" % (RESULT_DIR / "price_forecast_metrics.csv"))
    print("图：fig_sample_day_compare.svg / fig_daily_mean_compare.svg / fig_r2_compare.svg")


if __name__ == "__main__":
    main()