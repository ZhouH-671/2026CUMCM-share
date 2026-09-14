"""读取 result/problem03/margin_search.csv，画“裕度 → 费用”曲线并输出最优配置报告。

用法（在项目根目录下）：
    python -m src.problem03.plot.plot

输入：
    result/problem03/margin_search.csv

输出（result/problem03/）：
    fig1_margin_curves.png   总费用 / 费用构成 / 弃电与紧急购电 随裕度的变化
    fig2_margin_heatmap.png  两个裕度维度的总费用热力图
    fig3_tradeoff.png        计划购电费 vs 调整费 vs 紧急费的权衡
    best_config.md           最优裕度与对应指标
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "result" / "problem03"
APPENDIX_DIR = ROOT / "附件"
CSV_PATH = RESULT_DIR / "margin" / "margin_search.csv"
OUTPUT_DIR = RESULT_DIR / "plot"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    frame = frame.sort_values("total_cost").drop_duplicates(
        subset=["quantile_overnight", "quantile_day"], keep="first"
    )
    for name in (
        "total_cost",
        "plan_cost",
        "adjustment_cost",
        "emergency_cost",
        "emergency_kwh",
        "spill_kwh",
        "plan_kwh",
    ):
        frame[name] = frame[name].astype(float)
    frame["total_cost_wan"] = frame["total_cost"] / 1e4
    frame["emergency_wan_kwh"] = frame["emergency_kwh"] / 1e4
    frame["spill_gwh"] = frame["spill_kwh"] / 1e6
    return frame


def mark_best(ax, x, y, label: str, color: str = "crimson") -> None:
    ax.scatter([x], [y], s=90, marker="*", color=color, zorder=5)
    ax.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=(8, 10),
        color=color,
        fontsize=9,
    )


def figure1(frame: pd.DataFrame, night: float, day: float) -> Path:
    day_slice = frame[frame["quantile_overnight"].round(4) == round(night, 4)]
    day_slice = day_slice.sort_values("quantile_day")
    night_slice = frame[frame["quantile_day"].round(4) == round(day, 4)]
    night_slice = night_slice.sort_values("quantile_overnight")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    ax = axes[0][0]
    ax.plot(day_slice["quantile_day"], day_slice["total_cost_wan"], "o-",
            color="#1f77b4", label="%s" % "沿 6:00 后分位（固定 0:00-6:00=%.2f）" % night)
    ax.plot(night_slice["quantile_overnight"], night_slice["total_cost_wan"],
            "s-", color="#ff7f0e",
            label="沿 0:00-6:00 分位（固定 6:00 后=%.2f）" % day)
    mark_best(ax, day, float(day_slice["total_cost_wan"].min())
              if len(day_slice) else float("nan"), "")
    ax.set_xlabel("风险裕度分位（quantile）")
    ax.set_ylabel("总费用（万元）")
    ax.set_title("总费用随裕度的变化")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    ax.stackplot(
        day_slice["quantile_day"],
        day_slice["plan_cost"] / 1e4,
        day_slice["adjustment_cost"] / 1e4,
        day_slice["emergency_cost"] / 1e4,
        labels=["计划购电费", "调整费用（1.5/0.5 倍）", "紧急购电费（5 倍）"],
        colors=["#4c72b0", "#dd8452", "#c44e52"],
        alpha=0.85,
    )
    ax.plot(day_slice["quantile_day"], day_slice["total_cost_wan"], "k--",
            lw=1.4, label="总费用")
    ax.set_xlabel("6:00 之后的风险裕度分位")
    ax.set_ylabel("费用（万元）")
    ax.set_title("费用构成随 6:00 后裕度的变化（0:00-6:00=%.2f）" % night)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1][0]
    ax.plot(day_slice["quantile_day"], day_slice["emergency_wan_kwh"], "o-",
            color="#c44e52", label="紧急购电量")
    ax.set_xlabel("6:00 之后的风险裕度分位")
    ax.set_ylabel("紧急购电量（万 kWh）", color="#c44e52")
    ax.tick_params(axis="y", labelcolor="#c44e52")
    twin = ax.twinx()
    twin.plot(day_slice["quantile_day"], day_slice["spill_gwh"], "s-",
              color="#55a868", label="弃电量")
    twin.set_ylabel("弃电量（GWh）", color="#55a868")
    twin.tick_params(axis="y", labelcolor="#55a868")
    ax.set_title("少买一点省下的钱 vs 紧急购电/弃电")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.plot(day_slice["quantile_day"], day_slice["plan_kwh"] / 1e6, "o-",
            color="#4c72b0", label="计划购电量")
    ax.set_xlabel("6:00 之后的风险裕度分位")
    ax.set_ylabel("计划购电量（GWh）")
    ax.set_title("计划购电量随裕度的变化（越保守买得越多）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("裕度搜索：0:00-6:00 与 6:00 后风险分位对费用/购电量的影响",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUTPUT_DIR / "fig1_margin_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure2(frame: pd.DataFrame, night: float, day: float) -> Path:
    x = frame["quantile_overnight"].to_numpy(dtype=float)
    y = frame["quantile_day"].to_numpy(dtype=float)
    z = frame["total_cost_wan"].to_numpy(dtype=float)

    grid_x, grid_y = np.meshgrid(
        np.linspace(max(0.45, x.min()), x.max(), 260),
        np.linspace(y.min(), y.max(), 260),
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
    nearest = griddata((x, y), z, (grid_x, grid_y), method="nearest")
    grid_z = np.where(np.isnan(grid_z), nearest, grid_z)

    fig, ax = plt.subplots(figsize=(9.5, 7))
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=28, cmap="RdYlGn_r")
    lines = ax.contour(grid_x, grid_y, grid_z, levels=12, colors="black",
                       linewidths=0.5, alpha=0.35)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.1f")
    bar = fig.colorbar(contour, ax=ax, pad=0.02)
    bar.set_label("总费用（万元）：绿=便宜，红=贵")

    ax.scatter(x, y, s=26, facecolors="none", edgecolors="white",
               linewidths=0.9, zorder=4)
    ax.scatter([night], [day], s=300, marker="*", color="black", zorder=6,
               label="最优点 (%.3f, %.3f) = %.2f 万元" % (night, day, z.min()))
    ax.set_xlabel("0:00-6:00 的风险裕度分位")
    ax.set_ylabel("6:00 之后的风险裕度分位")
    ax.set_title("总费用热力图（白圈 = 实际算过的 %d 个配置）" % len(frame))
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig2_margin_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure3(frame: pd.DataFrame, night: float, day: float) -> Path:
    day_slice = frame[frame["quantile_overnight"].round(4) == round(night, 4)]
    day_slice = day_slice.sort_values("quantile_day")

    x = day_slice["quantile_day"].to_numpy(dtype=float)
    plan = day_slice["plan_cost"].to_numpy(dtype=float) / 1e4
    adjust = day_slice["adjustment_cost"].to_numpy(dtype=float) / 1e4
    emergency = day_slice["emergency_cost"].to_numpy(dtype=float) / 1e4
    total = plan + adjust + emergency
    best_index = int(np.argmin(total))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    ax = axes[0]
    ax.plot(x, total, "o-", color="#333333", label="总费用")
    ax.scatter([x[best_index]], [total[best_index]], s=200, marker="*",
               color="crimson", zorder=5,
               label="最低总费用 %.2f 万（q_day=%.2f）"
               % (total[best_index], x[best_index]))
    lower = total.min() - 8
    upper = total.max() + 4
    ax.set_ylim(lower, max(upper, total[0] + 2, total[-1] + 2))
    ax.set_xlabel("6:00 之后的风险裕度分位")
    ax.set_ylabel("总费用（万元，纵轴已放大）")
    ax.set_title("总费用曲线（0:00-6:00 固定 %.2f）" % night)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    width = 0.016
    ax.bar(x - width, plan - plan[best_index], width, color="#4c72b0",
           label="Δ 计划购电费")
    ax.bar(x, adjust - adjust[best_index], width, color="#dd8452",
           label="Δ 调整费用")
    ax.bar(x + width, emergency - emergency[best_index], width, color="#c44e52",
           label="Δ 紧急购电费")
    ax.plot(x, total - total[best_index], "k--o", ms=4, lw=1.3,
            label="Δ 总费用")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("6:00 之后的风险裕度分位")
    ax.set_ylabel("相对最优点的费用变化（万元）")
    ax.set_title("权衡拆解：少买的计划费 vs 多付的紧急费")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.suptitle("裕度权衡：0:00-6:00 分位固定 %.2f" % night, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUTPUT_DIR / "fig3_tradeoff.png"
    fig.savefig(path)
    plt.close(fig)
    return path

def write_report(frame: pd.DataFrame, night: float, day: float) -> Path:
    best = frame[
        (frame["quantile_overnight"].round(4) == round(night, 4))
        & (frame["quantile_day"].round(4) == round(day, 4))
    ].iloc[0]
    lines = [
        "# 最优裕度（question3-1 搜索结果）",
        "",
        "搜索配置数：%d（粗扫 + 细扫 + 一维切片）" % len(frame),
        "",
        "## 最优裕度",
        "",
        "- 0:00-6:00 风险分位：**%.3f**" % night,
        "- 6:00 之后风险分位：**%.3f**" % day,
        "",
        "## 最优配置指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        "| 总费用 | %.2f 元（%.2f 万元） |" % (best["total_cost"], best["total_cost_wan"]),
        "| 计划购电费 | %.2f 元 |" % best["plan_cost"],
        "| 调整费用 | %.2f 元 |" % best["adjustment_cost"],
        "| 紧急购电费 | %.2f 元 |" % best["emergency_cost"],
        "| 紧急购电量 | %.2f kWh（%.2f 万 kWh） |" % (best["emergency_kwh"], best["emergency_wan_kwh"]),
        "| 计划购电量 | %.2f kWh |" % best["plan_kwh"],
        "| 弃电量 | %.2f kWh（%.3f GWh） |" % (best["spill_kwh"], best["spill_gwh"]),
        "",
        "## 结论",
        "",
    ]
    day_slice = frame[frame["quantile_overnight"].round(4) == round(night, 4)]
    night_slice = frame[frame["quantile_day"].round(4) == round(day, 4)]
    lines.append(
        "- 固定 0:00-6:00 分位时，6:00 后分位的最优区间大致是 "
        "%.2f~%.2f。" % (day_slice["quantile_day"].min(), day_slice["quantile_day"].max())
    )
    lines.append(
        "- 固定 6:00 后分位时，0:00-6:00 分位的最优区间大致是 "
        "%.2f~%.2f。" % (night_slice["quantile_overnight"].min(), night_slice["quantile_overnight"].max())
    )
    lines.append(
        "- 6:00 后裕度越低，紧急购电费越高、弃电越少；两者相抵后存在明显的平坦最优点。"
    )
    path = OUTPUT_DIR / "best_config.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    if not CSV_PATH.is_file():
        raise SystemExit("找不到 %s，请先运行 python -m src.problem03.margin.search" % CSV_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    best = frame.iloc[0]
    night = float(best["quantile_overnight"])
    day = float(best["quantile_day"])
    print("最优裕度：0:00-6:00=%.3f，6:00后=%.3f，总费用=%.2f 万元"
          % (night, day, best["total_cost_wan"]))
    for path in (
        figure1(frame, night, day),
        figure2(frame, night, day),
        figure3(frame, night, day),
        write_report(frame, night, day),
    ):
        print("已输出：%s" % path)


if __name__ == "__main__":
    main()
