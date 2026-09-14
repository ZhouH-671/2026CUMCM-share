"""把裕度搜索结果导出成 Excel，方便在 Excel 里直接画等高线/热力图。

用法（在项目根目录下）：
    python -m src.problem03.margin.export_margin

输入：result/problem03/margin_search.csv
输出：result/problem03/margin_search.xlsx，含 7 个 sheet
    1) 全部配置      67 条完整搜索结果
    2) 粗扫透视表    5x6 规则网格（总费用，万元）—— 可直接做 Excel 等高线图
    3) 插值网格-总费用
    4) 插值网格-紧急购电
    5) 插值网格-弃电
    6) 白天切片      固定夜间 0.60，白天分位 0.20~0.80
    7) 夜间切片      固定白天 0.60，夜间分位 0.45~0.95
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "result" / "problem03" / "margin"
APPENDIX_DIR = ROOT / "附件"
CSV_PATH = RESULT_DIR / "margin_search.csv"
XLSX_PATH = RESULT_DIR / "margin_search.xlsx"

HEADER_FILL = PatternFill("solid", fgColor="D9EAF7")


def style_sheet(sheet) -> None:
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = Font(name="宋体", size=10, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    sheet.freeze_panes = "B2"
    for column in range(1, sheet.max_column + 1):
        sheet.column_dimensions[get_column_letter(column)].width = 13


def main() -> None:
    frame = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    frame["total_cost_wan"] = frame["total_cost"] / 1e4
    frame["emergency_wan_kwh"] = frame["emergency_kwh"] / 1e4
    frame["spill_gwh"] = frame["spill_kwh"] / 1e6
    frame = frame.sort_values(["quantile_overnight", "quantile_day"])

    detail = frame[
        [
            "stage",
            "quantile_overnight",
            "quantile_day",
            "total_cost_wan",
            "plan_cost",
            "adjustment_cost",
            "emergency_cost",
            "emergency_wan_kwh",
            "spill_gwh",
            "plan_kwh",
            "adjusted_kwh",
            "charge_kwh",
            "discharge_kwh",
            "storage_end",
            "seconds",
        ]
    ].reset_index(drop=True)
    detail.columns = [
        "阶段", "0:00-6:00分位", "6:00后分位", "总费用(万元)",
        "计划购电费(元)", "调整费用(元)", "紧急购电费(元)",
        "紧急购电量(万kWh)", "弃电量(GWh)", "计划购电量(kWh)",
        "调整后购电量(kWh)", "充电量(kWh)", "放电量(kWh)",
        "期末储电量(kWh)", "耗时(秒)",
    ]

    coarse = frame[frame["stage"] == "coarse"]
    coarse_pivot = coarse.pivot_table(
        index="quantile_overnight",
        columns="quantile_day",
        values="total_cost_wan",
    )
    coarse_pivot = coarse_pivot.round(2)
    coarse_pivot.index.name = "0:00-6:00分位 \\ 6:00后分位"

    x = frame["quantile_overnight"].to_numpy(dtype=float)
    y = frame["quantile_day"].to_numpy(dtype=float)
    gx, gy = np.meshgrid(
        np.round(np.arange(0.45, 0.9501, 0.005), 3),
        np.round(np.arange(0.20, 0.8001, 0.005), 3),
    )

    def interpolate(values: np.ndarray) -> np.ndarray:
        linear = griddata((x, y), values, (gx, gy), method="linear")
        nearest = griddata((x, y), values, (gx, gy), method="nearest")
        return np.round(np.where(np.isnan(linear), nearest, linear), 3)

    grid_cost = interpolate(frame["total_cost_wan"].to_numpy(dtype=float))
    grid_emg = interpolate(frame["emergency_wan_kwh"].to_numpy(dtype=float))
    grid_spill = interpolate(frame["spill_gwh"].to_numpy(dtype=float))
    grid_total = pd.DataFrame(
        grid_cost,
        index=pd.Index(gy[:, 0], name="6:00后分位 \\ 0:00-6:00分位"),
        columns=pd.Index(gx[0], name="0:00-6:00分位"),
    )
    grid_emg_df = pd.DataFrame(grid_emg, index=grid_total.index, columns=grid_total.columns)
    grid_spill_df = pd.DataFrame(grid_spill, index=grid_total.index, columns=grid_total.columns)

    night = float(frame.loc[frame["total_cost"].idxmin(), "quantile_overnight"])
    day = float(frame.loc[frame["total_cost"].idxmin(), "quantile_day"])

    day_slice = frame[frame["quantile_overnight"].round(4) == round(night, 4)]
    day_slice = day_slice.sort_values("quantile_day")[
        ["quantile_day", "total_cost_wan", "plan_cost", "adjustment_cost",
         "emergency_cost", "emergency_wan_kwh", "spill_gwh"]
    ].reset_index(drop=True)
    day_slice.columns = ["6:00后分位", "总费用(万元)", "计划购电费(元)",
                         "调整费用(元)", "紧急购电费(元)", "紧急购电量(万kWh)",
                         "弃电量(GWh)"]

    night_slice = frame[frame["quantile_day"].round(4) == round(day, 4)]
    night_slice = night_slice.sort_values("quantile_overnight")[
        ["quantile_overnight", "total_cost_wan", "plan_cost", "adjustment_cost",
         "emergency_cost", "emergency_wan_kwh", "spill_gwh"]
    ].reset_index(drop=True)
    night_slice.columns = ["0:00-6:00分位", "总费用(万元)", "计划购电费(元)",
                           "调整费用(元)", "紧急购电费(元)", "紧急购电量(万kWh)",
                           "弃电量(GWh)"]

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="全部配置", index=False)
        coarse_pivot.to_excel(writer, sheet_name="粗扫透视表")
        grid_total.to_excel(writer, sheet_name="插值网格-总费用")
        grid_emg_df.to_excel(writer, sheet_name="插值网格-紧急购电")
        grid_spill_df.to_excel(writer, sheet_name="插值网格-弃电")
        day_slice.to_excel(writer, sheet_name="白天切片", index=False)
        night_slice.to_excel(writer, sheet_name="夜间切片", index=False)
        for sheet in writer.book.worksheets:
            style_sheet(sheet)
            if sheet.title == "全部配置":
                for row in sheet.iter_rows(min_row=2):
                    for cell in row[3:9]:
                        cell.number_format = "0.00"

    print("已输出：%s" % XLSX_PATH)
    print("sheet 数：%d" % len(pd.ExcelFile(XLSX_PATH).sheet_names))
    print("全部配置：%d 行" % len(detail))
    print("插值网格：%d x %d（0.005 步长）" % grid_total.shape)


if __name__ == "__main__":
    main()
