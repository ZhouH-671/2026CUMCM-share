"""第4版光伏模型：1月选参数，2月起在线滚动预测。

原方法：最近7天同时刻均值。
新方法：物理太阳赤纬特征 + 光伏滞后/均值特征 + 岭回归，
并可选择与原方法按权重融合。岭参数和融合权重仅由1月验证集选择。

用法：
    python comparison_pv.py
    python comparison_pv.py --date 2026-01-03 --time 14:00
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import openpyxl


FORECAST_DAYS = 5
FEATURE_DAYS = 14
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
BLEND_GRID = tuple(value / 10 for value in range(11))


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "附件" / "附件2.xlsx").is_file():
            return candidate
    raise FileNotFoundError(f"从 {start} 向上未找到“附件/附件2.xlsx”")


def clock_text(value) -> str:
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return f"{value.hour:02d}:{value.minute:02d}"
    text = str(value).strip()
    if text.endswith("+1"):
        hour, minute = map(int, text[:-2].split(":")[:2])
        return f"{hour + 24:02d}:{minute:02d}"
    return f"{int(text.split(':')[0]):02d}:{int(text.split(':')[1]):02d}"


def parse_target_time(value: str) -> str:
    hour, minute = map(int, value.strip().split(":")[:2])
    if not (0 <= hour <= 24 and 0 <= minute < 60 and minute % 10 == 0):
        raise ValueError(f"时间必须是 00:00 至 24:00 间的 10 分钟节点：{value!r}")
    return f"{hour:02d}:{minute:02d}"


def interval_label(column: int) -> str:
    start_hour, start_minute = divmod(column * 10, 60)
    end_hour, end_minute = divmod((column + 1) * 10, 60)
    return f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"


def load_history(path: Path) -> tuple[list[date], list[str], np.ndarray]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.worksheets[1].iter_rows(values_only=True))
    finally:
        workbook.close()
    dates = []
    values = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        if not isinstance(row[0], datetime):
            raise TypeError(f"日期格式错误：{row[0]!r}")
        if any(value is None for value in row[1:]):
            raise ValueError(f"{row[0].date()} 存在缺失光伏数据")
        dates.append(row[0].date())
        values.append([max(0.0, float(value)) for value in row[1:]])
    if len(dates) != 365:
        raise ValueError(f"光伏数据应有365天，实际为{len(dates)}天")
    return dates, [clock_text(value) for value in rows[0][1:]], np.asarray(values)


def original_forecast_day(
    history_dates: list[date], history: np.ndarray, target: date
) -> np.ndarray:
    horizon = (target - history_dates[-1]).days
    if not 1 <= horizon <= FORECAST_DAYS:
        raise ValueError(
            f"目标日期必须在 {history_dates[-1]} 后 1-{FORECAST_DAYS} 天内"
        )
    return np.maximum(0.0, history[-7:].mean(axis=0))


def save_forecast(path: Path, forecasts: list[tuple[date, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["日期", "时间段", "预测光伏功率（kW）"])
        for target, interval, value in forecasts:
            writer.writerow([target.isoformat(), interval, f"{value:.6f}"])


def build_features(history: np.ndarray, target_day_of_year: int) -> np.ndarray:
    if len(history) < FEATURE_DAYS:
        raise ValueError(f"至少需要 {FEATURE_DAYS} 天历史数据")

    declination = np.radians(
        23.44 * np.sin(2 * np.pi * (target_day_of_year - 81) / 365)
    )
    rows = []
    for column in range(history.shape[1]):
        rows.append(
            [
                history[-1, column],
                history[-2, column],
                history[-3, column],
                history[-7, column],
                history[-14, column],
                history[-3:, column].mean(),
                history[-7:, column].mean(),
                history[-14:, column].mean(),
                np.sin(declination),
                np.cos(declination),
                np.sin(2 * np.pi * column / history.shape[1]),
                np.cos(2 * np.pi * column / history.shape[1]),
                1.0,
            ]
        )
    return np.asarray(rows)


class ExpandingRidgePVV4:
    def __init__(self, dates: list[date], history: np.ndarray, ridge_lambda: float):
        self.dates = dates
        self.history = history
        self.ridge_lambda = ridge_lambda
        self.features = {
            index: build_features(
                history[:index], dates[index].timetuple().tm_yday
            )
            for index in range(FEATURE_DAYS, len(dates))
        }
        self.historical_ridge = self._fit_expanding()
        self.coefficient = self._final_coefficient()

    def _solve(self, gram: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.linalg.solve(
            gram + self.ridge_lambda * np.eye(gram.shape[0]), target
        )

    def _fit_expanding(self) -> np.ndarray:
        prediction = np.full_like(self.history, np.nan)
        gram = np.zeros((self.features[FEATURE_DAYS].shape[1],) * 2)
        target = np.zeros(gram.shape[0])
        for index in range(FEATURE_DAYS, len(self.dates)):
            if index > FEATURE_DAYS:
                prediction[index] = np.maximum(
                    0.0, self.features[index] @ self._solve(gram, target)
                )
            current_x = self.features[index]
            gram += current_x.T @ current_x
            target += current_x.T @ self.history[index]
        return prediction

    def _final_coefficient(self) -> np.ndarray:
        gram = np.zeros((self.features[FEATURE_DAYS].shape[1],) * 2)
        target = np.zeros(gram.shape[0])
        for index in range(FEATURE_DAYS, len(self.dates)):
            current_x = self.features[index]
            gram += current_x.T @ current_x
            target += current_x.T @ self.history[index]
        return self._solve(gram, target)

    def predict_values(self, target: date) -> np.ndarray:
        horizon = (target - self.dates[-1]).days
        if not 1 <= horizon <= FORECAST_DAYS:
            raise ValueError(
                f"目标日期必须在 {self.dates[-1]} 后 1-{FORECAST_DAYS} 天内"
            )
        features = build_features(self.history, target.timetuple().tm_yday)
        return np.maximum(0.0, features @ self.coefficient)


def original_historical_prediction(history: np.ndarray) -> np.ndarray:
    prediction = np.full_like(history, np.nan)
    for index in range(7, len(history)):
        prediction[index] = history[index - 7:index].mean(axis=0)
    return prediction


def select_parameters(
    dates: list[date],
    history: np.ndarray,
    original: np.ndarray,
) -> tuple[float, float, list[dict]]:
    validation_days = [
        index
        for index, day in enumerate(dates)
        if day.month == 1 and index >= 20
    ]
    daylight_columns = np.arange(36, 108)
    records = []
    best = None

    for ridge_lambda in LAMBDA_GRID:
        ridge = ExpandingRidgePVV4(dates, history, ridge_lambda).historical_ridge
        for blend in BLEND_GRID:
            prediction = blend * original + (1 - blend) * ridge
            error = (
                history[validation_days][:, daylight_columns]
                - prediction[validation_days][:, daylight_columns]
            )
            mae = float(np.mean(np.abs(error)))
            records.append(
                {"lambda": ridge_lambda, "blend": blend, "MAE": mae}
            )
            if best is None or mae < best["MAE"]:
                best = records[-1]
    if best is None:
        raise RuntimeError("参数选择失败")
    return float(best["lambda"]), float(best["blend"]), records


def plot_comparison(
    dates: list[date],
    actual: np.ndarray,
    original: np.ndarray,
    improved: np.ndarray,
    original_future: np.ndarray,
    improved_future: np.ndarray,
    output: Path,
    *,
    time_label: str,
) -> tuple[float, float]:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    evaluation = np.asarray([day.month >= 2 for day in dates])
    evaluation &= np.isfinite(improved)
    original_mae = float(np.mean(np.abs(actual[evaluation] - original[evaluation])))
    improved_mae = float(
        np.mean(np.abs(actual[evaluation] - improved[evaluation]))
    )
    future_dates = [
        dates[-1] + timedelta(days=offset)
        for offset in range(1, FORECAST_DAYS + 1)
    ]

    figure, axis = plt.subplots(figsize=(15, 6.5))
    axis.axvspan(dates[0], date(2025, 1, 31), color="#eceff1", alpha=0.7)
    axis.plot(dates, actual, color="#1f2933", linewidth=1.1, label="实际值")
    axis.plot(
        dates,
        original,
        color="#9aa0a6",
        linewidth=0.9,
        linestyle="--",
        label=f"原方法（2月起 MAE={original_mae:.1f} kW）",
    )
    axis.plot(
        dates,
        improved,
        color="#1677b8",
        linewidth=1.15,
        label=f"pv4 新方法（2月起 MAE={improved_mae:.1f} kW）",
    )
    axis.plot(
        [dates[-1], *future_dates],
        [original[-1], *original_future],
        color="#9aa0a6",
        linestyle=":",
        linewidth=1.1,
    )
    axis.plot(
        [dates[-1], *future_dates],
        [improved[-1], *improved_future],
        color="#e09f00",
        linewidth=2.3,
        label=f"未来 {FORECAST_DAYS} 天新方法",
    )
    axis.axvline(dates[-1], color="#777777", linestyle="--", linewidth=1.0)
    axis.set_title(f"第4版光伏：实际值与新旧方法对比（每天 {time_label}）", fontsize=15, pad=12)
    axis.set_xlabel("日期")
    axis.set_ylabel("光伏实际功率（kW）")
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.grid(color="#d8d8d8", linewidth=0.7, alpha=0.75)
    axis.legend(loc="upper left")
    figure.autofmt_xdate(rotation=35, ha="right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return original_mae, improved_mae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--time", default="14:00")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    target_time = parse_target_time(args.time)
    root = find_project_root(Path(__file__).resolve().parent)
    dates, time_labels, history = load_history(root / "附件" / "附件2.xlsx")
    if target_time not in time_labels:
        raise ValueError(f"不存在结束时刻为 {target_time} 的 10 分钟时段")
    column = time_labels.index(target_time)

    original = original_historical_prediction(history)
    selected_lambda, selected_blend, records = select_parameters(
        dates, history, original
    )
    selected_model = ExpandingRidgePVV4(dates, history, selected_lambda)
    improved = (
        selected_blend * original
        + (1 - selected_blend) * selected_model.historical_ridge
    )
    future_dates = [
        dates[-1] + timedelta(days=offset)
        for offset in range(1, FORECAST_DAYS + 1)
    ]
    ridge_future = np.asarray(
        [selected_model.predict_values(target) for target in future_dates]
    )
    original_future_profiles = np.asarray(
        [
            original_forecast_day(dates, history, target)
            for target in future_dates
        ]
    )
    improved_future_profiles = (
        selected_blend * original_future_profiles
        + (1 - selected_blend) * ridge_future
    )
    forecasts = [
        (target, interval_label(index), float(profile[index]))
        for target, profile in zip(future_dates, improved_future_profiles)
        for index in range(history.shape[1])
    ]

    output_dir = args.output_dir or root / "output" / "question2"
    csv_path = output_dir / "pv4_forecast_5days.csv"
    save_forecast(csv_path, forecasts)
    plot_path = output_dir / f"pv4_{target_time.replace(':', '-')}_comparison.png"
    original_mae, improved_mae = plot_comparison(
        dates,
        history[:, column],
        original[:, column],
        improved[:, column],
        original_future_profiles[:, column],
        improved_future_profiles[:, column],
        plot_path,
        time_label=target_time,
    )
    best = min(records, key=lambda item: item["MAE"])
    print(
        f"1月选择：lambda={selected_lambda}，原方法融合权重={selected_blend}，"
        f"验证MAE={best['MAE']:.4f} kW"
    )
    print(f"2月起原方法 MAE：{original_mae:.4f} kW")
    print(f"2月起 pv4 新方法 MAE：{improved_mae:.4f} kW")
    print(f"5天预测CSV：{csv_path}")
    print(f"对比图：{plot_path}")

    if args.date is not None:
        if not 1 <= (args.date - dates[-1]).days <= FORECAST_DAYS:
            raise ValueError(
                f"{args.date} 不在历史截止日后的 1-{FORECAST_DAYS} 天内"
            )
        target_index = next(
            index for index, target in enumerate(future_dates) if target == args.date
        )
        print(
            f"{args.date} {interval_label(column)}："
            f"{improved_future_profiles[target_index, column]:.6f} kW"
        )


if __name__ == "__main__":
    main()
