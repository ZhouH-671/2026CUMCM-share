"""第4版负载模型：1月选参数，2月起严格滚动预测。

原方法：最近两个相同星期几的同时刻均值。
新方法：固定31天滚动岭回归，岭参数仅由1月数据选择。

用法：
    python comparison_load.py
    python comparison_load.py --date 2026-01-03 --time 14:00
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from datetime import datetime, time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import openpyxl


FORECAST_DAYS = 5
FEATURE_DAYS = 7
TRAIN_WINDOW_DAYS = 31
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)


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
        rows = list(workbook.worksheets[0].iter_rows(values_only=True))
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
            raise ValueError(f"{row[0].date()} 存在缺失负载")
        dates.append(row[0].date())
        values.append([float(value) for value in row[1:]])
    if len(dates) != 365:
        raise ValueError(f"负载数据应有365天，实际为{len(dates)}天")
    return dates, [clock_text(value) for value in rows[0][1:]], np.asarray(values)


def original_forecast_day(
    history_dates: list[date], history: np.ndarray, target: date
) -> np.ndarray:
    same_weekday = [
        index
        for index, history_date in enumerate(history_dates)
        if history_date < target and history_date.weekday() == target.weekday()
    ]
    if not same_weekday:
        raise ValueError("历史数据中没有可用于预测的相同星期几")
    return history[same_weekday[-2:]].mean(axis=0)


def save_forecast(path: Path, forecasts: list[tuple[date, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["日期", "时间段", "预测负载功率（kW）"])
        for target, interval, value in forecasts:
            writer.writerow([target.isoformat(), interval, f"{value:.6f}"])


def build_features(
    history: np.ndarray, history_dates: list[date], target: date
) -> np.ndarray:
    if len(history) < FEATURE_DAYS:
        raise ValueError(f"至少需要 {FEATURE_DAYS} 天历史数据")

    same_weekday = [
        index
        for index, history_date in enumerate(history_dates)
        if history_date.weekday() == target.weekday()
    ]
    day_of_year = target.timetuple().tm_yday
    weekday_angle = 2 * np.pi * target.weekday() / 7
    annual_angle = 2 * np.pi * day_of_year / 365
    rows = []
    for column in range(history.shape[1]):
        rows.append(
            [
                history[-1, column],
                history[-2, column],
                history[-3, column],
                history[-7, column],
                history[-3:, column].mean(),
                history[-7:, column].mean(),
                history[same_weekday[-2:], column].mean(),
                np.sin(weekday_angle),
                np.cos(weekday_angle),
                np.sin(annual_angle),
                np.cos(annual_angle),
                np.sin(2 * np.pi * column / history.shape[1]),
                np.cos(2 * np.pi * column / history.shape[1]),
                1.0,
            ]
        )
    return np.asarray(rows)


class RollingRidgeLoadV4:
    def __init__(
        self,
        dates: list[date],
        history: np.ndarray,
        ridge_lambda: float,
        train_window: int = TRAIN_WINDOW_DAYS,
    ):
        self.dates = dates
        self.history = history
        self.ridge_lambda = ridge_lambda
        self.train_window = train_window
        self.features = {
            index: build_features(history[:index], dates[:index], dates[index])
            for index in range(FEATURE_DAYS, len(dates))
        }
        self.historical_prediction, self.coefficient = self._fit_rolling()

    def _solve(self, gram: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.linalg.solve(
            gram + self.ridge_lambda * np.eye(gram.shape[0]), target
        )

    def _fit_rolling(self) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.full_like(self.history, np.nan)
        gram = np.zeros((self.features[FEATURE_DAYS].shape[1],) * 2)
        target = np.zeros(gram.shape[0])
        left = FEATURE_DAYS

        for index in range(FEATURE_DAYS, len(self.dates)):
            while index - left > self.train_window:
                old_x = self.features[left]
                old_y = self.history[left]
                gram -= old_x.T @ old_x
                target -= old_x.T @ old_y
                left += 1

            minimum = min(FEATURE_DAYS, self.train_window)
            if index - left >= minimum:
                coefficient = self._solve(gram, target)
                prediction[index] = self.features[index] @ coefficient
            else:
                prediction[index] = original_forecast_day(
                    self.dates[:index], self.history[:index], self.dates[index]
                )

            current_x = self.features[index]
            gram += current_x.T @ current_x
            target += current_x.T @ self.history[index]

        return prediction, self._solve(gram, target)

    def predict(self, target: date) -> tuple[np.ndarray, int]:
        horizon = (target - self.dates[-1]).days
        if not 1 <= horizon <= FORECAST_DAYS:
            raise ValueError(
                f"目标日期必须在 {self.dates[-1]} 后 1-{FORECAST_DAYS} 天内"
            )
        return build_features(self.history, self.dates, target), horizon

    def predict_values(self, target: date) -> tuple[np.ndarray, int]:
        features, horizon = self.predict(target)
        return features @ self.coefficient, horizon


def original_historical_prediction(
    dates: list[date], history: np.ndarray, column: int
) -> np.ndarray:
    prediction = np.full(len(dates), np.nan)
    for index in range(1, len(dates)):
        try:
            values = original_forecast_day(
                dates[:index], history[:index], dates[index]
            )
        except ValueError:
            continue
        prediction[index] = values[column]
    return prediction


def select_lambda(
    dates: list[date], history: np.ndarray
) -> tuple[float, dict[float, float]]:
    validation = [
        index
        for index, day in enumerate(dates)
        if day.month == 1 and index >= 14
    ]
    scores = {}
    for ridge_lambda in LAMBDA_GRID:
        model = RollingRidgeLoadV4(dates, history, ridge_lambda)
        prediction = model.historical_prediction[validation]
        actual = history[validation]
        scores[ridge_lambda] = float(np.mean(np.abs(actual - prediction)))
    selected = min(scores, key=scores.get)
    return selected, scores


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
        color="#c44e52",
        linewidth=1.15,
        label=f"load4 新方法（2月起 MAE={improved_mae:.1f} kW）",
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
    axis.set_title(f"第4版小区负载：实际值与新旧方法对比（每天 {time_label}）", fontsize=15, pad=12)
    axis.set_xlabel("日期")
    axis.set_ylabel("小区负载功率（kW）")
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

    selected_lambda, selection_scores = select_lambda(dates, history)
    model = RollingRidgeLoadV4(dates, history, selected_lambda)
    original = original_historical_prediction(dates, history, column)
    future_dates = [
        dates[-1] + timedelta(days=offset)
        for offset in range(1, FORECAST_DAYS + 1)
    ]
    improved_profiles = np.asarray(
        [model.predict_values(target)[0] for target in future_dates]
    )
    improved_future = improved_profiles[:, column]
    original_future = np.asarray(
        [
            original_forecast_day(dates, history, target)[column]
            for target in future_dates
        ]
    )
    forecasts = [
        (target, interval_label(index), float(profile[index]))
        for target, profile in zip(future_dates, improved_profiles)
        for index in range(history.shape[1])
    ]

    output_dir = args.output_dir or root / "output" / "question2"
    csv_path = output_dir / "load4_forecast_5days.csv"
    save_forecast(csv_path, forecasts)
    plot_path = output_dir / f"load4_{target_time.replace(':', '-')}_comparison.png"
    original_mae, improved_mae = plot_comparison(
        dates,
        history[:, column],
        original,
        model.historical_prediction[:, column],
        original_future,
        improved_future,
        plot_path,
        time_label=target_time,
    )
    print("1月选参得分：", {key: round(value, 4) for key, value in selection_scores.items()})
    print(f"1月选定岭参数：{selected_lambda}")
    print(f"2月起原方法 MAE：{original_mae:.4f} kW")
    print(f"2月起 load4 新方法 MAE：{improved_mae:.4f} kW")
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
        print(f"{args.date} {interval_label(column)}：{improved_future[target_index]:.6f} kW")


if __name__ == "__main__":
    main()
