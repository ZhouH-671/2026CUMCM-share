"""裕度（风险分位）搜索：先粗扫、再细扫、最后一维切片，结果写入 result/problem03/。

裕度 = segmented_risk_margin 的两个分位参数：
    --quantile-overnight  0:00-6:00（36 个 10 分钟时段，无调整窗口）
    --quantile-day        6:00-24:00（108 个时段，有 1~3 次 1.5 倍调整窗口）

用法（在项目根目录下）：
    python -m src.problem03.margin.search --dry-run
    python -m src.problem03.margin.search
    python -m src.problem03.margin.search --stage coarse
    python -m src.problem03.margin.search --workers 8

结果逐条追加到 result/problem03/margin/margin_search.csv，重复配置自动跳过，中断后可续跑。
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "result" / "problem03" / "margin"
APPENDIX_DIR = ROOT / "附件"
CSV_PATH = RESULT_DIR / "margin_search.csv"

from src.problem03 import problem


FIELDS = [
    "stage",
    "quantile_overnight",
    "quantile_day",
    "days",
    "plan_kwh",
    "adjusted_kwh",
    "plan_cost",
    "adjustment_cost",
    "emergency_kwh",
    "emergency_cost",
    "spill_kwh",
    "charge_kwh",
    "discharge_kwh",
    "total_cost",
    "storage_end",
    "seconds",
]

KEY = ("quantile_overnight", "quantile_day")


def _init_worker() -> None:
    problem.load_inputs()


def _evaluate(job: tuple) -> dict:
    stage, qn, qd = job
    summary = problem.run_simulation(
        quantile_overnight=qn,
        quantile_day=qd,
        use_correction=True,
    )
    row = {name: summary[name] for name in FIELDS if name in summary}
    row["stage"] = stage
    row["quantile_overnight"] = round(float(qn), 4)
    row["quantile_day"] = round(float(qd), 4)
    return row


def load_existing() -> dict:
    if not CSV_PATH.is_file():
        return {}
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (round(float(row["quantile_overnight"]), 4),
             round(float(row["quantile_day"]), 4)): row
            for row in csv.DictReader(handle)
        }


def append_rows(rows: list) -> None:
    if not rows:
        return
    exists = CSV_PATH.is_file()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDS})


def coarse_jobs() -> list:
    nights = [0.50, 0.60, 0.70, 0.80, 0.90]
    days = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    return [("coarse", n, d) for n in nights for d in days]


def fine_jobs(night: float, day: float, step: float = 0.025) -> list:
    offsets = [-2 * step, -step, 0.0, step, 2 * step]
    jobs = []
    for dn in offsets:
        for dd in offsets:
            qn = round(min(0.95, max(0.05, night + dn)), 4)
            qd = round(min(0.95, max(0.05, day + dd)), 4)
            jobs.append(("fine", qn, qd))
    return jobs


def slice_jobs(night: float, day: float) -> list:
    jobs = []
    value = 0.20
    while value <= 0.8001:
        jobs.append(("slice-day", round(night, 4), round(value, 4)))
        value += 0.05
    value = 0.45
    while value <= 0.9501:
        jobs.append(("slice-night", round(value, 4), round(day, 4)))
        value += 0.05
    return jobs


def best_of(rows: dict) -> tuple:
    best = min(rows.values(), key=lambda r: float(r["total_cost"]))
    return float(best["quantile_overnight"]), float(best["quantile_day"]), best


def run_batch(jobs: list, workers: int, existing: dict) -> dict:
    pending = [
        job for job in jobs
        if (round(job[1], 4), round(job[2], 4)) not in existing
    ]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"  跳过已算过的 {skipped} 个配置")
    if not pending:
        print("  没有新配置需要计算")
        return existing

    total = len(pending)
    done = 0
    started = time.perf_counter()
    print(f"  开始计算 {total} 个配置，进程数 {workers}")
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker
    ) as pool:
        futures = {pool.submit(_evaluate, job): job for job in pending}
        for future in as_completed(futures):
            row = future.result()
            append_rows([row])
            existing[(row["quantile_overnight"], row["quantile_day"])] = row
            done += 1
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (total - done)
            print(
                "  [%3d/%3d] %-11s qn=%.3f qd=%.3f  总费用=%.2f万  "
                "紧急=%.2f万kWh  弃电=%.3fGWh  (%.0fs, ETA %.0fs)"
                % (
                    done,
                    total,
                    row["stage"],
                    row["quantile_overnight"],
                    row["quantile_day"],
                    float(row["total_cost"]) / 1e4,
                    float(row["emergency_kwh"]) / 1e4,
                    float(row["spill_kwh"]) / 1e6,
                    row["seconds"],
                    eta,
                ),
                flush=True,
            )
    return existing


def main() -> None:
    global CSV_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["coarse", "fine", "slice", "all"],
        default="all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()

    CSV_PATH = args.csv

    existing = load_existing()
    print(f"已有结果：{len(existing)} 条（{CSV_PATH}）")

    if args.stage in ("coarse", "all"):
        print("阶段 A：粗扫 0:00-6:00 分位 × 6:00 后分位")
        if args.dry_run:
            for job in coarse_jobs():
                print("   ", job)
        else:
            existing = run_batch(coarse_jobs(), args.workers, existing)

    if args.stage in ("fine", "slice", "all"):
        if not existing:
            if not args.dry_run:
                raise SystemExit("还没有任何结果，请先跑 coarse 阶段")
            night, day, best = 0.70, 0.60, None
        else:
            night, day, best = best_of(existing)
        if best is not None:
            print(
                "当前最优：qn=%.3f qd=%.3f 总费用=%.2f 万"
                % (night, day, float(best["total_cost"]) / 1e4)
            )

    if args.stage in ("fine", "all"):
        print("阶段 B：在最优点附近细扫（步长 0.025）")
        jobs = fine_jobs(night, day)
        if args.dry_run:
            for job in jobs:
                print("   ", job)
        else:
            existing = run_batch(jobs, args.workers, existing)
            night, day, best = best_of(existing)
            print(
                "细扫后最优：qn=%.3f qd=%.3f 总费用=%.2f 万"
                % (night, day, float(best["total_cost"]) / 1e4)
            )

    if args.stage in ("slice", "all"):
        print("阶段 C：沿两个方向做一维切片（供画曲线）")
        jobs = slice_jobs(night, day)
        if args.dry_run:
            for job in jobs:
                print("   ", job)
        else:
            existing = run_batch(jobs, args.workers, existing)
            night, day, best = best_of(existing)
            print(
                "最终最优：qn=%.3f qd=%.3f 总费用=%.2f 万"
                % (night, day, float(best["total_cost"]) / 1e4)
            )

    if not args.dry_run and existing:
        night, day, best = best_of(existing)
        print("=" * 60)
        print("搜索完成，共 %d 个配置" % len(existing))
        print(
            "最优裕度：0:00-6:00 = %.3f，6:00 之后 = %.3f"
            % (night, day)
        )
        print("总费用：%.2f 元（%.2f 万元）" % (float(best["total_cost"]), float(best["total_cost"]) / 1e4))
        print("计划购电费：%.2f 元" % float(best["plan_cost"]))
        print("调整费用：%.2f 元" % float(best["adjustment_cost"]))
        print("紧急购电费：%.2f 元（%.2f 万 kWh）" % (float(best["emergency_cost"]), float(best["emergency_kwh"]) / 1e4))
        print("弃电量：%.2f GWh" % (float(best["spill_kwh"]) / 1e6))
        print("结果表：%s" % CSV_PATH)


if __name__ == "__main__":
    main()
