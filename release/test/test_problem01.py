"""针对能量口径、互斥约束及不可行性的独立小算例。运行 python -m unittest -v test_question1。"""

from datetime import time
from pathlib import Path
import tempfile
import unittest

import numpy as np
import openpyxl

from src.problem01.problem import (
    ROOT,
    Battery,
    Inputs,
    build_model,
    clock_label,
    solve_model,
    time_minutes,
    validate_solution,
    write_result_workbook,
)


def inputs(price, load, pv):
    return Inputs(*(np.array(values, dtype=float) for values in (price, load, pv)))


class DispatchTests(unittest.TestCase):
    def test_flat_price_no_pv_does_not_cycle(self):
        data = inputs([0.5, 0.5], [6, 6], [0, 0])
        model = build_model(data)
        result, _ = solve_model(model)
        self.assertEqual(result.status, 0)
        self.assertAlmostEqual(result.fun, 1.0, places=6)
        self.assertAlmostEqual(result.x[model.blocks["c"]].sum(), 0.0, places=6)
        validate_solution(data, Battery(), model, result.x)

    def test_price_shift_includes_both_efficiencies(self):
        data = inputs([0.1, 1.0], [0, 6], [0, 0])
        model = build_model(data)
        result, _ = solve_model(model)
        self.assertEqual(result.status, 0)
        self.assertAlmostEqual(result.fun, 0.1 / 0.81, places=6)
        self.assertAlmostEqual(result.x[model.blocks["c"]][0], 1 / 0.81, places=6)
        self.assertAlmostEqual(result.x[model.blocks["d"]][1], 1.0, places=6)
        validate_solution(data, Battery(), model, result.x)
        # 固定最优充放电模式，内层 LP 应恢复相同最优费用。
        inner, _ = solve_model(model, fixed_z=np.rint(result.x[model.blocks["z"]]))
        self.assertEqual(inner.status, 0)
        self.assertAlmostEqual(inner.fun, result.fun, places=6)

    def test_full_battery_surplus_cannot_be_dissipated_by_cycling(self):
        data = inputs([1.0], [0], [360])  # 10 分钟富余 60 kWh，松弛后的模式约束允许循环耗散。
        battery = Battery(initial=10800)
        model = build_model(data, battery)
        exact, _ = solve_model(model)
        self.assertEqual(exact.status, 2)  # 不弃光、不售电，且不允许同时充放电：不可行。
        relaxed, _ = solve_model(model, relax=True)
        self.assertEqual(relaxed.status, 0)
        self.assertGreater(relaxed.x[model.blocks["c"]][0], 0)
        self.assertGreater(relaxed.x[model.blocks["d"]][0], 0)
        with self.assertRaises(ValueError):
            validate_solution(data, battery, model, relaxed.x)

    def test_bad_fixed_mode_is_infeasible(self):
        data = inputs([1.0], [0], [600])
        result, _ = solve_model(build_model(data), fixed_z=np.array([0]))
        self.assertEqual(result.status, 2)

    def test_cannot_borrow_energy_before_it_is_charged(self):
        data = inputs([1.0, 0.1], [6, 0], [0, 0])
        battery = Battery(initial=1200)
        model = build_model(data, battery)
        result, _ = solve_model(model)
        self.assertEqual(result.status, 0)
        self.assertAlmostEqual(result.fun, 1.0, places=6)
        validate_solution(data, battery, model, result.x)

    def test_mixed_time_formats(self):
        self.assertEqual(time_minutes(time(0, 10)), 10)
        self.assertEqual(time_minutes("10:10"), 610)
        self.assertEqual(time_minutes("0:00+1"), 1440)
        self.assertEqual(time_minutes(1 / 144), 10)


class WorkbookOutputTests(unittest.TestCase):
    def test_result_workbook_matches_template_format(self):
        payload = {
            "summary": {
                "storage_initial_kwh": 6000.0,
                "storage_final_kwh": 6000.0,
            },
            "four_hour_summary": [
                {
                    "interval": f"{clock_label(240 * i)}-{clock_label(240 * (i + 1))}",
                    "charge_kwh": float(i + 1),
                    "discharge_kwh": float(i + 2),
                }
                for i in range(6)
            ],
            "ten_minute_dispatch": [
                {"purchase_kwh": float(i + 1) / 10} for i in range(144)
            ],
        }
        template = ROOT / "附件" / "附件5" / "result1.xlsx"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result1_chatgpt.xlsx"
            write_result_workbook(template, output, payload)

            workbook = openpyxl.load_workbook(output, data_only=True)
            try:
                plan = workbook["计划购电量"]
                self.assertEqual(plan["A2"].value, "0:00-0:10")
                self.assertAlmostEqual(plan["B2"].value, 0.1)
                self.assertEqual(plan["A145"].value, "23:50-0:00+1")
                self.assertAlmostEqual(plan["B145"].value, 14.4)

                battery_sheet = workbook["充放电量"]
                self.assertAlmostEqual(battery_sheet["B2"].value, 1.0)
                self.assertAlmostEqual(battery_sheet["C2"].value, 2.0)
                self.assertAlmostEqual(battery_sheet["B7"].value, 6.0)
                self.assertAlmostEqual(battery_sheet["C7"].value, 7.0)
                self.assertAlmostEqual(battery_sheet["E2"].value, 6000.0)
                self.assertAlmostEqual(battery_sheet["E3"].value, 6000.0)
            finally:
                workbook.close()


if __name__ == "__main__":
    unittest.main()
