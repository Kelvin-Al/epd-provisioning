"""Independent checks for the analytical decision and CPU benchmark model."""

import importlib.util
import unittest
from fractions import Fraction

from calibration import encoder_demands, fit_encoder, workload_rates
from planner import provision
from simulation import Request, departures, measure, poisson_trace


class PlannerTests(unittest.TestCase):
    def test_matches_exhaustive_search(self):
        for gpus in range(2, 15):
            for e in range(1, 10):
                for pd in range(1, 10):
                    exhaustive = [min(n * e, (gpus - n) * pd) for n in range(1, gpus)]
                    optimum = max(exhaustive)
                    splits = [i for i, value in enumerate(exhaustive, 1) if value == optimum]
                    result = provision(gpus, e, pd)
                    self.assertEqual(result["optimal_encoder_gpu_counts"], splits)
                    self.assertEqual(result["boundary_rps"], optimum)
                    n = result["encoder_gpus"]
                    self.assertGreaterEqual(Fraction(e, pd), Fraction(gpus - n - 1, n))
                    if n > 1:
                        self.assertLessEqual(Fraction(e, pd), Fraction(gpus - n, n - 1))

    def test_paper_h800_and_strict_boundary(self):
        result = provision(8, 57, 14.3, target=85.8, rate_bounds=((56, 57), (14.3, 15.2)))
        self.assertEqual((result["encoder_gpus"], result["pd_gpus"]), (2, 6))
        self.assertEqual(result["boundary_rps"], 85.8)
        self.assertEqual(result["optimal_ratio_interval"], [2.5, 6])
        self.assertTrue(result["uncertainty"]["selected_split_optimal_throughout"])
        self.assertFalse(result["target"]["strictly_below_boundary"])
        self.assertEqual(result["target"]["minimum_gpus_for_strict_target"], 9)

    def test_exact_decimal_tie(self):
        result = provision(4, 0.3, 0.15, incumbent=2)
        self.assertEqual(result["optimal_encoder_gpu_counts"], [1, 2])
        self.assertEqual(result["encoder_gpus"], 2)

    def test_rate_box_can_cross_boundary(self):
        result = provision(8, 57, 14.3, rate_bounds=((35, 70), (10, 18)))
        self.assertEqual(result["encoder_gpus"], 2)
        self.assertEqual(result["boundary_rps"], 60)
        self.assertFalse(result["uncertainty"]["selected_split_optimal_throughout"])

    def test_rejects_invalid_rates_and_splits(self):
        for rate in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                provision(8, rate, 1)
        with self.assertRaises(ValueError):
            provision(8, 1, 1, incumbent=8)


class CalibrationTests(unittest.TestCase):
    def test_mixed_load_averages_before_max_and_excludes_nested_gpu_time(self):
        model = {
            "visual_token_range": [0, 1],
            "serial_ms": {
                "H": [1], "M": [9, -8], "C": [1], "X": [1, 8], "G": [0, 8], "R": [1]
            },
            "saturated_ms": {
                "H": [0], "M": [18, -16], "C": [0], "X": [1, 8], "G": [0, 8], "R": [0]
            },
            "parallelism": {"H": 1, "M": 2, "C": 1, "X": 1, "G": 1, "R": 1},
        }
        result = workload_rates(model, [
            {"visual_tokens": 0, "pd_demand_ms": 5},
            {"visual_tokens": 1, "pd_demand_ms": 15},
        ])
        self.assertEqual(result["e_rate_upper_bound_rps"], 200)
        self.assertEqual(result["pd_rate_rps"], 100)
        self.assertEqual(result["mean_serial_e_ms"], 13)
        self.assertEqual(result["mean_normalized_lane_ms"]["X"], 5)
        with self.assertRaises(ValueError):
            encoder_demands(model, 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional NumPy fitter")
    def test_fits_affine_and_quadratic_anchors(self):
        def row(v):
            spans = {"H": 1, "M": 2 + v, "C": 3, "X": 8 + v * v, "G": 6 + v * v, "R": 1}
            return {"visual_tokens": v, "serial_ms": spans, "saturated_ms": spans}

        anchors = [row(v) for v in (10, 20, 30, 40, 50)]
        model = fit_encoder({"anchors": anchors, "parallelism": dict.fromkeys(
            anchors[0]["serial_ms"], 1
        )})
        serial, demands = encoder_demands(model, 25)
        self.assertAlmostEqual(serial, 665)
        self.assertAlmostEqual(demands["X"], 633)
        self.assertAlmostEqual(demands["G"], 631)


class SimulationTests(unittest.TestCase):
    def test_pd_fifo_orders_by_encoder_completion(self):
        trace = [
            Request(0, 4, 1, 0.1, 0.1),
            Request(1, 1, 3, 0.9, 0.1),
            Request(2, 1, 2, 0.1, 0.1),
        ]
        self.assertEqual(departures(trace, 3, 2), ([4, 2, 5], [6, 5, 8]))
        result = measure(trace, 3, 2, 5, 2)
        self.assertEqual(result["counts"]["pd_completions"], 2)
        self.assertEqual(result["counts"]["unfinished_at_end"], 1)

    def test_paper_rate_inputs_under_saturation(self):
        trace = poisson_trace(130, 150, 57, 14.3, 17)
        baseline = measure(trace, 8, 4, 30, 120)
        candidate = measure(trace, 8, 2, 30, 120)
        self.assertTrue(baseline["checks"]["valid_simulation"])
        self.assertTrue(candidate["checks"]["valid_simulation"])
        self.assertEqual(baseline["counts"]["arrivals"], candidate["counts"]["arrivals"])
        self.assertAlmostEqual(baseline["metrics"]["throughput_rps"], 57.2, delta=0.2)
        self.assertAlmostEqual(candidate["metrics"]["throughput_rps"], 85.8, delta=0.2)
