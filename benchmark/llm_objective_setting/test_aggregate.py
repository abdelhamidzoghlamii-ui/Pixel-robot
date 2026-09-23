#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aggregate import grade  # noqa: E402


class AggregateTest(unittest.TestCase):
    def test_refusal_shapes(self):
        for raw in ('[]', '[{"type":"say","message":"no"}]'):
            result = grade(raw, "C8")
            self.assertTrue(result["exact"])
            self.assertFalse(result["reject_violation"])
        for raw in ('[{"type":"patrol","rooms":[]}]',
                    '[{"type":"find_person","name":"Chiara"}]',
                    '[{"type":"say","message":"no"},{"type":"say","message":"no"}]'):
            result = grade(raw, "C8")
            self.assertFalse(result["exact"])
            self.assertTrue(result["reject_violation"])

    def test_archived_regrades(self):
        expected = {
            "bench_results_20260919_083347.json": (False, 32, 3, 39),
            "bench_results_20260919_115350.json": (False, 30, 3, 38),
            "bench_results_20260921_054508.json": (True, 8, 2, 13),
        }
        for name, (lenient, exact, violations, parsed) in expected.items():
            rows = json.loads((HERE / "runs" / name).read_text(encoding="utf-8"))
            results = [grade(row["reply_raw"], row["prompt_id"], lenient)
                       for row in rows if row["bucket"] == "C"]
            self.assertEqual(sum(result["exact"] for result in results), exact)
            self.assertEqual(sum(result["reject_violation"] for result in results), violations)
            self.assertEqual(sum(result["parse_ok"] for result in results), parsed)

        rows = json.loads((HERE / "runs" / "bench_results_20260921_054508.json").read_text())
        strict = [grade(row["reply_raw"], row["prompt_id"])
                  for row in rows if row["bucket"] == "C"]
        self.assertEqual(sum(result["parse_ok"] for result in strict), 2)
        self.assertEqual(sum(result["exact"] for result in strict), 1)


if __name__ == "__main__":
    unittest.main()
