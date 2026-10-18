import json
import os
import tempfile
import unittest
from toolkit.anomaly_scanner import log_scan_history, build_report

def make_stats(l2, mx):
    #helper to build a stats dict without needing a real tensor
    return {"mean": 0.0, "std": 1.0, "l2_norm": l2, "max_abs": mx, "has_nan": False, "has_inf": False}

def make_report(name, count):
    return {"checkpoint": name, "timestamp": "2026-10-18T00:00:00+00:00", "anomaly_count": count, "anomalies": []}

class TestScanHistory(unittest.TestCase):

    def setUp(self):
        #temp dir so tests never touch the real scan_history.json
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "scan_history.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_appends_entries(self):
        log_scan_history(make_report("a.safetensors", 0), self.path)
        log_scan_history(make_report("b.safetensors", 2), self.path)
        with open(self.path) as f:
            history = json.load(f)
        self.assertEqual([h["checkpoint"] for h in history], ["a.safetensors", "b.safetensors"])
        self.assertEqual(history[1], {"checkpoint": "b.safetensors", "timestamp": "2026-10-18T00:00:00+00:00", "anomaly_count": 2})

    def test_survives_corrupt_file(self):
        with open(self.path, "w") as f:
            f.write("{not valid json")
        log_scan_history(make_report("a.safetensors", 1), self.path)
        with open(self.path) as f:
            history = json.load(f)
        self.assertEqual(len(history), 1)

class TestReportGrouping(unittest.TestCase):

    def test_two_reasons_on_one_tensor_grouped(self):
        #l2_norm and max_abs both deviate on the same tensor - should be one entry, counted once
        base = {"a": make_stats(100.0, 1.0), "b": make_stats(50.0, 1.0)}
        cur = {"a": make_stats(150.0, 5.0), "b": make_stats(50.0, 1.0)}
        report = build_report("model.safetensors", cur, base)
        self.assertEqual(report["anomaly_count"], 1)
        self.assertEqual(report["anomalies"], [{"tensor": "a", "reasons": ["l2_norm_deviation", "max_abs_deviation"]}])

if __name__ == "__main__":
    unittest.main()
