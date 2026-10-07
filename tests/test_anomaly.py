import unittest
from toolkit.anomaly_scanner import find_baseline_anomalies

def make_stats(l2, mx):
    #helper to build a stats dict without needing a real tensor
    return {"mean": 0.0, "std": 1.0, "l2_norm": l2, "max_abs": mx, "has_nan": False, "has_inf": False}

class TestBaselineAnomalies(unittest.TestCase):

    def test_identical_profiles(self):
        #compare same profile against itself - nothing should be flagged
        p = {"a": make_stats(100.0, 1.0)}
        self.assertEqual(find_baseline_anomalies(p, p), [])

    def test_sparse_spike_caught_by_max_abs(self):
        #l2_norm barely moves (0.2%) but max_abs jumps, mimicing the 5-neuron backdoor
        base = {"a": make_stats(100.0, 1.0)}
        cur = {"a": make_stats(100.2, 11.0)}
        self.assertEqual(find_baseline_anomalies(cur, base), [("a", "max_abs_deviation")])

    def test_missing_from_baseline(self):
        #tensor name is not in baseline and should be flagged
        base = {"a": make_stats(100.0, 1.0)}
        cur = {"b": make_stats(100.0, 1.0)}
        self.assertEqual(find_baseline_anomalies(cur, base), [("b", "missing_from_baseline")])

if __name__ == "__main__":
    unittest.main()