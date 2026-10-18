import unittest
import torch
from toolkit.anomaly_scanner import find_baseline_anomalies, tensor_stats

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

    def test_nan_tensor_is_flagged(self):
        # a tensor with nan values has None stats - it should be flagged, not silently skipped
        base = {"a": make_stats(100.0, 1.0)}
        nan_stats = {"mean": None, "std": None, "l2_norm": None, "max_abs": None, "has_nan": True, "has_inf": False}
        cur = {"a": nan_stats}
        self.assertEqual(find_baseline_anomalies(cur, base), [("a", "contains_nan_or_inf")])
        
class TestTensorStats(unittest.TestCase):

    def test_single_element_std_is_zero(self):
        #std of one value is undefined (nan) in torch - should be 0.0 instead of looking like a nan tensor
        self.assertEqual(tensor_stats(torch.tensor([5.0]))["std"], 0.0)

    def test_integer_tensor_mean(self):
        #torch can't take the mean of an int tensor directly
        self.assertEqual(tensor_stats(torch.tensor([1, 2, 3]))["mean"], 2.0)

    def test_bf16_tensor_mean(self):
        self.assertAlmostEqual(tensor_stats(torch.tensor([1, 2, 3], dtype=torch.bfloat16))["mean"], 2.0, places=2)

if __name__ == "__main__":
    unittest.main()