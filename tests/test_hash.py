import unittest
import torch
from toolkit.hash_checkpoint import hash_tensor

class TestHashTensor(unittest.TestCase):

    def test_reshape_changes_hash(self):
        #same bytes, different shape - must not hash the same
        t = torch.arange(24, dtype=torch.float32)
        self.assertNotEqual(hash_tensor(t.reshape(4, 6)), hash_tensor(t.reshape(6, 4)))

    def test_reinterpreted_dtype_changes_hash(self):
        #same bytes read as int32 instead of float32 - must not hash the same
        t = torch.arange(24, dtype=torch.float32)
        self.assertNotEqual(hash_tensor(t), hash_tensor(t.view(torch.int32)))

    def test_same_tensor_same_hash(self):
        t = torch.arange(24, dtype=torch.float32)
        self.assertEqual(hash_tensor(t), hash_tensor(t.clone()))

    def test_non_contiguous_matches_contiguous(self):
        #a transposed view has different memory layout but the same logical values as its copy
        t = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        self.assertEqual(hash_tensor(t), hash_tensor(t.contiguous()))

if __name__ == "__main__":
    unittest.main()