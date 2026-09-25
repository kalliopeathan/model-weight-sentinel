import unittest
from toolkit.diff_checkpoint import diff_manifests

class TestDiffManifests(unittest.TestCase):

    def test_no_changes(self):
        #same manifest compared to itself (should be unchanged)
        m = {"a": "hash1", "b": "hash2"}
        result = diff_manifests(m, m)
        self.assertEqual(result.added, [])
        self.assertEqual(result.removed, [])
        self.assertEqual(result.changed, [])
        self.assertEqual(sorted(result.unchanged), ["a", "b"])

    def test_changed_key(self):
        #same key, different hash (should be flagged as changed)
        m_a = {"a": "hash1"}
        m_b = {"a": "hash2"}
        result = diff_manifests(m_a, m_b)
        self.assertEqual(result.changed, ["a"])

    def test_added_removed_keys(self):
        #no overlapping keys ("a" removed, "b" added)
        m_a = {"a": "hash1"}
        m_b = {"b": "hash2"}
        result = diff_manifests(m_a, m_b)
        self.assertEqual(result.added, ["b"])
        self.assertEqual(result.removed, ["a"])

if __name__ == "__main__":
    unittest.main()