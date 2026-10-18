import unittest
from toolkit.hash_checkpoint import require_file

class TestRequireFile(unittest.TestCase):

    def test_missing_file_carries_path(self):
        #the path must be on .filename so the cli never has to parse the error message
        with self.assertRaises(FileNotFoundError) as ctx:
            require_file("nope.safetensors")
        self.assertEqual(ctx.exception.filename, "nope.safetensors")

if __name__ == "__main__":
    unittest.main()
