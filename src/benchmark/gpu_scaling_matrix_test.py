import unittest
from src.benchmark.gpu_scaling_matrix import parse_profiles

class ParseProfilesTest(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(parse_profiles("1x1,2x1,1x2"), [(1,1),(2,1),(1,2)])

    def test_bad(self):
        for value in ("0x1", "1:2", "", "2x-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_profiles(value)

if __name__ == "__main__": unittest.main()
