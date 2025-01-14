import unittest
import pandas as pd
from source.missing_value_comparison import compare_missing_values

class TestMissingValueComparison(unittest.TestCase):
    def setUp(self):
        # Örnek veri setleri oluşturun
        data_original = {'A': [1, 2, None, 4], 'B': [None, 2, 3, 4]}
        data_processed = {'A': [1, 2, 3, 4], 'B': [1, 2, 3, 4]}
        self.df_original = pd.DataFrame(data_original)
        self.df_processed = pd.DataFrame(data_processed)

    def test_compare_missing_values(self):
        comparison = compare_missing_values(self.df_original, self.df_processed)
        expected = pd.DataFrame({
            'Original Missing': {'A': 1, 'B': 1},
            'Processed Missing': {'A': 0, 'B': 0},
            'Difference': {'A': -1, 'B': -1}
        })
        pd.testing.assert_frame_equal(comparison, expected)

if __name__ == '__main__':
    unittest.main()
