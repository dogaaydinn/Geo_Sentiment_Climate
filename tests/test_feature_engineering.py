# tests/test_feature_engineering.py

import unittest
import pandas as pd
from source.feature_engineering import scale_features, create_interaction_terms


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Örnek veri setleri oluşturun
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        }
        self.df = pd.DataFrame(data)

    def test_scale_features_standard(self):
        scaled_df = scale_features(self.df.copy(), cols=['feature1', 'feature2'], method='standard')
        self.assertAlmostEqual(scaled_df['feature1'].mean(), 0, places=1)
        self.assertAlmostEqual(scaled_df['feature1'].std(), 1, places=1)

    def test_scale_features_minmax(self):
        scaled_df = scale_features(self.df.copy(), cols=['feature1', 'feature2'], method='minmax')
        self.assertAlmostEqual(scaled_df['feature1'].min(), 0)
        self.assertAlmostEqual(scaled_df['feature1'].max(), 1)

    def test_scale_features_invalid_method(self):
        with self.assertRaises(ValueError):
            scale_features(self.df.copy(), cols=['feature1'], method='invalid')

    def test_create_interaction_terms_multiply(self):
        interaction = create_interaction_terms(self.df, col1='feature1', col2='feature2', operation='multiply')
        expected = pd.Series([5, 8, 9, 8, 5], name='feature1_feature2_multiply')
        pd.testing.assert_series_equal(interaction, expected)

    def test_create_interaction_terms_add(self):
        interaction = create_interaction_terms(self.df, col1='feature1', col2='feature2', operation='add')
        expected = pd.Series([6, 6, 6, 6, 6], name='feature1_feature2_add')
        pd.testing.assert_series_equal(interaction, expected)

    def test_create_interaction_terms_invalid_operation(self):
        with self.assertRaises(ValueError):
            create_interaction_terms(self.df, col1='feature1', col2='feature2', operation='divide')

    def test_create_interaction_terms_missing_column(self):
        with self.assertRaises(KeyError):
            create_interaction_terms(self.df, col1='feature1', col2='non_existent', operation='multiply')


if __name__ == '__main__':
    unittest.main()
