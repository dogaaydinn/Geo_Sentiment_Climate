import pytest
import pandas as pd


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, None],
        "C": ["a", "b", "c"]
    })


def test_missing_values(sample_data):
    assert sample_data.isnull().sum().sum() == 1


def test_column_names(sample_data):
    expected_columns = ["A", "B", "C"]
    assert list(sample_data.columns) == expected_columns
