import pytest
import pandas as pd
from source.feature_engineering import scale_features, create_interaction_terms


def test_scale_features():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    scaled_df = scale_features(df, cols=["col1", "col2"], method="standard")
    assert scaled_df["col1"].mean() == pytest.approx(0, 0.1)


def test_create_interaction_terms():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    interaction = create_interaction_terms(df, col1="col1", col2="col2", operation="multiply")
    assert (interaction == pd.Series([4, 10, 18])).all()
