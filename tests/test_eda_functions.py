import pandas as pd
from source.eda_exploration import basic_info

def test_basic_info_output():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    info = basic_info(df)
    assert "Shape" in info
    assert info["Shape"] == (2, 2)
