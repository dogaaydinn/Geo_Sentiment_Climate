import sys
import os
import pandas as pd

# Add the parent directory of 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.data_ingestion import ingest_data

# Test the ingest_data function while mocking the raw data directory
def test_ingest_data():
    df = ingest_data(raw_dir="../data/raw", max_rows=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0