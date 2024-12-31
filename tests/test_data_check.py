import os
from source.data_check import check_raw_data


def test_run_data_check():
    raw_dir = "../data/raw"
    processed_dir = "../data/processed"
    check_raw_data(raw_dir, processed_dir)
    assert os.path.exists(processed_dir), "Processed directory oluşturulmalı!"