import pytest
from source.utils.config_loader import load_config

@pytest.fixture
def config_path():
    return "../config/settings.yml"


def test_config_load(config_path):
    config = load_config(config_path)
    assert "project_name" in config
    assert "data" in config
    assert "logging" in config
