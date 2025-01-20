import os
from source.config.config_loader import load_config

def get_config(config_dir: str = "../config", config_file: str = "settings.yml"):
    config_path = os.path.abspath(os.path.join(config_dir, config_file))
    return load_config(config_path)

# Initialize the config once and reuse it
config = get_config()