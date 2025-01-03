import logging

import yaml

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

