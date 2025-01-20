import yaml
import logging

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            if config is None:
                raise ValueError("Configuration file is empty")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {e}")
    return None

def check_required_keys(cfg: dict, required_keys: list) -> None:

    missing_keys = [key for key in required_keys if key not in cfg.get("paths", {})]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")
