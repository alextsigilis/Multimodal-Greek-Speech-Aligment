import json

def load_config_from_file(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
