#
#   utils.py
#
# Small, dependency-light helpers shared across the package.
#
# Author: Alexandros Tsingilis
#
import json
import random

import numpy as np
import torch


def load_json(path):
    """Load a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


# Kept for backward compatibility with the old ``utils.load_config_from_file``.
load_config_from_file = load_json


def save_json(obj, path):
    """Write ``obj`` to ``path`` as indented JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def set_seed(seed: int):
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available device name: ``cuda``, ``mps`` or ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"
