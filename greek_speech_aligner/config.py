#
#   config.py
#
# Aligner configuration files.
#
# Format (also written next to every checkpoint as ``config.json``):
#
#   {
#     "adapter-type": "cnn",            # cnn | mlp | lstm | e5mlp
#     "kwargs": { ... },                # adapter constructor kwargs
#     "init_tau": 0.07,                 # optional, AlignmentModel kwargs
#     "gamma": 0.1                      # optional
#   }
#
# Author: Alexandros Tsingilis
#
from greek_speech_aligner.models import AlignmentModel, get_adapter_class
from greek_speech_aligner.utils import load_json

MODEL_KEYS = ("init_tau", "gamma")


def load_config(path) -> dict:
    """Load and validate an aligner config file."""
    cfg = load_json(path)
    get_adapter_type(cfg)  # validate early
    return cfg


def get_adapter_type(cfg: dict) -> str:
    adapter_type = cfg.get("adapter-type") or cfg.get("adapter_type")
    if adapter_type is None:
        raise ValueError("Config must contain an 'adapter-type' key (cnn, mlp, lstm, e5mlp).")
    return adapter_type


def build_adapter(cfg: dict):
    """Instantiate the adapter described by ``cfg``."""
    kwargs = cfg.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError("Config 'kwargs' must be a JSON object.")
    return get_adapter_class(get_adapter_type(cfg))(**kwargs)


def model_kwargs(cfg: dict) -> dict:
    """Extract the optional ``AlignmentModel`` kwargs (init_tau, gamma)."""
    return {k: cfg[k] for k in MODEL_KEYS if k in cfg}


def build_model(cfg: dict) -> AlignmentModel:
    """Instantiate ``AlignmentModel`` (adapter + temperature) from ``cfg``."""
    return AlignmentModel(adapter=build_adapter(cfg), **model_kwargs(cfg))
