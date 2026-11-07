#
#   checkpoint.py
#
# Loading trained aligners from Lightning (or plain PyTorch) checkpoints.
#
# Author: Alexandros Tsingilis
#
import os

import torch

from greek_speech_aligner.config import build_model, load_config


def _strip_prefix(state_dict, prefix="model."):
    """Remove the LightningModule attribute prefix (``model.``) from keys."""
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_model_from_checkpoint(ckpt_path, config):
    """
    Build an ``AlignmentModel`` from ``config`` and load weights from ``ckpt_path``.

    Falls back to loading only shape-compatible parameters (with a warning)
    when a strict load fails.
    """
    model = build_model(config)
    # Lightning checkpoints contain non-tensor metadata (hparams, loop state);
    # they are produced locally by this project, so a full unpickle is used.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    state_dict = _strip_prefix(state_dict)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[WARN] Strict load failed: {e}")
        model_state = model.state_dict()
        compatible, skipped = {}, []
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                compatible[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(model_state[k].shape) if k in model_state else None))
        if not compatible:
            raise RuntimeError("No compatible parameters found to load from checkpoint.")
        model.load_state_dict(compatible, strict=False)
        print(f"[OK] Loaded {len(compatible)} compatible parameters; skipped {len(skipped)} mismatched/unexpected keys.")
        for key, ck_shape, md_shape in skipped[:10]:
            print(f"  - Skipped {key}: ckpt={ck_shape} model={md_shape}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more skipped keys")

    return model.eval()


def load_aligner(ckpt_dir, ckpt_name, config_name="config.json"):
    """Load ``<ckpt_dir>/<ckpt_name>`` using ``<ckpt_dir>/<config_name>``."""
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    config = load_config(os.path.join(ckpt_dir, config_name))
    return load_model_from_checkpoint(ckpt_path, config)
