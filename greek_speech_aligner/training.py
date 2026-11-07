#
#   training.py
#
# Lightning training of the aligner (+ temperature) with the bidirectional
# contrastive objective. Defaults follow Section 4.2 of the paper:
# AdamW, lr 5e-4, weight decay 5e-5, batch size 128, 40 epochs,
# validation after every epoch, gamma = 0.1.
#
# Author: Alexandros Tsingilis
#
import os

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger

from greek_speech_aligner.data import SpeechTextDataModule
from greek_speech_aligner.models import AlignmentModel

DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 5e-5
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 40


class AlignmentLitModule(L.LightningModule):
    """Lightning wrapper around ``AlignmentModel`` (precomputed embeddings)."""

    def __init__(self,
                 adapter: nn.Module,
                 lr: float = DEFAULT_LR,
                 weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 init_tau: float = 0.07,
                 gamma: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["adapter"])
        self.model = AlignmentModel(adapter=adapter, init_tau=init_tau, gamma=gamma)

    def forward(self, speech, mask, text):
        return self.model(speech, mask, text)

    def _shared_step(self, batch, stage):
        out = self.model(batch["speech"], batch["mask"], batch["text"])
        self.log(f"{stage}_loss", out["loss"], prog_bar=True,
                 batch_size=batch["speech"].size(0), on_epoch=True, on_step=False)
        return out

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch, "train")
        self.log("log_tau", self.model.log_tau, on_epoch=True, on_step=False)
        self.log("tau", self.model.tau, on_epoch=True, on_step=False)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        # Only the aligner and the temperature are trained; frozen encoder
        # parameters (e.g. inside E5MLPAdapter) are excluded.
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def default_precision():
    """Mixed fp16 on CUDA, full precision elsewhere (CPU/MPS)."""
    return "16-mixed" if torch.cuda.is_available() else "32-true"


def run_training(train_ds, val_ds, adapter,
                 batch_size=DEFAULT_BATCH_SIZE,
                 lr=DEFAULT_LR,
                 weight_decay=DEFAULT_WEIGHT_DECAY,
                 init_tau=0.07,
                 gamma=0.1,
                 max_epochs=DEFAULT_EPOCHS,
                 input_checkpoint=None,
                 checkpoint_dir="./checkpoint",
                 output_checkpoint_name="model.ckpt",
                 logger_name="alignment",
                 log_dir=None,
                 accumulate_grad_batches=1,
                 num_workers=4,
                 val_check_interval=1.0,
                 precision=None):
    """Train (or resume) a speech–text alignment model with Lightning.

    The adapter's ``config.json`` (including ``init_tau`` and ``gamma``) is
    written to ``checkpoint_dir`` before training starts, so that
    evaluation can rebuild the exact model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if log_dir is None:
        log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if hasattr(adapter, "save_config"):
        adapter.save_config(checkpoint_dir)
        _add_model_kwargs_to_config(checkpoint_dir, init_tau=init_tau, gamma=gamma)

    resume_path = None
    if input_checkpoint:
        candidate = os.path.join(checkpoint_dir, input_checkpoint)
        if os.path.exists(candidate):
            resume_path = candidate
        else:
            raise FileNotFoundError(f"Input checkpoint not found: {candidate}")

    print(f"Resuming from {resume_path}" if resume_path else "Training from scratch.")
    # Trainer.fit(ckpt_path=...) restores weights, optimizer and loop state.
    lit_model = AlignmentLitModule(adapter=adapter, lr=lr, weight_decay=weight_decay,
                                   init_tau=init_tau, gamma=gamma)
    dm = SpeechTextDataModule(train_ds=train_ds, val_ds=val_ds,
                              batch_size=batch_size, num_workers=num_workers)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision=precision or default_precision(),
        log_every_n_steps=10,
        val_check_interval=val_check_interval,
        logger=TensorBoardLogger(log_dir, name=logger_name),
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(lit_model, datamodule=dm, ckpt_path=resume_path)

    output_ckpt_path = os.path.join(checkpoint_dir, output_checkpoint_name)
    trainer.save_checkpoint(output_ckpt_path)
    print(f"Model checkpoint saved to {output_ckpt_path}")
    print(f"TensorBoard logs saved to {os.path.join(log_dir, logger_name)}")
    return lit_model, trainer


def _add_model_kwargs_to_config(checkpoint_dir, **model_kwargs):
    from greek_speech_aligner.utils import load_json, save_json

    path = os.path.join(checkpoint_dir, "config.json")
    cfg = load_json(path)
    cfg.update(model_kwargs)
    save_json(cfg, path)
