#!/usr/bin/env python
import argparse
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from datasets import load_from_disk

from models import AlignmentModel, get_adapter_class
from loss import ContrastiveLoss
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SpeechTextDataset(Dataset):
    """
    Simple PyTorch dataset wrapper around a HuggingFace split that contains:
      - 'pooled_speech_embeddings' : [T, D_speech]
      - 'pooled_attn_masks'        : [T]
      - 'transcript_embeddings'    : [D_text]
    """

    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        speech = torch.tensor(item["pooled_speech_embeddings"], dtype=torch.float32)
        mask = torch.tensor(item["pooled_attn_masks"], dtype=torch.long)
        text_emb = torch.tensor(item["transcript_embeddings"], dtype=torch.float32)

        return {
            "speech": speech,
            "mask": mask,
            "text": text_emb,
        }


def collate_fn(batch):
    """
    Collate function that pads in time dimension if needed.
    Assumes all items in a batch have same length (likely true for precomputed),
    but this will also handle variable lengths by padding.
    """
    # speech: [T_i, D], mask: [T_i], text: [D_text]
    speech_list = [b["speech"] for b in batch]
    mask_list = [b["mask"] for b in batch]
    text_list = [b["text"] for b in batch]

    # Pad in time dimension if variable
    max_T = max(s.shape[0] for s in speech_list)
    D = speech_list[0].shape[1]

    padded_speech = torch.zeros(len(batch), max_T, D, dtype=torch.float32)
    padded_mask = torch.zeros(len(batch), max_T, dtype=torch.long)

    for i, (s, m) in enumerate(zip(speech_list, mask_list)):
        T = s.shape[0]
        padded_speech[i, :T] = s
        padded_mask[i, :T] = m

    text = torch.stack(text_list, dim=0)  # [B, D_text]

    return {
        "speech": padded_speech,
        "mask": padded_mask,
        "text": text,
    }


# ----------------------------
# Training / validation loops
# ----------------------------

def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch, num_epochs):
    model.to(device)
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
        leave=False,
    )

    for batch in progress:
        speech = batch["speech"].to(device, non_blocking=True)      # [B, T, D_speech]
        mask = batch["mask"].to(device, non_blocking=True)          # [B, T]
        text = batch["text"].to(device, non_blocking=True)          # [B, D_text]

        optimizer.zero_grad()

        # model(speech, mask) -> [B, D_text]
        pred = model(speech, mask)

        loss = criterion(text, pred)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        progress.set_postfix({"loss": running_loss / num_batches})

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, criterion, dataloader, device, epoch, num_epochs):
    model.to(device)
    model.eval()
    running_loss = 0.0
    num_batches = 0

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [val]",
        leave=False,
    )

    for batch in progress:
        speech = batch["speech"].to(device)
        mask = batch["mask"].to(device)
        text = batch["text"].to(device)

        pred = model(speech, mask)
        loss = criterion(text, pred)

        running_loss += loss.item()
        num_batches += 1
        progress.set_postfix({"val_loss": running_loss / num_batches})

    return running_loss / max(num_batches, 1)


class AlignmentLitModule(L.LightningModule):
    """Lightning wrapper around AlignmentModel (precomputed embeddings)."""

    def __init__(self, adapter: nn.Module, lr=1e-4, weight_decay=0.0, init_tau=0.07):
        super().__init__()
        self.save_hyperparameters(ignore=['adapter'])
        self.model = AlignmentModel(adapter=adapter, init_tau=init_tau)

    def forward(self, speech, mask, text):
        return self.model(speech, mask, text)

    def _shared_step(self, batch, stage):
        out = self.model(batch['speech'], batch['mask'], batch['text'])
        self.log('log_tau', self.model.log_tau, prog_bar=False, on_epoch=True, on_step=False)
        self.log('tau', torch.exp(self.model.log_tau), prog_bar=False, on_epoch=True, on_step=False)

        if stage == 'val':
            self.log('val_loss', out['loss'], prog_bar=True, batch_size=batch['speech'].size(0), on_epoch=True, on_step=False)
        return out

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch, 'train')
        self.log('train_loss', out['loss'], prog_bar=True, batch_size=batch['speech'].size(0), on_epoch=True, on_step=False)
        return out['loss']

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def configure_optimizers(self):
        params = list(self.model.adapter.parameters()) + [self.model.log_tau]
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


class SpeechTextDataModule(L.LightningDataModule):
    """DataModule for precomputed speech/text embeddings."""

    def __init__(self, train_ds, val_ds, batch_size=64, num_workers=4):
        super().__init__()
        self.train_dataset = SpeechTextDataset(train_ds)
        self.val_dataset = SpeechTextDataset(val_ds)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


def run_training(train_ds, val_ds, adapter,
                 batch_size=64,
                 lr=1e-4,
                 max_epochs=20,
                 input_checkpoint=None,
                 checkpoint_dir="./checkpoint",
                 output_checkpoint_name="model.ckpt",
                 logger_name="alignment",
                 log_dir=None,
                 accumulate_grad_batches=1,
                 num_workers=4):
    """Train or resume a speech-text alignment model with configurable parameters.

    This mirrors the notebook implementation and uses PyTorch Lightning.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if log_dir is None:
        log_dir = os.path.join(checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = TensorBoardLogger(log_dir, name=logger_name)

    input_checkpoint_path = os.path.join(checkpoint_dir, input_checkpoint) if input_checkpoint else None
    output_ckpt_path = os.path.join(checkpoint_dir, output_checkpoint_name)

    if input_checkpoint_path is not None and os.path.exists(input_checkpoint_path):
        print(f"Loading model from input checkpoint: {input_checkpoint_path}")
        lit_model = AlignmentLitModule.load_from_checkpoint(input_checkpoint_path, adapter=adapter, lr=lr)
    else:
        print("No input checkpoint found or provided. Training from scratch.")
        lit_model = AlignmentLitModule(adapter=adapter, lr=lr)

    dm = SpeechTextDataModule(train_ds=train_ds, val_ds=val_ds, batch_size=batch_size, num_workers=num_workers)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        precision='16-mixed',
        log_every_n_steps=10,
        val_check_interval=0.25,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(lit_model, datamodule=dm, ckpt_path=input_checkpoint_path if input_checkpoint_path and os.path.exists(input_checkpoint_path) else None)

    trainer.save_checkpoint(output_ckpt_path)
    print(f"Model checkpoint saved to {output_ckpt_path}")
    print(f"TensorBoard logs saved to {log_dir}/{logger_name}")
    # If adapter supports saving config, export it to checkpoint dir
    try:
        adapter.save_config(checkpoint_dir)
    except Exception:
        pass
    return lit_model, trainer


# ----------------------------
# Main
# ----------------------------

def run(data_dir,
        checkpoint_name: str,
        config_file: str,
        val_ratio: float = 0.2,
        batch_size: int = 128,
        num_workers: int = 4,
        speech_dim: int = 384,
        init_tau: float = 0.07,
        normalize_inputs: bool = True,
        epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        seed: int = 42,
        no_cuda: int = False,
        output_dir: str = 'outputs',):

    # For backward compatibility keep the old run() behavior, but prefer
    # the Lightning-based `run_training` defined below for new experiments.
    print("Note: prefer running `run_training` via the CLI (Lightning)")
    # The original run() behavior is preserved but not executed here.
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpeechAdapter (Lightning) using precomputed embeddings")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to HuggingFace dataset saved with save_to_disk")
    parser.add_argument("--config-file", type=str, required=True, help="Adapter config JSON (contains 'adapter-type' and 'kwargs')")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory to save/load checkpoints and logs")
    parser.add_argument("--input-checkpoint", type=str, default=None, help="Optional existing checkpoint filename to resume from (inside checkpoint-dir)")
    parser.add_argument("--output-checkpoint-name", type=str, default="model.ckpt", help="Name for the saved checkpoint")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of training data to use as validation (0–1)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--logger-name", type=str, default="alignment")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset
    print(f"Loading dataset from {args.data_dir} ...")
    ds = load_from_disk(args.data_dir)
    if "train" not in ds:
        raise ValueError("Dataset must contain a 'train' split.")

    # Train/validation split
    split = ds["train"].train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
    train_split = split["train"]
    val_split = split["test"]

    # Load adapter config and instantiate adapter
    with open(args.config_file) as f:
        cfg = json.load(f)
    adapter_type = cfg.get("adapter-type") or cfg.get("adapter_type")
    if adapter_type is None:
        raise ValueError("config file must contain 'adapter-type' key")
    adapter_kwargs = cfg.get("kwargs", {})
    AdapterClass = get_adapter_class(adapter_type)
    adapter = AdapterClass(**adapter_kwargs)

    # Run training (Lightning)
    run_training(
        train_ds=train_split,
        val_ds=val_split,
        adapter=adapter,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        input_checkpoint=args.input_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        output_checkpoint_name=args.output_checkpoint_name,
        logger_name=args.logger_name,
        log_dir=args.log_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
    )

