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

from models import SpeechEncoder
from loss import ContrastiveLoss


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
        text_encoder_id: str = "intfloat/multilingual-e5-small",
        init_tau: float = 0.07,
        normalize_inputs: bool = True,
        epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        seed: int = 42,
        no_cuda: int = False,
        output_dir: str = 'outputs',):

    output_path = os.path.join(output_dir, checkpoint_name)
    print('making dir:', output_path)
    os.makedirs(output_path, exist_ok=True)
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # ----------------------------
    # Load dataset from disk
    # ----------------------------
    print(f"Loading dataset from {data_dir} ...")
    ds = load_from_disk(data_dir)

    if "train" not in ds:
        raise ValueError("Dataset must contain a 'train' split.")

    # ----------------------------
    # Train/validation split
    # ----------------------------
    print(f"Splitting train into train/val with val_ratio={val_ratio} ...")
    split = ds["train"].train_test_split(
        test_size=val_ratio,
        seed=seed,
        shuffle=True,
    )

    train_split = split["train"]
    val_split = split["test"]

    print(f"Train size: {len(train_split)}, Val size: {len(val_split)}")

    # ----------------------------
    # Datasets & DataLoaders
    # ----------------------------
    train_dataset = SpeechTextDataset(train_split)
    val_dataset = SpeechTextDataset(val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # ----------------------------
    # Model & loss
    # ----------------------------
    print("Initializing model and loss...")

    # Load configuration
    with open(config_file) as f:
        config = json.load(f, parse_int=int, parse_float=float)

    model = SpeechEncoder(**config).to(device)

    criterion = ContrastiveLoss(
        init_tau=init_tau,
        normalize_inputs=normalize_inputs,
    ).to(device)

    # Only train parameters of aligner + log_tau
    params = list(model.aligner.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, epochs
        )
        val_loss = validate(
            model, criterion, val_loader, device, epoch, epochs
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
        )

    # ----------------------------
    # Plot losses
    # ----------------------------
    epochs = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Contrastive training loss")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(output_path, "loss_curve.png")
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Saved loss curve to: {plot_path}")

    # ----------------------------
    # Save checkpoint
    # ----------------------------
    os.makedirs(output_path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
    }
    torch.save(checkpoint, os.path.join(output_path, 'checkpoint'))

    print(f"Saved checkpoint to: {output_path}")

    # ----------------------------
    # Save arguments
    # ----------------------------
    args_path = os.path.join(output_path, 'args.json')
    with open(args_path, 'w') as f:
        json.dump({
            'data_dir': data_dir,
            'checkpoint_name': checkpoint_name,
            'config': config,
            'val_ratio': val_ratio,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'speech_dim': speech_dim,
            'text_encoder_id': text_encoder_id,
            'init_tau': init_tau,
            'normalize_inputs': normalize_inputs,
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'seed': seed,
            'no_cuda': no_cuda,
            'output_dir': output_dir,
        }, f, indent=1)

if __name__ == "__main__":
    #
    # CMD ARGS
    #
    parser = argparse.ArgumentParser(description="Train SpeechAdapter with contrastive loss")
    # Data / splits
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to HuggingFace dataset saved with save_to_disk")
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of training data to use as validation (0–1)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    # Model / loss hyperparameters
    parser.add_argument("--speech-dim", type=int, default=384,
                        help="Dimensionality of pooled speech embeddings (in_dim of aligner)")
    parser.add_argument("--text-encoder-id", type=str, default="intfloat/multilingual-e5-small",
                        help="HuggingFace ID of frozen text encoder")
    parser.add_argument("--init-tau", type=float, default=0.07,
                        help="Initial temperature for contrastive loss")
    parser.add_argument("--normalize-inputs", action="store_true",
                        help="Assume inputs to loss are already L2-normalized")
    # Optimization
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force training on CPU even if CUDA is available")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save plots and checkpoints")
    # Parse the arguments
    args = parser.parse_args()

    run(**vars(args))

