#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader

from models import mean_pooling, CnnAdapter, AlignmentModel
from models import get_adapter_class
from preprocess import run as precompute


# ---------------------------------------------------------
# Embedding computation with progress bar
# ---------------------------------------------------------

# Dataset wrapper (from train.py)
class SpeechTextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        speech = torch.tensor(item["pooled_speech_embeddings"], dtype=torch.float32)
        mask = torch.tensor(item["pooled_attn_masks"], dtype=torch.long)
        text_emb = torch.tensor(item["transcript_embeddings"], dtype=torch.float32)
        return {"speech": speech, "mask": mask, "text": text_emb}

def collate_fn(batch):
    speech_list = [b["speech"] for b in batch]
    mask_list = [b["mask"] for b in batch]
    text_list = [b["text"] for b in batch]
    max_T = max(s.shape[0] for s in speech_list)
    D = speech_list[0].shape[1]
    padded_speech = torch.zeros(len(batch), max_T, D, dtype=torch.float32)
    padded_mask = torch.zeros(len(batch), max_T, dtype=torch.long)
    for i, (s, m) in enumerate(zip(speech_list, mask_list)):
        T = s.shape[0]
        padded_speech[i, :T] = s
        padded_mask[i, :T] = m
    text = torch.stack(text_list, dim=0)
    return {"speech": padded_speech, "mask": padded_mask, "text": text}

@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    model.eval()
    all_text = []
    all_speech = []
    for batch in tqdm(dataloader, desc="Computing embeddings", dynamic_ncols=True):
        speech = batch["speech"].to(device)
        mask = batch["mask"].to(device)
        text = batch["text"].to(device)
        pred_output = model(speech, mask, text)
        all_text.append(text.cpu())
        all_speech.append(pred_output['speech_emb'].cpu())
    return torch.cat(all_text), torch.cat(all_speech)


# ---------------------------------------------------------
# Recall@K
# ---------------------------------------------------------

def compute_similarity_matrix(gt_embs, pred_embs, device="cpu", batch_size=256):
    """Compute full similarity matrix S where S[i,j] = dot(gt_embs[i], pred_embs[j]).

    Both input tensors are expected as 2D (N, D) and (M, D). The function
    normalizes embeddings (L2) and computes the matrix in query batches to
    limit peak memory usage.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    n = gt_embs.shape[0]
    m = pred_embs.shape[0]

    # Pre-normalize pred embeddings and move to device once
    pred_norm = pred_embs.to(device)
    pred_norm = pred_norm / pred_norm.norm(dim=1, keepdim=True).clamp(min=1e-9)

    sim = torch.empty((n, m), dtype=pred_norm.dtype, device=device)

    for i in tqdm(range(0, n, batch_size), desc="Similarity matrix", dynamic_ncols=True):
        batch_end = min(i + batch_size, n)
        batch_gt = gt_embs[i:batch_end].to(device)
        batch_gt = batch_gt / batch_gt.norm(dim=1, keepdim=True).clamp(min=1e-9)
        sim[i:batch_end] = batch_gt @ pred_norm.T

    return sim


def recall_at_k_batched(sim_matrix, k, batch_size=256, device="cpu"):
    """Compute Recall@k given a precomputed similarity matrix.

    sim_matrix: torch.Tensor [N, M]
    Returns fraction of queries whose correct index (diagonal) is in top-k.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    sim = sim_matrix.to(device)
    n = sim.shape[0]
    k_eff = min(k, sim.shape[1])
    hits = 0.0
    for i in tqdm(range(0, n, batch_size), desc=f"Recall@{k}", dynamic_ncols=True):
        batch = sim[i:i+batch_size]
        topk = torch.topk(batch, k=k_eff, dim=1).indices
        row_indices = torch.arange(i, i + batch.shape[0], device=device).unsqueeze(1)
        batch_hits = (topk == row_indices).any(dim=1).float().sum().item()
        hits += batch_hits
    return float(hits) / float(n)

def mrr_batched(sim_matrix, batch_size=256, device="cpu"):
    """Compute Mean Reciprocal Rank (MRR) from a precomputed similarity matrix.

    sim_matrix: torch.Tensor [N, M]
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    sim = sim_matrix.to(device)
    n = sim.shape[0]
    rr_sum = 0.0
    for i in tqdm(range(0, n, batch_size), desc="MRR", dynamic_ncols=True):
        batch = sim[i:i+batch_size]
        batch_size_actual = batch.shape[0]
        batch_indices = torch.arange(i, i + batch_size_actual, device=device)
        # For each row, get the rank of the correct index
        sorted_indices = torch.argsort(batch, dim=1, descending=True)
        # Find positions where sorted_indices == batch_indices
        # This yields positions (row, col) where col is the rank index
        # We search per-row for the column equal to batch_indices
        # To do this efficiently, compare and nonzero
        matches = (sorted_indices == batch_indices.unsqueeze(1))
        ranks = matches.float().argmax(dim=1) + 1  # 1-based rank
        rr_sum += (1.0 / ranks.float()).sum().item()
    return float(rr_sum) / float(n)


def plot_recall_curve(recall_dict, mrr, output_path):
    ks = list(recall_dict.keys())
    vals = [recall_dict[k] for k in ks]
    plt.figure()
    plt.plot(ks, vals, marker="o", label="Recall@K")
    plt.axhline(mrr, color="red", linestyle="--", label=f"MRR={mrr:.4f}")
    plt.xlabel("K")
    plt.ylabel("Recall@K (text→speech)")
    plt.title("Text→Speech Retrieval Recall Curve + MRR")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[OK] Saved plot → {output_path}")


# ---------------------------------------------------------
# Load model & config from checkpoint directory
# ---------------------------------------------------------
def load_model_from_checkpoint(ckpt_path, config):
    # Select adapter type based only on `adapter-type` in config.json (preferred)
    adapter_type = config.get("adapter-type", "cnn")
    adapter_kwargs = config.get("kwargs", {}) if isinstance(config.get("kwargs", {}), dict) else {}
    AdapterClass = get_adapter_class(adapter_type)
    adapter = AdapterClass(**adapter_kwargs)
    model = AlignmentModel(adapter=adapter, init_tau=config.get("init_tau", 0.07))
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # Handle LightningModule checkpoints with 'model.' prefix
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint["model_state_dict"]
    # If keys are prefixed with 'model.', strip it
    if any(k.startswith("model.") for k in state_dict.keys()):
        new_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        state_dict = new_state_dict
    # Try strict loading first; if shapes mismatch, fall back to selective loading
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[WARN] Strict load failed: {e}")
        model_state = model.state_dict()
        compatible_state = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    compatible_state[k] = v
                else:
                    skipped.append((k, v.shape, model_state[k].shape))
            else:
                # unexpected key in checkpoint
                skipped.append((k, v.shape, None))
        if compatible_state:
            model.load_state_dict(compatible_state, strict=False)
            print(f"[OK] Loaded {len(compatible_state)} compatible parameters; skipped {len(skipped)} mismatched/unexpected keys.")
            for key, ck_shape, md_shape in skipped[:10]:
                print(f"  - Skipped {key}: ckpt={ck_shape} model={md_shape}")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped)-10} more skipped keys")
        else:
            raise RuntimeError("No compatible parameters found to load from checkpoint.")
    model.eval()
    return model
# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def run(ckpt_dir, ckpt_name, data_dir, k_values, batch_size=128, num_workers=4):
    # Paths: config.json is expected to live in the same directory as the checkpoint
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    # Full checkpoint path
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    # Load model
    model = load_model_from_checkpoint(ckpt_path, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = model.to(device)
    # Load dataset
    print(f"[INFO] Loading dataset from {data_dir} ...")
    ds = load_from_disk(data_dir)
    if "test" not in ds:
        raise ValueError("Dataset must include a 'test' split.")
    test_dataset = SpeechTextDataset(ds["test"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    # Compute all embeddings
    print("[INFO] Computing embeddings for test set...")
    gt_embs, pred_embs = compute_embeddings(model, test_loader, device)

    # Compute full similarity matrix (queries x passages)
    print("[INFO] Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(gt_embs, pred_embs, device=device, batch_size=256)

    # Evaluate recall@k in batches using the similarity matrix
    recall_scores = {}
    print("\n===== TEXT → SPEECH RETRIEVAL =====")
    for k in k_values:
        score = recall_at_k_batched(sim_matrix, k, batch_size=256, device=device)
        recall_scores[k] = score
        print(f"Recall@{k}: {score:.4f}")
    # Evaluate MRR
    mrr_score = mrr_batched(sim_matrix, batch_size=256, device=device)
    print(f"MRR: {mrr_score:.4f}")
    # Output directory with checkpoint name
    checkpoint_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_dir = os.path.join(os.path.dirname(ckpt_path), f"evaluation_results_{checkpoint_name}")
    os.makedirs(output_dir, exist_ok=True)
    # Save results as CSV
    df = pd.DataFrame({"k": list(recall_scores.keys()), "recall": list(recall_scores.values())})
    df.to_csv(os.path.join(output_dir, "recall_at_k.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"recall": recall_scores, "mrr": mrr_score}, f, indent=2)
    # Plot
    plot_recall_curve(recall_scores, mrr_score, os.path.join(output_dir, "recall_curve_text_to_speech.png"))
    print(f"[OK] Results saved to {output_dir}")
    return recall_scores, mrr_score


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Text→Speech Retrieval Evaluation")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing the checkpoint and config.json")
    parser.add_argument("--ckpt-name", type=str, required=True, help="Checkpoint filename (e.g. cnn_alignment_model.ckpt)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the preprocessed hparl dataset")
    parser.add_argument("--k", type=int, nargs="+", required=True, help="Values of K for Recall@K, e.g. --k 1 5 10 20 50")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    run(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        data_dir=args.data_dir,
        k_values=args.k,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

