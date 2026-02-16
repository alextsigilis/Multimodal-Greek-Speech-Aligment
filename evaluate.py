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

def recall_at_k_batched(gt_embs, pred_embs, k, batch_size=256, device="cpu"):
    n = gt_embs.shape[0]
    hits = 0
    for i in tqdm(range(0, n, batch_size), desc=f"Recall@{k}", dynamic_ncols=True):
        batch_gt = gt_embs[i:i+batch_size].to(device)
        # Normalize
        batch_gt = batch_gt / batch_gt.norm(dim=1, keepdim=True)
        pred_norm = pred_embs / pred_embs.norm(dim=1, keepdim=True)
        # Compute similarity
        sim = batch_gt @ pred_norm.T  # [B, N]
        topk = torch.topk(sim, k=k, dim=1).indices
        row_indices = torch.arange(i, min(i+batch_size, n)).unsqueeze(1).to(device)
        batch_hits = (topk == row_indices).any(dim=1).float().sum().item()
        hits += batch_hits
    return hits / n

def mrr_batched(gt_embs, pred_embs, batch_size=256, device="cpu"):
    n = gt_embs.shape[0]
    rr_sum = 0.0
    for i in tqdm(range(0, n, batch_size), desc="MRR", dynamic_ncols=True):
        batch_gt = gt_embs[i:i+batch_size].to(device)
        batch_size_actual = batch_gt.shape[0]
        batch_indices = torch.arange(i, i+batch_size_actual).to(device)
        # Normalize
        batch_gt = batch_gt / batch_gt.norm(dim=1, keepdim=True)
        pred_norm = pred_embs / pred_embs.norm(dim=1, keepdim=True)
        sim = batch_gt @ pred_norm.T  # [B, N]
        # For each row, get the rank of the correct index
        sorted_indices = torch.argsort(sim, dim=1, descending=True)
        ranks = (sorted_indices == batch_indices.unsqueeze(1)).nonzero(as_tuple=False)[:,1] + 1
        rr_sum += (1.0 / ranks.float()).sum().item()
    return rr_sum / n


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
    # Select adapter type based on config
    adapter_type = config.get("adapter_type", "cnn")
    adapter_kwargs = config.get("kwargs", {})
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

def main(ckpt_dir, ckpt_name, data_dir, k_values, batch_size=128, num_workers=4):
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
    gt_embs = gt_embs.to(device)
    pred_embs = pred_embs.to(device)
    # Evaluate recall@k in batches
    recall_scores = {}
    print("\n===== TEXT → SPEECH RETRIEVAL =====")
    for k in k_values:
        score = recall_at_k_batched(gt_embs, pred_embs, k, batch_size=256, device=device)
        recall_scores[k] = score
        print(f"Recall@{k}: {score:.4f}")
    # Evaluate MRR
    mrr_score = mrr_batched(gt_embs, pred_embs, batch_size=256, device=device)
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
    main(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        data_dir=args.data_dir,
        k_values=args.k,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

