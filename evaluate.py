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
from utils import load_config_from_file

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

def compute_similarity_matrix(gt_embs, pred_embs, device="cpu", batch_size=256):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    n = gt_embs.shape[0]
    m = pred_embs.shape[0]

    pred_norm = pred_embs.to(device)
    pred_norm = pred_norm / pred_norm.norm(dim=1, keepdim=True).clamp(min=1e-9)

    sim = torch.empty((n, m), dtype=pred_norm.dtype, device=device)

    for i in tqdm(range(0, n, batch_size), desc="Similarity matrix", dynamic_ncols=True):
        batch_end = min(i + batch_size, n)
        batch_gt = gt_embs[i:batch_end].to(device)
        batch_gt = batch_gt / batch_gt.norm(dim=1, keepdim=True).clamp(min=1e-9)
        sim[i:batch_end] = batch_gt @ pred_norm.T

    return sim

def compute_recall_at_k(sim_matrix, k, device="cpu"):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    sim = sim_matrix.to(device)
    n = sim.shape[0]
    k_eff = min(k, sim.shape[1])
    topk = torch.topk(sim, k=k_eff, dim=1).indices
    row_indices = torch.arange(n, device=device).unsqueeze(1)
    hits = (topk == row_indices).any(dim=1).float().sum().item()
    return float(hits) / float(n)

def compute_mrr(sim_matrix, device="cpu"):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    sim = sim_matrix.to(device)
    n = sim.shape[0]
    sorted_indices = torch.argsort(sim, dim=1, descending=True)
    row_indices = torch.arange(n, device=device).unsqueeze(1)
    matches = (sorted_indices == row_indices)
    ranks = matches.float().argmax(dim=1) + 1
    mrr = (1.0 / ranks.float()).mean().item()
    return mrr

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

def load_model_from_checkpoint(ckpt_path, config):
    adapter_type = config.get("adapter-type", "cnn")
    adapter_kwargs = config.get("kwargs", {}) if isinstance(config.get("kwargs", {}), dict) else {}
    AdapterClass = get_adapter_class(adapter_type)
    adapter = AdapterClass(**adapter_kwargs)
    model = AlignmentModel(adapter=adapter, init_tau=config.get("init_tau", 0.07))
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint["model_state_dict"]
    if any(k.startswith("model.") for k in state_dict.keys()):
        new_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        state_dict = new_state_dict
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

def run(ckpt_dir, ckpt_name, data_dir, k_values, batch_size=128, num_workers=4):
    config_path = os.path.join(ckpt_dir, "config.json")
    config = load_config_from_file(config_path)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    model = load_model_from_checkpoint(ckpt_path, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = model.to(device)
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
    gt_embs, pred_embs = compute_embeddings(model, test_loader, device)
    sim_matrix = compute_similarity_matrix(gt_embs, pred_embs, device=device, batch_size=256)

    recall_scores = {}
    print("\n===== TEXT → SPEECH RETRIEVAL =====")
    for k in k_values:
        score = compute_recall_at_k(sim_matrix, k, device=device)
        recall_scores[k] = score
        print(f"Recall@{k}: {score:.4f}")
    mrr_score = compute_mrr(sim_matrix, device=device)
    print(f"MRR: {mrr_score:.4f}")

    checkpoint_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    output_dir = os.path.join(os.path.dirname(ckpt_path), f"evaluation_results_{checkpoint_name}")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({"k": list(recall_scores.keys()), "recall": list(recall_scores.values())})
    df.to_csv(os.path.join(output_dir, "recall_at_k.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"recall": recall_scores, "mrr": mrr_score}, f, indent=2)

    plot_recall_curve(recall_scores, mrr_score, os.path.join(output_dir, "recall_curve_text_to_speech.png"))
    print(f"[OK] Results saved to {output_dir}")
    return recall_scores, mrr_score

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
