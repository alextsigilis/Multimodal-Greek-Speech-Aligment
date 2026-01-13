#!/usr/bin/env python3
import os
import json
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader

from models import SpeechEncoder
from train import SpeechTextDataset, collate_fn
from loss import ContrastiveLoss


# ---------------------------------------------------------
# Embedding computation with progress bar
# ---------------------------------------------------------
@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    model.eval()
    all_text = []
    all_speech = []

    progress = tqdm(dataloader, desc="Computing embeddings", dynamic_ncols=True)

    for batch in progress:
        speech = batch["speech"].to(device)
        mask = batch["mask"].to(device)
        text = batch["text"].to(device)

        pred_speech = model(speech, mask)

        all_text.append(text.cpu())
        all_speech.append(pred_speech.cpu())

    return torch.cat(all_text), torch.cat(all_speech)


# ---------------------------------------------------------
# Recall@K
# ---------------------------------------------------------
def recall_at_k(sim_matrix, k):
    # sim_matrix: [N, N]
    topk = torch.topk(sim_matrix, k=k, dim=1).indices
    correct = torch.arange(sim_matrix.size(0)).unsqueeze(1)
    hits = (topk == correct).any(dim=1).float()
    return hits.mean().item()


def plot_recall_curve(recall_dict, output_path):
    ks = list(recall_dict.keys())
    vals = [recall_dict[k] for k in ks]

    plt.figure()
    plt.plot(ks, vals, marker="o")
    plt.xlabel("K")
    plt.ylabel("Recall@K (text→speech)")
    plt.title("Text→Speech Retrieval Recall Curve")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[OK] Saved plot → {output_path}")


# ---------------------------------------------------------
# Load model & config from checkpoint directory
# ---------------------------------------------------------
def load_from_checkpoint(path):
    """
    Load a trained SpeechEncoder and ContrastiveLoss from a checkpoint directory.

    Parameters
    ----------
    path : str
        Directory containing 'checkpoint' and 'args.json'.

    Returns
    -------
    model : SpeechEncoder
        The reconstructed model with loaded weights.
    criterion : ContrastiveLoss
        The reconstructed loss with loaded state (including learned temperature).
    args : dict
        The arguments dictionary loaded from 'args.json' (metadata).
    """
    cp_file = os.path.join(path, 'checkpoint')
    json_file = os.path.join(path, 'args.json')

    # Read the checkpoint metadata (including model config and loss hyperparams)
    with open(json_file, 'r') as f:
        args = json.load(f)
        config = args['config']

    # Instantiate model from saved config
    model = SpeechEncoder(**config)

    # Instantiate loss using saved hyperparameters (fallbacks are the training defaults)
    init_tau = args.get('init_tau', 0.07)
    normalize_inputs = args.get('normalize_inputs', True)
    criterion = ContrastiveLoss(
        init_tau=init_tau,
        normalize_inputs=normalize_inputs,
    )

    # Load weights
    checkpoint = torch.load(cp_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])

    # Put them in eval mode by default (you can switch to train() later if needed)
    model.eval()
    criterion.eval()

    return model, criterion, args
# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------
def main(checkpoint_dir, k_values):
    # Load model + config
    model, _, cfg = load_from_checkpoint(checkpoint_dir)

    # Output directory for evaluation
    output_dir = os.path.join(checkpoint_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"[INFO] Loading dataset from {cfg['data_dir']} ...")
    ds = load_from_disk(cfg["data_dir"])
    if "test" not in ds:
        raise ValueError("Dataset must include a 'test' split.")

    # Build dataset + dataloader
    test_dataset = SpeechTextDataset(ds["test"])

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = model.to(device)

    # Compute embeddings (with progress bar)
    text_embs, speech_embs = compute_embeddings(model, test_loader, device)

    # Cosine similarity (text→speech)
    text_norm = text_embs / text_embs.norm(dim=1, keepdim=True)
    speech_norm = speech_embs / speech_embs.norm(dim=1, keepdim=True)
    sim_matrix = text_norm @ speech_norm.T

    # Evaluate recall@k
    recall_scores = {}
    print("\n===== TEXT → SPEECH RETRIEVAL =====")

    for k in tqdm(k_values, desc="Computing Recall@K", dynamic_ncols=True):
        score = recall_at_k(sim_matrix, k)
        recall_scores[k] = score
        print(f"Recall@{k}: {score:.4f}")

    # Save results
    out_json = os.path.join(output_dir, "text_to_speech_recall.json")
    with open(out_json, "w") as f:
        json.dump(recall_scores, f, indent=2)
    print(f"[OK] Saved recall scores → {out_json}")

    # Plot
    plot_recall_curve(
        recall_scores,
        os.path.join(output_dir, "recall_curve_text_to_speech.png")
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text→Speech Retrieval Evaluation")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing the testing dataset")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing args.json and checkpoint")
    parser.add_argument("--k", type=int, nargs="+", required=True,
                        help="Values of K for Recall@K, e.g. --k 1 5 10 20 50")

    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        k_values=args.k
    )

