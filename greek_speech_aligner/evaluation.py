#
#   evaluation.py
#
# Text→speech retrieval evaluation (Section 4.3 of the paper):
# every ground-truth transcript is a query, its paired recording is the only
# relevant item, and Recall@K / MRR are computed over the full test split.
#
# Author: Alexandros Tsingilis
#
import json
import os

import torch
from tqdm.auto import tqdm


@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    """Return ``(text_embs, speech_embs)`` for every item of ``dataloader``."""
    model.eval()
    all_text, all_speech = [], []
    for batch in tqdm(dataloader, desc="Computing embeddings", dynamic_ncols=True):
        speech = batch["speech"].to(device)
        mask = batch["mask"].to(device)
        all_speech.append(model.encode(speech, mask).cpu())
        all_text.append(batch["text"])
    return torch.cat(all_text), torch.cat(all_speech)


def compute_similarity_matrix(query_embs, passage_embs, device="cpu", batch_size=256):
    """Cosine similarity ``[N_queries, N_passages]``, computed in query batches."""
    device = torch.device(device)
    passages = passage_embs.to(device).float()
    passages = passages / passages.norm(dim=1, keepdim=True).clamp(min=1e-9)

    n = query_embs.shape[0]
    sim = torch.empty((n, passages.shape[0]), dtype=passages.dtype, device=device)
    for i in tqdm(range(0, n, batch_size), desc="Similarity matrix", dynamic_ncols=True,
                  disable=n <= batch_size):
        q = query_embs[i:i + batch_size].to(device).float()
        q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-9)
        sim[i:i + batch_size] = q @ passages.T
    return sim


def compute_ranks(sim_matrix):
    """One-indexed rank of the relevant passage (the diagonal) for each query."""
    n = sim_matrix.shape[0]
    idx = torch.arange(n, device=sim_matrix.device)
    positives = sim_matrix[idx, idx].unsqueeze(1)
    # Number of passages scoring strictly higher than the positive, plus one.
    return (sim_matrix > positives).sum(dim=1) + 1


def compute_recall_at_k(sim_matrix, k, device="cpu"):
    """Recall@K (Eq. 12)."""
    ranks = compute_ranks(sim_matrix.to(device))
    return (ranks <= k).float().mean().item()


def compute_mrr(sim_matrix, device="cpu"):
    """Mean reciprocal rank (Eq. 13)."""
    ranks = compute_ranks(sim_matrix.to(device))
    return (1.0 / ranks.float()).mean().item()


def plot_recall_curve(recall_dict, mrr, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(recall_dict)
    fig, ax = plt.subplots()
    ax.plot(ks, [recall_dict[k] for k in ks], marker="o", label="Recall@K")
    ax.axhline(mrr, color="red", linestyle="--", label=f"MRR={mrr:.4f}")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (text→speech)")
    ax.set_title("Text→Speech Retrieval Recall Curve + MRR")
    ax.grid(True)
    ax.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved plot → {output_path}")


def run_evaluation(ckpt_dir, ckpt_name, data_dir, k_values, batch_size=128, num_workers=4,
                   split="test"):
    """Evaluate ``<ckpt_dir>/<ckpt_name>`` on ``split`` of the preprocessed dataset."""
    import pandas as pd
    from datasets import load_from_disk

    from greek_speech_aligner.checkpoint import load_aligner
    from greek_speech_aligner.data import make_dataloader

    model = load_aligner(ckpt_dir, ckpt_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = model.to(device)

    ds = load_from_disk(data_dir)
    if split not in ds:
        raise ValueError(f"Dataset must include a '{split}' split.")
    loader = make_dataloader(ds[split], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    text_embs, speech_embs = compute_embeddings(model, loader, device)
    sim_matrix = compute_similarity_matrix(text_embs, speech_embs, device=device, batch_size=256)
    ranks = compute_ranks(sim_matrix)

    print("\n===== TEXT → SPEECH RETRIEVAL =====")
    recall_scores = {}
    for k in sorted(set(k_values)):
        recall_scores[k] = (ranks <= k).float().mean().item()
        print(f"Recall@{k}: {recall_scores[k]:.4f}")
    mrr_score = (1.0 / ranks.float()).mean().item()
    print(f"MRR: {mrr_score:.4f}")

    checkpoint_name = os.path.splitext(ckpt_name)[0]
    output_dir = os.path.join(ckpt_dir, f"evaluation_results_{checkpoint_name}")
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({"k": list(recall_scores), "recall": list(recall_scores.values())}) \
        .to_csv(os.path.join(output_dir, "recall_at_k.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"recall": recall_scores, "mrr": mrr_score}, f, indent=2)
    plot_recall_curve(recall_scores, mrr_score,
                      os.path.join(output_dir, "recall_curve_text_to_speech.png"))
    print(f"[OK] Results saved to {output_dir}")
    return recall_scores, mrr_score
