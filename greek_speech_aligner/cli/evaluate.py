#!/usr/bin/env python
"""Text→speech retrieval evaluation (Recall@K, MRR)."""
import argparse

from greek_speech_aligner.evaluation import run_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Text→Speech Retrieval Evaluation")
    parser.add_argument("--ckpt-dir", required=True, help="Directory containing the checkpoint and config.json")
    parser.add_argument("--ckpt-name", required=True, help="Checkpoint filename (e.g. model.ckpt)")
    parser.add_argument("--data-dir", required=True, help="Directory containing the preprocessed dataset")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 15, 20], help="Values of K for Recall@K (default: 1 5 15 20)")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    run_evaluation(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        data_dir=args.data_dir,
        k_values=args.k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
    )


if __name__ == "__main__":
    main()
