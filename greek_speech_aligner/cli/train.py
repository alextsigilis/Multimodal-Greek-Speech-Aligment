#!/usr/bin/env python
"""Train an aligner on the preprocessed dataset."""
import argparse
import json
import os

from greek_speech_aligner import training
from greek_speech_aligner.config import build_adapter, load_config
from greek_speech_aligner.utils import set_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a speech→text aligner (Lightning) on precomputed embeddings")
    parser.add_argument("--data-dir", required=True, help="Path to the HuggingFace dataset saved with save_to_disk")
    parser.add_argument("--config-file", required=True, help="Aligner config JSON ('adapter-type', 'kwargs', optional 'init_tau', 'gamma')")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to save/load checkpoints, config.json, args.json and logs")
    parser.add_argument("--input-checkpoint", default=None, help="Existing checkpoint filename (inside --checkpoint-dir) to resume from")
    parser.add_argument("--output-checkpoint-name", default="model.ckpt", help="Name for the saved checkpoint")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of the train split held out for validation (default: 0.2)")
    parser.add_argument("--batch-size", type=int, default=training.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=training.DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=training.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--init-tau", type=float, default=None, help="Initial temperature (default: config 'init_tau' or 0.07)")
    parser.add_argument("--gamma", type=float, default=None, help="Weight of the speech→text loss (default: config 'gamma' or 0.1)")
    parser.add_argument("--epochs", type=int, default=training.DEFAULT_EPOCHS)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--val-check-interval", type=float, default=1.0, help="Validation interval as a fraction of an epoch (default: 1.0 = every epoch)")
    parser.add_argument("--precision", default=None, help="Lightning precision (default: 16-mixed on CUDA, 32-true otherwise)")
    parser.add_argument("--logger-name", default="alignment")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from datasets import load_from_disk

    set_seed(args.seed)

    cfg = load_config(args.config_file)
    init_tau = args.init_tau if args.init_tau is not None else cfg.get("init_tau", 0.07)
    gamma = args.gamma if args.gamma is not None else cfg.get("gamma", 0.1)
    adapter = build_adapter(cfg)

    print(f"Loading dataset from {args.data_dir} ...")
    ds = load_from_disk(args.data_dir)
    if "train" not in ds:
        raise ValueError("Dataset must contain a 'train' split.")
    split = ds["train"].train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "init_tau": init_tau, "gamma": gamma, "config": cfg}, f, indent=4)

    training.run_training(
        train_ds=split["train"],
        val_ds=split["test"],
        adapter=adapter,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        init_tau=init_tau,
        gamma=gamma,
        max_epochs=args.epochs,
        input_checkpoint=args.input_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        output_checkpoint_name=args.output_checkpoint_name,
        logger_name=args.logger_name,
        log_dir=args.log_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
        val_check_interval=args.val_check_interval,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
