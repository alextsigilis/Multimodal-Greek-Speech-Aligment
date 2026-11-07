#!/usr/bin/env python
"""Precompute pooled Whisper states and E5 transcript embeddings."""
import argparse

from greek_speech_aligner.data.preprocessing import preprocess_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute pooled Whisper-tiny encoder states and multilingual-e5-small "
                    "transcript embeddings from a HuggingFace dataset with 'sentence' and "
                    "'audio' columns, and save it with save_to_disk."
    )
    parser.add_argument("--dataset-url", required=True, help="HuggingFace dataset id, e.g. ddamianos/hparl")
    parser.add_argument("--target-sr", default=16_000, type=int, help="Target sampling rate (default: 16000)")
    parser.add_argument("--kernel-size", default=8, type=int, help="Temporal pooling window (default: 8; paper's principal setting: 32)")
    parser.add_argument("--stride", default=4, type=int, help="Temporal pooling stride (default: 4; paper's principal setting: 16)")
    parser.add_argument("--batch-size", default=4, type=int, help="Batch size for Dataset.map() and encoder inference (default: 4)")
    parser.add_argument("--cache-dir", default="./cache", help="Parent directory of the output when --out-dir is not given (default: ./cache)")
    parser.add_argument("--out-dir", default=None, help="Explicit output directory (default: derived from the arguments inside --cache-dir)")
    args = parser.parse_args(argv)

    try:  # optional: HF_TOKEN etc. from a .env file
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    preprocess_dataset(**vars(args))


if __name__ == "__main__":
    main()
