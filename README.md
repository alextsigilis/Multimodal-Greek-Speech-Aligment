# Multimodal Greek Speech Aligner

Code for training and evaluating a contrastive aligner that maps Greek speech representations
to text embeddings (and vice-versa). The project uses precomputed Whisper encoder hidden
states for speech and E5 embeddings for text.

**Contents & key files**
- `preprocess.py` — prepare dataset and precompute `pooled_speech_embeddings`, `pooled_attn_masks`, `transcript_embeddings` and save to disk.
- `train.py` / `train.ipynb` — training script and notebook; uses `models.py` and Lightning to train adapters and saves checkpoints and `args.json`/`config.json` under the checkpoint directory.
- `evaluate.py` — evaluation script (text→speech retrieval). Computes embeddings, similarity matrix, Recall@K and MRR.
- `models.py` — adapter implementations (`CnnAdapter`, `MlpAdapter`, `LstmAdapter`) and `AlignmentModel`.
- `configs/` — example configs (adapter kwargs, etc.).

Getting started
---------------
1. Create a Python environment and install requirements.

	 pip install -r requirements.txt

2. Prepare the dataset (HParl) and precompute embeddings. Default paths in the notebooks assume `/mnt/h/hparl-preprocessed`.

	 # run preprocessing (example)
	 python preprocess.py --data-dir /path/to/raw_hparl --out-dir /mnt/h/hparl-preprocessed

Training
--------
Train an adapter (example):

```bash
python train.py \
	--data-dir /mnt/h/hparl-preprocessed \
	--checkpoint-name model_0.ckpt \
	--config-file configs/cnn-aligner.json \
	--epochs 40 \
	--batch-size 128
```

- Checkpoints and a `config.json` are saved under the checkpoint directory (e.g. `/mnt/h/outputs/checkpoints/cnn_aligner_0/config.json`).
- `train.py` also writes an `args.json` containing the full training arguments and `config` used.

Config format
-------------
Each checkpoint directory contains a `config.json` describing the adapter to instantiate. Current format:

```json
{
	"adapter-type": "cnn",    // one of: "cnn", "mlp", "lstm"
	"kwargs": { ... }         // adapter constructor kwargs (speech_dim, text_dim, hidden_dim, ...)
}
```

Evaluation
----------
Run retrieval evaluation using a checkpoint directory and checkpoint filename:

```bash
python evaluate.py \
	--ckpt-dir /mnt/h/outputs/checkpoints/cnn_aligner_0/ \
	--ckpt-name model_0.ckpt \
	--data-dir /mnt/h/hparl-preprocessed/ \
	--k 1 3 5 10 15 20
```

What evaluation does
- Loads `config.json` from `--ckpt-dir` and instantiates the adapter + `AlignmentModel`.
- Loads the checkpoint `--ckpt-name` from `--ckpt-dir` and attempts to load compatible parameters.
- Computes speech and text embeddings for the test split, builds a similarity matrix S (queries = rows = ground-truth text embeddings, passages = columns = speech embeddings), then computes Recall@K and MRR.

Adapters
--------
- `CnnAdapter`: multi-kernel temporal convolutions + pooling.
- `MlpAdapter`: per-timestep MLP.
- `LstmAdapter`: LSTM-based sequence adapter (bidirectional option).

Notes & troubleshooting
-----------------------
- If you see warnings about missing/unexpected keys when loading a checkpoint, it usually means the adapter architecture in `config.json` does not match the adapter that was saved in the checkpoint. Ensure `adapter-type` and `kwargs` match the training configuration.
- The evaluation script will try to load only compatible parameters when shapes differ, but results will be invalid if the adapter architectures differ significantly.
- Large evaluation runs may require GPU and sufficient memory for the similarity matrix; `evaluate.py` computes the matrix in query batches to reduce peak memory but still requires O(N*M) memory for the final matrix on the chosen device.

Contact
-------
See the repository owner for questions or feature requests.