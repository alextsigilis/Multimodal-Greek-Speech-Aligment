# Multimodal Greek Speech Aligner

Code for training and evaluating a lightweight contrastive aligner for ASR-free Greek
spoken-content retrieval. A frozen **Whisper-tiny** encoder provides speech representations,
a frozen **multilingual-e5-small** encoder provides the text space, and a small trainable
aligner (MLP, temporal CNN or BiLSTM) maps the speech sequence into that space. Only the
aligner and a temperature parameter are trained, with a bidirectional InfoNCE objective
(`L = γ·L_s→t + (1−γ)·L_t→s`, γ = 0.1) using in-batch negatives.

Dataset: `ddamianos/hparl` (Hellenic Parliamentary Speech Corpus).

## Layout

```
greek_speech_aligner/
├── models/
│   ├── adapters.py      # MlpAdapter, CnnAdapter, LstmAdapter, E5MLPAdapter, get_adapter_class
│   ├── alignment.py     # AlignmentModel (aligner + mean pooling + learnable τ + loss)
│   └── pooling.py       # mask-aware mean pooling
├── data/
│   ├── preprocessing.py # Whisper / E5 embedding, temporal pooling, preprocess_dataset()
│   └── dataset.py       # SpeechTextDataset, collate_fn, SpeechTextDataModule
├── losses.py            # bidirectional_contrastive_loss, ContrastiveLoss
├── config.py            # config.json loading, build_adapter / build_model
├── checkpoint.py        # load_model_from_checkpoint, load_aligner
├── training.py          # AlignmentLitModule, run_training
├── evaluation.py        # similarity matrix, Recall@K, MRR, run_evaluation
├── pipelines.py         # AlignmentPipeline: search a long recording by text query
├── utils.py             # seeding, device selection, JSON helpers
└── cli/                 # preprocess / train / evaluate entry points
configs/                 # example aligner configs (cnn, mlp, lstm)
```

## Getting started

```bash
pip install -r requirements.txt
pip install -e .          # optional: installs the package and the gsa-* commands
```

All commands below are run from the repository root (or use the `gsa-preprocess`,
`gsa-train`, `gsa-evaluate` scripts after `pip install -e .`).

### 1. Preprocess

Computes `pooled_speech_embeddings`, `pooled_attn_masks` and `transcript_embeddings` for every
split and saves the dataset with `save_to_disk`. The paper's principal setting uses temporal
pooling with kernel 32 / stride 16.

```bash
python -m greek_speech_aligner.cli.preprocess \
    --dataset-url ddamianos/hparl \
    --kernel-size 32 --stride 16 \
    --out-dir /mnt/h/hparl-preprocessed
```

### 2. Train

```bash
python -m greek_speech_aligner.cli.train \
    --data-dir /mnt/h/hparl-preprocessed \
    --config-file configs/cnn-aligner.json \
    --checkpoint-dir /mnt/h/outputs/checkpoints/cnn_aligner_0 \
    --output-checkpoint-name model.ckpt
```

Defaults follow the paper: AdamW, lr 5e-4, weight decay 5e-5, batch size 128, 40 epochs,
validation every epoch, 20 % of the train split held out for validation. The checkpoint
directory receives the checkpoint, `config.json` (aligner + `init_tau` / `gamma`),
`args.json` (all training arguments) and TensorBoard logs.

### 3. Evaluate (text → speech retrieval)

```bash
python -m greek_speech_aligner.cli.evaluate \
    --ckpt-dir /mnt/h/outputs/checkpoints/cnn_aligner_0 \
    --ckpt-name model.ckpt \
    --data-dir /mnt/h/hparl-preprocessed \
    --k 1 5 15 20
```

Each test transcript is a query and its paired recording is the only relevant passage.
Recall@K and MRR are printed and saved (CSV, JSON, plot) under
`<ckpt-dir>/evaluation_results_<ckpt-name>/`.

## Config format

```jsonc
{
  "adapter-type": "cnn",   // one of: cnn, mlp, lstm, e5mlp
  "kwargs": { ... },       // adapter constructor kwargs (speech_dim, text_dim, hidden_dim, ...)
  "init_tau": 0.07,        // optional
  "gamma": 0.1             // optional, weight of the speech→text loss term
}
```

## Using a trained aligner from Python

```python
from transformers import AutoFeatureExtractor, AutoModel
from sentence_transformers import SentenceTransformer
from greek_speech_aligner.checkpoint import load_aligner
from greek_speech_aligner.pipelines import AlignmentPipeline

aligner = load_aligner("/mnt/h/outputs/checkpoints/cnn_aligner_0", "model.ckpt")
pipe = AlignmentPipeline(
    AutoFeatureExtractor.from_pretrained("openai/whisper-tiny"),
    AutoModel.from_pretrained("openai/whisper-tiny"),
    aligner,
    SentenceTransformer("intfloat/multilingual-e5-small"),
    kernel_size=32, stride=16,      # must match preprocessing used for training
)
pipe.load_file("session.wav")
for idx, _audio in pipe.search("προϋπολογισμός", top_k=5):
    print(idx, pipe.segment_time(idx))
```

## Notes & troubleshooting

- Warnings about missing/unexpected keys when loading a checkpoint usually mean the
  `config.json` does not match the trained architecture. The loader then falls back to
  loading only shape-compatible parameters, and results are not meaningful.
- `config.json` files written before v0.2.0 store the CNN stride as `pooling_stride`; it is
  still accepted.
- Datasets preprocessed before v0.2.0 have incorrect speech masks (see `CHANGELOG.md`) and
  should be regenerated.
- The similarity matrix is `N × M` on the chosen device; it is built in query batches but the
  full matrix is kept in memory.

See [CHANGELOG.md](CHANGELOG.md) for the list of changes.
