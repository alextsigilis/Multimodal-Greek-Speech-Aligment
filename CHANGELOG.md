# Changelog

## [0.2.0] — 2026-11-07

### Restructured

The flat scripts were reorganised into an installable package, `greek_speech_aligner`:

| Before                | After                                                               |
|-----------------------|---------------------------------------------------------------------|
| `models.py`           | `models/adapters.py`, `models/alignment.py`, `models/pooling.py`    |
| `loss.py`             | `losses.py`                                                         |
| `preprocess.py`       | `data/preprocessing.py` + `cli/preprocess.py`                       |
| `train.py`            | `data/dataset.py`, `training.py` + `cli/train.py`                   |
| `evaluate.py`         | `evaluation.py`, `checkpoint.py` + `cli/evaluate.py`                |
| `pipelines.py`        | `pipelines.py`                                                      |
| `utils.py`            | `utils.py`, `config.py`                                             |

- CLIs now run as `python -m greek_speech_aligner.cli.{preprocess,train,evaluate}`
  (or `gsa-preprocess` / `gsa-train` / `gsa-evaluate` after `pip install -e .`).
- Added `pyproject.toml`.
- The duplicated `SpeechTextDataset` / `collate_fn` in `train.py` and `evaluate.py`
  were merged into `data/dataset.py`.
- Adapters share a `BaseAdapter` with one `save_config` implementation and a
  registry (`ADAPTERS`) behind `get_adapter_class`.
- The loss is in one place, `losses.bidirectional_contrastive_loss`, used by
  `AlignmentModel`. Checkpoint keys (`adapter.*`, `log_tau`) are unchanged, so
  existing checkpoints still load.
- `pipelines.py` no longer imports the local `evaluate` module by name, which
  could clash with the HuggingFace `evaluate` package.
- Removed dead code: `train_one_epoch`, `validate`, the no-op `run()` in
  `train.py`, unused imports, `pdb` imports and breakpoints, and the empty
  "ASR-based pipeline" stub.

### Fixed

- **Whisper attention mask was misaligned with the encoder output**
  (`embed_speech`). The feature-extractor mask is defined on log-Mel frames
  (3000 per 30 s), while the encoder output has 1500 frames. The mask was
  truncated to its first 1500 entries instead of subsampled by the encoder
  stride, so about twice as many frames as were really there were marked
  valid. For example, a 5 s clip had 500 "valid" frames instead of 250, and the
  extra ones were padding. Mean pooling therefore averaged in padding states.
  **Datasets made with the old `preprocess.py` should be regenerated**, and
  models trained on them retrained, before comparing numbers.
- **`CnnAdapter` saved its stride as `pooling_stride`** while the constructor
  reads `pool_stride`. The unknown key was swallowed by `**kwargs`, so a
  checkpoint trained with a non-default stride was silently rebuilt with
  stride 2 at evaluation time. The key is now `pool_stride`, and the legacy
  `pooling_stride` is still accepted when reading old `config.json` files.
- **`--weight-decay` was ignored**: it was parsed but never passed to the
  LightningModule, so training always used weight decay 0.
- **`init_tau` / `gamma` from the config were ignored** during training and
  evaluation. They are now read from the config (or `--init-tau` / `--gamma`)
  and written to the checkpoint's `config.json`.
- **`configs/cnn-aligner.json` could not be used**: it was a flat kwargs dict
  without `adapter-type`, which `train.py` requires. It now uses the documented
  format.
- **`E5MLPAdapter` was unusable**: it returned an already pooled `[B, D]`
  tensor, which `AlignmentModel` then tried to mean-pool again with a
  `[B, T]` mask, giving a shape error. It now returns per-token states. It also
  defaulted to `device="cuda"`, which crashed on machines without CUDA.
- **Missing `args.json`**: the README said training writes `args.json`, but it
  never did. It is now written to the checkpoint directory.
- `adapter.save_config` errors were silently swallowed (`except: pass`). The
  config is now written before training starts, and errors are raised.
- A missing `--input-checkpoint` silently started training from scratch. It now
  raises `FileNotFoundError`.
- Resuming loaded the weights twice (`load_from_checkpoint` and then
  `Trainer.fit(ckpt_path=...)`). Only `fit(ckpt_path=...)` is used now.
- `precision="16-mixed"` was hard-coded and fails or warns on CPU and MPS. It now
  defaults to `16-mixed` on CUDA and `32-true` elsewhere (`--precision`).
- The checkpoint loader replaced the first `"model."` substring in *every* key,
  including keys that do not start with it. It now only strips a leading prefix.
  A checkpoint with neither `state_dict` nor `model_state_dict` raised a
  `KeyError`.
- `torch.load` now passes `weights_only=False` explicitly. PyTorch ≥ 2.6
  defaults to `True`, which can reject Lightning checkpoint metadata.
- Evaluation ran the full training forward pass (loss included) only to read
  the speech embeddings. It now calls `AlignmentModel.encode`. Ranks are computed
  once, without a full `argsort` of the N×N matrix.
- `AlignmentPipeline`: `top_k` larger than the number of segments crashed in
  `torch.topk`. `overlap >= seg_length` caused an endless loop or
  `ValueError`. Empty audio files are now rejected. `itertools.batched`
  (Python ≥ 3.12 only) was replaced. Added `segment_time(idx)` to get the
  start and end time of a segment.
- `mean_pooling` casts the mask to the hidden-state dtype, so bool, int16 and long
  masks all behave the same.
- `CnnAdapter` now rejects even kernel sizes. With `padding=k//2`, the parallel
  branches would get different lengths and `torch.cat` would fail.
- `preprocess.py`: the `--batch-size` help said the default was 256, but it is 4.
  The header said embeddings were stored as fp16 and masks as int16, which was
  not true. Added `--out-dir`, which the README already used. `python-dotenv` is
  now optional and listed in `requirements.txt`.
- Matplotlib uses the non-interactive `Agg` backend in evaluation, and figures
  are closed after saving.

### Changed

- Training defaults now match the paper (Section 4.2): lr `5e-4`, weight decay
  `5e-5`, batch size 128, 40 epochs, validation once per epoch (it used to be
  every 0.25 epoch; see `--val-check-interval`).
- The training dataloader drops the last incomplete batch. This keeps `B − 1`
  in-batch negatives per step and avoids batch-of-one steps with zero loss.
- `evaluate --k` defaults to `1 5 15 20`, the cut-offs reported in the paper.
- Added `configs/mlp-aligner.json` and `configs/lstm-aligner.json`. They follow
  the structure described in Table 1 of the paper.

### Known issues / to check

- **Parameter counts do not match Table 1 of the paper.** With the shipped
  configs, the counts are: CNN (2 stages, 256 channels, kernels 3/5/7) **4.72 M**,
  MLP (five 256-unit hidden layers) **0.46 M**, BiLSTM (3 layers, 256 units)
  **4.67 M**. The paper reports 1.7 M, 1.2 M and 1.8 M. Either the
  architecture descriptions or the counts in the paper need revisiting.
- `LstmAdapter` does not pack padded sequences, so the backward direction of a
  BiLSTM reads padding frames first. Mean pooling masks them out, but the hidden
  states of valid frames are still affected. This was left unchanged so that
  existing results stay reproducible.
- Preprocessed embeddings are stored as Python floats (float64 in Arrow), and
  the raw `audio` column is kept. This makes the preprocessed dataset much
  larger than necessary.
