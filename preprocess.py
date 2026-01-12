#   precompute.py
#
# Precompute speech and transcript embeddings from the original hparl dataset
# Embeddings stored as float16, masks stored as int16
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import os, re
import argparse
import torch
import librosa
import numpy as np
from datasets import load_dataset, DatasetDict, Features, Array2D, Sequence, Value
from transformers import AutoModel, AutoFeatureExtractor
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# Default arguments
BASE = '/mnt/h'
SAVE_DIR = 'hparl_precomputed'
SE_MODEL_ID = 'openai/whisper-tiny'
TE_MODEL_ID = 'intfloat/multilingual-e5-small'
BATCH_SIZE = 256
TARGET_SR = 16_000  # 16kHz


def resample(batch, target_sr):
    audios = batch['audio']
    waveforms = [np.array(audio['array']) for audio in audios]
    sampling_rates = [audio['sampling_rate'] for audio in audios]
    speech = [
        librosa.resample(wf, orig_sr=sr, target_sr=target_sr)
        for wf, sr in zip(waveforms, sampling_rates)
    ]
    return speech


def embed_speech(preprocessor, model, speech, sr, device):
    inputs = preprocessor(
        speech,
        sampling_rate=sr,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )

    with torch.no_grad():
        input_features = inputs.input_features.to(device)
        attn_mask = inputs.attention_mask.to(device)

        enc_out = model.encoder(
            input_features=input_features,
            attention_mask=attn_mask
        )

    speech_emb = enc_out.last_hidden_state.cpu()
    T_max = speech_emb.size(1)
    attn_mask = attn_mask[:, :T_max].to(torch.int16).cpu()

    return speech_emb, attn_mask


def masked_mean_pool_time(
    speech_emb,
    attn_mask,
    kernel_size=8,
    stride=4,
    eps=1e-5
):
    B, T, D = speech_emb.shape
    x = speech_emb.permute(0, 2, 1)
    valid = attn_mask.unsqueeze(1).to(x.dtype)

    num = F.avg_pool1d(x * valid, kernel_size, stride=stride)
    den = F.avg_pool1d(valid, kernel_size, stride=stride)

    pooled_emb = num / den.clamp(min=eps)
    pooled_emb = torch.where(den > 0, pooled_emb, torch.zeros_like(pooled_emb))

    # mask is binary (0/1)
    new_mask = (den.squeeze(1) > 0)

    return pooled_emb.permute(0, 2, 1), new_mask


def embed_transcripts(encoder, batch, batch_size):
    sentences = [
        re.sub(r'\[.+\]', '', s).strip()
        for s in batch['sentence']
    ]
    queries = [f'query: {s}' for s in sentences]

    return encoder.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

def run(
    base: str = BASE,
    save_dir: str = SAVE_DIR,
    speech_encoder: str = SE_MODEL_ID,
    text_encoder: str = TE_MODEL_ID,
    target_sr: int = TARGET_SR,
    kernel_size: int = 8,
    stride: int = 4,
    batch_size: int = BATCH_SIZE
):
    # HuggingFace caches
    os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
    os.environ["HF_HOME"] = base
    os.environ["TRANSFORMERS_CACHE"] = f"{base}/models"

    # Load dataset
    orig_ds = load_dataset(
        'ddamianos/hparl',
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )

    # Encoders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    speech_preprocessor = AutoFeatureExtractor.from_pretrained(speech_encoder)
    speech_model = AutoModel.from_pretrained(speech_encoder).to(device).eval()
    text_encoder = SentenceTransformer(text_encoder, device=device)

    # Find out resulting shapes
    test_batch = orig_ds['test'][0:1]
    test_speech = resample(test_batch, target_sr)
    test_sp_emb, test_attn_mask = embed_speech(speech_preprocessor,
                                               speech_model,
                                               test_speech,
                                               target_sr,
                                               device)
    pooled_test_sp_emb, pooled_test_mask = masked_mean_pool_time(test_sp_emb,
                                                                 test_attn_mask,
                                                                 kernel_size=kernel_size,
                                                                 stride=stride)
    test_txt_emb = embed_transcripts(text_encoder,
                                     test_batch,
                                     batch_size=1)
    _, time_dim, speech_dim = pooled_test_sp_emb.size()
    _, text_dim = test_txt_emb.size()

    def precompute(batch):
        speech = resample(batch, target_sr)

        speech_emb, attn_mask = embed_speech(
            speech_preprocessor,
            speech_model,
            speech,
            target_sr,
            device
        )

        pooled_emb, new_mask = masked_mean_pool_time(
            speech_emb,
            attn_mask,
            kernel_size=kernel_size,
            stride=stride
        )

        transcript_emb = embed_transcripts(
            text_encoder,
            batch,
            batch_size=batch_size
        )

        # ↓↓↓ FINAL STORAGE DTYPES ↓↓↓
        return {
            'pooled_speech_embeddings': pooled_emb.tolist(),
            'pooled_attn_masks': new_mask.tolist(),
            'transcript_embeddings': transcript_emb.tolist(),
        }

    features = Features({
        'pooled_speech_embeddings': Array2D(shape=(time_dim, speech_dim),
                                            dtype='float16'),
        'pooled_attn_masks': Sequence(Value('int16')),
        'transcript_embeddings': Sequence(Value('float16'))
    })

    tmp = orig_ds.map(
        precompute,
        batched=True,
        batch_size=batch_size,
        num_proc=None,
        features=features,
        remove_columns=orig_ds['train'].column_names,
        desc="Precomputing pooled speech + text embeddings"
    )

    clean = DatasetDict({
        name: split.remove_columns([
            c for c in split.column_names
            if c not in [
                'pooled_speech_embeddings',
                'pooled_attn_masks',
                'transcript_embeddings'
            ]
        ])
        for name, split in tmp.items()
    })

    clean.save_to_disk(f"{base}/{save_dir}")
    print("Saved.")
    return clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Precompute pooled speech embeddings (fp16 embeddings, int16 masks)"
    )
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("-se", "--speech-encoder", default=SE_MODEL_ID)
    parser.add_argument("-te", "--text-encoder", default=TE_MODEL_ID)
    parser.add_argument("-sr", "--target-sr", default=TARGET_SR)
    parser.add_argument("--kernel-size", default=8, type=int)
    parser.add_argument("--stride", default=4, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)

    args = parser.parse_args()
    run(**vars(args))

