import os, re, string, unicodedata
import argparse
import torch
import librosa
import numpy as np
from datasets import load_dataset, DatasetDict, Features, Array2D, Sequence, Value
from transformers import AutoModel, AutoFeatureExtractor
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

BASE = '/mnt/h'
SAVE_DIR = 'hparl_precomputed'
SE_MODEL_ID = 'openai/whisper-tiny'
TE_MODEL_ID = 'intfloat/multilingual-e5-small'
KERNEL_SIZE = 8
STRIDE = 4
BATCH_SIZE = 256
TARGET_SR = 16_000  # 16kHz

def resample(batch, target_sr):
    """Resample a batch of audio waveforms to the target sampling rate."""
    audios = batch['audio']
    waveforms = [np.array(audio['array']) for audio in audios]
    sampling_rates = [audio['sampling_rate'] for audio in audios]
    speech = [
        librosa.resample(wf, orig_sr=sr, target_sr=target_sr)
        for wf, sr in zip(waveforms, sampling_rates)
    ]
    return speech

def embed_speech(preprocessor, model, speech, sr, device):
    """Encode raw speech waveforms into Whisper encoder hidden states."""
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

def masked_mean_pool_time(speech_emb, attn_mask, kernel_size=8, stride=4, eps=1e-5):
    """Temporally downsample speech embeddings via masked average pooling."""
    B, T, D = speech_emb.shape
    x = speech_emb.permute(0, 2, 1)
    valid = attn_mask.unsqueeze(1).to(x.dtype)

    num = F.avg_pool1d(x * valid, kernel_size, stride=stride)
    den = F.avg_pool1d(valid, kernel_size, stride=stride)

    pooled_emb = num / den.clamp(min=eps)
    pooled_emb = torch.where(den > 0, pooled_emb, torch.zeros_like(pooled_emb))

    new_mask = (den.squeeze(1) > 0)

    return pooled_emb.permute(0, 2, 1), new_mask

def clean_text(text):
    """Clean text: remove bracket tokens, punctuation, normalize unicode, lowercase."""
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def embed_transcripts(encoder, batch, batch_size=BATCH_SIZE):
    """Encode transcript sentences into normalised E5 embeddings."""
    sentences = [clean_text(s) for s in batch['sentence']]
    queries = [f'query: {s}' for s in sentences]

    return encoder.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_tensor=True
    )

def preprocess_batch(batch, speech_preprocessor, speech_model, te_model, device, target_sr=TARGET_SR, kernel_size=KERNEL_SIZE, stride=STRIDE, batch_size=BATCH_SIZE):
    """Preprocess a batch of audio and transcript data to produce pooled speech embeddings, attention masks, and transcript embeddings."""
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
        te_model,
        batch,
        batch_size=batch_size
    )

    return {
        'pooled_speech_embeddings': pooled_emb.tolist(),
        'pooled_attn_masks': new_mask.tolist(),
        'transcript_embeddings': transcript_emb.tolist(),
    }

def run(base: str = BASE, save_dir: str = SAVE_DIR, speech_encoder: str = SE_MODEL_ID, text_encoder: str = TE_MODEL_ID, target_sr: int = TARGET_SR, kernel_size: int = 8, stride: int = 4, batch_size: int = BATCH_SIZE):
    """Run the full preprocessing pipeline."""
    os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
    os.environ["HF_HOME"] = base
    os.environ["TRANSFORMERS_CACHE"] = f"{base}/models"

    orig_ds = load_dataset(
        'ddamianos/hparl',
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    speech_preprocessor = AutoFeatureExtractor.from_pretrained(speech_encoder)
    speech_model = AutoModel.from_pretrained(speech_encoder).to(device).eval()
    te_model = SentenceTransformer(text_encoder, device=device)

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
    test_txt_emb = embed_transcripts(te_model,
                                     test_batch,
                                     batch_size=1)
    _, time_dim, speech_dim = pooled_test_sp_emb.size()
    _, text_dim = test_txt_emb.size()

    features = Features({
        'pooled_speech_embeddings': Array2D(shape=(time_dim, speech_dim),
                                            dtype='float16'),
        'pooled_attn_masks': Sequence(Value('int16')),
        'transcript_embeddings': Sequence(Value('float16'))
    })

    precompute = lambda batch: preprocess_batch(batch,
                                                speech_preprocessor,
                                                speech_model,
                                                te_model,
                                                target_sr,
                                                kernel_size,
                                                stride,
                                                batch_size,
                                                device)

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
        name: split.remove_columns([c for c in split.column_names
                                    if c not in ['pooled_speech_embeddings',
                                                 'pooled_attn_masks',
                                                 'transcript_embeddings',
                                                 'utt_id']
        ])
        for name, split in tmp.items()
    })

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    clean.save_to_disk(save_dir)
    print("Saved.")
    return tmp
