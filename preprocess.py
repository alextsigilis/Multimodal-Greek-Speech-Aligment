#   precompute.py
#
# Precompute speech and transcript embeddings from the original hparl dataset
# Embeddings stored as float16, masks stored as int16
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import os, re, string, unicodedata
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
    """Resample a batch of audio waveforms to the target sampling rate.

    Parameters
    ----------
    batch : dict
        A batch from the HuggingFace dataset containing an ``'audio'`` key,
        where each element has ``'array'`` (waveform) and
        ``'sampling_rate'`` fields.
    target_sr : int
        Target sampling rate in Hz (e.g. 16000).

    Returns
    -------
    list[np.ndarray]
        List of resampled waveforms.
    """
    audios = batch['audio']
    waveforms = [np.array(audio['array']) for audio in audios]
    sampling_rates = [audio['sampling_rate'] for audio in audios]
    speech = [
        librosa.resample(wf, orig_sr=sr, target_sr=target_sr)
        for wf, sr in zip(waveforms, sampling_rates)
    ]
    return speech


def embed_speech(preprocessor, model, speech, sr, device):
    """Encode raw speech waveforms into Whisper encoder hidden states.

    Parameters
    ----------
    preprocessor : AutoFeatureExtractor
        Whisper feature extractor that converts waveforms to log-mel
        spectrograms.
    model : AutoModel
        Whisper model (only the encoder is used).
    speech : list[np.ndarray]
        List of resampled waveforms.
    sr : int
        Sampling rate of the waveforms.
    device : str
        Device to run inference on (``'cuda'`` or ``'cpu'``).

    Returns
    -------
    speech_emb : torch.Tensor
        Encoder hidden states of shape ``[B, T, D]`` (CPU, float32).
    attn_mask : torch.Tensor
        Attention mask of shape ``[B, T]`` (CPU, int16).
    """
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
    """Temporally downsample speech embeddings via masked average pooling.

    Applies 1-D average pooling along the time axis while respecting the
    attention mask so that padding frames do not contribute.

    Parameters
    ----------
    speech_emb : torch.Tensor
        Speech hidden states of shape ``[B, T, D]``.
    attn_mask : torch.Tensor
        Binary attention mask of shape ``[B, T]``.
    kernel_size : int, optional
        Pooling window size (default: 8).
    stride : int, optional
        Pooling stride (default: 4).
    eps : float, optional
        Small constant to avoid division by zero (default: 1e-5).

    Returns
    -------
    pooled_emb : torch.Tensor
        Downsampled embeddings of shape ``[B, T', D]``.
    new_mask : torch.Tensor
        Binary mask of shape ``[B, T']`` indicating valid pooled frames.
    """
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


def clean_text(text):
    """Clean text: remove bracket tokens, punctuation, normalize unicode, lowercase."""
    text = re.sub(r'\[[^\]]+\]', '', text)  # remove [UNK] etc. (non-greedy)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def embed_transcripts(encoder, batch, batch_size=BATCH_SIZE):
    """Encode transcript sentences into normalised E5 embeddings.

    Each sentence is cleaned via :func:`clean_text`, prepended with the
    ``'query: '`` prefix required by E5, and encoded with the
    SentenceTransformer model.

    Parameters
    ----------
    encoder : SentenceTransformer
        The multilingual-e5-small sentence transformer.
    batch : dict
        A batch dict containing a ``'sentence'`` key with raw transcripts.
    batch_size : int
        Encoding batch size passed to ``encoder.encode``.

    Returns
    -------
    torch.Tensor
        L2-normalised transcript embeddings of shape ``[B, D]``.
    """
    sentences = [clean_text(s) for s in batch['sentence']]
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
    """Run the full preprocessing pipeline.

    Loads the raw ``ddamianos/hparl`` dataset, computes pooled Whisper
    encoder embeddings for speech and normalised E5 embeddings for
    transcripts, then saves the result as an Arrow dataset.

    Parameters
    ----------
    base : str
        Root directory for HuggingFace caches and output.
    save_dir : str
        Sub-directory under *base* where the precomputed dataset is saved.
    speech_encoder : str
        HuggingFace model ID for the speech encoder (default: whisper-tiny).
    text_encoder : str
        HuggingFace model ID for the text encoder (default:
        multilingual-e5-small).
    target_sr : int
        Target sampling rate for audio resampling (default: 16000).
    kernel_size : int
        Temporal pooling window size (default: 8).
    stride : int
        Temporal pooling stride (default: 4).
    batch_size : int
        Batch size for dataset ``.map()`` and encoder inference.

    Returns
    -------
    DatasetDict
        The precomputed dataset with columns ``pooled_speech_embeddings``,
        ``pooled_attn_masks``, and ``transcript_embeddings``.
    """
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
    te_model = SentenceTransformer(text_encoder, device=device)

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
    test_txt_emb = embed_transcripts(te_model,
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
            te_model,
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

