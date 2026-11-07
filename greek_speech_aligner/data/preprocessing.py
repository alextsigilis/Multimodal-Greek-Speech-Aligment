#   preprocessing.py
#
# Precompute (temporally pooled) Whisper encoder states and E5 transcript
# embeddings from a HuggingFace speech dataset (e.g. ddamianos/hparl) and
# save the result with ``DatasetDict.save_to_disk``.
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import re
import string
import unicodedata

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from greek_speech_aligner.utils import get_device

# Default encoders (both frozen)
SE_MODEL_ID = 'openai/whisper-tiny'
TE_MODEL_ID = 'intfloat/multilingual-e5-small'

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
    
    # Resample each waveform to target_sr
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
        Device to run inference on (``'cuda'``, ``'mps'``, or ``'cpu'``).

    Returns
    -------
    speech_emb : torch.Tensor
        Encoder hidden states of shape ``[B, T, D]`` (CPU, float32).
    attn_mask : torch.Tensor
        Attention mask of shape ``[B, T]`` (CPU, int16), at the resolution of
        the encoder output (i.e. downsampled from the log-Mel frame rate by
        the encoder's convolutional stride, which is 2 for Whisper).
    """
    # Ensure we're passing the correct sampling rate to the preprocessor
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
    # The feature-extractor mask is defined on log-Mel frames (3000 per 30 s),
    # while the encoder output has T_max frames (1500): subsample the mask by
    # the encoder stride instead of truncating it, otherwise ~half of the
    # padding frames would be marked as valid.
    enc_stride = max(1, attn_mask.size(1) // T_max)
    attn_mask = attn_mask[:, ::enc_stride][:, :T_max].to(torch.int16).cpu()

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


def embed_transcripts(encoder, batch, batch_size=4):
    """Encode transcript sentences into normalised E5 embeddings.

    Each sentence is cleaned via :func:`clean_text`, prepended with the
    ``'query: '`` prefix expected by E5, and encoded with the
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

def preprocess_batch(batch,
                     speech_preprocessor,
                     speech_model,
                     te_model,
                     device,
                     target_sr=16_000,
                     kernel_size=8,
                     stride=4,
                     batch_size=4):
    
    """
    Preprocess a batch of audio and transcript data to produce pooled speech embeddings, attention masks, and transcript embeddings.

    Parameters
    ----------
    batch : dict
        Batch from the dataset containing 'audio' and 'sentence' fields.
    speech_preprocessor : AutoFeatureExtractor
        Feature extractor for the speech encoder.
    speech_model : AutoModel
        Pretrained speech model (e.g., Whisper).
    te_model : SentenceTransformer
        Pretrained text encoder (e.g., multilingual-e5-small).
    target_sr : int
        Target sampling rate for audio.
    kernel_size : int
        Pooling window size for temporal pooling.
    stride : int
        Pooling stride for temporal pooling.
    batch_size : int
        Batch size for text encoder.
    device : str
        Device to run inference on ('cuda', 'mps', or 'cpu').

    Returns
    -------
    dict
        Dictionary with keys:
            'pooled_speech_embeddings': List of pooled speech embeddings (float)
            'pooled_attn_masks': List of pooled attention masks (int)
            'transcript_embeddings': List of transcript embeddings (float)
    """
    # Resample audio to the target sampling rate
    speech = resample(batch, target_sr)
    # Extract speech embeddings and attention masks from the speech model
    speech_emb, attn_mask = embed_speech(
        speech_preprocessor,
        speech_model,
        speech,
        target_sr,
        device
    )
    # Pool speech embeddings and downsample attention masks
    pooled_emb, new_mask = masked_mean_pool_time(
        speech_emb,
        attn_mask,
        kernel_size=kernel_size,
        stride=stride
    )
    # Encode transcript sentences into normalized embeddings
    transcript_emb = embed_transcripts(
        te_model,
        batch,
        batch_size=batch_size
    )
    # Convert tensors to lists for storage in datasets
    return {
        'pooled_speech_embeddings': pooled_emb.tolist(),
        'pooled_attn_masks': new_mask.tolist(),
        'transcript_embeddings': transcript_emb.tolist(),
    }


def default_output_dir(dataset_url, cache_dir, target_sr, kernel_size, stride):
    """Directory name that encodes the dataset, encoders and pooling setup."""
    return (
        f"{cache_dir}/{dataset_url.replace('/', '--')}"
        f"_{TE_MODEL_ID.replace('/', '--')}_{SE_MODEL_ID.replace('/', '--')}"
        f"_sr{target_sr}_ksz{kernel_size}_strd{stride}"
    )


def preprocess_dataset(
    dataset_url: str,
    target_sr: int = 16_000,
    kernel_size: int = 8,
    stride: int = 4,
    batch_size: int = 4,
    cache_dir: str = "./cache",
    out_dir: str = None,
):
    """Run the full preprocessing pipeline.

    Loads the raw dataset, computes pooled Whisper encoder embeddings for
    speech and normalised E5 embeddings for transcripts, and saves the result
    with ``save_to_disk``.

    Parameters
    ----------
    dataset_url : str
        HuggingFace dataset id (must have ``audio`` and ``sentence`` columns).
    target_sr : int
        Target sampling rate for audio resampling (default: 16000).
    kernel_size, stride : int
        Temporal average-pooling window and stride (paper: (8, 4) and (32, 16)).
    batch_size : int
        Batch size for ``Dataset.map`` and encoder inference.
    cache_dir : str
        Parent directory for the output when ``out_dir`` is not given.
    out_dir : str, optional
        Explicit output directory.

    Returns
    -------
    (DatasetDict, str)
        The precomputed dataset and the directory it was saved to.
    """
    from datasets import DatasetDict, load_dataset
    from sentence_transformers import SentenceTransformer
    from transformers import AutoFeatureExtractor, AutoModel

    device = get_device()
    print(f"Using {device}")

    speech_preprocessor = AutoFeatureExtractor.from_pretrained(SE_MODEL_ID)
    speech_model = AutoModel.from_pretrained(SE_MODEL_ID).to(device).eval()
    te_model = SentenceTransformer(TE_MODEL_ID, device=device)

    dataset = load_dataset(dataset_url)
    ds_dir = out_dir or default_output_dir(dataset_url, cache_dir, target_sr, kernel_size, stride)

    processed = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={
            "speech_preprocessor": speech_preprocessor,
            "speech_model": speech_model,
            "te_model": te_model,
            "device": device,
            "target_sr": target_sr,
            "kernel_size": kernel_size,
            "stride": stride,
            "batch_size": batch_size,
        },
        desc="Precomputing pooled speech + text embeddings",
    )
    processed = DatasetDict(dict(processed.items()))
    processed.save_to_disk(ds_dir)
    print(f"Preprocessed dataset saved in {ds_dir}.")
    return processed, ds_dir
