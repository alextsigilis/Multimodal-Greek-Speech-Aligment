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
import torch.nn.functional as F
from datasets import DatasetDict, Features, Array2D, Sequence, Value, load_dataset
from transformers import AutoModel, AutoFeatureExtractor
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default arguments
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
        Attention mask of shape ``[B, T]`` (CPU, int16).
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


def embed_transcripts(encoder, batch, batch_size=4):
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

def get_device():
    """Detect and return the best available device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

def run(
    dataset_url: str,
    target_sr: int = 16_000,
    kernel_size: int = 8,
    stride: int = 4,
    batch_size: int = 4,
    cache_dir: str = "./cache"
):
    """Run the full preprocessing pipeline.

    Loads the raw dataset from dataset_url, computes pooled Whisper
    encoder embeddings for speech and normalised E5 embeddings for
    transcripts, then saves the result as an Arrow dataset.

    Parameters
    ----------
    dataset_url : str
        HuggingFace dataset URL to load.
    target_sr : int
        Target sampling rate for audio resampling (default: 16000).
    kernel_size : int
        Temporal pooling window size (default: 8).
    stride : int
        Temporal pooling stride (default: 4).
    batch_size : int
        Batch size for dataset ``.map()`` and encoder inference.
    cache_dir : str
        Directory to store cached files (default: "./cache").

    Returns
    -------
    DatasetDict
        The precomputed dataset with columns ``pooled_speech_embeddings``,
        ``pooled_attn_masks``, and ``transcript_embeddings``.
    """
    # Encoders
    device = get_device()
    print(f"Using {device}")

    speech_preprocessor = AutoFeatureExtractor.from_pretrained(SE_MODEL_ID)
    speech_model = AutoModel.from_pretrained(SE_MODEL_ID).to(device).eval()
    te_model = SentenceTransformer(TE_MODEL_ID, device=device)

    # Find out resulting shapes
    dataset = load_dataset(dataset_url)
    test_batch = dataset['test'][0:1]
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

    # Remove features specification to let datasets handle types automatically
    # This prevents the KeyError: 'path' issue by not forcing specific feature types
    ds_dir = f"{cache_dir}/{dataset_url.replace('/', '--')}_{TE_MODEL_ID.replace('/', '--')}_{SE_MODEL_ID.replace('/', '--')}_sr{target_sr}_ksz{kernel_size}_strd{stride}"

    tmp = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=None,
        fn_kwargs={
            "speech_preprocessor": speech_preprocessor,
            "speech_model": speech_model,
            "te_model": te_model,
            "device": device,
            "target_sr": target_sr,
            "kernel_size": kernel_size,
            "stride": stride,
            "batch_size": batch_size
        },
        load_from_cache_file=True,
        cache_file_names=cache_file_names,
        desc="Precomputing pooled speech + text embeddings"
    )

    clean = DatasetDict({
        name: split
        for name, split in tmp.items()
    })

    clean.save_to_disk(ds_dir)

    # Cache the dataset using Hugging Face's built-in caching instead of save_to_disk
    # The dataset will be cached automatically by Hugging Face
    print(f"Preprocessed dataset saved in {ds_dir}.")
    
    return clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Precompute pooled speech embeddings (fp16 embeddings, int16 masks) from HuggingFace datasets containing 'sentence' and 'audio' columns. The audio column must be cast to Audio type."
    )
    parser.add_argument("--dataset-url", required=True, help="HuggingFace dataset URL")
    parser.add_argument("--target-sr", default=16_000, type=int, help="Target sampling rate for audio resampling (default: 16000)")
    parser.add_argument("--kernel-size", default=8, type=int, help="Temporal pooling window size applied to pooled speech embeddings (default: 8)")
    parser.add_argument("--stride", default=4, type=int, help="Temporal pooling stride applied to pooled speech embeddings (default: 4)")
    parser.add_argument("--batch-size", default=4, type=int, help="Batch size for dataset .map() and encoder inference (default: 256)")
    parser.add_argument("--cache-dir", default="./cache", help="Directory to store cached files (default: ./cache)")

    args = parser.parse_args()
    
    # Load the dataset using the HuggingFace datasets library
    run(**vars(args))
