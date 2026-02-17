#
#   model.py
#
# Definition of the model for Speech - Transcript aligment.
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import torch
import torch.nn as nn, torch.nn.functional as F
from itertools import pairwise
from transformers import AutoModel
import os
import json

def get_adapter_class(adapter_type):
    """
    Returns the adapter class corresponding to the adapter_type string.
    Supported types: 'cnn', 'linear', 'mlp'.
    Extend this function to add new adapters.

    Args:
        adapter_type (str): The type of adapter (case-insensitive).

    Returns:
        nn.Module class: The adapter class.

    Raises:
        ValueError: If adapter_type is unknown.
    """
    adapter_type = str(adapter_type).strip().lower()
    match adapter_type:
        case "cnn":
            return CnnAdapter
        case "mlp":
            return MlpAdapter
        case "lstm":
            return LstmAdapter
        case _:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

def mean_pooling(hidden_state, mask):
    '''
    Mean pooling with an attention mask.

    Computes a mask-aware mean over the temporal dimension of a sequence of 
    hidden states. Only positions where the mask is 1 contribute to the mean.

    Parameters
    ----------
    hidden_state : torch.Tensor
        Input tensor of shape ``[B, T, D]`` containing the sequence of hidden
        representations (e.g., token embeddings from a transformer).

    mask : torch.Tensor
        Attention mask of shape ``[B, T]`` with values in ``{0, 1}``.
        Positions with value ``1`` are included in the pooling operation,
        while positions with value ``0`` are ignored.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B, D]`` containing the mean-pooled representations
        for each example in the batch.
    '''
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    masked_hidden = hidden_state * mask
    sum_hidden = masked_hidden.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    pooled = sum_hidden / lengths
    return pooled

class MlpAdapter(nn.Module):
    """
    MLP Adapter.

    Maps per-frame speech embeddings to per-frame text-like embeddings using
    a simple feed-forward network applied independently to each time step.

    Input:  Tensor [B, T, speech_dim]
    Output: Tensor [B, T, text_dim], mask unchanged
    """
    def __init__(self,
                 speech_dim: int = 384,
                 text_dim: int = 384,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 **kwargs):

        super().__init__()

        speech_dim = int(speech_dim)
        text_dim = int(text_dim)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)

        layers = []
        in_dim = speech_dim
        # Build (num_layers-1) hidden blocks, then final linear to text_dim
        for i in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final projection to text_dim
        layers.append(nn.Linear(in_dim, text_dim))

        self.net = nn.Sequential(*layers)
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        # store constructor args for config export
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = float(dropout)
        self._config = {
            "speech_dim": int(speech_dim),
            "text_dim": int(text_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(self.dropout_rate),
        }

    def save_config(self, dir_path, filename="config.json"):
        os.makedirs(dir_path, exist_ok=True)
        cfg = {"adapter-type": "mlp", "kwargs": self._config}
        with open(os.path.join(dir_path, filename), "w") as f:
            json.dump(cfg, f, indent=4)

    def forward(self, speech_embs, attn_mask):
        # speech_embs: [B, T, D]
        B, T, D = speech_embs.shape
        x = speech_embs.reshape(-1, D)       # [B*T, D]
        x = self.net(x)                      # [B*T, text_dim]
        x = x.view(B, T, -1)                 # [B, T, text_dim]
        return x, attn_mask

class CnnAdapter(nn.Module):
    '''
    CNN Adapter with multi-kernel convolutions and temporal downsampling.

    input:  speech_embs [B, T, speech_dim]
    output: aligned_embs [B, T', text_dim]
            pooled_mask  [B, T']
    '''

    def __init__(self,
                 speech_dim: int = 384,
                 text_dim: int = 384,
                 hidden_dim: int = 256,
                 kernel_sizes=(3, 5, 7),
                 num_layers: int = 2,
                 pool_stride: int = 2,
                 dropout: float = 0.1,
                 **kwargs):

        super().__init__()

        speech_dim = int(speech_dim)
        text_dim = int(text_dim)
        hidden_dim = int(hidden_dim)
        kernel_sizes = [int(k) for k in kernel_sizes]

        self.kernel_sizes = kernel_sizes
        self.pool_stride = pool_stride
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_channels = speech_dim

        for _ in range(num_layers):
            layer_convs = nn.ModuleList([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=k,
                    padding=k // 2
                )
                for k in kernel_sizes
            ])
            self.convs.append(layer_convs)
            self.norms.append(nn.LayerNorm(hidden_dim * len(kernel_sizes)))
            in_channels = hidden_dim * len(kernel_sizes)

        self.pool = nn.MaxPool1d(
            kernel_size=pool_stride,
            stride=pool_stride
        )

        self.proj = nn.Conv1d(in_channels, text_dim, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))
        self.act = nn.GELU()

        self.speech_dim = speech_dim
        self.text_dim = text_dim
        # store constructor args for config export
        self._config = {
            "speech_dim": int(speech_dim),
            "text_dim": int(text_dim),
            "hidden_dim": int(hidden_dim),
            "kernel_sizes": list(self.kernel_sizes),
            "num_layers": int(num_layers),
            "pooling_stride": int(self.pool_stride),
            "dropout": float(dropout),
        }

    def save_config(self, dir_path, filename="config.json"):
        os.makedirs(dir_path, exist_ok=True)
        cfg = {"adapter-type": "cnn", "kwargs": self._config}
        with open(os.path.join(dir_path, filename), "w") as f:
            json.dump(cfg, f, indent=4)

    def forward(self, speech_embs, attn_mask):
        """
        speech_embs: [B, T, speech_dim]
        attn_mask:   [B, T]
        """

        # [B, C, T]
        x = speech_embs.transpose(1, 2)
        mask = attn_mask

        for convs, norm in zip(self.convs, self.norms):
            # Multi-kernel conv
            feats = [self.act(conv(x)) for conv in convs]
            x = torch.cat(feats, dim=1)   # [B, C', T]
            x = self.dropout(x)

            # Pool in time
            x = self.pool(x)
            mask = self._pool_mask(mask)

            # LayerNorm over channels
            x = x.transpose(1, 2)          # [B, T', C']
            x = norm(x)
            x = x.transpose(1, 2)          # [B, C', T']

        x = self.proj(x)                  # [B, text_dim, T']
        x = x.transpose(1, 2)             # [B, T', text_dim]

        return x, mask

    def _pool_mask(self, mask):
        """
        Downsample attention mask consistently with temporal pooling.
        """
        if mask is None:
            return None

        mask = mask.unsqueeze(1).float()  # [B, 1, T]
        mask = F.max_pool1d(
            mask,
            kernel_size=self.pool_stride,
            stride=self.pool_stride,
        )
        return mask.squeeze(1).long()

class LstmAdapter(nn.Module):
    """
    LSTM Adapter.

    Maps per-frame speech embeddings to per-frame text-like embeddings using
    a bidirectional (optional) LSTM followed by a linear projection per timestep.

    Input:  Tensor [B, T, speech_dim]
    Output: Tensor [B, T, text_dim], mask unchanged
    """
    def __init__(self,
                 speech_dim: int = 384,
                 text_dim: int = 384,
                 hidden_dim: int = 256,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.0,
                 **kwargs):

        super().__init__()

        speech_dim = int(speech_dim)
        text_dim = int(text_dim)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        bidirectional = bool(bidirectional)
        dropout = float(dropout)

        self.lstm = nn.LSTM(
            input_size=speech_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, text_dim)

        self.speech_dim = speech_dim
        self.text_dim = text_dim

        # store constructor args for config export
        self._config = {
            "speech_dim": int(speech_dim),
            "text_dim": int(text_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "bidirectional": bool(bidirectional),
            "dropout": float(dropout),
        }

    def save_config(self, dir_path, filename="config.json"):
        os.makedirs(dir_path, exist_ok=True)
        cfg = {"adapter-type": "lstm", "kwargs": self._config}
        with open(os.path.join(dir_path, filename), "w") as f:
            json.dump(cfg, f, indent=4)

    def forward(self, speech_embs, attn_mask):
        # speech_embs: [B, T, D]
        x, _ = self.lstm(speech_embs)   # [B, T, out_dim]
        x = self.proj(x)                # [B, T, text_dim]
        return x, attn_mask


class AlignmentModel(nn.Module):
    """
    Alignment model operating on precomputed embeddings.

    Takes precomputed Whisper encoder hidden states and E5 text embeddings.
    Only the adapter and learnable temperature are trainable.

    Parameters
    ----------
    adapter : nn.Module
        A speech adapter module (e.g. ``CnnAdapter``, ``LstmAdapter``) that
        maps ``[B, T, speech_dim]`` → ``[B, T', text_dim]``.
    init_tau : float
        Initial temperature for contrastive loss (default: 0.07).
    gamma : float
        Weighting factor for the contrastive loss (default: 0.1).
    """

    def __init__(self,
                 adapter: nn.Module,
                 init_tau: float = 0.07,
                 gamma: float = 0.1,
                 **kwargs):
        super().__init__()

        self.adapter = adapter
        self.gamma = gamma
        self.log_tau = nn.Parameter(
            torch.log(torch.tensor(init_tau, dtype=torch.float32))
        )

    def forward(self, speech_hidden, speech_mask, text_emb):
        """
        Parameters
        ----------
        speech_hidden : Tensor [B, T, speech_dim]
            Precomputed Whisper encoder hidden states.
        speech_mask : Tensor [B, T]
            Attention mask (1 = real frame, 0 = padding).
        text_emb : Tensor [B, text_dim]
            Precomputed E5 text embeddings.

        Returns
        -------
        dict with keys: loss, logits, speech_emb, text_emb
        """
        aligned_speech, aligned_mask = self.adapter(speech_hidden, speech_mask)
        speech_emb = mean_pooling(aligned_speech, aligned_mask)  # [B, D]

        speech_emb = F.normalize(speech_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        tau = torch.exp(self.log_tau)
        logits = (speech_emb @ text_emb.T) / tau  # [B, B]

        labels = torch.arange(logits.size(0), device=logits.device)
        loss = self.gamma * F.cross_entropy(logits, labels) + (1-self.gamma) * F.cross_entropy(logits.T, labels)

        return {
            'loss': loss,
            'logits': logits,
            'speech_emb': speech_emb,
            'text_emb': text_emb,
        }
    
    def encode(self, speech_hidden, speech_mask):
        aligned_speech, aligned_mask = self.adapter(speech_hidden, speech_mask)
        speech_emb = mean_pooling(aligned_speech, aligned_mask)  # [B, D]
        return F.normalize(speech_emb, dim=-1)
