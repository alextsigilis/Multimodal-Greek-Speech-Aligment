#
#   adapters.py
#
# Trainable aligners A_theta that map frozen Whisper encoder states
# [B, T, speech_dim] to a sequence in the E5 text space [B, T', text_dim]
# (Section 3.3 of the paper).
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAdapter(nn.Module):
    """
    Common base class for all aligners.

    Subclasses set ``adapter_type`` (the key used in ``config.json``) and
    populate ``self._config`` with the constructor kwargs needed to rebuild
    the module. ``forward(speech_embs, attn_mask)`` must return
    ``(aligned [B, T', text_dim], mask [B, T'])``.
    """

    adapter_type: str = None

    @property
    def config(self) -> dict:
        return {"adapter-type": self.adapter_type, "kwargs": dict(self._config)}

    def save_config(self, dir_path, filename="config.json"):
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, filename), "w") as f:
            json.dump(self.config, f, indent=4)


class MlpAdapter(BaseAdapter):
    """
    Frame-wise MLP aligner.

    Applies the same feed-forward network independently to every time step.
    ``num_layers`` counts linear layers, i.e. ``num_layers - 1`` hidden
    blocks (Linear + GELU + Dropout) followed by a projection to ``text_dim``.

    Input:  ``[B, T, speech_dim]``
    Output: ``[B, T, text_dim]``, mask unchanged
    """

    adapter_type = "mlp"

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
        dropout = float(dropout)

        layers = []
        in_dim = speech_dim
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, text_dim))

        self.net = nn.Sequential(*layers)
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self._config = {
            "speech_dim": speech_dim,
            "text_dim": text_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    def forward(self, speech_embs, attn_mask):
        # nn.Linear acts on the last dimension, so no reshape is needed.
        return self.net(speech_embs), attn_mask


class CnnAdapter(BaseAdapter):
    """
    Temporal CNN aligner with parallel multi-kernel convolutions.

    Each stage applies one Conv1d per kernel size in parallel, concatenates
    the outputs on the channel axis, max-pools in time by ``pool_stride`` and
    applies LayerNorm. A final 1x1 convolution projects to ``text_dim``.

    Input:  ``[B, T, speech_dim]``
    Output: ``[B, T', text_dim]`` and the correspondingly pooled mask ``[B, T']``
    """

    adapter_type = "cnn"

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

        # Configs written by older versions stored the stride under
        # "pooling_stride", which was silently swallowed by **kwargs.
        if "pooling_stride" in kwargs:
            pool_stride = kwargs.pop("pooling_stride")

        speech_dim = int(speech_dim)
        text_dim = int(text_dim)
        hidden_dim = int(hidden_dim)
        kernel_sizes = [int(k) for k in kernel_sizes]
        num_layers = int(num_layers)
        pool_stride = int(pool_stride)
        dropout = float(dropout)

        for k in kernel_sizes:
            if k % 2 == 0:
                # padding=k//2 only preserves the length for odd kernels; even
                # kernels would make the branches disagree in length.
                raise ValueError(f"CnnAdapter kernel sizes must be odd, got {k}")

        self.kernel_sizes = kernel_sizes
        self.pool_stride = pool_stride
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_channels = speech_dim
        for _ in range(num_layers):
            self.convs.append(nn.ModuleList([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]))
            self.norms.append(nn.LayerNorm(hidden_dim * len(kernel_sizes)))
            in_channels = hidden_dim * len(kernel_sizes)

        self.pool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.proj = nn.Conv1d(in_channels, text_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self._config = {
            "speech_dim": speech_dim,
            "text_dim": text_dim,
            "hidden_dim": hidden_dim,
            "kernel_sizes": list(kernel_sizes),
            "num_layers": num_layers,
            "pool_stride": pool_stride,
            "dropout": dropout,
        }

    def forward(self, speech_embs, attn_mask):
        x = speech_embs.transpose(1, 2)  # [B, C, T]
        mask = attn_mask

        for convs, norm in zip(self.convs, self.norms):
            x = torch.cat([self.act(conv(x)) for conv in convs], dim=1)  # [B, C', T]
            x = self.dropout(x)
            x = self.pool(x)
            mask = self._pool_mask(mask)
            x = norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm over channels

        x = self.proj(x)          # [B, text_dim, T']
        return x.transpose(1, 2), mask

    def _pool_mask(self, mask):
        """Downsample the mask consistently with the temporal max-pooling."""
        if mask is None:
            return None
        mask = F.max_pool1d(mask.unsqueeze(1).float(),
                            kernel_size=self.pool_stride,
                            stride=self.pool_stride)
        return mask.squeeze(1).long()


class LstmAdapter(BaseAdapter):
    """
    (Bi)LSTM aligner followed by a per-timestep projection to ``text_dim``.

    Input:  ``[B, T, speech_dim]``
    Output: ``[B, T, text_dim]``, mask unchanged
    """

    adapter_type = "lstm"

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
        self.proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), text_dim)

        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self._config = {
            "speech_dim": speech_dim,
            "text_dim": text_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
        }

    def forward(self, speech_embs, attn_mask):
        x, _ = self.lstm(speech_embs)   # [B, T, out_dim]
        return self.proj(x), attn_mask


class E5MLPAdapter(BaseAdapter):
    """
    Experimental aligner: a linear projection (+ dropout) whose output is fed
    as pseudo token embeddings into a frozen multilingual-e5-small encoder.
    Returns the per-token last hidden states so that ``AlignmentModel``
    applies the usual mask-aware mean pooling.
    """

    adapter_type = "e5mlp"

    def __init__(self,
                 speech_dim: int = 384,
                 text_dim: int = 384,
                 dropout: float = 0.1,
                 e5_model_name: str = "intfloat/multilingual-e5-small",
                 device: str = None,
                 **kwargs):
        super().__init__()
        from sentence_transformers import SentenceTransformer  # heavy, lazy import

        self.speech_dim = int(speech_dim)
        self.text_dim = int(text_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.linear = nn.Linear(self.speech_dim, self.text_dim)
        # Load on CPU by default; the module is moved with .to() like any other.
        self.e5 = SentenceTransformer(e5_model_name, device=device or "cpu")
        for param in self.e5.parameters():
            param.requires_grad = False
        self._config = {
            "speech_dim": self.speech_dim,
            "text_dim": self.text_dim,
            "dropout": float(dropout),
            "e5_model_name": e5_model_name,
        }

    def forward(self, speech_embs, attn_mask):
        x = self.dropout(self.linear(speech_embs))  # [B, T, text_dim]
        transformer = self.e5._first_module().auto_model
        outputs = transformer(inputs_embeds=x, attention_mask=attn_mask.long())
        return outputs.last_hidden_state, attn_mask


ADAPTERS = {
    cls.adapter_type: cls
    for cls in (CnnAdapter, MlpAdapter, LstmAdapter, E5MLPAdapter)
}


def get_adapter_class(adapter_type):
    """
    Return the adapter class registered under ``adapter_type``
    (case-insensitive). Supported: ``cnn``, ``mlp``, ``lstm``, ``e5mlp``.
    """
    key = str(adapter_type).strip().lower()
    try:
        return ADAPTERS[key]
    except KeyError:
        raise ValueError(
            f"Unknown adapter_type: {adapter_type!r}. Expected one of {sorted(ADAPTERS)}"
        ) from None
