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

class LinearAligner(nn.Module):
    '''
    Linear Aligner.
    It's purpose is to transform a sequence of speech embedding vectors to pseudo-tokens
    for the text encoder.

    input: Tensor [B, L, speech_dim]
    out: Tensor [B, L, text_dim]
    '''
    def __init__(self,
                 speech_dim = 384,
                 text_dim = 384,
                 **kwargs):

        super().__init__()

        # Make sure inputs are the correct type
        speech_dim = int(speech_dim)
        text_dim = int(text_dim)

        self.linear_layer = nn.Linear(speech_dim, text_dim)
        self.speech_dim = speech_dim
        self.text_dim = text_dim

    def forward(self, speech_embs, attn_mask):
        return self.linear_layer(speech_embs), attn_mask

class MlpAligner(nn.Module):
    '''
    MLP Aligner.
    It's purpose is to transform a sequence of speech embedding vectors to pseudo-tokens
    for the text encoder.

    input: Tensor [B, L, speech_dim]
    out: Tensor [B, L, text_dim]
    '''
    def __init__(self,
                 speech_dim = 384,
                 text_dim = 384,
                 hidden_dims = (500,),
                 dropout = 0.1,
                 **kwargs):

        super().__init__()

        # Make sure dims are int
        speech_dim = int(speech_dim)
        tex_dim = int(text_dim)
        hidden_dims = [int(d) for d in hidden_dims]

        dim_pairs = pairwise([speech_dim, *hidden_dims, text_dim])
        self.dropout = nn.Dropout(dropout) # NOTE: Hard code!
        self.layers = nn.ModuleList([
            nn.Linear(d1, d2)
            for d1, d2 in self.layers
        ])

    def forward(self, speech_embs, attn_mask):
        h = speech_embs
        for layer in self.layers:
            h = layer(h)
            h = nn.GELU(h)
        return h, attn_mask


class CnnAligner(nn.Module):
    '''
    CNN Aligner with multi-kernel convolutions and temporal downsampling.

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
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.speech_dim = speech_dim
        self.text_dim = text_dim

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

class LstmAligner(nn.Module):
    '''
    LSTM Aligner with optional temporal downsampling.

    input:  speech_embs [B, T, speech_dim]
            attn_mask   [B, T]
    output: aligned_embs [B, T', text_dim]
            pooled_mask  [B, T']
    '''

    def __init__(self,
                 speech_dim: int = 384,
                 text_dim: int = 384,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 pool_stride: int = 1,
                 dropout: float = 0.1,
                 **kwargs):

        super().__init__()

        speech_dim = int(speech_dim)
        text_dim = int(text_dim)
        hidden_dim = int(hidden_dim)

        self.pool_stride = int(pool_stride)
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=speech_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.proj = nn.Linear(lstm_out_dim, text_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, speech_embs, attn_mask):
        """
        speech_embs: [B, T, speech_dim]
        attn_mask:   [B, T]
        """

        # Compute true lengths from mask
        lengths = attn_mask.sum(dim=1).cpu()

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            speech_embs,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.lstm(packed)

        # Unpack
        T = attn_mask.size(1)
        
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=T,
        )
        
        x = self.dropout(x)
        x = self.proj(x)  # [B, T, text_dim]

        # Optional temporal downsampling
        if self.pool_stride > 1:
            x = x[:, ::self.pool_stride, :]
            attn_mask = attn_mask[:, ::self.pool_stride]

        return x, attn_mask


class E5Aligner(nn.Module):
    """
    E5-based Aligner.

    Transforms speech embeddings into pseudo-token embeddings using a linear
    projection, then feeds them into a pretrained multilingual-e5-small
    transformer encoder.

    input:
        speech_embs : [B, T, speech_dim]
        attn_mask   : [B, T]

    output:
        aligned_embs : [B, T, text_dim]
        attn_mask    : [B, T]
    """

    def __init__(
        self,
        speech_dim: int = 384,
        text_dim: int = 384,
        text_encoder_id: str = "intfloat/multilingual-e5-small",
        fine_tune_layers: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.speech_dim = int(speech_dim)
        self.text_dim = int(text_dim)

        # Linear projection: speech → E5 token space
        self.proj = nn.Linear(self.speech_dim, self.text_dim)

        # Load E5 encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_id)

        # -------------------------------------------------
        # Freeze everything
        # -------------------------------------------------
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # Unfreeze first N transformer layers
        # -------------------------------------------------
        encoder_layers = self.text_encoder.encoder.layer
        for layer in encoder_layers[:fine_tune_layers]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, speech_embs, attn_mask):
        """
        speech_embs: [B, T, speech_dim]
        attn_mask:   [B, T]
        """

        # Project speech frames to pseudo-token embeddings
        token_embs = self.proj(speech_embs)  # [B, T, text_dim]

        # Feed directly as inputs_embeds
        outputs = self.text_encoder(
            inputs_embeds=token_embs,
            attention_mask=attn_mask,
            return_dict=True,
        )

        # Contextualized pseudo-tokens
        hidden_states = outputs.last_hidden_state  # [B, T, text_dim]

        return hidden_states, attn_mask

# 
# The Final Speech Encoder
# 

class SpeechEncoder(nn.Module):

    name2model = {
        'linear': LinearAligner,
        'mlp' : MlpAligner,
        'cnn' : CnnAligner,
        'lstm': LstmAligner,
        'e5'  : E5Aligner
    }

    def __init__(self,
                 aligner_type: str = LinearAligner,
                 **kwargs):

        super().__init__()

        Aligner = self.name2model[aligner_type]
        self.aligner = Aligner(**kwargs)

    def forward(self, speech_embs, attn_mask):
        aligned_speech_embs, attn_mask = self.aligner(speech_embs, attn_mask)
        return mean_pooling(aligned_speech_embs, attn_mask)
