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
        return self.linear_layer(speech_embs)

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
                 **kwargs):

        super().__init__()

        # Make sure dims are int
        speech_dim = int(speech_dim)
        tex_dim = int(text_dim)
        hidden_dims = [int(d) for d in hidden_dims]

        dim_pairs = pairwise([speech_dim, *hidden_dims, text_dim])
        dropout = nn.Dropout(0.1) # NOTE: Hard code!
        self.layers = nn.ModuleList([
            nn.Linear(d1, d2)
            for d1, d2 in self.layers
        ])

    def forward(self, speech_embs, attn_mask):
        h = speech_embs
        for layer in self.layers:
            h = layer(h)
            h = nn.GELU(h)
        return h


class SpeechEncoder(nn.Module):

    name2model = {
        'linear': LinearAligner,
        'mlp' : MlpAligner,
    }

    def __init__(self,
                 aligner_type: str = LinearAligner,
                 **kwargs):

        super().__init__()

        Aligner = self.name2model[aligner_type]
        self.aligner = Aligner(**kwargs)

    def forward(self, speech_embs, attn_mask):
        aligned_speech_embs = self.aligner(speech_embs, attn_mask)
        return mean_pooling(aligned_speech_embs, attn_mask)
