#
#   alignment.py
#
# Speech–text alignment model: aligner + mean pooling + learnable temperature,
# trained with the bidirectional contrastive objective.
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from greek_speech_aligner.losses import bidirectional_contrastive_loss
from greek_speech_aligner.models.pooling import mean_pooling


class AlignmentModel(nn.Module):
    """
    Alignment model operating on precomputed embeddings.

    Takes precomputed (temporally downsampled) Whisper encoder hidden states
    and E5 text embeddings. Only the adapter and the temperature are trainable.

    Parameters
    ----------
    adapter : nn.Module
        Aligner mapping ``[B, T, speech_dim]`` → ``[B, T', text_dim]``.
    init_tau : float
        Initial contrastive temperature (default: 0.07).
    gamma : float
        Weight of the speech→text loss term (default: 0.1, as in the paper).
    """

    def __init__(self, adapter: nn.Module, init_tau: float = 0.07, gamma: float = 0.1, **kwargs):
        super().__init__()
        self.adapter = adapter
        self.gamma = float(gamma)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau, dtype=torch.float32)))

    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def encode(self, speech_hidden, speech_mask):
        """Return L2-normalised speech embeddings ``[B, text_dim]``."""
        aligned_speech, aligned_mask = self.adapter(speech_hidden, speech_mask)
        speech_emb = mean_pooling(aligned_speech, aligned_mask)
        return F.normalize(speech_emb, dim=-1)

    def forward(self, speech_hidden, speech_mask, text_emb):
        """
        Parameters
        ----------
        speech_hidden : Tensor [B, T, speech_dim]
        speech_mask : Tensor [B, T]   (1 = real frame, 0 = padding)
        text_emb : Tensor [B, text_dim]

        Returns
        -------
        dict with keys ``loss``, ``logits``, ``speech_emb``, ``text_emb``.
        """
        speech_emb = self.encode(speech_hidden, speech_mask)
        text_emb = F.normalize(text_emb, dim=-1)

        # Rows = speech, columns = text.
        logits = (speech_emb @ text_emb.T) / self.tau
        loss = bidirectional_contrastive_loss(logits, self.gamma)

        return {
            "loss": loss,
            "logits": logits,
            "speech_emb": speech_emb,
            "text_emb": text_emb,
        }
