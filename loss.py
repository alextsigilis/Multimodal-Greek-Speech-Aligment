#   loss.py
#
# Contrastive Loss function.
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import torch
import torch.nn as nn, torch.nn.functional as F
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss for aligning paired embeddings.

    This module computes:
        - text → speech cross-entropy
        - speech → text cross-entropy
        - final loss = average of the two

    Parameters
    ----------
    init_tau : float, default=0.07
        Initial value for the temperature τ. We store log(τ) internally
        and optimize log_tau as a learnable parameter for stability.
    normalize_inputs : bool, default=True
        If True, assume embeddings are already L2-normalized.
        If False, normalize them inside the similarity call.
    """

    def __init__(self, init_tau: float = 0.07, normalize_inputs: bool = True):
        super().__init__()
        self.normalize_inputs = normalize_inputs

        # We optimize log_tau for numerical stability
        log_tau = torch.log(torch.tensor(init_tau, dtype=torch.float32))
        self.log_tau = nn.Parameter(log_tau)

    def _similarity(self, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise similarity matrix [B, B] using dot product.
        Optionally normalizes y and p first.
        """
        if not self.normalize_inputs:
            y = F.normalize(y, dim=-1)
            p = F.normalize(p, dim=-1)

        return y @ p.T

    def forward(self, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss.

        Parameters
        ----------
        y : Tensor of shape [B, D]
            Text embeddings.
        p : Tensor of shape [B, D]
            Speech embeddings.

        Returns
        -------
        loss : scalar Tensor
        """
        tau = torch.exp(self.log_tau)
        logits = self._similarity(y, p) / tau

        labels = torch.arange(y.size(0), device=y.device)

        loss_y_to_p = F.cross_entropy(logits, labels)
        loss_p_to_y = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_y_to_p + loss_p_to_y)

