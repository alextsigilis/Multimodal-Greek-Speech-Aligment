#
#   losses.py
#
# Bidirectional contrastive (InfoNCE) objective with in-batch negatives
# (Section 3.4 of the paper, Eqs. 6-10).
#
# Author: Alexandros Tsingilis
# Date: 28 Nov 2025
#
import torch
import torch.nn as nn
import torch.nn.functional as F


def bidirectional_contrastive_loss(logits: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    """
    Weighted bidirectional InfoNCE loss.

    Parameters
    ----------
    logits : Tensor [B, B]
        Scaled similarities ``l_ij = s_ij / tau`` with **speech on the rows**
        and **text on the columns**; the diagonal holds the positive pairs.
    gamma : float
        Weight of the speech→text term. The paper uses ``gamma = 0.1`` so that
        the text→speech (retrieval) direction dominates:
        ``L = gamma * L_s→t + (1 - gamma) * L_t→s``.
    """
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_s2t = F.cross_entropy(logits, labels)    # softmax over texts, per speech row
    loss_t2s = F.cross_entropy(logits.T, labels)  # softmax over speech, per text row
    return gamma * loss_s2t + (1.0 - gamma) * loss_t2s


class ContrastiveLoss(nn.Module):
    """
    Stand-alone contrastive loss module with its own learnable temperature.

    ``AlignmentModel`` computes the loss itself (so that ``log_tau`` lives on
    the model and in its checkpoints); this module is kept for experiments that
    need the loss detached from the model.

    Parameters
    ----------
    init_tau : float
        Initial temperature; ``log(tau)`` is optimised for stability.
    gamma : float
        Weight of the speech→text term (0.5 = symmetric InfoNCE).
    normalize_inputs : bool
        If True, inputs are assumed to be already L2-normalised. If False, they
        are normalised before computing the similarity.
    """

    def __init__(self, init_tau: float = 0.07, gamma: float = 0.5, normalize_inputs: bool = True):
        super().__init__()
        self.gamma = gamma
        self.normalize_inputs = normalize_inputs
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau, dtype=torch.float32)))

    def forward(self, text_emb: torch.Tensor, speech_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text_emb : Tensor [B, D]
        speech_emb : Tensor [B, D]
        """
        if not self.normalize_inputs:
            text_emb = F.normalize(text_emb, dim=-1)
            speech_emb = F.normalize(speech_emb, dim=-1)
        logits = (speech_emb @ text_emb.T) / torch.exp(self.log_tau)
        return bidirectional_contrastive_loss(logits, self.gamma)
