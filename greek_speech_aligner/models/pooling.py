#
#   pooling.py
#
# Mask-aware temporal pooling, used to collapse the aligner output
# sequence into a single speech vector (Eq. 5 in the paper).
#
# Author: Alexandros Tsingilis
#
import torch


def mean_pooling(hidden_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling with an attention mask.

    Computes a mask-aware mean over the temporal dimension of a sequence of
    hidden states. Only positions where the mask is 1 contribute to the mean.

    Parameters
    ----------
    hidden_state : torch.Tensor
        Tensor of shape ``[B, T, D]``.
    mask : torch.Tensor
        Mask of shape ``[B, T]`` with values in ``{0, 1}`` (bool or integer).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[B, D]`` with the mean-pooled representations.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    mask = mask.to(hidden_state.dtype)
    sum_hidden = (hidden_state * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / lengths
