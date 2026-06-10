from __future__ import annotations

import torch


def occupancy_from_logits(
    logits: torch.Tensor, *, threshold: float = 0.5
) -> torch.Tensor:
    return torch.sigmoid(logits) >= threshold


def voxel_iou(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Mean intersection-over-union across a batch of binary occupancy grids.

    An example where both prediction and target are empty counts as IoU 1.0.
    """
    pred = predicted.bool().flatten(start_dim=1)
    targ = target.bool().flatten(start_dim=1)
    intersection = (pred & targ).sum(dim=1).float()
    union = (pred | targ).sum(dim=1).float()
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return float(iou.mean())
