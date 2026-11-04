"""Signed coordinate changes on the model's normalized gesture timeline."""
import torch
from torch import nn


class MotionDeltaFeatures(nn.Module):
    """Expose movement without treating missing observations as displacement.

    Input/output: (batch, time, features). Changes are scaled to gesture progress,
    not seconds, because complete recordings are resampled to a fixed timeline.
    Visibility bits never become movement features.
    """
    def __init__(self, input_size: int):
        super().__init__()
        if input_size not in (63, 126, 150, 180):
            raise ValueError('Motion deltas require a supported hand/body contract')
        self.input_size = input_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta = (features[:, 1:]-features[:, :-1]) * (features.shape[1]-1)
        parts = []
        for start in range(0, min(self.input_size, 126), 63):
            hand = features[:, :, start:start+63]
            present = torch.any(hand != 0, dim=-1)
            valid = (present[:, 1:] & present[:, :-1]).unsqueeze(-1)
            parts.append(delta[:, :, start:start+63] * valid)
        if self.input_size > 126:
            points = features[:, :, 126:].reshape(features.shape[0], features.shape[1], -1, 3)
            valid = ((points[:, 1:, :, 2] == 1) & (points[:, :-1, :, 2] == 1)).unsqueeze(-1)
            xy = (points[:, 1:, :, :2]-points[:, :-1, :, :2]) * (features.shape[1]-1) * valid
            context = torch.cat((xy, torch.zeros_like(xy[..., :1])), dim=-1)
            parts.append(context.flatten(start_dim=2))
        changes = torch.cat(parts, dim=-1)
        return torch.cat((torch.zeros_like(features[:, :1]), changes), dim=1)
