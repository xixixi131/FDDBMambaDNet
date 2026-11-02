import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """Edge preservation loss using Laplacian operator."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))

    def _laplacian(self, x):
        C = x.shape[1]
        kernel = self.kernel.expand(C, -1, -1, -1)
        return F.conv2d(x, kernel, padding=1, groups=C)

    def forward(self, pred, gt):
        return (self._laplacian(pred) - self._laplacian(gt)).abs().mean()
