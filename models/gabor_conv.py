import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborConv2d(nn.Module):
    """Convolution with learnable Gabor-modulated kernels."""

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 n_orientations=4, n_scales=3, stride=1, padding=None, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        n_gabor = n_orientations * n_scales
        self.n_gabor = n_gabor

        theta_init = torch.linspace(0, math.pi, n_orientations + 1)[:-1]
        theta_init = theta_init.repeat(n_scales)
        self.theta = nn.Parameter(theta_init)

        sigma_init = torch.ones(n_gabor) * (kernel_size / 4.0)
        for s in range(n_scales):
            sigma_init[s * n_orientations:(s + 1) * n_orientations] *= (1.0 + 0.5 * s)
        self.sigma = nn.Parameter(sigma_init)

        lambd_init = torch.ones(n_gabor) * (kernel_size / 2.0)
        for s in range(n_scales):
            lambd_init[s * n_orientations:(s + 1) * n_orientations] *= (1.0 + 0.3 * s)
        self.lambd = nn.Parameter(lambd_init)

        self.gamma = nn.Parameter(torch.ones(n_gabor) * 0.5)
        self.psi = nn.Parameter(torch.zeros(n_gabor))

        self.envelope = nn.Parameter(
            torch.randn(out_channels, in_channels // 1, kernel_size, kernel_size) * 0.02
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

    def _make_gabor_bank(self):
        filters = []
        with torch.amp.autocast(device_type='cuda', enabled=False):
            for i in range(self.n_gabor):
                theta = self.theta[i].float()
                sigma = self.sigma[i].float().clamp(min=0.5)
                lambd = self.lambd[i].float().clamp(min=1.0)
                gamma = self.gamma[i].float().clamp(min=0.1, max=2.0)
                psi = self.psi[i].float()

                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                x_prime = self.grid_x * cos_t + self.grid_y * sin_t
                y_prime = -self.grid_x * sin_t + self.grid_y * cos_t

                gaussian = torch.exp(-0.5 * (x_prime ** 2 + (gamma ** 2) * y_prime ** 2) / (sigma ** 2))
                sinusoid = torch.cos(2 * math.pi * x_prime / lambd + psi)
                gabor = gaussian * sinusoid
                filters.append(gabor)

            bank = torch.stack(filters, dim=0)
        return bank

    def forward(self, x):
        gabor_bank = self._make_gabor_bank().to(dtype=x.dtype)
        n_gabor = self.n_gabor
        out_c, in_c, kh, kw = self.envelope.shape

        gabor_idx = torch.arange(out_c, device=x.device) % n_gabor
        selected = gabor_bank[gabor_idx]
        selected = selected.unsqueeze(1).expand(-1, in_c, -1, -1)

        weight = self.envelope * selected
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)
