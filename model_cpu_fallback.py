"""
Pure-PyTorch fallback for NonexpansiveDenoiser.

Replicates the exact same mathematical operations as the CUDA-accelerated
model.py, but uses native PyTorch ops (torch.roll) instead of custom C++/CUDA
kernels.  This allows the Gradio web demo to run on CPU or on GPU instances
where the custom kernel was not compiled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelWeightPredictor(nn.Module):
    def __init__(self, in_channels=6, mid_channels=16, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for _ in range(max(0, num_layers - 2)):
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=1))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        layers.append(nn.Conv2d(mid_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class NonexpansiveDenoiserFallback(nn.Module):
    """Drop-in replacement that uses torch.roll instead of custom CUDA kernels."""

    def __init__(self, radius=7, hidden_channels=16, num_layers=2):
        super().__init__()
        self.radius = radius
        self.window_size = 2 * radius + 1
        self.total_shifts = self.window_size ** 2

        # Feature extractor (identical architecture to model.py)
        fe_layers = []
        fe_layers.append(nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1))
        fe_layers.append(nn.InstanceNorm2d(hidden_channels))
        fe_layers.append(nn.LeakyReLU(0.2, inplace=True))

        for _ in range(max(0, num_layers - 2)):
            fe_layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            fe_layers.append(nn.InstanceNorm2d(hidden_channels))
            fe_layers.append(nn.LeakyReLU(0.2, inplace=True))

        fe_layers.append(nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1))
        self.feature_extractor = nn.Sequential(*fe_layers)

        # Weight predictor (identical architecture to model.py)
        self.weight_net = ParallelWeightPredictor(
            in_channels=6, mid_channels=hidden_channels, num_layers=num_layers
        )

        # Pre-compute shift offsets (same logic as model.py __init__)
        dy_grid, dx_grid = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )

        flat_idx_grid = (dy_grid + radius) * self.window_size + (dx_grid + radius)

        is_center = (dy_grid == 0) & (dx_grid == 0)
        is_forward = (dy_grid > 0) | ((dy_grid == 0) & (dx_grid > 0))
        valid_fwd = is_center | is_forward
        valid_inv = is_forward

        fwd_dy = dy_grid[valid_fwd].to(torch.int32)
        fwd_dx = dx_grid[valid_fwd].to(torch.int32)
        fwd_flat_indices = flat_idx_grid[valid_fwd].to(torch.long)

        inv_dy = -dy_grid[valid_inv]
        inv_dx = -dx_grid[valid_inv]
        inv_flat_indices = (
            (inv_dy + radius) * self.window_size + (inv_dx + radius)
        ).to(torch.long)

        inv_neg_dy = inv_dy.to(torch.int32)
        inv_neg_dx = inv_dx.to(torch.int32)
        inv_source_idx = torch.arange(1, len(fwd_dy), dtype=torch.long)

        self.num_fwd = len(fwd_dy)
        self.num_inv = len(inv_flat_indices)

        self.register_buffer("fwd_dy", fwd_dy)
        self.register_buffer("fwd_dx", fwd_dx)
        self.register_buffer("fwd_flat_indices", fwd_flat_indices)

        self.register_buffer("inv_flat_indices", inv_flat_indices)
        self.register_buffer("inv_source_idx", inv_source_idx)
        self.register_buffer("inv_neg_dy", inv_neg_dy)
        self.register_buffer("inv_neg_dx", inv_neg_dx)

    def _generate_shifts(self, tensor, dy_shifts, dx_shifts):
        """Pure-PyTorch replacement for agg_module.generate_shifts."""
        shifts = []
        for dy, dx in zip(dy_shifts.tolist(), dx_shifts.tolist()):
            shifted = torch.roll(tensor, shifts=(-dy, -dx), dims=(2, 3))
            shifts.append(shifted)
        return torch.stack(shifts, dim=0)  # (num_shifts, B, C, H, W)

    def forward(self, noisy_x):
        B, C, H, W = noisy_x.shape

        guide = self.feature_extractor(noisy_x)

        # Generate shifted guide features using torch.roll
        guide_shifted = self._generate_shifts(guide, self.fwd_dy, self.fwd_dx)

        center_rep = guide.unsqueeze(0).expand(self.num_fwd, -1, -1, -1, -1)
        center_rep = center_rep.reshape(self.num_fwd * B, C, H, W)

        guide_shifted_flat = guide_shifted.reshape(self.num_fwd * B, C, H, W)

        net_input = torch.cat([center_rep, guide_shifted_flat], dim=1)

        fwd_weights = self.weight_net(net_input)
        fwd_weights = fwd_weights.view(self.num_fwd, B, 1, H, W)

        all_weights = torch.empty(
            self.total_shifts, B, 1, H, W, device=noisy_x.device, dtype=noisy_x.dtype
        )

        # Assign forward weights
        all_weights[self.fwd_flat_indices] = fwd_weights

        # Scatter inverse weights (pure-PyTorch replacement for agg_module.scatter_inv_weights)
        for i in range(self.num_inv):
            src_idx = self.inv_source_idx[i].item()
            neg_dy = self.inv_neg_dy[i].item()
            neg_dx = self.inv_neg_dx[i].item()
            dst_shift = self.inv_flat_indices[i].item()

            src_weight = fwd_weights[src_idx]  # (B, 1, H, W)
            shifted_weight = torch.roll(src_weight, shifts=(-neg_dy, -neg_dx), dims=(2, 3))
            all_weights[dst_shift] = shifted_weight

        # Fused aggregation (pure-PyTorch replacement for agg_module.fused_aggregate)
        numerator = torch.zeros_like(noisy_x)
        denominator = torch.zeros_like(noisy_x)

        for s in range(self.total_shifts):
            dy = (s // self.window_size) - self.radius
            dx = (s % self.window_size) - self.radius

            shifted_noisy = torch.roll(noisy_x, shifts=(-dy, -dx), dims=(2, 3))
            w = all_weights[s]  # (B, 1, H, W)

            numerator += w * shifted_noisy
            denominator += w.expand_as(noisy_x)

        return numerator / (denominator + 1e-8)
