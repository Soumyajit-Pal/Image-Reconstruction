import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

try:
    import agg_module
except ImportError:
    from torch.utils.cpp_extension import load

    agg_module = load(
        name="agg_module",
        sources=[
            os.path.join(os.path.dirname(__file__), "csrc", "binding.cpp"),
            os.path.join(os.path.dirname(__file__), "csrc", "aggregation.cu"),
        ],
        extra_cflags=["/Zc:preprocessor"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=100, patch_size=128, noise_sigma=25):
        super().__init__()
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.noise_sigma = noise_sigma / 255.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean_tensor = torch.rand(3, self.patch_size, self.patch_size)
        noise = torch.randn_like(clean_tensor) * self.noise_sigma
        noisy_tensor = clean_tensor + noise
        noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)
        return noisy_tensor, clean_tensor


def get_dataloader(batch_size=4, patch_size=128, noise_sigma=25):
    dataset = SyntheticDataset(
        num_samples=100, patch_size=patch_size, noise_sigma=noise_sigma
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


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


class NonexpansiveDenoiser(nn.Module):
    def __init__(self, radius=7, hidden_channels=16, num_layers=2):
        super().__init__()
        self.radius = radius
        self.window_size = 2 * radius + 1
        self.total_shifts = self.window_size**2

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

        self.weight_net = ParallelWeightPredictor(
            in_channels=6, mid_channels=hidden_channels, num_layers=num_layers
        )

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


    def forward(self, noisy_x):
        B, C, H, W = noisy_x.shape

        guide = self.feature_extractor(noisy_x)

        guide_shifted = agg_module.generate_shifts(guide, self.fwd_dy, self.fwd_dx)

        center_rep = guide.unsqueeze(0).expand(self.num_fwd, -1, -1, -1, -1)
        center_rep = center_rep.reshape(self.num_fwd * B, C, H, W)

        guide_shifted_flat = guide_shifted.reshape(self.num_fwd * B, C, H, W)

        net_input = torch.cat([center_rep, guide_shifted_flat], dim=1)

        fwd_weights = self.weight_net(net_input)
        fwd_weights = fwd_weights.view(self.num_fwd, B, 1, H, W)

        all_weights = torch.empty(
            self.total_shifts, B, 1, H, W, device=noisy_x.device, dtype=noisy_x.dtype
        )

        all_weights[self.fwd_flat_indices] = fwd_weights

        # Single CUDA kernel: scatter all inverse-shifted weights in parallel
        agg_module.scatter_inv_weights(
            fwd_weights,
            all_weights,
            self.inv_flat_indices,
            self.inv_source_idx,
            self.inv_neg_dy,
            self.inv_neg_dx,
        )

        # Single CUDA kernel: fused aggregation (computes shifts on-the-fly, no
        # intermediate all_noisy_shifted tensor needed)
        denoised_output = agg_module.fused_aggregate(noisy_x, all_weights, self.radius)
        return denoised_output


def train_model(epochs=1, batch_size=4, learning_rate=1e-4, radius=7, hidden_channels=16, num_layers=2):
    print("🏟️ Starting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(batch_size=batch_size, patch_size=128)

    model = NonexpansiveDenoiser(
        radius=radius, hidden_channels=hidden_channels, num_layers=num_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (noisy_batch, clean_batch) in enumerate(dataloader):
            noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)

            optimizer.zero_grad()
            predicted_clean = model(noisy_batch)
            loss = criterion(predicted_clean, clean_batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"📅 Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), f"weights/denoiser_unified.pth")
    return model


def evaluate_model(model):
    print("🏟️ Starting evaluation...")
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataloader = get_dataloader(batch_size=4, patch_size=128)
    noisy_img, clean_img = next(iter(dataloader))
    noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)

    with torch.no_grad():
        denoised_img = model(noisy_img)

    os.makedirs("assets", exist_ok=True)
    save_image(noisy_img, "assets/corrupted_input.png")
    save_image(denoised_img, "assets/denoised_output.png")
    save_image(clean_img, "assets/perfect_target.png")

    clean_np = clean_img[0].cpu().numpy().transpose(1, 2, 0)
    denoised_np = denoised_img[0].cpu().numpy().transpose(1, 2, 0)

    psnr_score = compute_psnr(clean_np, denoised_np, data_range=1.0)
    ssim_score = compute_ssim(clean_np, denoised_np, channel_axis=2, data_range=1.0)

    print(f"🏆 PSNR Score : {psnr_score:.2f} dB")
    print(f"🏆 SSIM Score : {ssim_score:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate toy_nn model.")
    parser.add_argument("--radius", type=int, default=7, help="Radius of the window.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--hidden_channels", type=int, default=16, help="Number of hidden channels in layers.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of conv layers in sub-networks.")
    args = parser.parse_args()

    model = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        radius=args.radius,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    )
    evaluate_model(model)
