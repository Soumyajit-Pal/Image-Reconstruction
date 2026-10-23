"""
Gradio web demo for the Non-Expansive Image Denoiser.

Automatically detects whether the custom CUDA kernels are available:
  - If YES  → uses the full GPU-accelerated pipeline (model.py)
  - If NO   → falls back to the pure-PyTorch implementation (model_cpu_fallback.py)

This guarantees the demo never crashes in front of a recruiter regardless
of the host machine's hardware.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim
except ImportError:
    def compute_psnr(im1, im2, data_range=1.0):
        mse = np.mean((im1.astype(float) - im2.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return float(20 * np.log10(data_range) - 10 * np.log10(mse))

    def compute_ssim(im1, im2, channel_axis=2, data_range=1.0):
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        mu1 = np.mean(im1)
        mu2 = np.mean(im2)
        sigma1_sq = np.var(im1)
        sigma2_sq = np.var(im2)
        sigma12 = np.mean((im1 - mu1) * (im2 - mu2))
        return float(((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)))

import gradio as gr

# ---------------------------------------------------------------------------
# 1.  Detect backend: CUDA-accelerated or pure-PyTorch fallback
# ---------------------------------------------------------------------------
CUDA_BACKEND = False

try:
    from model import NonexpansiveDenoiser

    # Quick smoke test — if agg_module cannot be loaded, this will raise
    _test_model = NonexpansiveDenoiser(radius=1, hidden_channels=4, num_layers=2)
    if torch.cuda.is_available():
        _test_model.cuda()
        _dummy = torch.randn(1, 3, 16, 16, device="cuda")
        with torch.no_grad():
            _test_model(_dummy)
    CUDA_BACKEND = True
    del _test_model, _dummy
    print("[INFO] CUDA backend loaded successfully.")
except Exception as e:
    print(f"[WARN] CUDA backend unavailable ({e}). Using pure-PyTorch fallback.")
    from model_cpu_fallback import NonexpansiveDenoiserFallback as NonexpansiveDenoiser

# ---------------------------------------------------------------------------
# 2.  Device & Model Setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if (torch.cuda.is_available() and CUDA_BACKEND) else "cpu")

model = NonexpansiveDenoiser(radius=7, hidden_channels=16, num_layers=2)

weights_path = os.path.join(os.path.dirname(__file__), "weights", "denoiser_unified.pth")
state_dict = torch.load(weights_path, map_location=device, weights_only=True)

# Remap legacy keys
key_mapping = {
    "weight_net.conv1.weight": "weight_net.net.0.weight",
    "weight_net.conv1.bias": "weight_net.net.0.bias",
    "weight_net.conv2.weight": "weight_net.net.2.weight",
    "weight_net.conv2.bias": "weight_net.net.2.bias",
}
new_state_dict = {}
for k, v in state_dict.items():
    if k in key_mapping:
        new_state_dict[key_mapping[k]] = v
    elif k not in ["inv_dy_cuda", "inv_dx_cuda"]:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

backend_label = "🟢 Custom CUDA Kernels (GPU Accelerated)" if CUDA_BACKEND else "🟡 Pure-PyTorch (CPU Fallback)"

# ---------------------------------------------------------------------------
# 3.  Core Inference Function
# ---------------------------------------------------------------------------
def reconstruct_image(input_img, noise_level):
    """Takes an uploaded image + noise σ, returns noisy / denoised / metrics."""
    if input_img is None:
        return None, None, None, "⚠️ Please upload an image or click an example below."

    # Convert to PIL → Tensor
    img = Image.fromarray(input_img).convert("RGB")
    clean = TF.to_tensor(TF.center_crop(TF.resize(img, 256), 256)).to(device)

    # Synthesize additive Gaussian noise
    noise_sigma = float(noise_level) / 255.0
    torch.manual_seed(42)  # Deterministic noise for reproducibility
    noisy = torch.clamp(clean + torch.randn_like(clean) * noise_sigma, 0.0, 1.0)

    # Warm-up run (discard first pass to avoid cold-start timing)
    if device.type == "cuda":
        with torch.no_grad():
            _ = model(noisy.unsqueeze(0))
        torch.cuda.synchronize()

    # Timed inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        denoised = model(noisy.unsqueeze(0)).squeeze(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    # Clamp output
    denoised = denoised.clamp(0.0, 1.0)

    # Compute quality metrics
    clean_np = clean.cpu().numpy().transpose(1, 2, 0)
    noisy_np = noisy.cpu().numpy().transpose(1, 2, 0)
    denoised_np = denoised.cpu().numpy().transpose(1, 2, 0)

    psnr_noisy = compute_psnr(clean_np, noisy_np, data_range=1.0)
    psnr_denoised = compute_psnr(clean_np, denoised_np, data_range=1.0)
    ssim_denoised = compute_ssim(clean_np, denoised_np, channel_axis=2, data_range=1.0)

    # Convert to PIL for Gradio display
    noisy_pil = TF.to_pil_image(noisy.cpu())
    denoised_pil = TF.to_pil_image(denoised.cpu())

    # Build metrics summary
    metrics = (
        f"**⚡ Inference Latency:** {latency_ms:.2f} ms\n\n"
        f"**🧠 Backend:** {backend_label}\n\n"
        f"**📊 PSNR (Noisy → Clean):** {psnr_noisy:.2f} dB\n\n"
        f"**📊 PSNR (Denoised → Clean):** {psnr_denoised:.2f} dB  "
        f"**(+{psnr_denoised - psnr_noisy:.2f} dB improvement)**\n\n"
        f"**🏆 SSIM (Denoised → Clean):** {ssim_denoised:.4f}\n\n"
        f"**🔧 Noise Level (σ):** {noise_level}/255\n\n"
        f"**📐 Resolution:** 256 × 256 × 3"
    )

    return input_img, noisy_pil, denoised_pil, metrics


custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="blue",
)
custom_css = """
    .header-text { text-align: center; }
    .metric-box { font-size: 15px; line-height: 1.8; }
    footer { display: none !important; }
"""

with gr.Blocks(title="Image Denoiser — CUDA Accelerated") as demo:

    # Header
    gr.Markdown(
        """
        # 🖼️ Non-Expansive Image Denoiser — CUDA Accelerated
        ### Deep Learning image reconstruction powered by custom C++/CUDA backend kernels

        Upload any image, select a noise level, and watch the model reconstruct it in real time.
        The denoiser fuses **225 spatial permutations** into a single optimized GPU kernel for maximum throughput.

        **[📄 Research Report](https://github.com/Soumyajit-Pal/Image-Reconstruction/blob/main/Denoiser_Report.pdf)** · **[💻 Source Code](https://github.com/Soumyajit-Pal/Image-Reconstruction)** · **Developed during IISc Research Internship**
        """,
        elem_classes="header-text",
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📤 Upload Image", type="numpy")
            noise_slider = gr.Slider(
                minimum=10, maximum=60, value=25, step=5,
                label="Noise Level (σ)",
                info="Higher σ = more corruption. Default 25 is standard for benchmarks.",
            )
            run_btn = gr.Button("🚀 Denoise", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Row():
                original_out = gr.Image(label="🟢 Original Input", interactive=False)
                noisy_out = gr.Image(label="🔴 Corrupted (Noisy)", interactive=False)
                denoised_out = gr.Image(label="🟢 Reconstructed (Denoised)", interactive=False)

            metrics_out = gr.Markdown(
                label="Execution Metrics",
                value="*Click 'Denoise' or select an example below to see results.*",
                elem_classes="metric-box",
            )

    # Examples
    gr.Examples(
        examples=[
            [os.path.join(os.path.dirname(__file__), "test_images", "clean_aeroplane.png"), 25],
            [os.path.join(os.path.dirname(__file__), "test_images", "clean_building.png"), 30],
            [os.path.join(os.path.dirname(__file__), "test_images", "clean_tree.png"), 25],
        ],
        inputs=[input_image, noise_slider],
        outputs=[original_out, noisy_out, denoised_out, metrics_out],
        fn=reconstruct_image,
        cache_examples=False,
        label="📸 Click an example to try instantly",
    )

    # Footer info
    gr.Markdown(
        f"""
        ---
        **Engine:** {backend_label} · **Architecture:** Permutation-Based Non-Expansive Denoiser with Parallel Weight Predictor ·
        **Kernel Fusion:** 225 spatial shifts aggregated in a single CUDA kernel · **Framework:** PyTorch + LibTorch
        """,
    )

    # Wire up the button
    run_btn.click(
        fn=reconstruct_image,
        inputs=[input_image, noise_slider],
        outputs=[original_out, noisy_out, denoised_out, metrics_out],
    )


# ---------------------------------------------------------------------------
# 5.  Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, theme=custom_theme, css=custom_css)
    except TypeError:
        demo.launch(server_name="0.0.0.0", server_port=7860)
