import time
import torch

from model import NonexpansiveDenoiser
from model_serial import NonexpansiveDenoiserSerial


def run_benchmark(radius=7, batch_size=1, patch_size=128, iterations=1, hidden_channels=16, num_layers=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}...")

    # --- Load parallel model ---
    print("Initializing Parallel model (CUDA shifts + ICC)...")
    model_parallel = NonexpansiveDenoiser(
        radius=radius, hidden_channels=hidden_channels, num_layers=num_layers
    ).to(device)
    model_parallel.eval()

    # --- Load serial model with identical weights ---
    print("Initializing Serial model (pure PyTorch loops)...")
    model_serial = NonexpansiveDenoiserSerial(
        radius=radius, hidden_channels=hidden_channels, num_layers=num_layers
    ).to(device)
    # Copy weights from parallel to serial so outputs are comparable
    serial_state = model_serial.state_dict()
    parallel_state = model_parallel.state_dict()
    # Only load matching keys (serial has no buffer tensors for shifts)
    compatible = {k: v for k, v in parallel_state.items() if k in serial_state}
    serial_state.update(compatible)
    model_serial.load_state_dict(serial_state)
    model_serial.eval()
    print(f"  Copied {len(compatible)} parameter tensors from parallel -> serial")

    # --- Shared input ---
    dummy_input = torch.rand(batch_size, 3, patch_size, patch_size).to(device)

    # --- Warm-up ---
    print("Warming up GPU...")
    for _ in range(3):
        with torch.no_grad():
            _ = model_parallel(dummy_input)
            _ = model_serial(dummy_input)
    torch.cuda.synchronize()

    # --- Benchmark parallel ---
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            out_parallel = model_parallel(dummy_input)
    torch.cuda.synchronize()
    time_parallel = (time.time() - start) * 1000 / iterations
    mem_parallel = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # --- Benchmark serial ---
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            out_serial = model_serial(dummy_input)
    torch.cuda.synchronize()
    time_serial = (time.time() - start) * 1000 / iterations
    mem_serial = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # --- Output difference ---
    max_diff = torch.max(torch.abs(out_parallel - out_serial)).item()

    # --- Results ---
    print("\n" + "=" * 55)
    print("BENCHMARK RESULTS (SAME NETWORK, SAME WEIGHTS)")
    print("=" * 55)
    print("toy_nn Parallel Model (CUDA Shifter + ICC):")
    print(f"   Time : {time_parallel:.2f} ms")
    print(f"   Peak Mem: {mem_parallel:.2f} MB")
    print("-" * 55)
    print("Serial Model (Pure PyTorch Loops):")
    print(f"   Time : {time_serial:.2f} ms")
    print(f"   Peak Mem: {mem_serial:.2f} MB")
    print("-" * 55)
    speedup = time_serial / time_parallel
    print(f"Speedup : {speedup:.2f}x faster")
    if mem_serial > 0 and mem_parallel > 0:
        print(f"Mem Ratio: {mem_parallel / mem_serial:.2f}x (parallel / serial)")
    print("-" * 55)
    print(f"Output Difference: {max_diff:.8f}")
    if max_diff < 1e-5:
        print("   (Outputs are mathematically identical!)")
    else:
        print("   Outputs differ more than expected")
    print("=" * 55)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark parallel vs serial model.")
    parser.add_argument("--radius", type=int, default=7, help="Radius of the window.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for benchmark.")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size (H and W).")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to average.")
    parser.add_argument("--hidden_channels", type=int, default=16, help="Number of hidden channels in layers.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of conv layers in sub-networks.")
    args = parser.parse_args()

    run_benchmark(
        radius=args.radius,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        iterations=args.iterations,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers
    )
