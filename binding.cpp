#include <torch/extension.h>



torch::Tensor generate_shifts_cuda(torch::Tensor input, torch::Tensor dy_shifts,
                                   torch::Tensor dx_shifts);

void scatter_inv_weights_cuda(
    torch::Tensor fwd_weights,
    torch::Tensor all_weights,
    torch::Tensor inv_flat_indices,
    torch::Tensor inv_source_idx,
    torch::Tensor inv_neg_dy,
    torch::Tensor inv_neg_dx);

torch::Tensor fused_aggregate_cuda(
    torch::Tensor noisy_x,
    torch::Tensor all_weights,
    int radius);



// Wrapper function for generate_shifts that checks if the input tensors are CUDA tensors and calls the corresponding CUDA function.
torch::Tensor generate_shifts(torch::Tensor input, torch::Tensor dy_shifts,
                              torch::Tensor dx_shifts) {
  TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(dy_shifts.device().is_cuda(), "dy_shifts must be a CUDA tensor");
  TORCH_CHECK(dx_shifts.device().is_cuda(), "dx_shifts must be a CUDA tensor");
  TORCH_CHECK(dy_shifts.dtype() == torch::kInt32, "dy_shifts must be int32");
  TORCH_CHECK(dx_shifts.dtype() == torch::kInt32, "dx_shifts must be int32");
  TORCH_CHECK(dy_shifts.size(0) == dx_shifts.size(0),
              "dy_shifts and dx_shifts must have same length");
  return generate_shifts_cuda(input, dy_shifts, dx_shifts);
}

// Wrapper for scatter_inv_weights — writes inverse-shifted weights into all_weights in-place.
void scatter_inv_weights(torch::Tensor fwd_weights,
                         torch::Tensor all_weights,
                         torch::Tensor inv_flat_indices,
                         torch::Tensor inv_source_idx,
                         torch::Tensor inv_neg_dy,
                         torch::Tensor inv_neg_dx) {
  TORCH_CHECK(fwd_weights.device().is_cuda(), "fwd_weights must be a CUDA tensor");
  TORCH_CHECK(all_weights.device().is_cuda(), "all_weights must be a CUDA tensor");
  TORCH_CHECK(inv_flat_indices.device().is_cuda(), "inv_flat_indices must be a CUDA tensor");
  TORCH_CHECK(inv_source_idx.device().is_cuda(), "inv_source_idx must be a CUDA tensor");
  TORCH_CHECK(inv_neg_dy.device().is_cuda(), "inv_neg_dy must be a CUDA tensor");
  TORCH_CHECK(inv_neg_dx.device().is_cuda(), "inv_neg_dx must be a CUDA tensor");
  scatter_inv_weights_cuda(fwd_weights, all_weights, inv_flat_indices,
                           inv_source_idx, inv_neg_dy, inv_neg_dx);
}

// Wrapper for fused_aggregate — computes denoised output in a single kernel.
torch::Tensor fused_aggregate(torch::Tensor noisy_x,
                              torch::Tensor all_weights,
                              int radius) {
  TORCH_CHECK(noisy_x.device().is_cuda(), "noisy_x must be a CUDA tensor");
  TORCH_CHECK(all_weights.device().is_cuda(), "all_weights must be a CUDA tensor");
  return fused_aggregate_cuda(noisy_x, all_weights, radius);
}

// Define the Pybind11 module and bind the C++ functions to Python, allowing them to be called from Python code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("generate_shifts", &generate_shifts,
        "GPU Parallel Circular Shift Generator (CUDA)");
  m.def("scatter_inv_weights", &scatter_inv_weights,
        "Scatter inverse-shifted weights into all_weights tensor (CUDA)");
  m.def("fused_aggregate", &fused_aggregate,
        "Fused weighted aggregation with on-the-fly shifts (CUDA)");
}
