#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void generate_shifts_kernel(const scalar_t *__restrict__ input,
                                       scalar_t *__restrict__ output,
                                       const int *__restrict__ dy_shifts,
                                       const int *__restrict__ dx_shifts,
                                       int num_shifts, int B, int C, int H,
                                       int W){

  const int total_elements = num_shifts * B * C * H * W;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const int BCHW = B * C * H * W;
  const int CHW = C * H * W;
  const int HW = H * W;

  const int shift_idx = idx / BCHW;
  const int remainder = idx % BCHW;
  
  const int b = remainder / CHW;
  const int rem2 = remainder % CHW;
  const int c = rem2 / HW;
  const int rem3 = rem2 % HW;
  const int h = rem3 / W;
  const int w = rem3 % W;

  const int dy = dy_shifts[shift_idx];
  const int dx = dx_shifts[shift_idx];

  int src_h = h - dy;
  if (src_h < 0) src_h += H;
  else if (src_h >= H) src_h -= H;

  int src_w = w - dx;
  if (src_w < 0) src_w += W;
  else if (src_w >= W) src_w -= W;

  const int src_idx = b * CHW + c * HW + src_h * W + src_w;
  output[idx] = input[src_idx];
}

torch::Tensor generate_shifts_cuda(torch::Tensor input, torch::Tensor dy_shifts,
                                   torch::Tensor dx_shifts)
{

  const int B = input.size(0);
  const int C = input.size(1);
  const int H = input.size(2);
  const int W = input.size(3);
  const int num_shifts = dy_shifts.size(0);

  auto output = torch::empty({num_shifts, B, C, H, W}, input.options());

  const int total_elements = num_shifts * B * C * H * W;
  const int threads_per_block = 256;
  const int blocks =
      (total_elements + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "generate_shifts_kernel", ([&]
                                                      { generate_shifts_kernel<scalar_t><<<blocks, threads_per_block>>>(
                                                            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                                            dy_shifts.data_ptr<int>(), dx_shifts.data_ptr<int>(), num_shifts, B,
                                                            C, H, W); }));

  return output;
}


template <typename scalar_t>
__global__ void scatter_inv_weights_kernel(
    const scalar_t *__restrict__ fwd_weights,  
    scalar_t *__restrict__ all_weights,        
    const int64_t *__restrict__ inv_flat_indices, 
    const int64_t *__restrict__ inv_source_idx,   
    const int *__restrict__ inv_neg_dy,           
    const int *__restrict__ inv_neg_dx,           
    int num_inv, int B, int H, int W,
    int total_shifts)
{
  const int total_elements = num_inv * B * H * W;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const int BHW = B * H * W;
  const int HW = H * W;

  const int inv_idx = idx / BHW;
  const int rem1 = idx % BHW;
  const int b = rem1 / HW;
  const int rem2 = rem1 % HW;
  const int h = rem2 / W;
  const int w = rem2 % W;

  const int64_t src_fwd_idx = inv_source_idx[inv_idx];

  const int neg_dy = inv_neg_dy[inv_idx];
  const int neg_dx = inv_neg_dx[inv_idx];

  int src_h = ((h - neg_dy) % H + H) % H;
  int src_w = ((w - neg_dx) % W + W) % W;

  const int64_t read_idx = src_fwd_idx * (B * HW) + b * HW + src_h * W + src_w;
  const int64_t dst_shift = inv_flat_indices[inv_idx];
  const int64_t write_idx = dst_shift * (B * HW) + b * HW + h * W + w;

  all_weights[write_idx] = fwd_weights[read_idx];
}

void scatter_inv_weights_cuda(
    torch::Tensor fwd_weights,      
    torch::Tensor all_weights,      
    torch::Tensor inv_flat_indices, 
    torch::Tensor inv_source_idx,   
    torch::Tensor inv_neg_dy,       
    torch::Tensor inv_neg_dx)       
{
  const int num_inv = inv_flat_indices.size(0);
  const int B = fwd_weights.size(1);

  const int H = fwd_weights.size(3);
  const int W = fwd_weights.size(4);
  const int total_shifts = all_weights.size(0);

  const int total_elements = num_inv * B * H * W;
  const int threads_per_block = 256;
  const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_FLOATING_TYPES(
      fwd_weights.scalar_type(), "scatter_inv_weights_kernel", ([&] {
        scatter_inv_weights_kernel<scalar_t><<<blocks, threads_per_block>>>(
            fwd_weights.data_ptr<scalar_t>(),
            all_weights.data_ptr<scalar_t>(),
            inv_flat_indices.data_ptr<int64_t>(),
            inv_source_idx.data_ptr<int64_t>(),
            inv_neg_dy.data_ptr<int>(),
            inv_neg_dx.data_ptr<int>(),
            num_inv, B, H, W, total_shifts);
      }));
}

template <typename scalar_t>
__global__ void fused_aggregate_kernel(
    const scalar_t* __restrict__ noisy_x,
    const scalar_t* __restrict__ all_weights,
    scalar_t* __restrict__ numerator,
    scalar_t* __restrict__ denominator,
    int B, int C, int H, int W, 
    int radius, int window_size, int total_shifts)
{
  const int total_threads = B * C * H * W * total_shifts;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= total_threads) return;

  const int CHW = C * H * W;
  const int HW = H * W;
  
  const int shift_idx = idx / (B * CHW);
  const int rem1 = idx % (B * CHW);
  const int b = rem1 / CHW;
  const int rem2 = rem1 % CHW;
  const int c = rem2 / HW;
  const int rem3 = rem2 % HW;
  const int h = rem3 / W;
  const int w = rem3 % W;

  const int dy = (shift_idx / window_size) - radius;
  const int dx = (shift_idx % window_size) - radius;

  int src_h = h - dy;
  if (src_h < 0) src_h += H;
  else if (src_h >= H) src_h -= H;

  int src_w = w - dx;
  if (src_w < 0) src_w += W;
  else if (src_w >= W) src_w -= W;

  const int weight_bhw = b * HW + h * W + w;
  const int weight_stride = B * HW;
  
  // Single memory load for weight
  scalar_t wt = all_weights[shift_idx * weight_stride + weight_bhw];
  
  // Single memory load for image pixel
  scalar_t noisy_val = noisy_x[b * CHW + c * HW + src_h * W + src_w];
  
  scalar_t val = wt * noisy_val;

  int out_idx = b * CHW + c * HW + h * W + w;
  
  // Atomically accumulate to global memory
  atomicAdd(&numerator[out_idx], val);

  // Denominator must be accumulated for all C channels
  atomicAdd(&denominator[out_idx], wt);
}

torch::Tensor fused_aggregate_cuda(
    torch::Tensor noisy_x,      
    torch::Tensor all_weights,   
    int radius)
{
  const int B = noisy_x.size(0);
  const int C = noisy_x.size(1);
  const int H = noisy_x.size(2);
  const int W = noisy_x.size(3);
  const int window_size = 2 * radius + 1;
  const int total_shifts = window_size * window_size;

  auto numerator = torch::zeros_like(noisy_x);
  auto denominator = torch::zeros_like(noisy_x);

  const int total_threads = B * C * H * W * total_shifts;
  const int threads_per_block = 256;
  const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  AT_DISPATCH_FLOATING_TYPES(
      noisy_x.scalar_type(), "fused_aggregate_kernel", ([&] {
        fused_aggregate_kernel<scalar_t><<<blocks, threads_per_block>>>(
            noisy_x.data_ptr<scalar_t>(),
            all_weights.data_ptr<scalar_t>(),
            numerator.data_ptr<scalar_t>(),
            denominator.data_ptr<scalar_t>(),
            B, C, H, W,
            radius, window_size, total_shifts);
      }));

  return numerator / (denominator + 1e-8);
}
