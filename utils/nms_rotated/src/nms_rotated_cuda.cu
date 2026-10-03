#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "box_iou_rotated_utils.h"

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__global__ void nms_rotated_cuda_kernel(const int n_boxes, const float iou_threshold,
                                        const T* dev_boxes, unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
  __shared__ T block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    #pragma unroll
    for (int k = 0; k < 5; ++k)
      block_boxes[threadIdx.x * 5 + k] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + k];
  }
  __syncthreads();
  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 5;
    unsigned long long t = 0;
    int start = (row_start == col_start) ? threadIdx.x + 1 : 0;
    for (int i = start; i < col_size; i++)
      if (single_box_iou_rotated<T>(cur_box, block_boxes + i * 5) > iou_threshold) t |= 1ULL << i;
    const int col_blocks = at::cuda::ATenCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

at::Tensor nms_rotated_cuda(const at::Tensor& dets, const at::Tensor& scores, float iou_threshold) {
  TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(dets.device());
  auto order_t = std::get<1>(scores.sort(0, true));
  auto dets_sorted = dets.index_select(0, order_t);
  auto dets_num = dets.size(0);
  const int col_blocks = at::cuda::ATenCeilDiv(static_cast<int>(dets_num), threadsPerBlock);
  auto mask = at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));
  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(dets_sorted.scalar_type(), "nms_rotated_kernel_cuda", [&] {
    nms_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
      dets_num, iou_threshold, dets_sorted.data_ptr<scalar_t>(),
      reinterpret_cast<unsigned long long*>(mask.data_ptr<int64_t>()));
  });
  auto mask_cpu = mask.to(at::kCPU);
  auto mask_host = reinterpret_cast<unsigned long long*>(mask_cpu.data_ptr<int64_t>());
  std::vector<unsigned long long> remv(col_blocks, 0);
  auto keep = at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  auto keep_out = keep.data_ptr<int64_t>();
  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock, inblock = i % threadsPerBlock;
    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      auto p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) remv[j] |= p[j];
    }
  }
  C10_CUDA_CHECK(cudaGetLastError());
  return order_t.index({keep.narrow(0, 0, num_to_keep).to(order_t.device(), keep.scalar_type())});
}
