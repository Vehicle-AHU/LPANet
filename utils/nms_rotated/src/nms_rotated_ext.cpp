#include <ATen/ATen.h>
#include <torch/extension.h>

#ifdef WITH_CUDA
at::Tensor nms_rotated_cuda(const at::Tensor& dets, const at::Tensor& scores, const float iou_threshold);
#endif
at::Tensor nms_rotated_cpu(const at::Tensor& dets, const at::Tensor& scores, const float iou_threshold);

inline at::Tensor nms_rotated(const at::Tensor& dets, const at::Tensor& scores, const float iou_threshold) {
  TORCH_CHECK(dets.device().is_cuda() == scores.device().is_cuda(), "dets and scores must be on the same device");
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    return nms_rotated_cuda(dets.contiguous(), scores.contiguous(), iou_threshold);
#else
    TORCH_CHECK(false, "nms_rotated was compiled without CUDA support");
#endif
  }
  return nms_rotated_cpu(dets.contiguous(), scores.contiguous(), iou_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_rotated", &nms_rotated, "NMS for rotated boxes");
}
