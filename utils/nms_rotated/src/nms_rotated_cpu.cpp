#include <torch/types.h>
#include "box_iou_rotated_utils.h"

template <typename scalar_t>
at::Tensor nms_rotated_cpu_kernel(const at::Tensor& dets, const at::Tensor& scores, const float iou_threshold) {
  TORCH_CHECK(dets.device().is_cpu(), "dets must be a CPU tensor");
  TORCH_CHECK(scores.device().is_cpu(), "scores must be a CPU tensor");
  TORCH_CHECK(dets.scalar_type() == scores.scalar_type(), "dets and scores must have the same dtype");
  if (dets.numel() == 0) return at::empty({0}, dets.options().dtype(at::kLong));
  auto order_t = std::get<1>(scores.sort(0, true));
  auto ndets = dets.size(0);
  auto suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  auto keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));
  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();
  int64_t num_to_keep = 0;
  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i]) continue;
    keep[num_to_keep++] = i;
    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j]) continue;
      auto ovr = single_box_iou_rotated<scalar_t>(dets[i].data_ptr<scalar_t>(), dets[j].data_ptr<scalar_t>());
      if (ovr >= iou_threshold) suppressed[j] = 1;
    }
  }
  return keep_t.narrow(0, 0, num_to_keep);
}

at::Tensor nms_rotated_cpu(const at::Tensor& dets, const at::Tensor& scores, const float iou_threshold) {
  auto result = at::empty({0}, dets.options());
  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_rotated", [&] {
    result = nms_rotated_cpu_kernel<scalar_t>(dets, scores, iou_threshold);
  });
  return result;
}
