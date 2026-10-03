import numpy as np
import torch

from . import nms_rotated_ext

def obb_nms(dets, scores, iou_thr, device_id=None):
    """
    RIoU NMS - iou_thr.
    Args:
        dets (tensor/array): (num, [cx cy w h θ]) θ∈[-pi/2, pi/2)
        scores (tensor/array): (num)
        iou_thr (float): (1)
    Returns:
        dets (tensor): (n_nms, [cx cy w h θ])
        inds (tensor): (n_nms), nms index of dets
    """
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.numel() == 0:
        inds = dets_th.new_zeros(0, dtype=torch.int64)
    else:
        too_small = dets_th[:, [2, 3]].min(1)[0] < 0.001
        if too_small.all():
            inds = dets_th.new_zeros(0, dtype=torch.int64)
        else:
            ori_inds = torch.arange(dets_th.size(0), device=dets_th.device)
            ori_inds = ori_inds[~too_small]
            dets_th = dets_th[~too_small]
            scores = scores[~too_small]
            inds = nms_rotated_ext.nms_rotated(dets_th, scores, iou_thr)
            inds = ori_inds[inds]

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds



def poly_nms(*args, **kwargs):
    raise NotImplementedError('LPANet does not use polygon NMS; only obb_nms is built in this adapted package.')
