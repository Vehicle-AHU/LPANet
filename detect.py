"""Paired RGB/IR inference for LPANet."""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr, increment_path,
                           non_max_suppression_obb, print_args, scale_polys)
from utils.plots import Annotator, colors
from utils.rboxs_utils import rbox2poly
from utils.semantic import load_class_embeddings
from utils.torch_utils import select_device, time_sync


def _paired_datasets(rgb_source, ir_source, imgsz, stride):
    rgb = LoadImages(str(rgb_source), img_size=imgsz, stride=stride, auto=True)
    ir = LoadImages(str(ir_source), img_size=imgsz, stride=stride, auto=True)
    if len(rgb.files) != len(ir.files):
        raise RuntimeError(f'RGB/IR inference count mismatch: {len(rgb.files)} vs {len(ir.files)}')
    rgb_keys = [Path(x).stem for x in rgb.files]
    ir_keys = [Path(x).stem for x in ir.files]
    if rgb_keys != ir_keys:
        raise RuntimeError('RGB/IR inference files are not paired by filename stem.')
    return rgb, ir


@torch.no_grad()
def run(weights,
        source_rgb,
        source_ir,
        semantic_embeddings=ROOT / 'weights/class_description_embedding_mpnet.pkl',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.40,
        max_det=1000,
        device='',
        save_txt=False,
        save_conf=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=2,
        hide_labels=False,
        hide_conf=False,
        half=False):
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    names = model.names
    imgsz = check_img_size(imgsz, s=stride)
    half &= device.type != 'cpu'
    model.half() if half else model.float()
    model.eval()

    nc = len(names)
    attr_vectors = load_class_embeddings(semantic_embeddings, device, nc=nc)
    attr_vectors = attr_vectors.half() if half else attr_vectors.float()

    rgb_dataset, ir_dataset = _paired_datasets(source_rgb, source_ir, imgsz, stride)
    dt = [0.0, 0.0, 0.0]
    seen = 0

    for rgb_item, ir_item in zip(rgb_dataset, ir_dataset):
        rgb_path, rgb_np, _, _, _ = rgb_item
        ir_path, ir_np, ir0, _, status = ir_item

        t1 = time_sync()
        rgb = torch.from_numpy(rgb_np).to(device)
        ir = torch.from_numpy(ir_np).to(device)
        rgb = rgb.half() if half else rgb.float()
        ir = ir.half() if half else ir.float()
        rgb /= 255.0
        ir /= 255.0
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
            ir = ir.unsqueeze(0)
        dt[0] += time_sync() - t1

        t2 = time_sync()
        outputs = model(rgb, ir, attr_vectors)
        detector_output = outputs[0]
        pred = detector_output[0] if isinstance(detector_output, (tuple, list)) else detector_output
        dt[1] += time_sync() - t2

        t3 = time_sync()
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres,
                                       classes=classes, agnostic=agnostic_nms,
                                       multi_label=True, max_det=max_det)
        dt[2] += time_sync() - t3

        for det in pred:
            seen += 1
            p = Path(ir_path)
            im0 = ir0.copy()
            save_path = save_dir / p.name
            txt_path = save_dir / 'labels' / p.stem
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                poly = rbox2poly(det[:, :5])
                poly = scale_polys(ir.shape[2:], poly, im0.shape)
                det_poly = torch.cat((poly, det[:, -2:]), dim=1)

                counts = []
                for c in det_poly[:, -1].unique():
                    n = int((det_poly[:, -1] == c).sum())
                    counts.append(f'{n} {names[int(c)]}')
                LOGGER.info(f"{status}{', '.join(counts)}")

                for *coords, conf, cls in reversed(det_poly):
                    coords = [float(x) for x in coords]
                    if save_txt:
                        line = (int(cls), *coords, float(conf)) if save_conf else (int(cls), *coords)
                        with open(str(txt_path) + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if not nosave:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.poly_label(coords, label, color=colors(c, True))

            if not nosave:
                cv2.imwrite(str(save_path), annotator.result())

    if seen:
        speed = tuple(x / seen * 1e3 for x in dt)
        LOGGER.info('Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image' % speed)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='LPANet .pt checkpoint')
    parser.add_argument('--source-rgb', type=str, required=True, help='RGB image/video directory or file')
    parser.add_argument('--source-ir', type=str, required=True, help='IR image/video directory or file')
    parser.add_argument('--semantic-embeddings', type=str,
                        default=str(ROOT / 'weights/class_description_embedding_mpnet.pkl'),
                        help='class semantic embedding pickle')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.40)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='', help='cuda device, e.g. 0 or cpu')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--project', default=ROOT / 'runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=2, type=int)
    parser.add_argument('--hide-labels', action='store_true')
    parser.add_argument('--hide-conf', action='store_true')
    parser.add_argument('--half', action='store_true')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    main(parse_opt())
