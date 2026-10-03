#!/usr/bin/env python3
from pathlib import Path
import argparse

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(path):
    return sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)


def label_path(img):
    for d in ('labelTxt', 'labels'):
        p = img.parent.parent / d / f'{img.stem}.txt'
        if p.exists():
            return p
    return img.parent.parent / 'labelTxt' / f'{img.stem}.txt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='DroneVehicle dataset root')
    ap.add_argument('--max-label-check', type=int, default=200)
    args = ap.parse_args()
    root = Path(args.root)
    print(f'root: {root}')
    if not root.is_dir():
        raise SystemExit(f'ERROR: dataset root does not exist: {root}')

    all_ok = True
    for split in ('train', 'val', 'test'):
        rgb_dir = root / 'rgb' / split / 'images'
        ir_dir = root / 'ir' / split / 'images'
        print(f'\n[{split}]')
        print(f'  RGB: {rgb_dir}')
        print(f'  IR : {ir_dir}')
        if not rgb_dir.is_dir() or not ir_dir.is_dir():
            print('  ERROR: expected image directory missing')
            all_ok = False
            continue
        rgb = collect_images(rgb_dir)
        ir = collect_images(ir_dir)
        rgb_map = {p.stem: p for p in rgb}
        ir_map = {p.stem: p for p in ir}
        common = sorted(rgb_map.keys() & ir_map.keys())
        print(f'  images: RGB={len(rgb)} IR={len(ir)} paired={len(common)}')
        miss_ir = sorted(rgb_map.keys() - ir_map.keys())
        miss_rgb = sorted(ir_map.keys() - rgb_map.keys())
        if miss_ir or miss_rgb:
            print(f'  ERROR: missing pairs: no-IR={len(miss_ir)}, no-RGB={len(miss_rgb)}')
            print(f'         examples no-IR={miss_ir[:5]} no-RGB={miss_rgb[:5]}')
            all_ok = False

        checked = 0
        missing_labels = 0
        bad_labels = 0
        class_ids = set()
        for stem in common:
            # LPANet trains/evaluates in the IR coordinate frame, but both modalities are checked.
            for img in (rgb_map[stem], ir_map[stem]):
                lp = label_path(img)
                if not lp.exists():
                    missing_labels += 1
                    continue
                if checked < args.max_label_check:
                    for line_no, line in enumerate(lp.read_text(errors='ignore').splitlines(), 1):
                        if not line.strip():
                            continue
                        parts = line.split()
                        try:
                            if len(parts) == 10:  # DOTA: x1..y4 class_name difficult
                                _ = [float(x) for x in parts[:8]]
                                class_ids.add(parts[8])
                                int(float(parts[9]))
                            elif len(parts) == 9:  # numeric: class_id x1..y4
                                cls = int(float(parts[0]))
                                _ = [float(x) for x in parts[1:9]]
                                class_ids.add(cls)
                            else:
                                raise ValueError(f'{len(parts)} fields')
                        except ValueError as e:
                            bad_labels += 1
                            print(f'  BAD LABEL: {lp}:{line_no} invalid OBB row ({e}); expected 10 DOTA fields or 9 numeric fields')
                            break
                    checked += 1
        print(f'  labels: missing={missing_labels}, sampled_bad={bad_labels}, sampled_class_ids={sorted(class_ids)}')
        numeric_ids = [c for c in class_ids if isinstance(c, int)]
        if missing_labels or bad_labels or any(c < 0 or c > 4 for c in numeric_ids):
            all_ok = False

    print('\nRESULT:', 'OK' if all_ok else 'FAILED')
    raise SystemExit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
