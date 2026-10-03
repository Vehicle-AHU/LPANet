"""Convert LPANet OBB JSON predictions to one DOTA/VOC-style text file per class."""

import argparse
import json
import shutil
from pathlib import Path

CLASSES = ['car', 'truck', 'bus', 'freight_car', 'van']


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', '--json_path', dest='json_path', required=True, type=Path)
    parser.add_argument('--save-path', '--save_path', dest='save_path', required=True, type=Path)
    return parser.parse_args()


def main(opt):
    with opt.json_path.open('r') as f:
        predictions = json.load(f)

    if opt.save_path.exists():
        shutil.rmtree(opt.save_path)
    opt.save_path.mkdir(parents=True)

    handles = {}
    try:
        for pred in predictions:
            category_id = int(pred['category_id'])
            class_index = category_id - 1  # LPANet JSON uses 1-based category IDs.
            if not 0 <= class_index < len(CLASSES):
                raise ValueError(f'Unexpected category_id={category_id}')
            class_name = CLASSES[class_index]
            poly = pred['poly']
            if len(poly) != 8:
                raise ValueError(f"Prediction for {pred.get('file_name')} has {len(poly)} polygon coordinates")
            path = opt.save_path / f'Task1_{class_name}.txt'
            if path not in handles:
                handles[path] = path.open('a')
            line = [pred['file_name'], pred['score'], *poly]
            handles[path].write(' '.join(map(str, line)) + '\n')
    finally:
        for f in handles.values():
            f.close()

    print(f'Converted {len(predictions)} predictions to {opt.save_path}')


if __name__ == '__main__':
    main(parse_opt())
