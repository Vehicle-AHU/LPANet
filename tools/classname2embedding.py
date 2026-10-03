"""Generate the class-description embeddings used by LPANet.

The paper constructs fine-grained descriptions for each object category and
encodes them with MPNet.  The released checkpoint uses a 768-D embedding
matrix whose row order must match the dataset YAML.

For exact reproduction of released results, use the released
``class_description_embedding_mpnet.pkl``.  This utility reproduces the paper
pipeline for users who need to regenerate the embeddings.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

DEFAULT_ORDER = ['car', 'truck', 'bus', 'freight_car', 'van']


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--description-file', type=Path,
        default=Path('data/class_descriptions_dronevehicle.json'),
        help='JSON mapping class names to the fine-grained descriptions used in the paper')
    parser.add_argument(
        '--classes', nargs='+', default=DEFAULT_ORDER,
        help='embedding row order; must match the dataset YAML')
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument(
        '--output', type=Path,
        default=Path('weights/class_description_embedding_mpnet.pkl'))
    return parser.parse_args()


def load_descriptions(path, classes):
    with Path(path).open('r', encoding='utf-8') as f:
        descriptions = json.load(f)
    missing = [name for name in classes if name not in descriptions]
    if missing:
        raise KeyError(f'Missing class descriptions for: {missing}')
    return [descriptions[name] for name in classes]


def main(opt):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            'Install sentence-transformers first: pip install sentence-transformers') from e

    texts = load_descriptions(opt.description_file, opt.classes)
    model = SentenceTransformer(opt.model)
    vectors = np.asarray(model.encode(texts), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 768:
        raise ValueError(f'LPANet expects 768-D MPNet embeddings, got {vectors.shape}.')

    opt.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'attr_words': list(opt.classes),
        'attr_descriptions': texts,
        'attr_vectors': vectors,
        'text_encoder': opt.model,
    }
    with opt.output.open('wb') as f:
        pickle.dump(payload, f)
    print(f'Saved {vectors.shape} class-description embeddings to {opt.output}')


if __name__ == '__main__':
    main(parse_opt())
