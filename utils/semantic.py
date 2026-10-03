from pathlib import Path
import pickle
import numpy as np
import torch


def load_class_embeddings(path, device, nc=None):
    """Load LPANet MPNet class embeddings and move them to the requested device."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Semantic embedding file not found: {path}\n"
            "LPANet requires the MPNet class-description embeddings. "
            "Place class_description_embedding_mpnet.pkl under weights/ or pass "
            "--semantic-embeddings /path/to/class_description_embedding_mpnet.pkl."
        )
    with path.open('rb') as f:
        obj = pickle.load(f)
    if hasattr(obj, 'attr_vectors'):
        arr = obj.attr_vectors
    elif isinstance(obj, dict) and 'attr_vectors' in obj:
        arr = obj['attr_vectors']
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        arr = obj
    else:
        raise KeyError(f"{path} does not contain 'attr_vectors'.")
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected semantic embeddings with shape [num_classes, dim], got {arr.shape}.")
    if nc is not None and arr.shape[0] != int(nc):
        raise ValueError(
            f"Semantic embedding class count ({arr.shape[0]}) != dataset nc ({nc}). "
            "The embedding row order must match data/DroneVehicle_poly.yaml."
        )
    return torch.from_numpy(arr).to(device)
