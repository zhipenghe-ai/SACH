"""Load the precomputed image, text, and label splits used by SACH."""

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.io.matlab import MatReadError
except ImportError:
    from scipy.io.matlab.miobase import MatReadError

torch.multiprocessing.set_sharing_strategy("file_system")

_DATASET_LAYOUT = {
    "MIRFlickr": ("MIRFlickr", "mir"),
    "NUSWIDE": ("NUS-WIDE", "nus"),
    "COCO": ("MSCOCO", "COCO"),
}


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.texts[index], self.labels[index], index

    def __len__(self):
        count = len(self.images)
        assert count == len(self.labels)
        return count


def _mat_split(path, suffix):
    arrays = sio.loadmat(str(path))
    labels = np.array(arrays["L_" + suffix], dtype=float)
    images = np.array(arrays["I_" + suffix], dtype=float)
    texts = np.array(arrays["T_" + suffix], dtype=float)
    return images, texts, labels


def _coco_database_split(path):
    with h5py.File(str(path), "r", libver="latest", swmr=True) as arrays:
        labels = np.array(arrays["L_db"], dtype=float).T
        images = np.array(arrays["I_db"], dtype=float).T
        texts = np.array(arrays["T_db"], dtype=float).T
    return images, texts, labels


def _split_path(dataset, data_root, split):
    folder, prefix = _DATASET_LAYOUT[dataset]
    return Path(data_root) / folder / f"{prefix}_{split}.mat"


def _data_loader(arrays, batch_size, shuffle=False):
    return DataLoader(
        CustomDataSet(*arrays), batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def _loaders(batch_size, train, query, database):
    splits = {"query": query, "train": train, "database": database}
    return {
        name: _data_loader(arrays, batch_size, shuffle=name == "train")
        for name, arrays in splits.items()
    }, train


def _training_loaders(dataset, batch_size, data_root):
    train = _mat_split(_split_path(dataset, data_root, "train"), "tr")
    query = _mat_split(_split_path(dataset, data_root, "query"), "te")
    database_path = _split_path(dataset, data_root, "database")
    database = (
        _coco_database_split(database_path) if dataset == "COCO" else _mat_split(database_path, "db")
    )
    return _loaders(batch_size, train, query, database)


def get_loader_flickr(batch_size, data_root="./datasets"):
    return _training_loaders("MIRFlickr", batch_size, data_root)


def get_loader_nus(batch_size, data_root="./datasets"):
    return _training_loaders("NUSWIDE", batch_size, data_root)


def get_loader_coco(batch_size, data_root="./datasets"):
    return _training_loaders("COCO", batch_size, data_root)


def _evaluation_split(path, suffix, *, hdf5=False):
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation dataset file: {path}")
    try:
        arrays = _coco_database_split(path) if hdf5 else _mat_split(path, suffix)
    except KeyError as exc:
        raise ValueError(f"Missing dataset field {exc} in {path}") from exc
    except (OSError, TypeError, IndexError, ValueError, MatReadError, NotImplementedError) as exc:
        raise ValueError(f"Could not read evaluation dataset file {path}: {exc}") from exc

    if any(array.ndim != 2 or not array.size for array in arrays):
        raise ValueError(f"Expected nonempty 2D image, text and label matrices in {path}")
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError(f"Image, text and label row counts differ in {path}")
    return arrays


def get_evaluation_loaders(dataset, batch_size, data_root="./datasets"):
    """Load the ordered query and gallery splits without reading training data."""
    if dataset not in _DATASET_LAYOUT:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose {', '.join(_DATASET_LAYOUT)}.")
    query_path = _split_path(dataset, data_root, "query")
    database_path = _split_path(dataset, data_root, "database")
    query = _evaluation_split(query_path, "te")
    database = _evaluation_split(database_path, "db", hdf5=dataset == "COCO")
    if tuple(array.shape[1] for array in query) != tuple(array.shape[1] for array in database):
        raise ValueError(f"Image, text or label widths differ between {query_path} and {database_path}")
    return _data_loader(query, batch_size), _data_loader(database, batch_size)
