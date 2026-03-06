"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets.

Uses ONLY urllib + gzip + numpy (stdlib + numpy).
No keras, no tensorflow, no torch required.

Downloads raw IDX files from public mirrors on first run and
caches them in ~/.cache/mnist_numpy/ as a single .npy file so
every subsequent run loads instantly without hitting the network.
"""

import os
import gzip
import struct
import urllib.request
import numpy as np

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mnist_numpy")

# sources for the raw .gz files (mirrors for reliability)
_SOURCES = {
    "mnist": {
        "train_images": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        ],
        "train_labels": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        ],
        "test_images": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        ],
        "test_labels": [
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        ],
    },
    "fashion_mnist": {
        "train_images": [
            "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
        ],
        "train_labels": [
            "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
        ],
        "test_images": [
            "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
        ],
        "test_labels": [
            "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
        ],
    },
}

# class names 
MNIST_LABELS = [str(i) for i in range(10)]

FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]


def _download(url: str, dest: str) -> bool:
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, \
             open(dest, "wb") as f:
            f.write(resp.read())
        return os.path.exists(dest) and os.path.getsize(dest) > 1_000
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return False


def _get_gz(name: str, key: str) -> str:
    filename = _SOURCES[name][key][0].split("/")[-1]
    local    = os.path.join(_CACHE_DIR, name, filename)

    if os.path.exists(local) and os.path.getsize(local) > 1_000:
        return local  

    print(f"  [DataLoader] Downloading {filename} ...", flush=True)
    for url in _SOURCES[name][key]:
        if _download(url, local):
            return local

    raise RuntimeError(
        f"All download mirrors failed for '{filename}' ({name}/{key}).\n"
        "Please check your internet connection, or manually place the "
        f".gz files in:  {os.path.join(_CACHE_DIR, name)}"
    )


def _parse_images(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"Not an IDX3 image file (magic={magic:#010x})")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def _parse_labels(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"Not an IDX1 label file (magic={magic:#010x})")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def _load_raw(name: str):
    """
    Return (X_train, y_train, X_test, y_test) for the given dataset.
    """
    cache = os.path.join(_CACHE_DIR, name, "data.npy")

    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True).item()
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

    X_train = _parse_images(_get_gz(name, "train_images"))
    y_train = _parse_labels(_get_gz(name, "train_labels"))
    X_test  = _parse_images(_get_gz(name, "test_images"))
    y_test  = _parse_labels(_get_gz(name, "test_labels"))

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
    })

    return X_train, y_train, X_test, y_test



def load_dataset(name: str, val_split: float = 0.1, seed: int = 42):
    """
    Load and preprocess MNIST or Fashion-MNIST.

    Args:
        name      : 'mnist' or 'fashion_mnist'
        val_split : fraction of training data to use as validation (default 0.1)
        seed      : random seed for a reproducible train/val split

    Returns:
        X_train, y_train : (N_train, 784) float64,  (N_train,) int64
        X_val,   y_val   : (N_val,   784) float64,  (N_val,)   int64
        X_test,  y_test  : (10000,   784) float64,  (10000,)   int64
        label_names      : list of 10 class-name strings
    """
    name = name.lower().strip().replace("-", "_")

    if name not in _SOURCES:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'."
        )

    X_raw_train, y_raw_train, X_test, y_test = _load_raw(name)
    label_names = MNIST_LABELS if name == "mnist" else FASHION_MNIST_LABELS

    # reproducible train / val split
    rng      = np.random.default_rng(seed)
    N        = X_raw_train.shape[0]
    val_size = int(N * val_split)
    idx      = rng.permutation(N)

    X_train = X_raw_train[idx[val_size:]]
    y_train = y_raw_train[idx[val_size:]]
    X_val   = X_raw_train[idx[:val_size]]
    y_val   = y_raw_train[idx[:val_size]]

    print(
        f"[DataLoader] {name.upper()}  "
        f"train={X_train.shape[0]}  "
        f"val={X_val.shape[0]}  "
        f"test={X_test.shape[0]}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, label_names
