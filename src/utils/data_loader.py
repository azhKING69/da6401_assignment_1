"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets via keras.datasets.

Uses lazy imports (inside the function) so keras is never imported
at module load time — avoids tensorflow import errors on the autograder.
"""

import numpy as np

# Fashion-MNIST class names
FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]
MNIST_LABELS = [str(i) for i in range(10)]


def load_dataset(name: str, val_split: float = 0.1, seed: int = 42):
    """
    Load, flatten, and normalise MNIST or Fashion-MNIST.
    Splits training data into train/val sets.

    Args:
        name      : 'mnist' or 'fashion_mnist'
        val_split : fraction of training data for validation (default 0.1)
        seed      : random seed for reproducible split

    Returns:
        X_train, y_train : (N_train, 784) float64, (N_train,) int64
        X_val,   y_val   : (N_val,   784) float64, (N_val,)   int64
        X_test,  y_test  : (10000,   784) float64, (10000,)   int64
        label_names      : list of 10 class-name strings
    """
    name = name.lower().strip().replace("-", "_")

    if name == "mnist":
        from keras.datasets import mnist
        (X_raw_train, y_raw_train), (X_test, y_test) = mnist.load_data()
        label_names = MNIST_LABELS

    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_raw_train, y_raw_train), (X_test, y_test) = fashion_mnist.load_data()
        label_names = FASHION_MNIST_LABELS

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'."
        )

    # Flatten 28×28 → 784 and normalise to [0, 1]
    X_raw_train = X_raw_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test      = X_test.reshape(-1, 784).astype(np.float64) / 255.0
    y_raw_train = y_raw_train.astype(np.int64)
    y_test      = y_test.astype(np.int64)

    # Reproducible train / val split
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
