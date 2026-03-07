"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets via keras.datasets.
Provides normalisation, flattening, and train/val/test splitting.
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
    Load and preprocess MNIST or Fashion-MNIST.

    Args:
        name      : 'mnist' or 'fashion_mnist'
        val_split : fraction of training data to use for validation
        seed      : random seed for reproducibility of train/val split

    Returns:
        X_train, y_train : training data and labels
        X_val,   y_val   : validation data and labels
        X_test,  y_test  : test data and labels
        label_names      : list of class names for interpretation
    """
    name = name.lower().strip()

    if name == "mnist":
        from keras.datasets import mnist
        (X_raw_train, y_raw_train), (X_test, y_test) = mnist.load_data()
        label_names = MNIST_LABELS

    elif name in ("fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist
        (X_raw_train, y_raw_train), (X_test, y_test) = fashion_mnist.load_data()
        label_names = FASHION_MNIST_LABELS

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'."
        )

    # Flatten 28x28 → 784 and normalise to [0, 1]
    X_raw_train = X_raw_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test      = X_test.reshape(-1, 784).astype(np.float64) / 255.0
    y_raw_train = y_raw_train.astype(np.int64)
    y_test      = y_test.astype(np.int64)

    # Reproducible train / validation split
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
        f"train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, label_names
