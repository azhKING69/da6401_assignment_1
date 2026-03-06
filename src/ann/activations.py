"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax, Linear
Each class caches values needed for the backward pass.
"""

import numpy as np


class ReLU:

    def forward(self, Z):
        self.mask = (Z > 0)
        return Z * self.mask

    def backward(self, dA):
        return dA * self.mask


class Sigmoid:

    def forward(self, Z):
        Z_clipped = np.clip(Z, -500, 500)
        self.A = 1.0 / (1.0 + np.exp(-Z_clipped))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1.0 - self.A)


class Tanh:

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        return dA * (1.0 - self.A ** 2)


class Softmax:

    def forward(self, Z):
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.A

    def backward(self, dA):
        s = np.sum(dA * self.A, axis=1, keepdims=True)  # (N, 1)
        return self.A * (dA - s)


class Linear:

    def forward(self, Z):
        return Z

    def backward(self, dA):
        return dA


def get_activation(name):
    
    registry = {
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softmax": Softmax,
        "linear": Linear,
        "identity": Linear,
    }
    key = name.lower().strip()
    if key not in registry:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[key]()