"""
Loss / Objective Functions and Their Derivatives
Implements: CrossEntropy (with internal softmax), MeanSquaredError (with internal softmax)
"""

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax."""
    Z = logits - np.max(logits, axis=1, keepdims=True)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer label array to one-hot matrix."""
    N = y.shape[0]
    oh = np.zeros((N, num_classes))
    oh[np.arange(N), y] = 1.0
    return oh


# CrossEntropy loss
class CrossEntropy:

    def forward(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        N = logits.shape[0]
        probs = _softmax(logits)                      # Clipped to avoid log(0)
        correct_log_probs = -np.log(
            np.clip(probs[np.arange(N), y_true], 1e-15, 1.0)
        )
        return float(np.mean(correct_log_probs))

    def backward(self, y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
        N = logits.shape[0]
        probs = _softmax(logits)                          
        grad = probs.copy()
        grad[np.arange(N), y_true] -= 1.0
        return grad


# Mean Squared Error (MSE) loss
class MeanSquaredError:
    """
    Softmax + Mean Squared Error loss.
    """

    def forward(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        N, C = logits.shape
        probs = _softmax(logits)                          
        y_oh = _one_hot(y_true, C)                        
        diff = probs - y_oh                               
        return float(np.sum(diff ** 2) / (2.0 * N))

    def backward(self, y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """
        Gradient of MSE loss w.r.t. logits via softmax Jacobian.
        """
        N, C = logits.shape
        probs = _softmax(logits)                           
        y_oh = _one_hot(y_true, C)                        

        delta = probs - y_oh                             
        S = np.sum(delta * probs, axis=1, keepdims=True)  

        grad = probs * (delta - S)                        
        return grad



_LOSS_REGISTRY = {
    "cross_entropy": CrossEntropy,
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
}


def get_loss(name: str):
    key = name.lower().strip()
    if key not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. "
            f"Choose from: {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[key]()