"""
Optimization Algorithms
Implements: SGD, Momentum, NAG (Nesterov Accelerated Gradient), RMSProp
"""

import numpy as np


# Stochastic Gradient Descent (SGD)
class SGD:

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers, grad_Ws, grad_bs):
        for i, layer in enumerate(reversed(layers)):
            eff_gW = grad_Ws[i] + self.weight_decay * layer.W
            layer.W -= self.lr * eff_gW
            layer.b -= self.lr * grad_bs[i]


# Momentum
class Momentum:

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None

    def _init_state(self, grad_Ws, grad_bs):
        self.vW = [np.zeros_like(g) for g in grad_Ws]
        self.vb = [np.zeros_like(g) for g in grad_bs]

    def step(self, layers, grad_Ws, grad_bs):
        if self.vW is None:
            self._init_state(grad_Ws, grad_bs)

        for i, layer in enumerate(reversed(layers)):
            eff_gW = grad_Ws[i] + self.weight_decay * layer.W
            self.vW[i] = self.beta * self.vW[i] - self.lr * eff_gW
            self.vb[i] = self.beta * self.vb[i] - self.lr * grad_bs[i]
            layer.W += self.vW[i]
            layer.b += self.vb[i]


# Nesterov Accelerated Gradient (NAG)
class NAG:

    def __init__(self, lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.vW = None
        self.vb = None
        self._saved_W = None
        self._saved_b = None

    def _init_state(self, layers):
      self.vW = [np.zeros_like(layer.W) for layer in reversed(layers)]
      self.vb = [np.zeros_like(layer.b) for layer in reversed(layers)]

    def apply_lookahead(self, layers):
        if self.vW is None:
            self._init_state(layers)

        self._saved_W = [layer.W.copy() for layer in layers]
        self._saved_b = [layer.b.copy() for layer in layers]
        for j, layer in enumerate(layers):
            v_idx = len(layers) - 1 - j
            layer.W = layer.W + self.beta * self.vW[v_idx]
            layer.b = layer.b + self.beta * self.vb[v_idx]

    def restore_weights(self, layers):
        for j, layer in enumerate(layers):
            layer.W = self._saved_W[j]
            layer.b = self._saved_b[j]

    def step(self, layers, grad_Ws, grad_bs):
        if self.vW is None:
            self._init_state(layers)

        for i, layer in enumerate(reversed(layers)):
            eff_gW = grad_Ws[i] + self.weight_decay * layer.W
            self.vW[i] = self.beta * self.vW[i] - self.lr * eff_gW
            self.vb[i] = self.beta * self.vb[i] - self.lr * grad_bs[i]
            layer.W += self.vW[i]
            layer.b += self.vb[i]


# RMSProp
class RMSProp:

    def __init__(
        self, lr: float = 0.001, rho: float = 0.9,
        eps: float = 1e-8, weight_decay: float = 0.0
    ):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.sW = None
        self.sb = None

    def _init_state(self, grad_Ws, grad_bs):
        self.sW = [np.zeros_like(g) for g in grad_Ws]
        self.sb = [np.zeros_like(g) for g in grad_bs]

    def step(self, layers, grad_Ws, grad_bs):
        if self.sW is None:
            self._init_state(grad_Ws, grad_bs)

        for i, layer in enumerate(reversed(layers)):
            eff_gW = grad_Ws[i] + self.weight_decay * layer.W
            self.sW[i] = self.rho * self.sW[i] + (1.0 - self.rho) * eff_gW ** 2
            self.sb[i] = self.rho * self.sb[i] + (1.0 - self.rho) * grad_bs[i] ** 2
            layer.W -= self.lr * eff_gW / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.lr * grad_bs[i] / (np.sqrt(self.sb[i]) + self.eps)



def get_optimizer(name: str, lr: float, weight_decay: float = 0.0, **kwargs):

    name = name.lower().strip()
    if name == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    elif name == "momentum":
        beta = kwargs.get("beta", 0.9)
        return Momentum(lr=lr, beta=beta, weight_decay=weight_decay)
    elif name == "nag":
        beta = kwargs.get("beta", 0.9)
        return NAG(lr=lr, beta=beta, weight_decay=weight_decay)
    elif name == "rmsprop":
        rho = kwargs.get("rho", 0.9)
        eps = kwargs.get("eps", 1e-8)
        return RMSProp(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: sgd, momentum, nag, rmsprop"
        )