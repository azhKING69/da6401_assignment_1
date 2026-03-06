"""
Neural Layer Implementation
"""

import numpy as np
from ann.activations import get_activation


class NeuralLayer:

    def __init__(self, input_size, output_size, activation=None, weight_init="xavier"):
        """
        Initializes a fully connected (Dense) layer.
        Args:
            input_size (int): Number of input features/neurons from the previous layer
            output_size (int): Number of neurons in this layer
            activation_name (str): Name of the activation function ('relu', 'sigmoid', 'tanh', 'softmax', or None)
            weight_init (str): Method to initialize weights ('random', 'zeros', 'xavier')
            
        Attributes:
            input_size (int): Number of input features/neurons from the previous layer
            output_size (int): Number of neurons in this layer
            activation_name (str): Name of the activation function ('relu', 'sigmoid', 'tanh', 'softmax', or None)
            weight_init (str): Method to initialize weights ('random', 'zeros', 'xavier')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation if activation else "linear"

        if activation and activation.lower() not in ("none", "linear", "identity", ""):
            self.activation = get_activation(activation)
        else:
            self.activation = None  

        # Weight Initialization 
        self._init_weights(weight_init)

        self.grad_W = None
        self.grad_b = None

        # Forward-pass cache
        self.X = None   # input to this layer
        self.Z = None   # pre-activation 
        self.A = None   # post-activation output

    def _init_weights(self, method: str):

        if method == "xavier":
            scale = np.sqrt(2.0 / (self.input_size + self.output_size))
            self.W = np.random.randn(self.input_size, self.output_size) * scale
        elif method == "random":
            self.W = np.random.randn(self.input_size, self.output_size) * 0.01
        else:
            raise ValueError(
                f"Unknown weight_init '{method}'. Choose 'xavier' or 'random'."
            )
        self.b = np.zeros((1, self.output_size))

    # Forward pass
    def forward(self, X: np.ndarray) -> np.ndarray:

        self.X = X  # cache input for backward
        self.Z = X @ self.W + self.b

        if self.activation is not None:
            self.A = self.activation.forward(self.Z)
        else:
            self.A = self.Z                

        return self.A

    # Backward pass
    def backward(self, dA: np.ndarray) -> np.ndarray:

        batch_size = self.X.shape[0]

        if self.activation is not None:
            dZ = self.activation.backward(dA)
        else:
            dZ = dA

        self.grad_W = (self.X.T @ dZ) / batch_size
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)

        # Gradient to propagate to the previous layer
        dX = dZ @ self.W.T 

        return dX