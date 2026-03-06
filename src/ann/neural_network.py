"""
Main Neural Network Model
Orchestrates layers, forward propagation, backward propagation, and training.
"""

import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss
from ann.optimizers import get_optimizer, NAG


class NeuralNetwork:
    """
    Neural Network class
    Handles forward and backward propagation loops
    
    Attributes:
        layers (list): List of neural layers
        loss_name (str): Name of the loss function
        
    Methods:
        forward(X): Forward propagation through all layers
        backward(y_true, y_pred): Backward propagation to compute gradients
    """
    def __init__(self, cli_args):
        """
        Build the network from CLI arguments.

        Args:
            cli_args: argparse.Namespace with all configuration fields.
        """
        self.cli_args = cli_args
        self.layers = []

        input_size = 784    # flattened 28x28
        output_size = 10    # MNIST / Fashion-MNIST classes

        # resolve hidden layer sizes 
        raw = cli_args.hidden_size
        if isinstance(raw, (list, tuple)):
            hidden_sizes = [int(h) for h in raw]
        else:
            hidden_sizes = [int(raw)]

        num_hidden = cli_args.num_layers

        # Extend or truncate to exactly num_hidden layers
        if len(hidden_sizes) < num_hidden:
            hidden_sizes += [hidden_sizes[-1]] * (num_hidden - len(hidden_sizes))
        hidden_sizes = hidden_sizes[:num_hidden]

        # build layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            act = None if is_last else cli_args.activation
            self.layers.append(
                NeuralLayer(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    activation=act,
                    weight_init=cli_args.weight_init,
                )
            )

        # loss and optimizer
        self.loss_fn = get_loss(cli_args.loss)
        self.optimizer = get_optimizer(
            cli_args.optimizer, lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay,
        )
        self.weight_decay = cli_args.weight_decay

        # stored grad arrays 
        self.grad_W = None
        self.grad_b = None


    def forward(self, X: np.ndarray) -> np.ndarray:

        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out   


    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):

        dA = self.loss_fn.backward(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            grad_W_list.append(layer.grad_W)   
            grad_b_list.append(layer.grad_b)

        n = len(grad_W_list)
        self.grad_W = np.empty(n, dtype=object)
        self.grad_b = np.empty(n, dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

   
    def _optimizer_step(self, X_batch, y_batch):

        if isinstance(self.optimizer, NAG):
            # 1. Shift to lookahead weights
            self.optimizer.apply_lookahead(self.layers)
            # 2. Forward + backward at lookahead position
            logits = self.forward(X_batch)
            self.backward(y_batch, logits)
            # 3. Restore original weights
            self.optimizer.restore_weights(self.layers)
        else:
            logits = self.forward(X_batch)
            self.backward(y_batch, logits)

        # 4. Apply parameter update
        self.optimizer.step(self.layers, self.grad_W, self.grad_b)

        # Recompute logits at the NEW weights for loss logging
        logits = self.forward(X_batch)
        loss = self.loss_fn.forward(y_batch, logits)
        return loss, logits

    # Training loop
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = None,
        batch_size: int = None,
        wandb_run=None,
    ):
        """
        Full training loop

        Args:
            X_train   : Training inputs,  shape (N_train, 784)
            y_train   : Training labels,  shape (N_train,)
            X_val     : Validation inputs, shape (N_val, 784)  [optional]
            y_val     : Validation labels, shape (N_val,)      [optional]
            epochs    : Override cli_args.epochs if provided.
            batch_size: Override cli_args.batch_size if provided.
            wandb_run : Active wandb run object for logging (or None).
        Returns:
            history (dict): Keys 'train_loss', 'val_loss', 'val_acc'
        """
        epochs = epochs or self.cli_args.epochs
        batch_size = batch_size or self.cli_args.batch_size
        N = X_train.shape[0]
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            # shuffle training data each epoch
            perm = np.random.permutation(N)
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, N, batch_size):
                X_batch = X_shuf[start: start + batch_size]
                y_batch = y_shuf[start: start + batch_size]

                loss, _ = self._optimizer_step(X_batch, y_batch)
                epoch_loss += loss
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_train_loss)

            # validation metrics
            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # console logging
            msg = (
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"train_loss: {avg_train_loss:.4f}"
            )
            if val_loss is not None:
                msg += f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
            print(msg)

            # wandb logging
            if wandb_run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                }
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                    log_dict["val_accuracy"] = val_acc
                wandb_run.log(log_dict)

        return history

    # Evaluate the model on a dataset
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute loss and accuracy on a dataset.

        Args:
            X : Inputs,  shape (N, 784)
            y : Integer labels, shape (N,)
        Returns:
            dict with keys 'loss', 'accuracy', 'predictions', 'logits'
        """
        logits = self.forward(X)                       
        loss = self.loss_fn.forward(y, logits)
        preds = np.argmax(logits, axis=1)
        acc = float(np.mean(preds == y))
        return {
            "loss": float(loss),
            "accuracy": acc,
            "predictions": preds,
            "logits": logits,
        }

    # Get weights
    def get_weights(self) -> dict:
        """
        Serialize all layer weights into a dictionary.

        Returns:
            dict: {'W0': ..., 'b0': ..., 'W1': ..., 'b1': ..., ...}
        """
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        """
        Restore layer weights from a serialized dictionary.

        Args:
            weight_dict: dict as produced by get_weights().
        """
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()