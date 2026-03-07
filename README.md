# DA6401 Assignment 1: NumPy Multi-Layer Perceptron

## Links

- **Weights and Biases Report:** https://api.wandb.ai/links/ce23b004-iitm/x8kc7jek
- **GitHub Repository:** https://github.com/azhKING69/da6401_assignment_1

---

**Multi-Layer Perceptron (MLP)** implemented from scratch using **only NumPy**. This project implements the complete training pipeline from forward propagation to various optimization strategies to classify the **MNIST** and **Fashion-MNIST** datasets.

## Assignment Overview

The objective is to build an MLP that:
- Uses **exclusively NumPy** for mathematical operations
- Implements forward propagation, backward propagation, and multiple optimization strategies
- Supports full configuration via command-line arguments
- Integrates with Weights & Biases for experiment tracking


---

## Project Structure

```
da6401_assignment_1/
├── README.md
├── requirements.txt
├── src/
│   ├── train.py              # Training script
│   ├── inference.py          # Inference & evaluation script
│   ├── best_model.npy        # Best model weights (saved after training)
│   ├── best_config.json      # Best hyperparameter configuration
│   ├── ann/
│   │   ├── neural_network.py  # Main NeuralNetwork class
│   │   ├── neural_layer.py   # Fully connected layer 
│   │   ├── activations.py    # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── objective_functions.py  # CrossEntropy, MSE
│   │   └── optimizers.py     # SGD, Momentum, NAG, RMSProp
│   └── utils/
│       └── data_loader.py    # Dataset loading & preprocessing
├── models/
└── notebooks/
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy` — core computations
- `tensorflow` or `keras` — for `keras.datasets` (MNIST, Fashion-MNIST)
- `scikit-learn` — metrics (accuracy, precision, recall, F1, confusion matrix)
- `wandb` — experiment tracking 

---

## Usage

### Training

Train the model with full CLI configuration:

```bash
cd src
python train.py -d mnist -e 30 -b 128 -l cross_entropy -o rmsprop \
  -lr 0.0016 -wd 0.0001 -nhl 3 -sz 128 128 128 \
  -a tanh -w_i xavier -w_p da6401_a1_
```

### Inference

Load a trained model and evaluate on the test set:

```bash
cd src
python inference.py --model_path best_model.npy -d mnist
```

The inference script loads `best_config.json` by default to match the model architecture. Can override with CLI arguments if needed.

---

## Command-Line Arguments

Both `train.py` and `inference.py` share the same CLI interface for compatibility.

| Argument | Short | Description | Default |
|----------|-------|--------------|---------|
| `--dataset` | `-d` | Dataset: `mnist` or `fashion_mnist` | `mnist` |
| `--epochs` | `-e` | Number of training epochs | `30` |
| `--batch_size` | `-b` | Mini-batch size | `128` |
| `--loss` | `-l` | Loss: `cross_entropy` or `mse` | `cross_entropy` |
| `--optimizer` | `-o` | Optimizer: `sgd`, `momentum`, `nag`, `rmsprop` | `rmsprop` |
| `--learning_rate` | `-lr` | Initial learning rate | `0.001553536703042097` |
| `--weight_decay` | `-wd` | L2 regularization coefficient | `0.0001` |
| `--num_layers` | `-nhl` | Number of hidden layers | `3` |
| `--hidden_size` | `-sz` | Neurons per hidden layer (list, e.g. `128 128 64`) | `128 128 128` |
| `--activation` | `-a` | Activation: `sigmoid`, `tanh`, `relu` | `tanh` |
| `--weight_init` | `-w_i` | Weight init: `random` or `xavier` | `xavier` |
| `--wandb_project` | `-w_p` | Weights & Biases project ID | `da6401_assignment1` |


---

## Implementation Details

### Model Output
The model returns **logits** (linear combination from the output layer), not softmax-activated outputs or one-hot encoding.

### Gradient Computation
- The `backward()` method computes and stores gradients in each layer.
- Each layer exposes `self.grad_W` and `self.grad_b` after every backward pass.
- Gradients flow from the last layer to the first.

### Model Save/Load
```python
from ann.neural_network import NeuralNetwork
import numpy as np

def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data

model = NeuralNetwork(cli_args)
weights = load_model(args.model_path)
model.set_weights(weights)

best_weights = model.get_weights()
np.save("best_model.npy", best_weights)
```

### Inference Metrics
The inference script outputs:
- **Accuracy**
- **Precision** (macro)
- **Recall** (macro)
- **F1-Score** (macro)
- **Loss**
- Per-class classification report and confusion matrix

---


## Best Model Configuration

The best model (selected by **test F1-score**) uses the following configuration:

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
| Epochs | 30 |
| Batch size | 128 |
| Loss | Cross Entropy |
| Optimizer | RMSProp |
| Learning rate | 0.001553536703042097 |
| Weight decay | 0.0001 |
| Hidden layers | 3 |
| Hidden sizes | [128, 128, 128] |
| Activation | Tanh |
| Weight init | Xavier |

**Reported performance:** Test Accuracy ≈ 97.87%, Test F1-Score ≈ 97.85%

---


