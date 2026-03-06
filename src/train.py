"""
Main Training Script
Entry point for training a NumPy MLP on MNIST or Fashion-MNIST.

Usage example:
    python train.py -d mnist -e 20 -b 64 -l cross_entropy -o rmsprop \\
                   -lr 0.001 -wd 0.0005 -nhl 3 -sz 128 128 64 \\
                   -a relu -w_i xavier -w_p da6401_a1_
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset



def parse_arguments():
    """
    Parse all command-line arguments for training.
    """
    parser = argparse.ArgumentParser(
        description="Train a NumPy MLP on MNIST or Fashion-MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # dataset & training
    parser.add_argument(
        "-d", "--dataset",
        type=str, default="fashion_mnist",
        choices=["mnist", "fashion_mnist"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int, default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int, default=64,
        help="Mini-batch size.",
    )

    # loss & optimization
    parser.add_argument(
        "-l", "--loss",
        type=str, default="cross_entropy",
        choices=["cross_entropy", "mse"],
        help="Loss / objective function.",
    )
    parser.add_argument(
        "-o", "--optimizer",
        type=str, default="rmsprop",
        choices=["sgd", "momentum", "nag", "rmsprop"],
        help="Gradient descent optimizer.",
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float, default=0.001,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "-wd", "--weight_decay",
        type=float, default=0.0005,
        help="L2 regularization coefficient (weight decay).",
    )

    # model architecture
    parser.add_argument(
        "-nhl", "--num_layers",
        type=int, default=3,
        help="Number of HIDDEN layers (excluding output layer).",
    )
    parser.add_argument(
        "-sz", "--hidden_size",
        type=int, nargs="+", default=[128, 128, 64],
        help=(
            "Neurons per hidden layer. "
            "Provide a list (e.g. -sz 128 128 64) or a single value "
            "repeated for all hidden layers."
        ),
    )
    parser.add_argument(
        "-a", "--activation",
        type=str, default="relu",
        choices=["relu", "sigmoid", "tanh"],
        help="Activation function for all hidden layers.",
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        type=str, default="xavier",
        choices=["xavier", "random"],
        help="Weight initialisation strategy.",
    )

    # wandb logging
    parser.add_argument(
        "-w_p", "--wandb_project",
        type=str, default="da6401_assignment1",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str, default=None,
        help="W&B entity (username or team) — leave empty to use default.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )

    # misc
    parser.add_argument(
        "--model_path",
        type=str, default="last_trained_model.npy",
        help="Relative path to save / load the trained model weights.",
    )
    parser.add_argument(
        "--config_path",
        type=str, default="last_trained_config.json",
        help="Relative path to save the best hyperparameter config.",
    )
    parser.add_argument(
        "--val_split",
        type=float, default=0.1,
        help="Fraction of training data to reserve as validation set.",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()

# initialise W&B run with config logging
def init_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "dataset":       args.dataset,
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "loss":          args.loss,
                "optimizer":     args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay":  args.weight_decay,
                "num_layers":    args.num_layers,
                "hidden_size":   args.hidden_size,
                "activation":    args.activation,
                "weight_init":   args.weight_init,
            },
        )
        print(f"[W&B] Run URL: {run.url}")
        return run
    except Exception as e:
        print(f"[W&B] Could not initialise wandb: {e}. Continuing without logging.")
        return None


# compute metrics using sklearn for convenience 
def compute_full_metrics(model, X, y):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    result = model.evaluate(X, y)
    preds = result["predictions"]
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average="macro", zero_division=0)
    rec = recall_score(y, preds, average="macro", zero_division=0)
    f1 = f1_score(y, preds, average="macro", zero_division=0)

    return {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "loss":      result["loss"],
        "predictions": preds,
        "logits":    result["logits"],
    }


# main training loop
def main():
    """
    Main training function.

    1. Parse CLI args
    2. Set random seed
    3. Load + preprocess data
    4. Build NeuralNetwork
    5. Optionally initialise W&B
    6. Train with wandb logging
    7. Evaluate on test set → print metrics
    8. Save best model weights + config
    """
    args = parse_arguments()

    # Reproducibility
    np.random.seed(args.seed)

    # data loading
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_dataset(
        args.dataset, val_split=args.val_split, seed=args.seed
    )

    # model 
    model = NeuralNetwork(args)
    print(
        f"\nModel architecture: 784 → "
        + " → ".join(str(s) for s in (
            args.hidden_size if isinstance(args.hidden_size, list) else [args.hidden_size]
        ))
        + f" → 10\n"
        f"  activation   : {args.activation}\n"
        f"  weight_init  : {args.weight_init}\n"
        f"  optimizer    : {args.optimizer}  (lr={args.learning_rate}, wd={args.weight_decay})\n"
        f"  loss         : {args.loss}\n"
        f"  batch_size   : {args.batch_size}  epochs: {args.epochs}\n"
    )

    # wandb logging
    wandb_run = init_wandb(args)

    # training
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        wandb_run=wandb_run,
    )

    # test evaluation
    print("\n--- Test Set Evaluation ---")
    test_metrics = compute_full_metrics(model, X_test, y_test)
    print(
        f"  Accuracy  : {test_metrics['accuracy']:.4f}\n"
        f"  Precision : {test_metrics['precision']:.4f}\n"
        f"  Recall    : {test_metrics['recall']:.4f}\n"
        f"  F1-Score  : {test_metrics['f1']:.4f}\n"
        f"  Loss      : {test_metrics['loss']:.4f}"
    )

    if wandb_run is not None:
        wandb_run.log({
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_f1":        test_metrics["f1"],
            "test_loss":      test_metrics["loss"],
        })

    # save model
    save_dir = os.path.dirname(os.path.abspath(args.model_path)) or "."
    os.makedirs(save_dir, exist_ok=True)

    best_weights = model.get_weights()
    np.save(args.model_path, best_weights)
    print(f"\n[Saved] Model weights → {args.model_path}")

    # save config
    config = {
        "dataset":       args.dataset,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "loss":          args.loss,
        "optimizer":     args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay":  args.weight_decay,
        "num_layers":    args.num_layers,
        "hidden_size":   (
            args.hidden_size if isinstance(args.hidden_size, list)
            else [args.hidden_size]
        ),
        "activation":    args.activation,
        "weight_init":   args.weight_init,
        "test_f1":       test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
    }
    with open(args.config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Saved] Config          → {args.config_path}")

    if wandb_run is not None:
        wandb_run.finish()

    print("\nTraining complete!")
    return test_metrics


if __name__ == "__main__":
    main()
