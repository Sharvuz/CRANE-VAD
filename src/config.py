"""
Configuration and Command-Line Argument Parser for CRANE-VAD.
"""

import argparse


def get_parser(dataset="ucf"):
    parser = argparse.ArgumentParser(description=f"CRANE-VAD Training and Testing on {dataset.upper()}")

    # Dataset & Paths
    parser.add_argument("--dataset", default=dataset, type=str, choices=["ucf", "xd"], help="Dataset name")
    parser.add_argument("--train-list", default=f"list/{dataset}_CLIP_rgb.csv", type=str, help="Path to train CSV")
    parser.add_argument("--test-list", default=f"list/{dataset}_CLIP_rgbtest.csv", type=str, help="Path to test CSV")
    parser.add_argument("--gt-path", default=f"list/gt_ucf.npy" if dataset == "ucf" else "list/gt.npy", type=str)
    parser.add_argument("--model-save-dir", default=f"models/{dataset}", type=str, help="Checkpoint output folder")
    parser.add_argument("--log-dir", default=f"logs/{dataset}", type=str, help="Log folder")

    # Architecture
    parser.add_argument("--d-model", default=512, type=int, help="CLIP feature dimension")
    parser.add_argument("--d-state", default=64, type=int, help="S4D state dimension")
    parser.add_argument("--n-layers", default=2, type=int, help="Number of BiS4D layers")
    parser.add_argument("--n-prototypes", default=16, type=int, help="Number of normal prototypes K")
    parser.add_argument("--tau", default=10.0, type=float, help="Attention temperature for prototype bank")
    parser.add_argument("--ema-momentum", default=0.996, type=float, help="EMA momentum for prototypes")
    parser.add_argument("--text-scale", default=10.0, type=float, help="Scaling factor for residual-to-text cosine")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")

    # Temporal sampling
    parser.add_argument("--t-max", default=200, type=int, help="Snippet length per video snippet during training")

    # Loss Hyperparameters
    parser.add_argument("--lambda-1", default=0.1, type=float, dest="lambda_1", help="Weight for compact loss")
    parser.add_argument("--lambda-2", default=0.1, type=float, dest="lambda_2", help="Weight for orthogonality loss")
    parser.add_argument("--lambda-3", default=0.1, type=float, dest="lambda_3", help="Weight for separation loss")
    parser.add_argument("--lambda-4", default=0.5, type=float, dest="lambda_4", help="Weight for semantic CE loss")
    parser.add_argument("--lambda-5", default=0.01, type=float, dest="lambda_5", help="Weight for evidential loss")
    parser.add_argument("--lambda-6", default=0.01, type=float, dest="lambda_6", help="Weight for smoothness/sparsity")
    parser.add_argument("--margin-eta", default=0.3, type=float, help="Separation margin for abnormal reconstruction")
    parser.add_argument("--top-k-ratio", default=0.1, type=float, help="Top-K ratio for MIL pooling")

    # Training parameters
    parser.add_argument("--batch-size", default=64 if dataset == "ucf" else 96, type=int, help="Batch size (total)")
    parser.add_argument("--epochs", default=50 if dataset == "ucf" else 30, type=int, help="Total training epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="Base learning rate for backbone and heads")
    parser.add_argument("--lr-proto", default=5e-4, type=float, help="Learning rate for normal prototype bank")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--eval-freq", default=5, type=int, help="Evaluation frequency in epochs")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader num workers")
    parser.add_argument("--device", default="cuda", type=str, help="Device to run on (cuda or cpu)")

    return parser
