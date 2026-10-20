"""
Evaluation script for CRANE-VAD on UCF-Crime test set.
Computes frame-level ROC-AUC and Average Precision (AP).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Support both direct script execution and package imports
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_CURR_DIR)
for _p in [_REPO_DIR, _CURR_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from model import CRANE_VAD
    from dataset import UCFDataset
    from config import get_parser
    from text_prompts import get_text_embeddings
    from utils.eval_utils import eval_ucf_auc
except ImportError:
    from src.model import CRANE_VAD
    from src.dataset import UCFDataset
    from src.config import get_parser
    from src.text_prompts import get_text_embeddings
    from src.utils.eval_utils import eval_ucf_auc


def evaluate_ucf(model, test_loader, device, gt_path=None):
    """
    Evaluates CRANE-VAD on UCF-Crime test videos.
    """
    model.eval()
    all_scores = []

    # Unwrap DataParallel if necessary
    raw_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0)  # (N_chunks, T_snippet, 512) or (T_snippet, 512)
            length = int(item[2])

            if visual.ndim == 2:
                visual = visual.unsqueeze(0)  # (1, T, 512)

            visual = visual.to(device)

            out = model(visual)
            scores = out["S_final"].reshape(-1)  # Flatten all chunks
            video_scores = scores[:length].cpu().numpy()
            all_scores.append(video_scores)

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"  [Eval UCF] Processed video [{i+1:03d}/{len(test_loader):03d}]", flush=True)

    if gt_path is not None and os.path.exists(gt_path):
        gt = np.load(gt_path)
        auc, ap = eval_ucf_auc(all_scores, gt)
        print(f"\n" + "=" * 60)
        print(f"  [UCF-CRIME RESULTS] ROC-AUC: {auc:.2f}% | AP: {ap:.2f}%")
        print("=" * 60 + "\n", flush=True)
        return auc, ap
    else:
        print(f"[Notice] GT path not found at {gt_path}. Skipping numeric score calculation.")
        return 0.0, 0.0


def main():
    parser = get_parser(dataset="ucf")
    parser.add_argument("--checkpoint", default="models/ucf/crane_ucf_best.pth", type=str)
    parser.add_argument("--model-path", dest="checkpoint", type=str, help="Alias for --checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[CRANE-VAD] Evaluating UCF-Crime on: {device}")

    # Build model
    model = CRANE_VAD(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        n_prototypes=args.n_prototypes,
        tau=args.tau,
        ema_momentum=args.ema_momentum,
        n_text_classes=14,
        text_scale=args.text_scale,
        dropout=args.dropout,
    )

    # Load text embeddings
    text_emb = get_text_embeddings(dataset="ucf", device=device)
    model.set_text_embeddings(text_emb)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    else:
        print(f"[Warning] Checkpoint {args.checkpoint} not found. Running with initial weights.")

    model = model.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)

    test_dataset = UCFDataset(clip_dim=args.t_max, file_path=args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    evaluate_ucf(model, test_loader, device, gt_path=args.gt_path)


if __name__ == "__main__":
    main()
