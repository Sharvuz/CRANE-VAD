"""
Evaluation script for CRANE-VAD on XD-Violence test set.
Computes frame-level Average Precision (AP) and ROC-AUC.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CRANE_VAD
from .dataset import XDDataset
from .config import get_parser
from .text_prompts import get_text_embeddings
from .utils.eval_utils import eval_xd_ap


def evaluate_xd(model, test_loader, device, gt_path=None):
    """
    Evaluates CRANE-VAD on XD-Violence test videos.
    """
    model.eval()
    all_scores = []

    with torch.no_grad():
        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0)
            length = int(item[2])

            if visual.ndim == 2:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            out = model(visual)
            scores = out["S_final"].reshape(-1)
            video_scores = scores[:length].cpu().numpy()
            all_scores.append(video_scores)

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"  [Eval XD] Processed video [{i+1:03d}/{len(test_loader):03d}]", flush=True)

    if gt_path is not None and os.path.exists(gt_path):
        gt = np.load(gt_path)
        ap, auc = eval_xd_ap(all_scores, gt)
        print(f"\n" + "=" * 60)
        print(f"  [XD-VIOLENCE RESULTS] Average Precision (AP): {ap:.2f}% | ROC-AUC: {auc:.2f}%")
        print("=" * 60 + "\n", flush=True)
        return ap, auc
    else:
        print(f"[Notice] GT path not found at {gt_path}. Skipping numeric score calculation.")
        return 0.0, 0.0


def main():
    parser = get_parser(dataset="xd")
    parser.add_argument("--checkpoint", default="models/xd/crane_xd_best.pth", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[CRANE-VAD] Evaluating XD-Violence on: {device}")

    model = CRANE_VAD(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        n_prototypes=args.n_prototypes,
        tau=args.tau,
        ema_momentum=args.ema_momentum,
        n_text_classes=7,
        text_scale=args.text_scale,
        dropout=args.dropout,
    )

    text_emb = get_text_embeddings(dataset="xd", device=device)
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

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)

    test_dataset = XDDataset(clip_dim=args.t_max, file_path=args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    evaluate_xd(model, test_loader, device, gt_path=args.gt_path)


if __name__ == "__main__":
    main()
