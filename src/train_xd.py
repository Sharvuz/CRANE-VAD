"""
Complete Training Script for CRANE-VAD on XD-Violence from Scratch.
Optimized for Dual Tesla T4 (DataParallel) on Kaggle with high batch size.
Target: Frame-level AP >= 86.5%.
"""

import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

# Support both direct script execution and package imports
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_CURR_DIR)
for _p in [_REPO_DIR, _CURR_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from model import CRANE_VAD
    from losses import compute_total_loss
    from dataset import XDDataset
    from config import get_parser
    from text_prompts import get_text_embeddings
    from test_xd import evaluate_xd
except ImportError:
    from src.model import CRANE_VAD
    from src.losses import compute_total_loss
    from src.dataset import XDDataset
    from src.config import get_parser
    from src.text_prompts import get_text_embeddings
    from src.test_xd import evaluate_xd


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_prototypes_kmeans(model, normal_loader, device, n_prototypes=16):
    """
    K-Means initialization on normal features for XD-Violence.
    Falls back gracefully to random normal samples if KMeans fails.
    """
    print(f"  [Init] Gathering normal snippet features for prototype initialization (K={n_prototypes})...", flush=True)
    features_list = []
    max_samples = 50000

    with torch.no_grad():
        for i, (feat, _, length) in enumerate(normal_loader):
            B, T, D = feat.shape
            feat_flat = feat.reshape(-1, D).numpy()
            features_list.append(feat_flat)
            if sum(len(f) for f in features_list) > max_samples:
                break

    all_features = np.concatenate(features_list, axis=0)

    try:
        from sklearn.cluster import KMeans
        print(f"  [Init] Running K-Means on {len(all_features)} normal snippets...", flush=True)
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42, n_init=5)
        kmeans.fit(all_features)
        init_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    except Exception as e:
        print(f"  [Notice] K-Means initialization skipped ({e}). Sampling random normal snippets directly...", flush=True)
        indices = np.random.choice(len(all_features), n_prototypes, replace=False)
        init_centers = torch.tensor(all_features[indices], dtype=torch.float32)

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.lgam.prototypes.data.copy_(init_centers)
    raw_model.lgam.proto_ema.copy_(init_centers)
    raw_model.lgam.ema_initialized.fill_(True)
    print("  [Init] Normal prototype memory initialized successfully!", flush=True)


def train():
    parser = get_parser(dataset="xd")
    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"\n" + "=" * 65)
    print(f"  CRANE-VAD: TRAINING ON XD-VIOLENCE FROM SCRATCH")
    print(f"  Device: {device} | Number of GPUs: {num_gpus}")
    print(f"  Batch Size: {args.batch_size} | Epochs: {args.epochs} | Base LR: {args.lr}")
    print("=" * 65 + "\n", flush=True)

    batch_half = max(1, args.batch_size // 2)
    normal_dataset = XDDataset(clip_dim=args.t_max, file_path=args.train_list, normal=True)
    abnormal_dataset = XDDataset(clip_dim=args.t_max, file_path=args.train_list, normal=False)

    normal_loader = DataLoader(
        normal_dataset, batch_size=batch_half, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    abnormal_loader = DataLoader(
        abnormal_dataset, batch_size=batch_half, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )

    test_dataset = XDDataset(clip_dim=args.t_max, file_path=args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # XD uses 7 text classes
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

    text_emb = get_text_embeddings(dataset="xd", cache_dir="list", device=device)
    model.set_text_embeddings(text_emb)

    init_prototypes_kmeans(model, normal_loader, device=device, n_prototypes=args.n_prototypes)

    model = model.to(device)

    if num_gpus > 1:
        print(f"🔥 Wrapping model with nn.DataParallel across {num_gpus} GPUs (Dual T4 mode)!")
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, "module") else model

    backbone_params = list(raw_model.temporal_encoder.parameters())
    head_params = list(raw_model.det_head.parameters()) + list(raw_model.r2t_head.parameters())
    memory_params = [raw_model.lgam.prototypes]
    other_params = [p for p in raw_model.lgam.parameters() if p is not raw_model.lgam.prototypes]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": memory_params, "lr": args.lr_proto, "weight_decay": 0.0},
        {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_ap = 0.0
    best_auc = 0.0

    steps_per_epoch = min(len(normal_loader), len(abnormal_loader))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_losses = {"total": 0.0, "mil": 0.0, "compact": 0.0, "sem": 0.0, "rec": 0.0}

        nor_iter = iter(normal_loader)
        abn_iter = iter(abnormal_loader)

        for step in range(steps_per_epoch):
            nor_feat, _, _ = next(nor_iter)
            abn_feat, _, _ = next(abn_iter)

            X = torch.cat([nor_feat, abn_feat], dim=0).to(device)
            Y = torch.cat([
                torch.zeros(batch_half, dtype=torch.long),
                torch.ones(batch_half, dtype=torch.long)
            ], dim=0).to(device)

            output = model(X, Y)

            loss, loss_dict = compute_total_loss(output, Y, model, args)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in running_losses:
                if k in loss_dict:
                    running_losses[k] += loss_dict[k]

        scheduler.step()
        epoch_time = time.time() - epoch_start

        avg_loss = running_losses["total"] / steps_per_epoch
        gw = output["gate_weights"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
            f"Loss: {avg_loss:.4f} (MIL: {running_losses['mil']/steps_per_epoch:.3f}, "
            f"Comp: {running_losses['compact']/steps_per_epoch:.3f}, "
            f"Sem: {running_losses['sem']/steps_per_epoch:.3f}) | "
            f"Gate: [Det: {gw['w_det']:.2f}, Rec: {gw['w_rec']:.2f}, Sem: {gw['w_sem']:.2f}]",
            flush=True
        )

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            ap, auc = evaluate_xd(model, test_loader, device=device, gt_path=args.gt_path)

            if ap > best_ap:
                best_ap = ap
                best_auc = auc
                save_path = os.path.join(args.model_save_dir, "crane_xd_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "ap": best_ap,
                    "auc": best_auc,
                }, save_path)
                print(f"  ⭐ New Best Model Saved to {save_path} (AP: {best_ap:.2f}%)!", flush=True)

            latest_path = os.path.join(args.model_save_dir, "crane_xd_latest.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "ap": ap,
                "auc": auc,
            }, latest_path)

    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETED! Best XD-Violence AP: {best_ap:.2f}% | ROC-AUC: {best_auc:.2f}%")
    print("=" * 65 + "\n", flush=True)


if __name__ == "__main__":
    train()
