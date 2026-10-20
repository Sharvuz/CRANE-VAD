"""
Complete Loss Function Suite for CRANE-VAD.
Enforces non-saturating MIL ranking, Label-Gated compactness on normal features,
prototype orthogonality, counterfactual separation, and evidential uncertainty calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mil_loss(S_det, Y, top_k_ratio: float = 0.1):
    """
    Non-saturating BCE on Top-K snippet mean scores.
    Args:
        S_det: Anomaly score predictions (B, T)
        Y: Video-level labels (B,) [0=normal, 1=abnormal]
        top_k_ratio: Fraction of top snippets to pool (default 0.1)
    """
    B, T = S_det.shape
    k = max(1, int(T * top_k_ratio))

    topk_scores, _ = torch.topk(S_det, k, dim=1)  # (B, k)
    video_scores = topk_scores.mean(dim=1)  # (B,)

    loss = F.binary_cross_entropy(video_scores.clamp(1e-6, 1.0 - 1e-6), Y.float())
    return loss


def compact_loss(H, prototypes, Y):
    """
    Compactness Loss: Pulls Y=0 (normal) video features toward nearest normal prototype.
    Strictly applied ONLY to normal videos to prevent prototype contamination.
    """
    normal_mask = (Y == 0)
    if normal_mask.sum() == 0:
        return torch.tensor(0.0, device=H.device)

    H_normal = H[normal_mask]  # (B_nor, T, D)
    P = F.normalize(prototypes, dim=-1)  # (K, D)

    # Flatten snippets
    H_flat = H_normal.reshape(-1, H_normal.size(-1))  # (B_nor*T, D)
    dist = torch.cdist(H_flat, P)  # (B_nor*T, K)
    min_dist, _ = dist.min(dim=-1)
    return min_dist.pow(2).mean()


def orth_loss(prototypes):
    """
    Orthogonality Loss: Enforces diversity among normal prototypes
    and prevents representation collapse.
    """
    P = F.normalize(prototypes, dim=-1)  # (K, D)
    gram = P @ P.T  # (K, K)
    I = torch.eye(P.size(0), device=P.device)
    return (gram - I).pow(2).sum()


def separation_loss(S_rec, Y, margin: float = 0.3, top_k_ratio: float = 0.1):
    """
    Separation Loss:
      - Normal videos (Y=0): Minimizes reconstruction error (S_rec -> 0)
      - Abnormal videos (Y=1): Pushes top-k reconstruction error above margin
    """
    B, T = S_rec.shape
    k = max(1, int(T * top_k_ratio))

    normal_mask = (Y == 0)
    abnormal_mask = (Y == 1)

    loss = torch.tensor(0.0, device=S_rec.device)

    if normal_mask.sum() > 0:
        loss = loss + S_rec[normal_mask].mean()

    if abnormal_mask.sum() > 0:
        S_abn = S_rec[abnormal_mask]
        topk_abn, _ = torch.topk(S_abn, k, dim=1)
        loss = loss + F.relu(margin - topk_abn.mean())

    return loss


def semantic_loss(logits_sem, S_det, Y, top_k_ratio: float = 0.1):
    """
    Semantic Cross-Entropy Loss:
      - All snippets from Y=0 videos: Aligned to Class 0 ("normal activity")
      - Top-K snippets from Y=1 videos: Pushed away from Class 0 (crime classes)
    """
    B, T, C = logits_sem.shape
    k = max(1, int(T * top_k_ratio))
    device = logits_sem.device

    loss = torch.tensor(0.0, device=device)
    count = 0

    for i in range(B):
        if Y[i] == 0:
            # Normal video: all snippets should predict Class 0
            targets = torch.zeros(T, dtype=torch.long, device=device)
            loss = loss + F.cross_entropy(logits_sem[i], targets)
            count += 1
        else:
            # Abnormal video: top-k snippets (by detached detector score) should NOT be normal
            _, topk_idx = torch.topk(S_det[i].detach(), k)
            topk_logits = logits_sem[i, topk_idx]  # (k, C)
            prob_normal = F.softmax(topk_logits, dim=-1)[:, 0]  # (k,)
            # Minimize probability of normal class on anomaly snippets
            loss = loss - torch.log(1.0 - prob_normal + 1e-6).mean()
            count += 1

    return loss / max(count, 1)


def evidential_reg_loss(alpha, beta, mu):
    """
    Regularization for Beta evidential heads:
    Penalizes overconfidence when predictions are ambiguous (mu ≈ 0.5).
    """
    ambiguity = 1.0 - (2.0 * mu - 1.0).abs()  # Peaks at mu=0.5
    kl_approx = (alpha - 1.0).pow(2) + (beta - 1.0).pow(2)
    return (ambiguity * kl_approx).mean()


def temporal_smooth_loss(S):
    """Encourages temporal smoothness between adjacent snippets."""
    return (S[:, 1:] - S[:, :-1]).pow(2).mean()


def sparsity_loss(S, Y):
    """Encourages sparse anomaly scores on normal videos to eliminate false alarms."""
    normal_mask = (Y == 0)
    if normal_mask.sum() == 0:
        return torch.tensor(0.0, device=S.device)
    return S[normal_mask].mean()


def compute_total_loss(model_output, Y, model, args):
    """
    Computes weighted total loss for CRANE-VAD.
    """
    S_det = model_output["S_det"]
    S_rec = model_output["S_rec"]
    S_sem = model_output["S_sem"]
    S_final = model_output["S_final"]
    H = model_output["H"]
    logits_sem = model_output["logits_sem"]
    beta_params = model_output["beta_params"]

    # Access base model (unwrapped if DataParallel)
    raw_model = model.module if hasattr(model, "module") else model
    prototypes = raw_model.lgam.prototypes

    # 1. Primary MIL ranking loss
    L_mil = mil_loss(S_final, Y, args.top_k_ratio)

    # 2. LGAM normal compactness loss (strictly Y=0)
    L_compact = compact_loss(H, prototypes, Y)

    # 3. Prototype orthogonality loss
    L_orth = orth_loss(prototypes)

    # 4. Separation loss on reconstruction branch
    L_sep = separation_loss(S_rec, Y, args.margin_eta, args.top_k_ratio)

    # 5. Semantic residual-to-text cross entropy loss
    L_sem = semantic_loss(logits_sem, S_det, Y, args.top_k_ratio)

    # 6. Evidential uncertainty regularization
    L_evid = torch.tensor(0.0, device=Y.device)
    for branch_name in ["det", "rec", "sem"]:
        if "beta_params" in model_output and branch_name in model_output["beta_params"]:
            alpha, beta_val = model_output["beta_params"][branch_name]
        else:
            alpha = model_output[f"alpha_{branch_name}"]
            beta_val = model_output[f"beta_{branch_name}"]
        mu = alpha / (alpha + beta_val)
        L_evid = L_evid + evidential_reg_loss(alpha, beta_val, mu)
    L_evid = L_evid / 3.0

    # 7. Smoothness & Sparsity
    L_smooth = temporal_smooth_loss(S_final)
    L_sparse = sparsity_loss(S_final, Y)

    # Total loss combination
    L_total = (
        L_mil
        + args.lambda_1 * L_compact
        + args.lambda_2 * L_orth
        + args.lambda_3 * L_sep
        + args.lambda_4 * L_sem
        + args.lambda_5 * L_evid
        + args.lambda_6 * (L_smooth + L_sparse)
    )

    loss_dict = {
        "total": L_total.item(),
        "mil": L_mil.item(),
        "compact": L_compact.item(),
        "orth": L_orth.item(),
        "sep": L_sep.item(),
        "sem": L_sem.item(),
        "evid": L_evid.item(),
        "smooth": L_smooth.item(),
        "sparse": L_sparse.item(),
    }

    return L_total, loss_dict
