"""
Evaluation utilities for UCF-Crime and XD-Violence.
Computes frame-level ROC-AUC and Average Precision (AP).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def eval_ucf_auc(scores_list, gt):
    """
    Computes frame-level ROC-AUC and AP on UCF-Crime.
    
    Args:
        scores_list: List of 1D numpy arrays, snippet-level anomaly scores per video.
        gt: 1D numpy array of frame-level ground-truth labels (0 or 1).
    
    Returns:
        auc: Frame-level ROC-AUC percentage (0-100)
        ap: Frame-level Average Precision percentage (0-100)
    """
    all_scores_flat = np.concatenate(scores_list, axis=0)
    frame_scores = np.repeat(all_scores_flat, 16)
    
    length = min(len(gt), len(frame_scores))
    auc = roc_auc_score(gt[:length], frame_scores[:length]) * 100.0
    ap = average_precision_score(gt[:length], frame_scores[:length]) * 100.0
    return auc, ap


def eval_xd_ap(scores_list, gt):
    """
    Computes frame-level Average Precision (AP) and ROC-AUC on XD-Violence.
    
    Args:
        scores_list: List of 1D numpy arrays, snippet-level anomaly scores per video.
        gt: 1D numpy array of frame-level ground-truth labels (0 or 1).
    
    Returns:
        ap: Frame-level Average Precision percentage (0-100)
        auc: Frame-level ROC-AUC percentage (0-100)
    """
    all_scores_flat = np.concatenate(scores_list, axis=0)
    frame_scores = np.repeat(all_scores_flat, 16)
    
    length = min(len(gt), len(frame_scores))
    ap = average_precision_score(gt[:length], frame_scores[:length]) * 100.0
    auc = roc_auc_score(gt[:length], frame_scores[:length]) * 100.0
    return ap, auc
