"""
Dataset Loaders for UCF-Crime and XD-Violence with CLIP Features.
Supports automatic Kaggle path resolution, normal/abnormal partitioning,
and uniform snippet sampling.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils.tools import process_feat, process_split

_PATH_CACHE = {}


def resolve_npy_path(path_str: str) -> str:
    """
    Resolves relative or alien paths in CSVs by searching
    Kaggle inputs, working dirs, and local workspaces.
    """
    if os.path.exists(path_str):
        return path_str

    if path_str in _PATH_CACHE:
        return _PATH_CACHE[path_str]

    basename = os.path.basename(str(path_str).replace("\\", "/"))

    # Quick scan on cache miss
    if not _PATH_CACHE:
        search_roots = ["/kaggle/input", "/kaggle/working", ".", ".."]
        for root_dir in search_roots:
            if os.path.exists(root_dir):
                for r, _, files in os.walk(root_dir, followlinks=False):
                    for f in files:
                        if f.endswith(".npy"):
                            fp = os.path.join(r, f)
                            _PATH_CACHE[f] = fp
                            # Normalize potential symbol variations
                            for char1, char2 in [("#", "+"), ("+", "#"), ("#", "_"), ("+", "_")]:
                                _PATH_CACHE[f.replace(char1, char2)] = fp

    if basename in _PATH_CACHE:
        return _PATH_CACHE[basename]

    for alt in [
        basename.replace("#", "+"),
        basename.replace("+", "#"),
        basename.replace("#", "_"),
        basename.replace("+", "_"),
        basename.replace("#", ""),
        basename.replace("+", ""),
    ]:
        if alt in _PATH_CACHE:
            return _PATH_CACHE[alt]

    return path_str


class UCFDataset(Dataset):
    """
    DataLoader for UCF-Crime Video Features (CLIP ViT-B/16, 512-d).
    Can filter normal (Y=0) or abnormal (Y=1) subsets for balanced training.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not test_mode:
            if normal:
                self.df = self.df.loc[self.df["label"] == "Normal"].reset_index(drop=True)
            else:
                self.df = self.df.loc[self.df["label"] != "Normal"].reset_index(drop=True)

        if len(self.df) > 0 and not os.path.exists(str(self.df.loc[0, "path"])):
            self.df["path"] = [resolve_npy_path(p) for p in self.df["path"]]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]["path"]
        label = self.df.loc[index]["label"]

        clip_feature = np.load(path)

        if not self.test_mode:
            feat, length = process_feat(clip_feature, self.clip_dim)
            return torch.tensor(feat, dtype=torch.float32), label, length
        else:
            feat, length = process_split(clip_feature, self.clip_dim)
            return torch.tensor(feat, dtype=torch.float32), label, length, path


class XDDataset(Dataset):
    """
    DataLoader for XD-Violence Video Features (CLIP ViT-B/16, 512-d).
    Can filter normal (Y=0) or abnormal (Y=1) subsets for balanced training.
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not test_mode:
            is_normal = self.df["label"].astype(str).str.strip().isin(["A", "Normal", "normal"])
            if normal:
                self.df = self.df.loc[is_normal].reset_index(drop=True)
            else:
                self.df = self.df.loc[~is_normal].reset_index(drop=True)

        if len(self.df) > 0 and not os.path.exists(str(self.df.loc[0, "path"])):
            self.df["path"] = [resolve_npy_path(p) for p in self.df["path"]]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]["path"]
        label = self.df.loc[index]["label"]

        clip_feature = np.load(path)

        if not self.test_mode:
            feat, length = process_feat(clip_feature, self.clip_dim)
            return torch.tensor(feat, dtype=torch.float32), label, length
        else:
            feat, length = process_split(clip_feature, self.clip_dim)
            return torch.tensor(feat, dtype=torch.float32), label, length, path
