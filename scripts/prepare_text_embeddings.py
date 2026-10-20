"""
Offline Pre-computation of CLIP Text Embeddings for UCF-Crime and XD-Violence.
Saves .pt files into the list/ directory so training can proceed offline.
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.text_prompts import UCF_TEXT_PROMPTS, XD_TEXT_PROMPTS, encode_text_prompts


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "list"))
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Prepare Text Embeddings] Using device: {device}")

    # 1. UCF-Crime
    print("\nEncoding 14 prompts for UCF-Crime...")
    ucf_emb = encode_text_prompts(UCF_TEXT_PROMPTS, device=device)
    ucf_path = os.path.join(out_dir, "ucf_text_embeddings.pt")
    torch.save(ucf_emb.cpu(), ucf_path)
    print(f"Saved UCF text embeddings to: {ucf_path} (shape: {tuple(ucf_emb.shape)})")

    # 2. XD-Violence
    print("\nEncoding 7 prompts for XD-Violence...")
    xd_emb = encode_text_prompts(XD_TEXT_PROMPTS, device=device)
    xd_path = os.path.join(out_dir, "xd_text_embeddings.pt")
    torch.save(xd_emb.cpu(), xd_path)
    print(f"Saved XD text embeddings to: {xd_path} (shape: {tuple(xd_emb.shape)})")

    print("\nDone! Text embeddings are ready for CRANE-VAD training.")


if __name__ == "__main__":
    main()
