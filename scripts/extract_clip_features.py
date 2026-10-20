"""
CLIP ViT-B/16 Video Feature Extraction Script.
Extracts 512-dimensional visual representations per 16-frame snippet.
"""

import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


def extract_features_from_video(video_path, model, preprocess, device, snippet_len=16):
    """
    Reads video frames using OpenCV, groups every snippet_len frames,
    and encodes them into normalized CLIP visual embeddings.
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required to extract features from video files.")

    cap = cv2.VideoCapture(video_path)
    frames = []
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        frames.append(preprocess(pil_img))

        if len(frames) == snippet_len:
            # Batch of frames for one snippet
            snippet_tensor = torch.stack(frames).to(device)  # (16, 3, 224, 224)
            with torch.no_grad():
                f = model.encode_image(snippet_tensor).float()
                # Snippet representation = average of frame embeddings
                snippet_feat = f.mean(dim=0, keepdim=True)
                snippet_feat = snippet_feat / snippet_feat.norm(dim=-1, keepdim=True)
                features.append(snippet_feat.cpu().numpy())
            frames = []

    cap.release()

    if len(frames) > 0:
        # Handle trailing frames
        snippet_tensor = torch.stack(frames).to(device)
        with torch.no_grad():
            f = model.encode_image(snippet_tensor).float()
            snippet_feat = f.mean(dim=0, keepdim=True)
            snippet_feat = snippet_feat / snippet_feat.norm(dim=-1, keepdim=True)
            features.append(snippet_feat.cpu().numpy())

    if len(features) == 0:
        return None

    return np.concatenate(features, axis=0)  # (T, 512)


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP ViT-B/16 Video Features")
    parser.add_argument("--video-dir", required=True, type=str, help="Folder containing video files (.mp4, .avi)")
    parser.add_argument("--out-dir", required=True, type=str, help="Folder to save .npy feature files")
    parser.add_argument("--snippet-len", default=16, type=int, help="Frames per snippet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    if not HAS_CLIP:
        raise ImportError("openai-clip must be installed: pip install git+https://github.com/openai/CLIP.git")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading CLIP ViT-B/16 on {args.device}...")
    model, preprocess = clip.load("ViT-B/16", device=args.device)
    model.eval()

    video_files = glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True) + \
                  glob.glob(os.path.join(args.video_dir, "**", "*.avi"), recursive=True)

    print(f"Found {len(video_files)} video files.")

    for i, v_path in enumerate(video_files):
        base_name = os.path.splitext(os.path.basename(v_path))[0]
        out_path = os.path.join(args.out_dir, f"{base_name}.npy")

        if os.path.exists(out_path):
            continue

        print(f"[{i+1}/{len(video_files)}] Processing: {base_name}...")
        feat = extract_features_from_video(v_path, model, preprocess, args.device, args.snippet_len)
        if feat is not None:
            np.save(out_path, feat)

    print("Feature extraction finished!")


if __name__ == "__main__":
    main()
