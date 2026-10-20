"""
CLIP Text Prompts and Text Embedding Generators for UCF-Crime and XD-Violence.
Enables Counterfactual Residual-to-Text Alignment without background interference.
"""

import os
import torch
import torch.nn.functional as F

# UCF-Crime: 14 classes (Class 0: Normal, Class 1-13: Crimes)
UCF_TEXT_PROMPTS = [
    "a surveillance video of normal daily activity",        # Class 0: Normal
    "a surveillance video of abuse and violence",           # Class 1: Abuse
    "a surveillance video of an arrest by police",          # Class 2: Arrest
    "a surveillance video of arson and fire",               # Class 3: Arson
    "a surveillance video of assault and fighting",         # Class 4: Assault
    "a surveillance video of an accident or crash",         # Class 5: Accident
    "a surveillance video of a burglary break-in",          # Class 6: Burglary
    "a surveillance video of an explosion",                 # Class 7: Explosion
    "a surveillance video of fighting in public",           # Class 8: Fighting
    "a surveillance video of robbery and theft",            # Class 9: Robbery
    "a surveillance video of shooting with guns",           # Class 10: Shooting
    "a surveillance video of shoplifting and stealing",     # Class 11: Shoplifting
    "a surveillance video of people stealing",              # Class 12: Stealing
    "a surveillance video of vandalism and destruction",    # Class 13: Vandalism
]

# XD-Violence: 7 classes (Class 0: Normal, Class 1-6: Violence types)
XD_TEXT_PROMPTS = [
    "a video showing normal peaceful activity",            # Class 0: Normal
    "a video showing a physical fight or brawl",           # Class 1: Fighting
    "a video showing a shooting incident",                 # Class 2: Shooting
    "a video showing a riot or mob violence",              # Class 3: Riot
    "a video showing abuse or bullying",                   # Class 4: Abuse
    "a video showing a car crash or accident",             # Class 5: Car crash
    "a video showing an explosion or bombing",             # Class 6: Explosion
]


def encode_text_prompts(prompts, clip_model=None, device="cpu"):
    """
    Encodes text prompts using OpenAI's CLIP text encoder.
    If clip_model is not provided, tries to load 'ViT-B/16'.
    """
    if clip_model is None:
        try:
            import clip
            clip_model, _ = clip.load("ViT-B/16", device=device)
        except Exception as e:
            print(f"[Warning] OpenAI CLIP not available ({e}). Creating random normalized text embeddings for testing.")
            emb = torch.randn(len(prompts), 512, device=device)
            return F.normalize(emb, dim=-1)

    import clip
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens).float()
        text_emb = F.normalize(text_emb, dim=-1)
    return text_emb


def get_text_embeddings(dataset="ucf", cache_dir="list", device="cpu"):
    """
    Loads pre-computed text embeddings from cache if available,
    otherwise computes them on-the-fly and saves to cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset}_text_embeddings.pt")

    if os.path.exists(cache_path):
        emb = torch.load(cache_path, map_location=device)
        return F.normalize(emb.float(), dim=-1)

    prompts = UCF_TEXT_PROMPTS if dataset.lower() == "ucf" else XD_TEXT_PROMPTS
    emb = encode_text_prompts(prompts, device=device)
    try:
        torch.save(emb.cpu(), cache_path)
    except Exception as e:
        print(f"[Notice] Could not write cache to {cache_path}: {e}")
    return emb
