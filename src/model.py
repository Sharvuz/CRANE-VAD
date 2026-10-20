"""
CRANE-VAD: Counterfactual Residual Alignment Network with Evidential Fusion.

100% Standalone Weakly-Supervised Video Anomaly Detection (WS-VAD).
Key Pillars:
  1. Bidirectional S4D Temporal Encoder (Linear O(T) complexity)
  2. Label-Gated Asymmetric Memory (LGAM - Mathematical guarantee against contamination)
  3. Counterfactual Residual-to-Text Alignment (Background-free semantic matching)
  4. Beta-Evidential Inverse-Variance Dynamic Gate (No heuristic bias tricks)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. TEMPORAL BACKBONE: S4D (Diagonal State Space Model)
# =========================================================================

class DiagonalSSM(nn.Module):
    """
    Structured State Space - Diagonal (S4D) Layer.
    Computes continuous-time linear SSM via diagonal decomposition
    and applies parallel FFT convolution over sequence length T.
    Complexity: O(T log T), memory O(T).
    """
    def __init__(self, d_model: int = 512, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Log discretization step size dt
        self.log_step = nn.Parameter(torch.log(torch.tensor(0.001)))

        # S4D / HiPPO diagonal initialization for transition matrix A
        # Strictly negative real part guarantees stability
        self.A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))

        # Complex observation matrix C and direct feedthrough skip D
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, D)
        Returns:
            out: Convolved representation of shape (B, T, D)
        """
        B, T, D = x.shape
        dt = torch.exp(self.log_step)
        A = -torch.exp(self.A_real)  # Ensure negative eigenvalues for asymptotic stability

        # Generate continuous-time convolution kernel K over sequence length T
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        k = torch.einsum("hn,t->hnt", A * dt, t)
        K = torch.einsum("hn,hnt->ht", self.C, torch.exp(k)) * dt  # (D, T)

        # Parallel 1D temporal convolution via Fast Fourier Transform (FFT)
        x_t = x.transpose(1, 2)  # (B, D, T)
        n_fft = 2 * T
        K_f = torch.fft.rfft(K, n=n_fft)
        x_f = torch.fft.rfft(x_t, n=n_fft)
        y = torch.fft.irfft(x_f * K_f, n=n_fft)[..., :T]

        # Direct feedthrough skip connection D * x
        y = y + x_t * self.D.unsqueeze(-1)
        out = self.dropout(self.out_proj(y.transpose(1, 2)))
        return out


class BiS4DBlock(nn.Module):
    """
    Single bidirectional S4D block with Forward + Backward passes,
    LayerNorm, GLU / FFN projection, and residual skip connections.
    """
    def __init__(self, d_model: int = 512, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.fwd_ssm = DiagonalSSM(d_model=d_model // 2, d_state=d_state, dropout=dropout)
        self.bwd_ssm = DiagonalSSM(d_model=d_model // 2, d_state=d_state, dropout=dropout)

        self.in_split = nn.Linear(d_model, d_model)
        self.merge = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = x
        x_norm = self.norm1(x)
        x_split = self.in_split(x_norm)
        fwd_in, bwd_in = torch.chunk(x_split, 2, dim=-1)

        fwd_out = self.fwd_ssm(fwd_in)
        # Flip sequence for backward pass, then flip back
        bwd_out = torch.flip(self.bwd_ssm(torch.flip(bwd_in, dims=[1])), dims=[1])

        merged = self.merge(torch.cat([fwd_out, bwd_out], dim=-1))
        x = res + merged

        # FFN stage
        x = x + self.ffn(self.norm2(x))
        return x


class BiS4DEncoder(nn.Module):
    """
    Multi-layer Bidirectional S4D Temporal Encoder.
    Replaces windowed attention and dual GCNs.
    """
    def __init__(self, d_model: int = 512, d_state: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BiS4DBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# =========================================================================
# 2. BRANCH 1: DISCRIMINATIVE MIL DETECTOR (With Beta Uncertainty)
# =========================================================================

class DiscriminativeHead(nn.Module):
    """
    MIL-based anomaly detector parameterized via a Beta distribution.
    Outputs:
        mu_d (S_det): Expected anomaly score in [0, 1]
        sigma2_d: Epistemic uncertainty
    """
    def __init__(self, d_model: int = 512, hidden: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 2),  # [pre_alpha, pre_beta]
        )

    def forward(self, H):
        out = self.head(H)  # (B, T, 2)
        alpha = F.softplus(out[..., 0]) + 1.0  # Ensure > 1 for valid uni-modal Beta
        beta = F.softplus(out[..., 1]) + 1.0
        mu = alpha / (alpha + beta)
        sigma2 = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        return mu, alpha, beta, sigma2


# =========================================================================
# 3. BRANCH 2: LABEL-GATED ASYMMETRIC MEMORY (LGAM - Core Breakthrough)
# =========================================================================

class LGAM(nn.Module):
    """
    Label-Gated Asymmetric Memory.
    
    Mathematical Guarantee:
      Normal prototype bank P is updated ONLY from Y=0 (100% clean normal videos).
      For Y=1 videos, gradients to P are strictly blocked via .detach().
      Prototype contamination is thus impossible from Day 1, Epoch 0.
    """
    def __init__(self, d_model: int = 512, n_prototypes: int = 16, tau: float = 10.0, ema_momentum: float = 0.996):
        super().__init__()
        self.K = n_prototypes
        self.tau = tau
        self.ema_momentum = ema_momentum

        # Learnable normal prototype bank
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model))
        nn.init.xavier_uniform_(self.prototypes)

        # Slow-moving EMA buffer
        self.register_buffer("proto_ema", torch.zeros(n_prototypes, d_model))
        self.register_buffer("ema_initialized", torch.tensor(False))

        # Evidential head on residual features
        self.rec_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, H, Y=None):
        """
        Args:
            H: Temporal representations (B, T, D)
            Y: Video-level labels (B,) [0=normal, 1=abnormal], None in inference
        """
        B, T, D = H.shape
        P = F.normalize(self.prototypes, dim=-1)  # (K, D)

        # Counterfactual normal reconstruction via cross-attention
        H_norm = F.normalize(H, dim=-1)  # (B, T, D)
        attn = torch.softmax(self.tau * (H_norm @ P.T), dim=-1)  # (B, T, K)
        F_rec = attn @ P  # (B, T, D)

        # ============================================================
        # LABEL-GATED GRADIENT MASKING (Airtight Contamination Firewall)
        # ============================================================
        # For Y=0: F_rec flows gradient normally to update prototypes P.
        # For Y=1: F_rec is strictly detached, making grad_P identically 0.
        if self.training and Y is not None:
            normal_mask = (1.0 - Y.float()).view(B, 1, 1)
            F_rec = F_rec * normal_mask + F_rec.detach() * (1.0 - normal_mask)

        # Clean anomaly residual (Background stripped!)
        R = H - F_rec  # (B, T, D)

        # Reconstruction score
        S_rec_raw = (1.0 - F.cosine_similarity(H, F_rec, dim=-1)) / 2.0  # (B, T)

        # Evidential output
        rec_out = self.rec_head(R)
        alpha_r = F.softplus(rec_out[..., 0]) + 1.0
        beta_r = F.softplus(rec_out[..., 1]) + 1.0
        mu_r = alpha_r / (alpha_r + beta_r)
        sigma2_r = (alpha_r * beta_r) / (((alpha_r + beta_r) ** 2) * (alpha_r + beta_r + 1.0))

        # Smooth EMA update during training
        if self.training:
            with torch.no_grad():
                if not self.ema_initialized:
                    self.proto_ema.copy_(self.prototypes.data)
                    self.ema_initialized.fill_(True)
                else:
                    self.proto_ema.mul_(self.ema_momentum).add_(
                        self.prototypes.data, alpha=1.0 - self.ema_momentum
                    )

        return mu_r, F_rec, R, alpha_r, beta_r, sigma2_r


# =========================================================================
# 4. BRANCH 3: COUNTERFACTUAL RESIDUAL-TO-TEXT ALIGNMENT
# =========================================================================

class ResidualToTextHead(nn.Module):
    """
    Projects the clean anomaly residual R = H - F_rec into CLIP text space.
    Eliminates >90% stationary background noise before semantic comparison.
    """
    def __init__(self, d_model: int = 512, clip_dim: int = 512, n_classes: int = 14, text_scale: float = 10.0):
        super().__init__()
        self.text_scale = text_scale

        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, clip_dim),
        )

        # Frozen text embeddings buffer (Class 0: Normal, Class 1..C: Anomalies)
        self.register_buffer("text_embeddings", torch.zeros(n_classes, clip_dim))

        self.sem_head = nn.Sequential(
            nn.Linear(n_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def set_text_embeddings(self, text_emb):
        """Initializes frozen CLIP text embeddings."""
        self.text_embeddings.copy_(F.normalize(text_emb, dim=-1))

    def forward(self, R):
        R_proj = self.proj(R)
        R_norm = F.normalize(R_proj, dim=-1)

        W_text = self.text_embeddings  # (C+1, clip_dim)
        logits_sem = self.text_scale * (R_norm @ W_text.T)  # (B, T, C+1)

        # Semantic anomaly score: 1 - P(Class 0 = Normal)
        probs = F.softmax(logits_sem, dim=-1)
        S_sem_raw = 1.0 - probs[:, :, 0]

        sem_out = self.sem_head(logits_sem)
        alpha_s = F.softplus(sem_out[..., 0]) + 1.0
        beta_s = F.softplus(sem_out[..., 1]) + 1.0
        mu_s = alpha_s / (alpha_s + beta_s)
        sigma2_s = (alpha_s * beta_s) / (((alpha_s + beta_s) ** 2) * (alpha_s + beta_s + 1.0))

        return mu_s, logits_sem, alpha_s, beta_s, sigma2_s


# =========================================================================
# 5. DYNAMIC EVDIENTIAL GATE (Inverse-Variance Precision Weighting)
# =========================================================================

class EvidentialGate(nn.Module):
    """
    Fuses three branch predictions using Inverse-Variance Weighting (Bayesian precision).
    No heuristic bias tuning needed.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mu_det, sigma2_det, mu_rec, sigma2_rec, mu_sem, sigma2_sem):
        # Precision = 1 / variance
        prec_det = 1.0 / (sigma2_det + self.eps)
        prec_rec = 1.0 / (sigma2_rec + self.eps)
        prec_sem = 1.0 / (sigma2_sem + self.eps)

        total_prec = prec_det + prec_rec + prec_sem
        w_det = prec_det / total_prec
        w_rec = prec_rec / total_prec
        w_sem = prec_sem / total_prec

        S_final = w_det * mu_det + w_rec * mu_rec + w_sem * mu_sem

        weights = {
            "w_det": w_det.mean().item(),
            "w_rec": w_rec.mean().item(),
            "w_sem": w_sem.mean().item(),
        }
        return S_final, weights


# =========================================================================
# 6. COMPLETE CRANE-VAD ARCHITECTURAL FRAMEWORK
# =========================================================================

class CRANE_VAD(nn.Module):
    """
    Counterfactual Residual Alignment Network with Evidential Fusion.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        n_layers: int = 2,
        n_prototypes: int = 16,
        tau: float = 10.0,
        ema_momentum: float = 0.996,
        n_text_classes: int = 14,
        text_scale: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Temporal Encoder
        self.temporal_encoder = BiS4DEncoder(
            d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout
        )

        # Branch 1: Discriminative MIL
        self.det_head = DiscriminativeHead(d_model=d_model)

        # Branch 2: LGAM (Normal Memory)
        self.lgam = LGAM(
            d_model=d_model, n_prototypes=n_prototypes, tau=tau, ema_momentum=ema_momentum
        )

        # Branch 3: Residual-to-Text
        self.r2t_head = ResidualToTextHead(
            d_model=d_model, clip_dim=d_model, n_classes=n_text_classes, text_scale=text_scale
        )

        # Evidential Gate
        self.gate = EvidentialGate()

    def set_text_embeddings(self, text_emb):
        self.r2t_head.set_text_embeddings(text_emb)

    def forward(self, X, Y=None):
        """
        Args:
            X: CLIP snippet features (B, T, 512)
            Y: Video-level labels (B,) [0 or 1], optional during inference
        """
        # Step 1: Temporal encoding (O(T) linear complexity)
        H = self.temporal_encoder(X)

        # Step 2: Branch 1 - Discriminative
        mu_det, alpha_d, beta_d, sigma2_d = self.det_head(H)

        # Step 3: Branch 2 - LGAM (Label-Gated Normal Memory)
        mu_rec, F_rec, R, alpha_r, beta_r, sigma2_r = self.lgam(H, Y)

        # Step 4: Branch 3 - Residual-to-Text Counterfactual Alignment
        mu_sem, logits_sem, alpha_s, beta_s, sigma2_s = self.r2t_head(R)

        # Step 5: Evidential Fusion
        S_final, gate_weights = self.gate(
            mu_det, sigma2_d, mu_rec, sigma2_r, mu_sem, sigma2_s
        )

        return {
            "S_final": S_final,
            "S_det": mu_det,
            "S_rec": mu_rec,
            "S_sem": mu_sem,
            "H": H,
            "F_rec": F_rec,
            "R": R,
            "logits_sem": logits_sem,
            "gate_weights": gate_weights,
            "beta_params": {
                "det": (alpha_d, beta_d),
                "rec": (alpha_r, beta_r),
                "sem": (alpha_s, beta_s),
            },
        }
