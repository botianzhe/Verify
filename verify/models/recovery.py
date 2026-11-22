"""
Watermark-Guided Face Recovery
==========================================

This module implements a recovery network that restores deepfake/forged facial regions
by jointly leveraging two sources of guidance:

1) Watermark semantics (bits): provide identity- and appearance-related priors
     via 1024 continuous VAE latent values. We encode values into tokens and a global CLS embedding
     and fuse them into image tokens using cross-attention and FiLM modulation.

2) Background context: provide in-situ texture, color, and illumination priors
     from clean regions. We explicitly build K/V only from background tokens
     and inject their responses into the forged (foreground) tokens.

Design overview
 - Patch-wise tokenization of the input image.
 - BitEncoder to derive continuous tokens and a global CLS from the latent vector.
 - Watermark-Conditioned Transformer blocks performing self-attention and cross-attention
     to watermark tokens, with per-head scaling derived from bit CLS and token-level mask gating.
 - Background-Aware Completion blocks performing cross-attention to background tokens only,
     with global and per-head scales from bit CLS; residuals only applied to forged tokens.
 - Token decoder upsamples back to full resolution; final pixel-space blending with the
     provided forgery mask and a light refinement to reduce seams.

Inputs
 - forged image: [B,3,256,256] explicitly in [-1,1] (v2)
 - forgery mask: [B,1,256,256] or [B,3,256,256] in [0,1] (1 means forged region)
 - latent values: [B,1024], continuous; legacy_mode retains old binary behavior

Outputs
 - recovered image: [B,3,256,256] in [-1,1]

Notes
 - Counterfactual streams and debias adapters are intentionally omitted in this version per request.
 - The localizer module is provided for potential future use; current workflow uses the given mask directly.
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using a Conv with stride=patch_size.

    - Input: x [B,3,H,W]
    - Output: tokens [B, N, C] with N = (H/ps)*(W/ps) and grid size (H/ps, W/ps)
    - Typical config: H=W=256, ps=16 -> N=256 tokens
    """
    def __init__(self, in_ch: int = 3, embed_dim: int = 512, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # x: [B,3,H,W] in [-1,1]
        x = self.proj(x)  # [B,C,H/ps,W/ps]
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x, (H, W)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BitEncoder(nn.Module):
    """
    Encode continuous latent scalars with two affine anchors + Transformer encoder.

    Rationale
    - The watermark carries identity/appearance priors. Turning bits into contextual tokens allows
        cross-attention to inject these priors into image tokens.

    Behavior
    - V2 preserves continuous scalar amplitudes and input gradients.
    - Only legacy_mode thresholds values at 0.5 for historical checkpoints.
    - A learnable CLS token aggregates global context for FiLM/per-head scaling.

    Returns
    - cls_token: [B, D] global embedding for the watermark sequence
    - bit_tokens: [B, L, D] per-bit tokens used as K/V in cross-attention
    """

    def __init__(self, wm_len: int, dim: int, depth: int = 3, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1, legacy_mode: bool = False):
        super().__init__()
        self.wm_len = wm_len
        self.legacy_mode = legacy_mode
        self.embed = nn.Embedding(2, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            batch_first=True,
            dropout=dropout,
            activation='gelu'
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = SinusoidalPositionalEncoding(dim, max_len=wm_len + 1)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if bits.ndim != 2 or bits.shape[1] != self.wm_len:
            raise ValueError(f'Expected latent [B,{self.wm_len}], got {tuple(bits.shape)}')
        if self.legacy_mode:
            x = self.embed((bits > 0.5).long())
        else:
            # Continuous affine scalar projection. Reuse the two old embedding
            # anchors (same state keys/shapes) without quantizing the latent.
            # For binary 0/1 this is exactly the previous embedding lookup;
            # negative and >1 VAE values remain continuous and differentiable.
            values = bits.to(self.embed.weight.dtype).unsqueeze(-1)
            x = self.embed.weight[0] + values * (self.embed.weight[1] - self.embed.weight[0])
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos(x)
        x = self.enc(x)
        x = self.norm(x)
        cls_token = x[:, 0]  # [B, D]
        tokens = x[:, 1:]    # [B, L, D]
        return cls_token, tokens


# Note: BitQuantizer removed per request.


class BitFiLM(nn.Module):
    """
    Feature-wise Linear Modulation from bit CLS embedding.

    Computes per-channel affine parameters and applies: y = x * (1 + gamma) + beta.
    This conditions token processing on watermark semantics in a lightweight way.
    """
    def __init__(self, dim: int, hidden: int = 4):
        super().__init__()
        hid = max(dim // hidden, 64)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(),
            nn.Linear(hid, dim * 2)
        )

    def forward(self, x: torch.Tensor, bit_cls: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D], bit_cls: [B,D]
        gamma_beta = self.mlp(bit_cls)  # [B, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# Note: Low-rank debias adapter removed per request.


class SpatialRouter(nn.Module):
    """
    Map bit tokens to a per-head routing gate. Kept for compatibility; can be disabled to use uniform gates.
    """
    def __init__(self, dim: int, num_patches: int, heads: int = 4, enabled: bool = False):
        super().__init__()
        self.heads = heads
        self.enabled = enabled
        if enabled:
            self.proj = nn.Linear(dim, dim)
            self.to_logits = nn.Linear(dim, heads)
            self.norm = nn.LayerNorm(dim)
        self.num_patches = num_patches

    def forward(self, bit_tokens: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.ones(bit_tokens.size(0), self.heads, device=bit_tokens.device)
        # bit_tokens: [B, L, D]
        x = self.norm(bit_tokens)
        x = self.proj(x)  # [B,L,D]
        logits = self.to_logits(x)  # [B,L,H]
        gate = logits.mean(dim=1).sigmoid()  # [B,H]
        return gate


class CounterfactualCrossBlock(nn.Module):
    """
    Watermark-Conditioned Transformer block (simplified, no counterfactual, no debias adapter).

    Design
    - Self-Attention -> Cross-Attention to watermark tokens (real only) with per-head gating.
    - FiLM and MLP for residual processing.
    - Token-level mask gating to restrict edits to forged regions.
    """
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout)
        )
        self.film = BitFiLM(dim)
        self.heads = heads
        self.head_scale = nn.Linear(dim, heads)
        # debias adapter removed

    def forward(
        self,
        img_tokens: torch.Tensor,
        bit_tokens: torch.Tensor,
        bit_cls: torch.Tensor,
        head_gate: torch.Tensor,
        token_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # img_tokens: [B,N,D]; bit_tokens: [B,L,D]; head_gate: [B,H]
        x = img_tokens
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        # Normalize before cross-modal fusion
        x = self.norm1(x)
        # Prepare K/V
        q = self.norm2(x)
        k = self.norm2(bit_tokens)
        v = k
        # Cross-attention (real only)
        real, _ = self.cross_attn(q, k, v, need_weights=False)
        B, N, D = real.shape
        H = self.heads
        real = real.view(B, N, H, D // H)
        # Per-head scales
        scale_h = (self.head_scale(bit_cls).sigmoid() * head_gate).view(B, 1, H, 1)
        residual = real * scale_h
        residual = residual.reshape(B, N, D)
        if token_gate is not None:
            # token_gate: [B,N,1], suppress cross-residual outside the provided mask
            residual = residual * token_gate
        x = x + residual
        # FiLM conditioned by bit cls embedding
        x = self.film(self.norm3(x), bit_cls) + x
        # MLP
        x = x + self.mlp(x)
        return x


class BackgroundAwareCompletionBlock(nn.Module):
    """
    Background-Aware Completion (BAC) block.

    Purpose
    - Leverage clean background regions as an in-situ prior to complete forged areas.
    - Preserve lighting, color tone, and local textures consistent with the immediate context.

    Design
    - Build K/V only from background tokens (masked by provided forgery mask).
    - Query is the full token grid; the residual is hard-gated by the foreground token mask so only forged tokens are updated.
    - Bit-CLS provides per-head and global scales (FiLM-like) to adapt how much background flows in.

    Inputs
    - img_tokens: [B,N,D] token grid
    - bit_cls: [B,D] watermark global embedding
    - token_gate: [B,N,1] foreground soft mask on token grid in [0,1]

    Output
    - updated tokens [B,N,D]
    """
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_bg = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        # scales from bit CLS: global gate and per-head gate
        self.global_scale = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.head_scale = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, heads))
        self.heads = heads

    def forward(self, img_tokens: torch.Tensor, bit_cls: torch.Tensor, token_gate: Optional[torch.Tensor]) -> torch.Tensor:
        if token_gate is None:
            return img_tokens
        # token_gate is foreground mask in [0,1] at token level
        x = img_tokens
        q = self.norm_q(x)
        k = self.norm_kv(x)
        v = k
        # Build key padding mask to drop foreground from background K/V: True -> ignore
        # token_gate: [B,N,1]; foreground positions near 1.0 will be masked out
        key_padding_mask = (token_gate.squeeze(-1) > 0.5)  # [B,N] boolean
        bg, _ = self.cross_attn_bg(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        B, N, D = bg.shape
        H = self.heads
        # per-head scaling from bit cls
        head_s = torch.sigmoid(self.head_scale(bit_cls)).view(B, 1, H, 1)
        # reshape residual to apply per-head gate
        bg = bg.view(B, N, H, D // H) * head_s
        bg = bg.view(B, N, D)
        # global scale and hard gating to foreground only
        g = torch.sigmoid(self.global_scale(bit_cls)).view(B, 1, 1)
        residual = bg * token_gate * g
        x = x + residual
        x = x + self.mlp(x)
        return x


class GeometricAligner(nn.Module):
    """
    Low-resolution geometric alignment head.

    Predicts a global affine transform (2x3) plus a residual dense flow at the bottleneck resolution
    to coarsely align forged regions before texture recovery. Conditioning uses the watermark bit CLS.

    Inputs
    - b: [B, C, H, W] bottleneck features (e.g., 16x16)
    - m_lr: [B, 1, H, W] low-res mask (1 = forged)
    - bit_cls: [B, C]

    Outputs
    - b_warp: [B, C, H, W] warped features
    - m_warp: [B, 1, H, W] warped mask
    """
    def __init__(self, dim: int, max_flow: float = 0.25, affine_scale: float = 0.15):
        super().__init__()
        self.max_flow = max_flow
        self.affine_scale = affine_scale
        hid = max(dim // 2, 128)
        # affine head
        self.affine_mlp = nn.Sequential(
            nn.Linear(dim * 2, hid), nn.GELU(),
            nn.Linear(hid, 6)
        )
        # residual flow head
        self.flow_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim // 2, 2, 3, padding=1), nn.Tanh()
        )

    def forward(self, b: torch.Tensor, m_lr: torch.Tensor, bit_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = b.shape
        # Global affine around identity
        pooled = torch.mean(b, dim=(2, 3))  # [B,C]
        aff_in = torch.cat([pooled, bit_cls], dim=-1)
        delta = self.affine_mlp(aff_in)  # [B,6]
        delta = torch.tanh(delta) * self.affine_scale
        theta = torch.zeros(B, 2, 3, device=b.device, dtype=b.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta = theta + delta.view(B, 2, 3)
        grid_aff = F.affine_grid(theta, size=b.size(), align_corners=False)  # [B,H,W,2]

        # Residual dense flow in normalized coordinates
        flow = self.flow_head(b) * self.max_flow  # [B,2,H,W]
        flow = flow.permute(0, 2, 3, 1)  # [B,H,W,2]
        grid = grid_aff + flow

        b_warp = F.grid_sample(b, grid, mode='bilinear', padding_mode='border', align_corners=False)
        m_warp = F.grid_sample(m_lr, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return b_warp, m_warp


class MultiGranularityLocalizer(nn.Module):
    """
    Multi-Granularity Forgery Localizer (MGFL).

    Problem: Spatial localization ambiguity (uncertain where to edit), background preservation.
    Motivation: Predict editability from coarse-to-fine, with uncertainty, guided by watermark semantics.
    Design: bit-conditioned FiLM on token grid; produce masks at 32/64/128/256 and combine via learned weights.
    Outputs: soft mask M in [0,1] and optional uncertainty map U in (0,1).
    """
    def __init__(self, dim: int, grid: Tuple[int, int]):
        super().__init__()
        H, W = grid
        self.H, self.W = H, W
        ch = dim
        self.film = BitFiLM(dim)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
        )
        # heads for different scales
        self.h32 = nn.Conv2d(ch, 1, 1)
        self.h64 = nn.Conv2d(ch, 1, 1)
        self.h128 = nn.Conv2d(ch, 1, 1)
        self.h256 = nn.Conv2d(ch, 1, 1)
        # scale combiners predicted from bit cls
        self.alpha = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))
        # uncertainty head at full res
        self.unc = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 1, 1)
        )

    # simplified upsampling: use bilinear pyramid instead of PixelShuffle chains

    def forward(self, tokens: torch.Tensor, bit_cls: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = tokens.shape
        x = self.film(tokens, bit_cls).transpose(1, 2).reshape(B, D, self.H, self.W)
        x = self.bottleneck(x)
        # Starting grid is typically 16x16 (patch_size=16 for 256x256). Build a bilinear pyramid.
        x32 = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        m32 = self.h32(x32)
        x64 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        m64 = self.h64(x64)
        x128 = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        m128 = self.h128(x128)
        x256 = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        m256 = self.h256(x256)
        # combine via learned weights
        a = torch.softmax(self.alpha(bit_cls), dim=-1).view(B, 4, 1, 1, 1)
        m32u = F.interpolate(m32, size=(256, 256), mode='bilinear', align_corners=False)
        m64u = F.interpolate(m64, size=(256, 256), mode='bilinear', align_corners=False)
        m128u = F.interpolate(m128, size=(256, 256), mode='bilinear', align_corners=False)
        m256u = F.interpolate(m256, size=(256, 256), mode='bilinear', align_corners=False)
        m_stack = torch.stack([m32u, m64u, m128u, m256u], dim=1)  # [B,4,1,256,256]
        mask = torch.sigmoid((a * m_stack).sum(dim=1))  # [B,1,256,256]
        # uncertainty
        U = torch.sigmoid(self.unc(x256))  # [B,1,256,256]
        return mask, U


class TokenDecoder(nn.Module):
    """
    Decode tokens [B,N,D] back to image via conv upsampling.
    """
    def __init__(self, dim: int, grid: Tuple[int, int], out_ch: int = 3):
        super().__init__()
        H, W = grid
        self.H, self.W = H, W
        ch = dim
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
        )
        # Upsample back to full res via two PixelShuffle stages (assuming patch_size=16 -> 16x upsample via 4x then 4x)
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.GELU(),
            nn.PixelShuffle(2),  # x2
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.GELU(),
            nn.PixelShuffle(2),  # x2
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.GELU(),
            nn.PixelShuffle(2),  # x2 -> total x8
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.GELU(),
            nn.PixelShuffle(2),  # x2 -> total x16
            nn.Conv2d(ch, out_ch, 1)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, self.H, self.W)
        x = self.conv(x)
        x = self.up(x)
        x = torch.tanh(x)
        return x


class BitGuidedRecoveryTransformer(nn.Module):
    """
    Watermark + Background-Guided Hybrid CNN-Transformer U-Net

    What changed
    - Add a strong CNN encoder-decoder with skip connections for local texture fidelity.
    - Keep a Transformer at the bottleneck operating on 16x16 tokens for global, watermark-guided reasoning.
    - Perform background-aware completion and watermark cross-attention on tokens, then decode with CNN.

    Contract
    - Inputs:
        forged image x_forged explicitly in [-1,1], shape [B,3,256,256];
        initial recovery x_init explicitly in [-1,1], shape [B,3,256,256];
        watermark bits b [B,wm_len] (now wm_len defaults to 1024);
        forgery mask M [B,1|3,256,256] in [0,1].
    - Output: recovered image in [-1,1], shape [B,3,256,256].
    - Optional: with return_aux=True returns dict with mask and uncertainty for training losses.
    """
    def __init__(
        self,
        wm_len: int = 1024,
        img_size: int = 256,
        patch_size: int = 16,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        bit_depth: int = 3,
        dropout: float = 0.1,
        return_aux: bool = False,
        use_router: bool = True,
        legacy_mode: bool = False,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.return_aux = return_aux
        self.heads = heads
        self.use_router = use_router
        self.legacy_mode = legacy_mode

        # =========================
        # CNN encoder/decoder (U-Net)
        # =========================
        ch = dim // 6  # base width (e.g., 64 when dim=384)
        self.enc_conv1 = nn.Sequential(
            # accept concatenated forged and initial images -> 6 channels
            nn.Conv2d(6, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
        )  # 256x256 -> ch
        self.down1 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)  # 128x128
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.GELU(),
        )
        self.down2 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)  # 64x64
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1), nn.GELU(),
        )
        self.down3 = nn.Conv2d(ch * 4, ch * 8, 3, stride=2, padding=1)  # 32x32
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch * 8, ch * 8, 3, padding=1), nn.GELU(),
        )
        self.down4 = nn.Conv2d(ch * 8, dim, 3, stride=2, padding=1)  # 16x16 -> dim
        self.enc_bottleneck = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
        )

        # Bit encoder
        self.bit_enc = BitEncoder(wm_len=wm_len, dim=dim, depth=bit_depth, heads=heads, dropout=dropout, legacy_mode=legacy_mode)

        # Transformer blocks at bottleneck
        self.blocks = nn.ModuleList([
            CounterfactualCrossBlock(dim=dim, heads=heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.router = SpatialRouter(dim=dim, num_patches=(img_size // patch_size) ** 2, heads=heads, enabled=False)

        # Background-aware completion block(s)
        self.bg_blocks = nn.ModuleList([
            BackgroundAwareCompletionBlock(dim=dim, heads=heads, dropout=dropout)
            for _ in range(max(1, depth // 3))
        ])

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(dim, ch * 8, 2, stride=2)  # 16->32
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(ch * 8 + ch * 8, ch * 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch * 8, ch * 8, 3, padding=1), nn.GELU(),
        )
        self.up2 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)  # 32->64
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(ch * 4 + ch * 4, ch * 4, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch * 4, ch * 4, 3, padding=1), nn.GELU(),
        )
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)  # 64->128
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(ch * 2 + ch * 2, ch * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.GELU(),
        )
        self.up4 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)  # 128->256
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(ch + ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
        )
        self.out_head = nn.Conv2d(ch, 3, 1)

        # Light residual refinement
        self.refine = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        # Geometric aligner at bottleneck
        self.aligner = GeometricAligner(dim=dim, max_flow=0.25, affine_scale=0.15)

    def forward(
        self, x_forged: torch.Tensor, x_init: torch.Tensor,
        wm_bits: torch.Tensor, forge_mask: torch.Tensor,
        *, return_components: bool = False,
    ):
        # Map [0,1] -> [-1,1] if needed for both images
        if self.legacy_mode and x_forged.min() >= 0.0 and x_forged.max() <= 1.0:
            x_forged = x_forged * 2 - 1
        if self.legacy_mode and x_init.min() >= 0.0 and x_init.max() <= 1.0:
            x_init = x_init * 2 - 1

        # Require mask for this version (explicit mask-aware editing)
        if forge_mask is None:
            raise ValueError("forge_mask is required for mask-aware recovery; no learned localizer is used.")

        # Normalize mask to single channel in [0,1]
        M = forge_mask
        if M.dim() == 4 and M.size(1) == 3:
            M = M.max(dim=1, keepdim=True)[0]
        M = torch.clamp(M, 0.0, 1.0)

        # -----------------
        # CNN encoder path
        # -----------------
        context = x_forged if self.legacy_mode else x_forged * (1 - M)
        x_in = torch.cat([context, x_init], dim=1)
        e1 = self.enc_conv1(x_in)             # 256x256, ch
        e2 = self.enc_conv2(self.down1(e1))    # 128x128, 2ch
        e3 = self.enc_conv3(self.down2(e2))    # 64x64, 4ch
        e4 = self.enc_conv4(self.down3(e3))    # 32x32, 8ch
        b  = self.enc_bottleneck(self.down4(e4))  # 16x16, dim

        Bz, Dz, Ht, Wt = b.shape  # Ht=Wt=img_size//16

        # Prepare low-res mask
        m_grid = F.interpolate(M, size=(Ht, Wt), mode='bilinear', align_corners=False)  # [B,1,Ht,Wt]
            
        # Bit encoder (needed for aligner)
        bit_cls, bit_tokens = self.bit_enc(wm_bits)
        cross_tokens_real = bit_tokens

        # Geometric alignment on bottleneck features and mask
        b, m_grid = self.aligner(b, m_grid, bit_cls)

        # Tokenize aligned bottleneck for transformer
        tokens = b.flatten(2).transpose(1, 2)  # [B,N,D]
        B, N, D = tokens.shape
        token_gate = m_grid.flatten(2).transpose(1, 2)  # [B,N,1]

        head_dim = self.router.heads if self.use_router else self.heads
        head_gate = self.router(cross_tokens_real) if self.use_router else torch.ones(tokens.size(0), head_dim, device=tokens.device)  # [B,H]

        # Transformer blocks with watermark cross-attention
        for blk in self.blocks:
            tokens = blk(tokens, cross_tokens_real, bit_cls, head_gate, token_gate=token_gate)
        tokens = self.norm(tokens)

        # Background-aware completion blocks (operate on tokens)
        for bg_blk in self.bg_blocks:
            tokens = bg_blk(tokens, bit_cls, token_gate)

        # Back to bottleneck feature map
        b_upd = tokens.transpose(1, 2).reshape(B, D, Ht, Wt)

        # -----------------
        # CNN decoder path
        # -----------------
        d1 = self.up1(b_upd)
        d1 = self.dec_conv1(torch.cat([d1, e4], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec_conv2(torch.cat([d2, e3], dim=1))
        d3 = self.up3(d2)
        d3 = self.dec_conv3(torch.cat([d3, e2], dim=1))
        d4 = self.up4(d3)
        d4 = self.dec_conv4(torch.cat([d4, e1], dim=1))
        pred = torch.tanh(self.out_head(d4))  # [-1,1]

        # Blend with input using mask; optional uncertainty (zeros for now)
        U = torch.zeros_like(M)
        M_eff = M #* (1.0 - 0.5 * U)
        out = x_forged * (1 - M_eff) + pred * M_eff
        # out = torch.tanh(self.refine(out) + out)
        out = torch.clamp(out, -1.0, 1.0)
        if return_components:
            # Diagnostics only; the default forward output is unchanged.
            return {"image": out, "prediction": pred, "training_image": out, "recovery_input": x_forged}
        if self.return_aux:
            return {"image": out, "mask": M, "uncertainty": U}
        return out


def build_recovery_model(wm_len: int = 1024, return_aux: bool = False, **kwargs) -> BitGuidedRecoveryTransformer:
    """Factory helper for external wiring."""
    return BitGuidedRecoveryTransformer(wm_len=wm_len, return_aux=return_aux, **kwargs)
