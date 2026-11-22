import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

# ------------------ 基础组件 ------------------ #

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, (H,W)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

def window_partition(x, window_size: int):
    B,H,W,C = x.shape
    x = x.view(B,
               H//window_size, window_size,
               W//window_size, window_size,
               C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H*W / window_size / window_size))
    x = windows.view(B,
                     H//window_size, W//window_size,
                     window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size: Tuple[int,int], num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # Follow original SwinUNet meshgrid convention for broader compatibility
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flat = torch.flatten(coords, 1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1,2,0).contiguous()
        rel_coords[:,:,0] += window_size[0]-1
        rel_coords[:,:,1] += window_size[1]-1
        rel_coords[:,:,0] *= 2*window_size[1]-1
        relative_position_index = rel_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask: Optional[torch.Tensor]=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))
        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N,N,-1).permute(2,0,1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1,self.num_heads,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path_prob=0.):
        super().__init__()
        H,W = input_resolution
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size if min(H,W) > window_size else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (window_size,window_size), num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob>0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), drop=drop)

        if self.shift_size > 0:
            attn_mask = self.create_mask(H,W)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def create_mask(self, H, W):
        device = self.norm1.weight.device
        img_mask = torch.zeros((1,H,W,1), device=device)
        cnt = 0
        ws = self.window_size
        ss = self.shift_size
        for h in (slice(0,-ws), slice(-ws,-ss), slice(-ss,None)):
            for w in (slice(0,-ws), slice(-ws,-ss), slice(-ss,None)):
                img_mask[:,h,w,:] = cnt; cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws*ws)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask!=0, float(-100.0)).masked_fill(mask==0, 0.0)
        return mask

    def forward(self, x, H, W):
        B,L,C = x.shape
        assert L == H*W
        shortcut = x
        x = self.norm1(x).view(B,H,W,C)
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size,-self.shift_size), dims=(1,2))
        else:
            shifted = x
        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size*self.window_size, C)
        attn_windows = self.attn(windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1,self.window_size,self.window_size,C)
        shifted_back = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_back, shifts=(self.shift_size,self.shift_size), dims=(1,2))
        else:
            x = shifted_back
        x = x.view(B,H*W,C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(4*dim)
    def forward(self, x, H, W):
        B,L,C = x.shape
        assert L == H*W
        # Enforce even H and W to align with original SwinUNet behavior
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B,H,W,C)
        x0 = x[:,0::2,0::2,:]
        x1 = x[:,1::2,0::2,:]
        x2 = x[:,0::2,1::2,:]
        x3 = x[:,1::2,1::2,:]
        x = torch.cat([x0,x1,x2,x3], dim=-1).view(B,-1,4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H//2, W//2

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path_probs=None, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(dim, input_resolution, num_heads, window_size,
                          shift_size=0 if (i%2==0) else window_size//2,
                          mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                          attn_drop=attn_drop, drop_path_prob=drop_path_probs[i] if drop_path_probs else 0.)
            )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x,H,W)
        x_out_before_down = x  # 保存下采样前
        if self.downsample is not None:
            x, H, W = self.downsample(x,H,W)
        return x, H, W, x_out_before_down

# ------------------ 条件模块 ------------------ #

class WatermarkBitEncoder(nn.Module):
    def __init__(self, bit_length=100, embed_dim=256, use_mlp=True):
        super().__init__()
        layers = [nn.Linear(bit_length, embed_dim)]
        if use_mlp:
            layers += [nn.GELU(), nn.Linear(embed_dim, embed_dim)]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, bits):
        if bits.dtype != torch.float32:
            bits = bits.float()
        centered = bits * 2.0 - 1.0
        token = self.net(centered)
        token = self.norm(token)
        return token

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, wm_dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(wm_dim, dim*2)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(wm_dim)

    def forward(self, x_tokens, wm_token_feat):
        B,N,C = x_tokens.shape
        x_n = self.norm_q(x_tokens)
        wm_n = self.norm_kv(wm_token_feat)
        q = self.q(x_n).view(B,N,self.num_heads,C//self.num_heads).transpose(1,2)
        kv = self.kv(wm_n).view(B,1,2,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        k,v = kv[0],kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        # Softmax over key dimension to match standard attention behavior
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return x_tokens + out

class FiLMFromToken(nn.Module):
    def __init__(self, feat_ch, token_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_ch*2)
        )
    def forward(self, x, token):
        gamma_beta = self.net(token)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta

class UpSampleMerge(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.skip_ch = skip_ch
        self.reduce = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if skip_ch > 0:
            self.fuse = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )
        else:
            self.fuse = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )

    def forward(self, x, skip=None):
        if skip is not None:
            target_size = skip.shape[-2:]
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
        else:
            target_size = (x.shape[-2] * 2, x.shape[-1] * 2)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = self.reduce(x)
        return self.fuse(x)

# ------------------ 主模型（修正版） ------------------ #

class SwinUNetWatermarkBits(nn.Module):
    """
    条件水印区域检测 (修正：特征采集在下采样前进行，保证通道 = stage_dims)
    输入:
      image: (B,3,512,512)
      watermark_bits: (B,bit_length)
    输出:
      watermark_mask_logits: (B,1,512,512)
    """
    def __init__(self,
                 img_size=512,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=(2,2,6,2),
                 num_heads=(3,6,12,24),
                 window_size=8,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 bit_length=100,
                 wm_token_dim=256,
                 cross_attn_heads=8,
                 film_on_stages=True,
                 seq_use_mlp=True,
                 use_similarity=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_similarity = use_similarity
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                        in_chans=in_chans, embed_dim=embed_dim)
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, steps=total_blocks).tolist()
        dp_ptr = 0
        self.layers = nn.ModuleList()
        self.stage_dims: List[int] = []
        resolution = self.patches_resolution
        dim = embed_dim
        for i, depth in enumerate(depths):
            layer = BasicLayer(dim=dim,
                               depth=depth,
                               input_resolution=resolution,
                               num_heads=num_heads[i],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=True,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path_probs=dpr[dp_ptr:dp_ptr+depth],
                               downsample=(i < len(depths)-1))
            dp_ptr += depth
            self.layers.append(layer)
            self.stage_dims.append(dim)
            if i < len(depths)-1:
                dim *= 2
                resolution = (resolution[0]//2, resolution[1]//2)
        self.norm = nn.LayerNorm(self.stage_dims[-1])

        # 水印比特编码
        self.wm_encoder = WatermarkBitEncoder(bit_length=bit_length,
                                              embed_dim=wm_token_dim,
                                              use_mlp=seq_use_mlp)
        self.wm_token_to_feat = nn.Linear(wm_token_dim, self.stage_dims[-1])

        self.cross_attn = CrossAttentionBlock(dim=self.stage_dims[-1],
                                              wm_dim=self.stage_dims[-1],
                                              num_heads=cross_attn_heads,
                                              attn_drop=attn_drop_rate,
                                              proj_drop=drop_rate)

        # encoder_channels: [C1, C2, C3, C4]
        self.encoder_channels = self.stage_dims
        C1,C2,C3,C4 = self.encoder_channels

        # skips: C3, C2, C1
        self.skip_channels = self.encoder_channels[:-1][::-1]

        # Up blocks specs (merge with skips then extra upsample to reach full size)
        up_specs = []
        prev_ch = C4
        for skip_ch in self.skip_channels:
            up_specs.append((prev_ch, skip_ch, skip_ch))
            prev_ch = skip_ch
        # add extra upsample stages: need log2(patch_size) more 2x upsampling
        extra_ups = int(math.log2(self.patch_size)) if self.patch_size > 1 else 0
        for _ in range(extra_ups):
            up_specs.append((prev_ch, 0, prev_ch))
            # prev_ch unchanged
        self.up_blocks = nn.ModuleList([
            UpSampleMerge(in_ch, skip_ch, out_ch) for (in_ch, skip_ch, out_ch) in up_specs
        ])

        self.film_on_stages = film_on_stages
        if film_on_stages:
            self.film_blocks = nn.ModuleList([
                FiLMFromToken(feat_ch=spec[2], token_dim=wm_token_dim) for spec in up_specs
            ])

        final_in_ch = prev_ch  # channels after last up block
        self.final_head = nn.Sequential(
            nn.Conv2d(final_in_ch, final_in_ch//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(final_in_ch//2, 1, 1)
        )

        # Per-upstage projection of wm token to stage channel dims for similarity computation
        self.sim_proj = nn.ModuleList([
            nn.Linear(wm_token_dim, spec[2]) for spec in up_specs
        ])
        # Fuse stacked similarity maps (upsampled to final size) into a single logit
        self.sim_fuse = nn.Sequential(
            nn.Conv2d(len(up_specs), max(2, len(up_specs)//2), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(2, len(up_specs)//2), 1, 1)
        )
        # Learnable scalar to weight similarity-derived logit
        self.sim_weight = nn.Parameter(torch.tensor(1.0))

    def forward_features(self, x):
        """
        收集每层下采样前的特征：确保通道顺序 == self.stage_dims
        """
        feats = []
        B = x.shape[0]
        x,(H,W) = self.patch_embed(x)  # (B, N, C1)
        for layer in self.layers:
            # 手动展开 BasicLayer 逻辑以获得下采样前特征
            # 重新执行 layer.blocks，然后保存下采样前，再执行 downsample
            for blk in layer.blocks:
                x = blk(x,H,W)
            feat_before_down = x.view(B,H,W,-1).permute(0,3,1,2).contiguous()
            feats.append(feat_before_down)
            if layer.downsample is not None:
                x,H,W = layer.downsample(x,H,W)
        # 最后一层归一化
        x = self.norm(x)
        feats[-1] = x.view(B,H,W,-1).permute(0,3,1,2).contiguous()
        # 确认通道匹配
        collected = [f.shape[1] for f in feats]
        assert collected == self.stage_dims, f"Collected channels {collected} != expected {self.stage_dims}"
        return feats  # [C1,C2,C3,C4]

    def forward(self, image, watermark_bits):
        feats = self.forward_features(image)
        deepest = feats[-1]  # C4
        B,Cd,Hd,Wd = deepest.shape

        wm_token = self.wm_encoder(watermark_bits)          # (B,Dw)
        wm_feat = self.wm_token_to_feat(wm_token).unsqueeze(1)  # (B,1,C4)

        tokens = deepest.flatten(2).transpose(1,2)          # (B,N,C4)
        tokens = self.cross_attn(tokens, wm_feat)
        deepest = tokens.transpose(1,2).view(B,Cd,Hd,Wd)

        skip_feats = feats[:-1][::-1]  # [C3,C2,C1]
        for k,(sf,exp) in enumerate(zip(skip_feats, self.skip_channels)):
            assert sf.shape[1] == exp, f"Skip {k} channel {sf.shape[1]} != expected {exp}"

        dec = deepest
        num_skips = len(skip_feats)
        sim_maps = []  # collect per-stage similarity maps
        for i, up in enumerate(self.up_blocks):
            skip = skip_feats[i] if i < num_skips else None
            dec = up(dec, skip)
            if self.film_on_stages:
                dec = self.film_blocks[i](dec, wm_token)
            # Compute similarity between per-location feature and projected wm vector
            if self.use_similarity:
                stage_vec = self.sim_proj[i](wm_token)  # (B, C_stage)
                # Normalize feature and vector for cosine-like similarity
                f_norm = F.normalize(dec, p=2, dim=1, eps=1e-6)
                v = F.normalize(stage_vec, p=2, dim=1, eps=1e-6).unsqueeze(-1).unsqueeze(-1)
                sim = (f_norm * v).sum(dim=1, keepdim=True)  # (B,1,H,W)
                sim_maps.append(sim)

        pred_logit = self.final_head(dec)  # (B,1,Hf,Wf)

        if self.use_similarity and len(sim_maps) > 0:
            target_size = pred_logit.shape[-2:]
            sims_up = [F.interpolate(s, size=target_size, mode='bilinear', align_corners=False)
                       for s in sim_maps]
            sims_stack = torch.cat(sims_up, dim=1)  # (B, S, Hf, Wf)
            sim_logit = self.sim_fuse(sims_stack)
            combined_logit = pred_logit + self.sim_weight * sim_logit
        else:
            combined_logit = pred_logit

        # Return probabilities directly; callers should not rely on an
        # internal ``last_logits`` attribute.
        return torch.sigmoid(combined_logit)
          
