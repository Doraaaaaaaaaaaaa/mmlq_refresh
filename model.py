"""
改进版 IAA 模型 —— 按照 claude思路.txt 的 9/10 方案实现

架构四阶段：
  阶段1: 多尺度特征提取
    - VisualEncoder : Swin-Base(img=224) → 4 级特征 → 投影到 768
    - TextEncoder   : Frozen BERT + 4 个轻量 Adapter → 4 级特征 768
  阶段2: 层级弱交互式融合 (WeakInteractionFusion × 4)
    每层: 模态内 CrossAttn → 跨模态 CrossAttn → 均值聚合
  阶段3: 动态属性推理 (DynamicAttributeModule)
    PARN (ResNet-50 冻结) → 11 个 attribute tokens → 2 层 CrossAttn
  阶段4: 预测头 → 10-bin Softmax 分布
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import logging, BertTokenizer, BertModel
import torchvision.models as tv_models

logging.set_verbosity_error()


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def disabled_train(self, mode=True):
    """让冻结模块在 model.train() 时也保持 eval 状态"""
    return self


# ─────────────────────────────────────────────────────────────────────────────
# Swin Transformer（从 test-main 引入，支持返回 4 个 stage 特征）
# ─────────────────────────────────────────────────────────────────────────────

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        # padding if needed
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0], 0, 0))
        x = self.proj(x)               # (B, embed_dim, H/P, W/P)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, C)
        x = self.norm(x)
        return x, Hp, Wp


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        # pad if odd
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, attn_mask, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        self.downsample = downsample

    def create_mask(self, x, H, W):
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            x = blk(x, attn_mask, H, W)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class SwinTransformer(nn.Module):
    """
    Swin-Base 224: embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]
    输出 4 个 stage 特征，形状 (B, N_i, C_i)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_c=in_chans,
                                      embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinStage(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging(int(embed_dim * 2 ** i_layer), norm_layer=nn.LayerNorm)
                if i_layer < self.num_layers - 1 else None,
            )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        stage_outs = []
        for layer in self.layers:
            x_out, H_out, W_out, x, H, W = layer(x, H, W)
            stage_outs.append(x_out)   # 保存每个 stage 的输出
        return stage_outs  # [(B,N1,C1),(B,N2,C2),(B,N3,C3),(B,N4,C4)]


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 1a：视觉编码器（原版 CLIP ViT-L/14，从本地 .pth 加载）
# ─────────────────────────────────────────────────────────────────────────────

class CLIPResidualAttentionBlock(nn.Module):
    """原版 CLIP ViT 的 ResidualAttentionBlock"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ln = self.ln_1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPViTL14(nn.Module):
    """
    原版 CLIP ViT-L/14，直接从 clip_vit_L.pth 加载权重。
    支持返回中间层隐藏状态。
    """
    def __init__(self, ckpt_path):
        super().__init__()
        # ViT-L/14 超参
        width, heads, layers = 1024, 16, 24
        patch_size, image_size = 14, 224
        num_patches = (image_size // patch_size) ** 2  # 256

        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(num_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = nn.ModuleList([
            CLIPResidualAttentionBlock(width, heads) for _ in range(layers)
        ])
        self.ln_post = nn.LayerNorm(width)

        # 加载权重
        self._load(ckpt_path)

    def _load(self, path):
        state = torch.load(path, map_location="cpu")
        # 原版 key → 当前 key 的映射
        new_state = {}
        for k, v in state.items():
            nk = k
            # transformer.resblocks.i.* → transformer.i.*
            nk = nk.replace("transformer.resblocks.", "transformer.")
            # attn.in_proj_weight 需要拆分成 q/k/v，但 nn.MultiheadAttention
            # 支持合并的 in_proj_weight，直接映射
            nk = nk.replace(".attn.in_proj_weight", ".attn.in_proj_weight")
            nk = nk.replace(".attn.in_proj_bias",   ".attn.in_proj_bias")
            nk = nk.replace(".attn.out_proj.",       ".attn.out_proj.")
            nk = nk.replace(".mlp.c_fc.",            ".mlp.0.")
            nk = nk.replace(".mlp.c_proj.",          ".mlp.2.")
            new_state[nk] = v
        miss, unexp = self.load_state_dict(new_state, strict=False)
        print(f"[CLIPViTL14] Loaded. Missing: {len(miss)}, Unexpected: {len(unexp)}")

    def forward(self, x, return_layers=(6, 12, 18, 24)):
        # Patch embedding
        x = self.conv1(x)                          # (B, width, H/14, W/14)
        x = x.flatten(2).transpose(1, 2)           # (B, 256, width)
        cls = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)             # (B, 257, width)
        x = x + self.positional_embedding.unsqueeze(0)
        x = self.ln_pre(x)

        hidden_states = []
        for i, blk in enumerate(self.transformer):
            x = blk(x)
            if (i + 1) in return_layers:           # 第 i+1 层（1-indexed）
                hidden_states.append(x)

        return hidden_states  # list of (B, 257, 1024)


class VisualEncoder(nn.Module):
    """
    CLIP ViT-L/14 (冻结，从本地 clip_vit_L.pth 加载)
    → 取第 6,12,18,24 层隐藏状态 → 投影到 query_size (768)
    输出 V = [V1(B,257,768), V2(B,257,768), V3(B,257,768), V4(B,257,768)]
    """
    LAYER_INDICES = (6, 12, 18, 24)
    CLIP_DIM = 1024
    CLIP_PTH  = r"C:\Users\admin\.cache\torch\hub\checkpoints\clip_vit_L.pth"

    def __init__(self, config):
        super().__init__()
        qs = config["query_size"]  # 768

        self.clip = CLIPViTL14(self.CLIP_PTH)

        # 冻结 CLIP
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip = self.clip.eval()
        self.clip.train = disabled_train

        # 4 个投影层：1024 → 768
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(self.CLIP_DIM, qs), nn.LayerNorm(qs))
            for _ in range(4)
        ])

    def forward(self, image):
        hidden_states = self.clip(image, return_layers=self.LAYER_INDICES)
        # hidden_states: list of 4 × (B, 257, 1024)
        V = [self.projectors[i](hidden_states[i]) for i in range(4)]
        return V  # 4 × (B, 257, 768)


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 1b：文本编码器（Frozen BERT + Adapter）
# ─────────────────────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Frozen BERT-base → 取第 3,6,9,12 层隐藏状态
    每层接独立轻量 Adapter（768→768）
    输出 T = [T1, T2, T3, T4]，各 (B, seq_len, 768)
    """
    LAYER_INDICES = [3, 6, 9, 12]   # 对应 hidden_states 下标（0=embedding）

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased",
                                              output_hidden_states=True)
        # 冻结 BERT
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert = self.bert.eval()
        self.bert.train = disabled_train

        qs = config["query_size"]   # 768
        # 轻量 Adapter：down-project → GELU → dropout → up-project + 残差
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(qs, qs),
                nn.GELU(),
                nn.Dropout(config.get("dropout", 0.1)),
                nn.Linear(qs, qs),
                nn.LayerNorm(qs),
            )
            for _ in range(4)
        ])

    def forward(self, captions):
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_caption_length"],
            return_tensors="pt",
        ).to(next(self.adapters.parameters()).device)

        with torch.no_grad():
            outputs = self.bert(**tokens)

        hidden_states = outputs.hidden_states   # tuple of 13 × (B, seq, 768)

        T = []
        for i, idx in enumerate(self.LAYER_INDICES):
            h = hidden_states[idx]              # (B, seq, 768)
            # Adapter + 残差
            t = h + self.adapters[i](h)
            T.append(t)

        # 同时返回 attention_mask 供后续 cross-attn padding mask 使用
        return T, tokens.attention_mask  # 4×(B,seq,768), (B,seq)


# ─────────────────────────────────────────────────────────────────────────────
# 通用 CrossAttnBlock
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttnBlock(nn.Module):
    """
    标准多头 Cross-Attention + FFN + LayerNorm（Pre-Norm 风格）
    Q 来自 query_embeds；K、V 来自 feats。
    若 feats=None 则退化为 Self-Attention。
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query_embeds, feats=None, key_padding_mask=None):
        """
        query_embeds : (B, Nq, D)
        feats        : (B, Nk, D)  — None → self-attn
        key_padding_mask: (B, Nk) bool，True 表示该位置是 padding
        """
        residual = query_embeds
        x = self.norm1(query_embeds)
        kv_src = self.norm1(feats) if feats is not None else x

        Q = self._split_heads(self.q_proj(x))           # (B, H, Nq, d)
        K = self._split_heads(self.k_proj(kv_src))      # (B, H, Nk, d)
        V = self._split_heads(self.v_proj(kv_src))

        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (B,H,Nq,Nk)

        if key_padding_mask is not None:
            # mask: True = 忽略，扩展到 (B,1,1,Nk)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, V)                      # (B, H, Nq, d)
        out = out.transpose(1, 2).contiguous().view(query_embeds.shape[0], -1, self.num_heads * self.head_dim)
        out = self.drop(self.out_proj(out))
        out = out + residual

        # FFN
        out = out + self.drop(self.ffn(self.norm2(out)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 2：层级弱交互式融合
# ─────────────────────────────────────────────────────────────────────────────

class WeakInteractionFusion(nn.Module):
    """
    单层弱交互式融合（对应一个 Swin stage 和一个 BERT 层级）

    Step 1 模态内提取：
        VQ = CrossAttn(VQ, V)    # 视觉 query 从当前层视觉特征提取
        TQ = CrossAttn(TQ, T)    # 文本 query 从当前层文本特征提取
    Step 2 跨模态交互：
        VQ_hat = CrossAttn(VQ, T)  # 视觉 query 看文本
        TQ_hat = CrossAttn(TQ, V)  # 文本 query 看视觉
    Step 3 均值融合（保持维度不变）：
        VQ_final = (VQ + VQ_hat) / 2
        TQ_final = (TQ + TQ_hat) / 2
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        # Step 1 模态内
        self.visual_intra = CrossAttnBlock(dim, num_heads, dropout)
        self.text_intra   = CrossAttnBlock(dim, num_heads, dropout)
        # Step 2 跨模态
        self.visual_cross = CrossAttnBlock(dim, num_heads, dropout)
        self.text_cross   = CrossAttnBlock(dim, num_heads, dropout)

    def forward(self, VQ, TQ, V, T, text_pad_mask=None):
        """
        VQ : (B, Nv_q, 768)  视觉可学习查询
        TQ : (B, Nt_q, 768)  文本可学习查询
        V  : (B, Nv,   768)  当前层视觉特征
        T  : (B, seq,  768)  当前层文本特征
        text_pad_mask: (B, seq) bool，True=padding
        """
        # Step 1
        VQ = self.visual_intra(VQ, V)
        TQ = self.text_intra(TQ, T, key_padding_mask=text_pad_mask)

        # Step 2
        VQ_hat = self.visual_cross(VQ, T, key_padding_mask=text_pad_mask)
        TQ_hat = self.text_cross(TQ, V)

        # Step 3 均值
        VQ_final = (VQ + VQ_hat) / 2.0
        TQ_final = (TQ + TQ_hat) / 2.0

        return VQ_final, TQ_final


# ─────────────────────────────────────────────────────────────────────────────
# 阶段 3：动态属性模块（PARN + 属性推理）
# ─────────────────────────────────────────────────────────────────────────────

class PARNAttributeEncoder(nn.Module):
    """
    ResNet-50 backbone → 共享 bottleneck → 11 个属性分数
    结构与 AMM-Net.pt 的 img_attr.* 权重完全对齐，可直接加载。
    输出：g (B, 2048) + scores (B, 11)
    """
    NUM_ATTRS = 11

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool

        self.fc1_1   = nn.Linear(2048, 256)
        self.bn1_1   = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.fc2_1   = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.fc3_1   = nn.Linear(64, self.NUM_ATTRS)

    def extract_attributes(self, image):
        """返回 (g, scores)，二者均不带梯度（供冻结模式使用）"""
        x = self.conv1(image)
        x = self.bn1(x);  x = self.relu(x);  x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        g = x.flatten(1)                              # (B, 2048)
        h = self.relu1_1(self.bn1_1(self.fc1_1(g)))  # (B, 256)
        h = self.relu2_1(self.fc2_1(h))               # (B, 64)
        scores = self.fc3_1(h)                         # (B, 11)
        return g, scores

    def load_from_ammnet(self, path):
        """从 AMM-Net.pt 加载 img_attr.* 权重"""
        import os
        if not os.path.exists(path):
            print(f"[PARN] AMM-Net.pt not found at {path}, using random init")
            return
        ckpt = torch.load(path, map_location="cpu")
        if any(k.startswith("img_attr.") for k in ckpt):
            state = {k[len("img_attr."):]: v for k, v in ckpt.items()
                     if k.startswith("img_attr.")}
        else:
            state = ckpt
        miss, unexp = self.load_state_dict(state, strict=False)
        print(f"[PARN] Loaded from {path}. Missing: {len(miss)}, Unexpected: {len(unexp)}")


class DynamicAttributeModule(nn.Module):
    """
    阶段 3：动态属性推理

    1. 调用冻结 PARN 提取当前图像属性 g (B,2048) + scores (B,11)
    2. 对每个属性 i 拼接 [g, score_i] → Linear → (B, 11, query_size)
    3. 用 F（融合后的多模态 query）做 2 层 CrossAttn，以 attr_tokens 为 K/V
       → F_hat（精炼后的美学特征）
    """
    def __init__(self, config):
        super().__init__()
        qs = config["query_size"]          # 768
        self.num_attrs = 11

        self.parn = PARNAttributeEncoder()

        # 可选加载预训练 PARN
        if config.get("parn_pretrained"):
            self.parn.load_from_ammnet(config["parn_pretrained"])

        # 冻结 PARN
        for p in self.parn.parameters():
            p.requires_grad = False
        self.parn = self.parn.eval()
        self.parn.train = disabled_train

        # 每个属性独立投影：concat(g, score_i) → qs
        self.attr_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048 + 1, qs),
                nn.LayerNorm(qs),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_attrs)
        ])

        # 属性推理：2 层 CrossAttn，Q=F，K/V=attr_tokens
        num_heads = config.get("num_attn_heads", 8)
        dropout   = config.get("dropout", 0.1)
        attr_depth = config.get("attr_reasoning_layers", 1)
        self.attr_reasoning = nn.ModuleList([
            CrossAttnBlock(qs, num_heads, dropout)
            for _ in range(attr_depth)
        ])

        # 图像下采样到 224（PARN 输入尺寸固定）
        self._parn_size = 224

    def forward(self, image, feat):
        """
        image : (B, 3, H, W)   — 可以是 224 或其他尺寸
        feat  : (B, Nq, 768)   — 融合后的多模态 query（避免与 F 模块命名冲突）
        """
        B = image.shape[0]

        # 保证 PARN 输入为 224×224
        if image.shape[-1] != self._parn_size or image.shape[-2] != self._parn_size:
            img_small = F.interpolate(
                image, size=(self._parn_size, self._parn_size),
                mode='bilinear', align_corners=False
            )
        else:
            img_small = image

        with torch.no_grad():
            g, scores = self.parn.extract_attributes(img_small)  # (B,2048),(B,11)

        # 生成 11 个 attribute tokens
        tokens = []
        for i in range(self.num_attrs):
            z_i = scores[:, i:i+1]                                     # (B, 1)
            a_i = self.attr_projs[i](torch.cat([g, z_i], dim=-1))      # (B, qs)
            tokens.append(a_i.unsqueeze(1))                             # (B, 1, qs)
        attr_tokens = torch.cat(tokens, dim=1)                          # (B, 11, qs)

        # 属性推理：Q=feat，K/V=attr_tokens
        H = feat
        for layer in self.attr_reasoning:
            H = layer(H, attr_tokens)

        return H  # (B, Nq, 768)


# ─────────────────────────────────────────────────────────────────────────────
# 完整模型
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalAttentionAggregator(nn.Module):
    """Aggregate four layer-level query outputs with self-attention."""

    def __init__(self, dim=768, num_layers=4, num_heads=8, depth=2, dropout=0.1):
        super().__init__()
        self.global_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.layer_pos = nn.Parameter(torch.zeros(1, num_layers + 1, dim))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm2": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                ),
            })
            for _ in range(depth)
        ])
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.global_token, std=0.02)
        nn.init.trunc_normal_(self.layer_pos, std=0.02)

    def forward(self, layer_queries):
        layer_tokens = torch.stack([q.mean(dim=1) for q in layer_queries], dim=1)
        B = layer_tokens.shape[0]
        x = torch.cat([self.global_token.expand(B, -1, -1), layer_tokens], dim=1)
        x = x + self.layer_pos[:, :x.shape[1], :]

        for block in self.blocks:
            residual = x
            h = block["norm1"](x)
            attn_out, _ = block["attn"](h, h, h, need_weights=False)
            x = residual + self.drop(attn_out)
            x = x + self.drop(block["ffn"](block["norm2"](x)))

        return self.out_norm(x[:, 0, :])


class ImprovedIAAModel(nn.Module):
    """
    四阶段改进 IAA 模型

    config 需包含以下字段：
      query_size       : 768
      num_visual_query : 每层视觉 query 数（默认 2）
      num_text_query   : 每层文本 query 数（默认 2）
      num_attn_heads   : 8
      max_caption_length : 512
      dropout          : 0.1
      vision_freeze    : bool（是否冻结 Swin）
      swin_pretrained  : str 或 ""（Swin 预训练权重路径）
      parn_pretrained  : str 或 ""（AMM-Net.pt 路径）
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        qs       = config["query_size"]           # 768
        nv       = config.get("num_visual_query", 2)
        nt       = config.get("num_text_query", 2)
        nh       = config.get("num_attn_heads", 8)
        dropout  = config.get("dropout", 0.1)
        num_layers = 4   # 固定 4 个层级

        # ── 阶段 1：编码器 ─────────────────────────────────────────────────
        self.visual_encoder = VisualEncoder(config)
        self.text_encoder   = TextEncoder(config)

        # ── 可学习 Query（每层独立） ─────────────────────────────────────────
        self.visual_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(1, nv, qs)) for _ in range(num_layers)
        ])
        self.text_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(1, nt, qs)) for _ in range(num_layers)
        ])
        # 初始化
        for p in self.visual_queries:
            nn.init.trunc_normal_(p, std=0.02)
        for p in self.text_queries:
            nn.init.trunc_normal_(p, std=0.02)

        # ── 阶段 2：4 层弱交互式融合 ─────────────────────────────────────────
        self.fusion_layers = nn.ModuleList([
            WeakInteractionFusion(dim=qs, num_heads=nh, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 4 层输出聚合后的层归一化
        hier_depth = config.get("hier_attn_layers", 2)
        self.visual_hier_agg = HierarchicalAttentionAggregator(
            dim=qs, num_layers=num_layers, num_heads=nh,
            depth=hier_depth, dropout=dropout,
        )
        self.text_hier_agg = HierarchicalAttentionAggregator(
            dim=qs, num_layers=num_layers, num_heads=nh,
            depth=hier_depth, dropout=dropout,
        )

        # ── 阶段 3：动态属性推理 ─────────────────────────────────────────────
        self.attribute_module = DynamicAttributeModule(config)

        # ── 阶段 4：预测头 ───────────────────────────────────────────────────
        # 输入维度 = qs（视觉 + 文本 query 均值后 cat，再平均 → qs）
        # 实际拼接视觉均值 + 文本均值 → 2*qs，先降维到 qs 再输出
        self.pred_proj = nn.Linear(qs * 2, qs)
        self.pred_head = nn.Sequential(
            nn.Linear(qs, qs),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(qs, 10),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, samples):
        """
        samples: dict
          "image"   : (B, 3, 224, 224)
          "caption" : list of str，长度 B
        """
        image    = samples["image"]
        captions = samples["caption"]
        B = image.shape[0]

        # ── 阶段 1 ──────────────────────────────────────────────────────────
        V_list = self.visual_encoder(image)        # 4 × (B, Ni, 768)
        T_list, text_attn_mask = self.text_encoder(captions)  # 4×(B,seq,768), (B,seq)

        # padding mask：True = 是 padding（需要忽略）
        pad_mask = (text_attn_mask == 0)           # (B, seq)

        # ── 阶段 2 ──────────────────────────────────────────────────────────
        all_VQ, all_TQ = [], []
        for l in range(4):
            VQ = self.visual_queries[l].expand(B, -1, -1)   # (B, nv, 768)
            TQ = self.text_queries[l].expand(B, -1, -1)     # (B, nt, 768)

            VQ, TQ = self.fusion_layers[l](
                VQ, TQ,
                V_list[l], T_list[l],
                text_pad_mask=pad_mask,
            )
            all_VQ.append(VQ)   # (B, nv, 768)
            all_TQ.append(TQ)   # (B, nt, 768)

        # 4 层输出在 query 维度上堆叠后取均值（等价于 mean pooling over layers）
        # all_VQ: list of 4 × (B, nv, 768) → stack → (B, 4*nv, 768)
        VQ_global = self.visual_hier_agg(all_VQ)             # (B, 768)
        TQ_global = self.text_hier_agg(all_TQ)               # (B, 768)

        # 取均值得到紧凑表示
        # 拼接成多模态 query，送入属性模块
        F_cat = torch.cat([VQ_global.unsqueeze(1), TQ_global.unsqueeze(1)], dim=1)  # (B,2,768)

        # ── 阶段 3 ──────────────────────────────────────────────────────────
        F_hat = self.attribute_module(image, F_cat)   # forward(image, feat): (B, 2, 768)

        # ── 阶段 4 ──────────────────────────────────────────────────────────
        # 均值聚合 F_hat 的两个 token（视觉 + 文本）
        fv = F_hat[:, 0, :]    # (B, 768)
        ft = F_hat[:, 1, :]    # (B, 768)
        feat = self.pred_proj(torch.cat([fv, ft], dim=-1))   # (B, 768)
        logits = self.pred_head(feat)                         # (B, 10)
        return self.softmax(logits)
