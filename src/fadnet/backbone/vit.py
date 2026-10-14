"""基础骨干装载（design §3.4 Foundation Transformer；Q8 许可注意）

优先级：
1) --ckpt 指定的 DINOv3 权重（Meta 官网申请后本地下载；用 torch.hub 结构挂载）
2) 本地已缓存的 dinov3 hub 模型
3) smoke 模式：随机初始化的 ViT-S/16（仅用于跑通管线，不产生有意义特征）
与 DinomalyReconBranch 的约定：backbone 暴露 .blocks（hook 中层特征），
patch 尺寸 16，token 序列中 patch tokens 排在最后。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, patch: int = 16, dim: int = 384):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)          # [B, N, D]


class Block(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Dropout(drop), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class SmokeViT(nn.Module):
    """随机初始化 ViT-S/16 结构（dim 384/12L/8 头），只在 --smoke 时使用。"""

    def __init__(self, img_size: int = 896, patch: int = 16, dim: int = 384,
                 depth: int = 12, heads: int = 8, n_registers: int = 4):
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch
        self.patch_embed = PatchEmbed(patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.registers = nn.Parameter(torch.zeros(1, n_registers, dim))
        n = self.grid * self.grid
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_registers + n, dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos(self, n_patches: int) -> torch.Tensor:
        """位置编码；输入尺寸与训练网格不同时双线性插值 patch 部分。"""
        p = self.pos_embed
        if n_patches == self.grid * self.grid:
            return p
        head = p[:, : 1 + self.registers.shape[1]]
        g0 = self.grid
        pe = p[:, head.shape[1]:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
        g = int(n_patches ** 0.5)
        pe = nn.functional.interpolate(pe, size=(g, g), mode="bilinear",
                                       align_corners=False)
        return torch.cat([head, pe.flatten(2).transpose(1, 2)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = self.patch_embed(x)
        t = torch.cat([self.cls_token.expand(b, -1, -1),
                       self.registers.expand(b, -1, -1), t], dim=1)
        t = t + self._pos(t.shape[1] - 1 - self.registers.shape[1])
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)


def build_backbone(name: str = "dinov3_vitsp16", ckpt: str | None = None,
                   img_size: int = 896, smoke: bool = False) -> nn.Module:
    """返回满足 .blocks-可 hook、冻结可用的 backbone。

    参数
    ----
    name : 'dinov3_vitsp16'（默认）| 'dinov3_vits16' | 'smoke_vits16'
    ckpt : 本地权重路径（Meta 申请下载后传入；torch.load 后若为完整模型直接装载）
    smoke: True 时强制走随机初始化 ViT（CI/冒烟）
    """
    if smoke or name == "smoke_vits16":
        m = SmokeViT(img_size=img_size)
        m.eval()
        return m
    if ckpt:
        # torch.hub 拉取的是结构代码（github.com 可达），权重从本地装载。
        model = torch.hub.load("facebookresearch/dinov3", name, pretrained=False)
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
    try:
        model = torch.hub.load("facebookresearch/dinov3", name, pretrained=True)
        model.eval()
        return model
    except Exception as e:                                        # noqa: BLE001
        raise RuntimeError(
            f"无法加载 {name}（DINOv3 权重需 Meta 官网申请后本地下载，见 design Q8）。"
            f"请用 --ckpt 指向本地权重，或 --smoke 冒烟。原始错误: {e}"
        ) from e
