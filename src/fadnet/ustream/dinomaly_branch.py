"""U 分支·重构通道 —— Dinomaly 风格（design §3.4.1）

对应 Dinomaly (CVPR 2025, https://github.com/guojiajeremy/Dinomaly) 的
四个"少即是多"要点：
  1) Foundation Transformer：冻结的 DINOv3 ViT（原作用 DINOv2；DINOv2 权重
     为 CC BY-NC 非商用，商用部署改用 DINOv3，接口等价，见 design Q8）
  2) Noisy Bottleneck：已经存在的 Dropout 完成全部噪声注入
  3) Linear Attention 解码器："天然不会聚焦"，不易对异常区域抄捷径
  4) Loose Reconstruction：分组宽松对应 + 余弦距离，不强制逐层逐点

训练只使用正常样本；异常图 = 各分组 (1 - cos) 均值 → 上采样 → 高斯平滑。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """线性注意力（Dinomaly 解码器构件）：elψ 特征映射，O(N·d²) 而非 O(N²·d)。"""

    def __init__(self, dim: int, heads: int = 8, **kw):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    @staticmethod
    def elu_feature(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.to_qkv(x).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # [B,H,N,Dh]
        q, k = self.elu_feature(q), self.elu_feature(k)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.to_out(out)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DinomalyReconBranch(nn.Module):
    """Dinomaly 式重构异常检测分支（冻结骨干 + 噪声瓶颈 + 线性注意力解码器）。"""

    def __init__(
        self,
        backbone: nn.Module,            # DINOv3 ViT（需暴露 forward_intermediate 或 hook）
        feat_dim: int = 384,            # ViT-S+/16 宽度
        recon_layers: tuple[int, ...] = (3, 5, 7, 9),
        decoder_layers: int = 3,
        noise_p: float = 0.2,           # Dinomaly Noisy Bottleneck = Dropout
        film: nn.Module | None = None,  # 可选 FiLM 布种条件（FC 嵌入）
    ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)     # Foundation Transformer 冻结
        self.recon_layers = recon_layers
        self.bottleneck = nn.Dropout(noise_p)  # ← Dinomaly 的"噪声注入全靠 Dropout"
        self.film = film
        self.decoders = nn.ModuleList(
            nn.Sequential(*[DecoderBlock(feat_dim) for _ in range(decoder_layers)])
            for _ in recon_layers
        )

    def _mid_tokens(self, x: torch.Tensor) -> list[torch.Tensor]:
        """取骨干中层 token（去掉 CLS/register，保留 patch 网格）。

        实现约定：backbone 通过 hook 把中层层输出写入 self._feats 字典。
        """
        feats: dict[int, torch.Tensor] = {}
        handles = []
        blocks = self.backbone.blocks  # timm/DINOv3 风格
        for i in self.recon_layers:
            def hook(_m, _in, out, i=i):
                feats[i] = out
            handles.append(blocks[i].register_forward_hook(hook))
        _ = self.backbone(x)
        for h in handles:
            h.remove()
        return [feats[i][:, -int(x.shape[-1] // 16 * (x.shape[-2] // 16)):, :]
                for i in self.recon_layers]

    def forward(self, x: torch.Tensor, fabric_embed: torch.Tensor | None = None):
        """返回 (enc_tokens per group, dec_tokens per group) 供训练/异常分使用。"""
        encs = self._mid_tokens(x)
        outs = []
        for enc, dec in zip(encs, self.decoders):
            t = self.bottleneck(enc)             # Noisy Bottleneck（训练期）
            if self.film is not None and fabric_embed is not None:
                t = self.film(t, fabric_embed)
            outs.append(dec(t))
        return encs, outs

    # ---------- 训练（design §5.2，正常样本 only） ----------
    def loss(self, encs, outs) -> torch.Tensor:
        """Loose Reconstruction：分组宽松对应的余弦距离。

        Dinomaly 原文不强制层对层/点对点；这里以组内分组为基准，
        并允许 ±1 组窗口内取最优匹配（loose），正常样本上最小化 1-cos。
        """
        loss = 0.0
        for i, (e, o) in enumerate(zip(encs, outs)):
            d = 1 - F.cosine_similarity(o, e, dim=-1)          # [B,N]
            cand = [d]
            if i > 0:
                cand.append(1 - F.cosine_similarity(o, encs[i - 1], dim=-1))
            if i + 1 < len(encs):
                cand.append(1 - F.cosine_similarity(o, encs[i + 1], dim=-1))
            loss = loss + torch.stack(cand).min(0).values.mean()
        return loss / len(encs)

    # ---------- 推理 ----------
    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor, fabric_embed: torch.Tensor | None = None
                    ) -> torch.Tensor:
        """异常热力图 A_rec：[B,1,H,W]（各组 1-cos 均值 → 上采样 → 高斯平滑）。"""
        self.eval()                                            # 推理期 Dropout 关闭
        encs, outs = self(x, fabric_embed)
        b, n, d = encs[0].shape
        g = int(n ** 0.5)
        maps = [1 - F.cosine_similarity(o, e, dim=-1).reshape(b, 1, g, g)
                for e, o in zip(encs, outs)]
        amap = torch.stack(maps).mean(0)
        amap = F.interpolate(amap, size=x.shape[-2:], mode="bilinear", align_corners=False)
        k = torch.exp(-(torch.arange(-6, 7, device=amap.device) ** 2) / (2 * 4.0 ** 2))
        k = (k / k.sum())
        amap = F.conv2d(amap, k.view(1, 1, 1, -1), padding=(0, 6))
        amap = F.conv2d(amap, k.view(1, 1, -1, 1), padding=(6, 0))
        return amap
