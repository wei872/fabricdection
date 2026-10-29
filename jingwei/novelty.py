"""Bounded online normal prototypes on SHARED frozen DINOv2 patch features.

This is a novelty heuristic, not a guarantee of open-set recall. No second ViT,
full patch memory bank, fabricated unknown class labels or anomaly training GT.
"""
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NormalPrototypes(nn.Module):
    def __init__(self, dim, k=32, momentum=.95, min_patches=1, max_regions=8):
        super().__init__()
        self.momentum = float(momentum)
        self.min_patches, self.max_regions = int(min_patches), int(max_regions)
        self.register_buffer("centers", torch.zeros(k, dim))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("threshold", torch.tensor(2.))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("calibration_shape", torch.zeros(2, dtype=torch.long))
        self.register_buffer("precision_bits", torch.tensor(32, dtype=torch.uint8))

    def _apply(self, fn):
        # Keep cosine centers and calibrated thresholds in FP32 even when the
        # encoder/head use FP16. Preserve original values before Module.half().
        centers, threshold = self.centers, self.threshold
        super()._apply(fn)
        self.centers = centers.to(device=self.centers.device, dtype=torch.float32)
        self.threshold = threshold.to(device=self.threshold.device, dtype=torch.float32)
        return self

    @torch.no_grad()
    def update(self, features):
        if features.numel() == 0:
            return
        patches = F.normalize(features.detach().float().permute(0, 2, 3, 1).reshape(-1, features.shape[1]), dim=1)
        # Bound training-time initialization/update work, without retaining images.
        if len(patches) > 2048:
            patches = patches[torch.linspace(0, len(patches) - 1, 2048, device=patches.device).long()]
        if self.updates.item() == 0:
            # Farthest-first seeds, with deterministic ties and no global RNG use.
            index = 0
            distance = patches.new_full((len(patches),), 2.)
            for center in self.centers:
                center.copy_(patches[index])
                distance = torch.minimum(distance, 1 - patches @ center)
                index = int(distance.argmax())
        else:
            assignment = (patches @ self.centers.float().T).argmax(1)
            sums = torch.zeros_like(self.centers, dtype=torch.float32)
            sums.index_add_(0, assignment, patches)
            counts = torch.bincount(assignment, minlength=len(self.centers)).float()
            used = counts > 0
            means = sums[used] / counts[used, None]
            self.centers[used] = F.normalize(self.momentum * self.centers[used].float()
                                            + (1 - self.momentum) * means, dim=1).to(self.centers.dtype)
        self.updates.add_(1)
        # Any update invalidates a previously calibrated threshold.
        self.calibrated.fill_(False)

    def score(self, features):
        if self.updates.item() == 0:
            raise RuntimeError("Normal prototypes are untrained; supply normaltrain during training")
        z = F.normalize(features.float(), dim=1)
        similarity = torch.einsum("bchw,kc->bkhw", z, F.normalize(self.centers.float(), dim=1))
        return (1 - similarity.amax(1)).clamp(0, 2)

    @torch.no_grad()
    def regions(self, features, img_size):
        maps = self.score(features)
        # One small device->host transfer per batch, not a full-resolution map.
        maps_np = maps.detach().float().cpu().numpy()
        threshold = float(self.threshold)
        result = []
        for score in maps_np:
            n, ids, stats, _ = cv2.connectedComponentsWithStats(
                np.ascontiguousarray(score > threshold, dtype=np.uint8), connectivity=8)
            regions = []
            h, w = score.shape
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area >= self.min_patches:
                    regions.append(([x * img_size[1] / w, y * img_size[0] / h,
                                     (x + bw) * img_size[1] / w, (y + bh) * img_size[0] / h],
                                    float(score[ids == i].max())))
            regions.sort(key=lambda item: -item[1])
            regions = regions[:self.max_regions]
            result.append(dict(unknown_boxes=features.new_tensor([r[0] for r in regions]).reshape(-1, 4),
                               unknown_scores=features.new_tensor([r[1] for r in regions])))
        return result
