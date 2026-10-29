"""Deterministic fusion + the large-defect safety net.

This module is the reason the design can *guarantee* that a large defect is
never silently dropped: the rule

    every connected anomalous region above an area threshold that is not
    covered by a detection box is emitted as a box (label = 未知异常).
    With ``fusion.emit_unknown_boxes`` this also covers SMALLER unknown defect
    types, not just huge anomalies; large components are never dropped.

is deterministic, not learned. Recall for large defects is therefore bounded
below by the anomaly map's recall, not by the detector's top-k or confidence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import box_iou


# --------------------------------------------------------------------------------------
# dependency-free ROI align (grid_sample) — avoids a torchvision dependency
# --------------------------------------------------------------------------------------
def roi_pool_features(feat: torch.Tensor, boxes: torch.Tensor, img_size,
                      out_size: int = 7) -> torch.Tensor:
    """Average-pooled ROI features.

    Args:
        feat: (1, C, Hf, Wf)
        boxes: (N, 4) xyxy in input-image pixels
        img_size: (H, W) of the input image
    Returns:
        (N, C) ROI descriptors.
    """
    if boxes.numel() == 0:
        return feat.new_zeros((0, feat.shape[1]))
    N = boxes.shape[0]
    H, W = img_size
    b = boxes.float()
    x0 = (b[:, 0] / W).clamp(0, 1)
    y0 = (b[:, 1] / H).clamp(0, 1)
    x1 = (b[:, 2] / W).clamp(0, 1)
    y1 = (b[:, 3] / H).clamp(0, 1)
    # guard against degenerate boxes
    x1 = torch.maximum(x1, x0 + 1.0 / W)
    y1 = torch.maximum(y1, y0 + 1.0 / H)

    t = torch.linspace(0, 1, out_size, device=feat.device, dtype=feat.dtype)
    gx = x0[:, None] + (x1 - x0)[:, None] * t[None, :]        # (N, out)
    gy = y0[:, None] + (y1 - y0)[:, None] * t[None, :]
    # explicit per-ROI sampling grid
    gx = gx[:, None, :].expand(N, out_size, out_size)
    gy = gy[:, :, None].expand(N, out_size, out_size)
    grid = torch.stack([gx * 2 - 1, gy * 2 - 1], -1)          # (N, out, out, 2), [-1,1]
    f = feat.expand(N, -1, -1, -1)
    sampled = F.grid_sample(f, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)               # (N, C, out, out)
    return sampled.mean(dim=(2, 3))


class ROIRefiner(nn.Module):
    """Small second-stage classifier for precise 19-class + 未知 discrimination."""

    def __init__(self, cfg, channels):
        super().__init__()
        self.nc = int(cfg.data.nc)
        hidden = int(cfg.fusion.roi_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(channels[1] + channels[3], hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.nc + 1))
        self.prior_weight = float(cfg.fusion.roi_prior_weight)

    def forward(self, feats, boxes, img_size, scores, labels):
        if boxes.numel() == 0:
            return scores, labels
        f3 = roi_pool_features(feats["n3"], boxes, img_size)
        f5 = roi_pool_features(feats["n5"], boxes, img_size)
        logits = self.mlp(torch.cat([f3, f5], -1))
        idx = torch.arange(boxes.shape[0], device=boxes.device)
        prior = torch.logit(scores.clamp(1e-4, 1 - 1e-4))
        logits = logits.clone()
        logits[idx, labels] += self.prior_weight * prior
        probs = logits.softmax(-1)
        return probs.max(-1).values, probs.argmax(-1)


# --------------------------------------------------------------------------------------
# Connected-component safety net
# --------------------------------------------------------------------------------------
def safety_net_numpy(anomaly: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                     labels: np.ndarray, img_size, cfg, unknown_label: int,
                     context: np.ndarray | None = None) -> dict:
    """Core deterministic rule. All arrays are numpy; boxes are xyxy pixels.

    Args:
        anomaly: (G, G) float in [0, 1], already gated by fabric/edge context
        boxes:   (N, 4) xyxy pixels
        scores:  (N,) float
        labels:  (N,) int
        context: (3, G, G) = [defect_fg, fabric_mask, fold_band], or None.
                 Learned local+global attention maps.  ``fold_band`` follows the
                 predicted cloth boundary (not fixed image-border pixels), so a
                 moving 布幅 / normal fold-in is rejected as background while
                 real edge defects (high defect_fg / high peak) still survive.
    """
    import cv2

    f = cfg.fusion
    fcfg = cfg.get("foreground", {}) or {}
    G = anomaly.shape[0]
    H, W = img_size
    step_x, step_y = W / G, H / G
    if context is None:
        defect_fg = np.ones_like(anomaly, np.float32)
        fabric_mask = np.ones_like(anomaly, np.float32)
        fold_band = np.zeros_like(anomaly, np.float32)
    else:
        defect_fg, fabric_mask, fold_band = (np.asarray(context[0], np.float32),
                                             np.asarray(context[1], np.float32),
                                             np.asarray(context[2], np.float32))
    fabric_thresh = float(fcfg.get("fabric_mask_thresh", 0.30))
    fold_thresh = float(fcfg.get("fold_band_thresh",
                                 fcfg.get("edge_band_thresh", 0.08)))
    fg_thresh = float(fcfg.get("defect_fg_thresh", 0.25))
    min_fabric_frac = float(fcfg.get("min_fabric_frac", 0.50))
    fold_aspect = float(fcfg.get("fold_aspect_ratio", 3.0))
    use_aspect_cue = bool(fcfg.get("use_aspect_cue", False))
    fold_peak_override = float(fcfg.get("fold_peak_override",
                                        fcfg.get("edge_peak_override", 0.80)))
    # Unknown defect types: emit a box for ANY anomalous component that is not
    # already covered by a supervised detection, not just for huge ones.
    emit_unknown = bool(f.get("emit_unknown_boxes", True))
    unknown_ratio = float(f.get("unknown_min_area_ratio",
                                f.get("large_area_ratio", 0.001)))
    unknown_peak_thresh = float(f.get("unknown_peak_thresh",
                                      f.get("anomaly_thresh", 0.5)))
    max_unknown = int(f.get("max_unknown_boxes", 20))
    mask = (anomaly >= float(f.anomaly_thresh)).astype(np.uint8)

    # dilate slightly to connect fragmented elongated defects (断纬 / 稀缝)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    n_cc, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    large_min = max(1, int(float(f.large_area_ratio) * G * G))
    unknown_min = max(1, int(float(unknown_ratio) * G * G))
    min_area_cells = min(large_min, unknown_min) if emit_unknown else large_min
    margin = float(f.box_margin)
    # Guard against a GLOBAL anomaly (whole fabric darker / light flicker / a
    # mis-calibrated threshold).  Such a frame produces ONE connected component
    # covering the entire image, and the old code duly emitted a full-frame
    # 未知异常 box -- useless for the operator and a guaranteed false alarm.
    # A real defect never fills the whole field of view, so components above
    # this fraction are treated as a global event, not a defect.
    max_area_ratio = float(f.get("max_component_area_ratio", 0.90))
    max_area_cells = int(max_area_ratio * G * G) if max_area_ratio < 1.0 else G * G
    n_global = 0

    # Large components are NEVER dropped (recall guarantee).  Smaller unknown
    # candidates are kept up to ``max_unknown_boxes`` by peak anomaly score.
    large_extra, unknown_extra = [], []
    boost = np.zeros(len(scores), np.float32)

    for i in range(1, n_cc):
        x, y, w, h, area = stats[i]
        if area < min_area_cells:
            continue
        if area > max_area_cells:
            n_global += 1                         # global event, see above
            continue
        is_large = area >= large_min
        is_unknown = (emit_unknown and (not is_large) and
                      area >= unknown_min)
        if not (is_large or is_unknown):
            continue
        region_mask = lab == i
        region = anomaly[region_mask]
        peak = float(region.max())
        if peak < float(f.anomaly_thresh):
            continue
        if is_unknown and peak < unknown_peak_thresh:
            continue                              # weak noise, not a defect

        # ---- foreground context: reject background / normal edge fold-in ----
        valid_frac = float((fabric_mask[region_mask] >= fabric_thresh).mean())
        fold_frac = float((fold_band[region_mask] >= fold_thresh).mean())
        fg_frac = float((defect_fg[region_mask] >= fg_thresh).mean())
        if valid_frac < min_fabric_frac:
            continue                              # outside the cloth / 背景
        aspect = max(w, h) / max(1, min(w, h))
        # Primary decision is the LEARNED dynamic fold_band + defect_fg.
        # The old aspect-ratio heuristic is optional and off by default.
        fold_like = (fold_frac > fold_thresh and fg_frac < fg_thresh)
        if use_aspect_cue:
            fold_like = fold_like and aspect >= fold_aspect
        # A real edge defect survives if the learned foreground says so, or if
        # the anomaly peak is simply too strong to be a normal fold shadow.
        if fold_like and peak < fold_peak_override:
            continue

        # component bbox -> pixel coords, with a small margin
        cx = (x + w / 2) * step_x
        cy = (y + h / 2) * step_y
        bw = max(w * step_x, float(f.min_box_px)) * margin
        bh = max(h * step_y, float(f.min_box_px)) * margin
        cb = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], np.float32)
        cb[[0, 2]] = np.clip(cb[[0, 2]], 0, W)
        cb[[1, 3]] = np.clip(cb[[1, 3]], 0, H)

        if len(boxes):
            tboxes = torch.as_tensor(boxes, dtype=torch.float32)
            iou = box_iou(tboxes, torch.as_tensor(cb[None], dtype=torch.float32))[:, 0].numpy()
            hit = iou > float(f.min_iou_overlap)
        else:
            iou = np.zeros(0, np.float32)
            hit = np.zeros(0, bool)

        score = min(0.999, float(f.unknown_score_scale) * peak)
        if hit.any():
            # supervised head already found it -> reinforce its confidence.
            # A fold-like low-foreground component gets only a weak boost.
            boost_gain = 1.0 if not fold_like else 0.25
            boost[hit] = np.maximum(boost[hit],
                                    float(f.score_boost) * peak * boost_gain)
        else:
            # *** the guarantee: emit a box anyway, labelled 未知异常 ***
            (large_extra if is_large else unknown_extra).append(
                (float(peak), cb, score, unknown_label))

    if max_unknown > 0 and len(unknown_extra) > max_unknown:
        unknown_extra.sort(key=lambda t: -t[0])
        unknown_extra = unknown_extra[:max_unknown]
    all_extra = large_extra + unknown_extra
    extra_boxes = [t[1] for t in all_extra]
    extra_scores = [t[2] for t in all_extra]
    extra_labels = [t[3] for t in all_extra]
    n_unknown_boxes = len(unknown_extra)

    if extra_boxes:
        boxes = np.concatenate([boxes, np.stack(extra_boxes, 0)], 0)
        scores = np.concatenate([scores, np.asarray(extra_scores, np.float32)], 0)
        labels = np.concatenate([labels, np.asarray(extra_labels, np.int64)], 0)
        boost = np.concatenate([boost, np.zeros(len(extra_boxes), np.float32)], 0)

    scores = np.clip(scores + boost * (1.0 - scores), 0.0, 1.0)
    return dict(boxes=boxes, scores=scores, labels=labels,
                n_safety_boxes=len(extra_boxes),
                n_unknown_boxes=n_unknown_boxes,
                n_global_components=n_global)


def fuse_detections(det: dict, anomaly: torch.Tensor, img_size, cfg,
                    unknown_label: int, feats=None, roi_refiner=None,
                    context=None) -> dict:
    """Torch wrapper around :func:`safety_net_numpy` (+ optional ROI refiner).

    ``context`` is (1,3,G,G) = [defect_fg, fabric_mask, fold_band] from the
    deployed local+global attention head; it is what suppresses background and
    normal edge fold-in without assuming fixed image-border pixels.
    """
    device = det["boxes"].device
    n_in = int(det["boxes"].shape[0])
    scale_idx = det.get("scale_idx")
    a = anomaly[0, 0].detach().cpu().numpy().astype(np.float32)
    ctx = context[0].detach().cpu().numpy().astype(np.float32) if context is not None else None
    out = safety_net_numpy(
        a,
        det["boxes"].detach().cpu().numpy().astype(np.float32),
        det["scores"].detach().cpu().numpy().astype(np.float32),
        det["labels"].detach().cpu().numpy().astype(np.int64),
        img_size, cfg, unknown_label, context=ctx)

    boxes = torch.as_tensor(out["boxes"], dtype=torch.float32, device=device)
    scores = torch.as_tensor(out["scores"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(out["labels"], dtype=torch.long, device=device)

    if roi_refiner is not None and feats is not None and boxes.numel():
        scores, labels = roi_refiner(feats, boxes, img_size, scores, labels)

    keep = scores >= float(cfg.fusion.final_score_thresh)
    # keep the (P3/P5) scale of every box: the FP breakdown per scale is what
    # tells you whether a small scale is flooding the frame.
    if scale_idx is not None and scale_idx.numel() == n_in:
        n_extra = int(boxes.shape[0]) - n_in
        if n_extra > 0:
            scale_idx = torch.cat([scale_idx.to(device),
                                   torch.full((n_extra,), -1, dtype=torch.long,
                                              device=device)])
        else:
            scale_idx = scale_idx.to(device)
    result = dict(boxes=boxes[keep], scores=scores[keep], labels=labels[keep],
                  n_safety_boxes=out["n_safety_boxes"],
                  n_unknown_boxes=out.get("n_unknown_boxes", 0),
                  n_global_components=out.get("n_global_components", 0))
    if scale_idx is not None and scale_idx.numel() == keep.numel():
        result["scale_idx"] = scale_idx[keep]
    return result
