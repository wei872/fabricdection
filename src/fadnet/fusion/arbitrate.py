"""F-Head：融合裁决规则引擎（design §3.6）——"大缺陷不漏"的核心。

规则（按序）：
1. 异常区提取（在 heatmap.py 完成）；
2. 大缺陷升级：面积 ≥ large_area 或 最高分 ≥ tau_large → grade=critical 必报；
3. 框-区合并：IoU>0 即关联，score=max(conf, w·anomaly)，类别取检测类；
4. 孤立异常区：输出 unknown_anomaly，review_flag=True，绝不丢弃；
5. 低置信孤立框：conf<tau_det_heavy 且无异常区支撑 → 降级 review（不计自动判废）；
6. 跨块归并：全局坐标 IoU≥0.5 去重，聚合米级台账。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..ustream.heatmap import AnomRegion

UNKNOWN_CLASS = "unknown_anomaly"


@dataclass
class DetBox:
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    cls: str


@dataclass
class Verdict:
    box: tuple[float, float, float, float]
    cls: str
    conf: float
    anomaly_score: float
    grade: str                      # critical | major | minor | review
    evidence: dict = field(default_factory=dict)   # {"det": bool, "anom": bool}

    def to_json(self) -> dict:
        return {"box_xywh": [self.box[0], self.box[1],
                             self.box[2] - self.box[0], self.box[3] - self.box[1]],
                **asdict(self)}


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    ar_a = (a[2] - a[0]) * (a[3] - a[1])
    ar_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar_a + ar_b - inter)


class FHead:
    def __init__(self, tau_det_low: float = 0.05, tau_det_heavy: float = 0.5,
                 large_area: int = 2500, tau_large: float = 0.9, w: float = 1.0):
        self.tau_det_low, self.tau_det_heavy = tau_det_low, tau_det_heavy
        self.large_area, self.tau_large, self.w = large_area, tau_large, w

    def fuse(self, dets: list[DetBox], regions: list[AnomRegion]) -> list[Verdict]:
        verdicts: list[Verdict] = []
        used_r: set[int] = set()
        dets = [d for d in dets if d.conf >= self.tau_det_low]           # 截面先低阈值
        for d in dets:
            db = (d.x0, d.y0, d.x1, d.y1)
            hit, hit_i = 0.0, -1
            for i, r in enumerate(regions):
                rb = (r.x0, r.y0, r.x1, r.y1)
                v = _iou(db, rb)
                if v > hit:                      # hit 初值 0：仅 IoU>0 才关联（规则 3）
                    hit, hit_i = v, i
            if hit_i >= 0:                                               # 规则 3
                r = regions[hit_i]
                used_r.add(hit_i)
                conf = max(d.conf, self.w * r.score)
                grade = "critical" if r.area >= self.large_area or r.score >= self.tau_large \
                    else ("major" if conf >= self.tau_det_heavy else "minor")
                verdicts.append(Verdict(db, d.cls, conf, r.score, grade,
                                        {"det": True, "anom": True}))
            elif d.conf >= self.tau_det_heavy:
                verdicts.append(Verdict(db, d.cls, d.conf, 0.0, "major",
                                        {"det": True, "anom": False}))
            else:
                verdicts.append(Verdict(db, d.cls, d.conf, 0.0, "review",   # 规则 5
                                        {"det": True, "anom": False}))
        for i, r in enumerate(regions):                                   # 规则 4/2
            if i in used_r:
                continue
            grade = "critical" if (r.area >= self.large_area or r.score >= self.tau_large) \
                else "review"
            verdicts.append(Verdict((r.x0, r.y0, r.x1, r.y1), UNKNOWN_CLASS,
                                    r.score, r.score, grade, {"det": False, "anom": True}))
        return verdicts

    @staticmethod
    def stitch(verdicts_global: list[Verdict], iou_th: float = 0.5) -> list[Verdict]:
        """跨块归并（规则 6）：全局坐标 IoU≥iou_th 去重，critical 优先。"""
        out = sorted(verdicts_global, key=lambda v: -v.conf)
        keep: list[Verdict] = []
        while out:
            cur = out.pop(0)
            keep.append(cur)
            out = [v for v in out if _iou(cur.box, v.box) < iou_th]
        return keep
