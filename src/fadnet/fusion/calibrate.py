"""逐类独立置信度标定（用户要求 2026-10：置信度与缺陷严重程度关联，且每类独立）。

设计：
  * 每个缺陷类 c 拥有独立温度系数 T_c：conf_cal = sigmoid(logit(conf)/T_c)，
    在验证集上按类拟合（最小化 NLL），使 conf_cal ≈ P(该类缺陷确实存在)。
  * 严重度 severity ∈[0,1] 为该类独立的单调组合：
        sev = w1·conf_cal + w2·clip(anom_z/6) + w3·log1p(area)/log1p(A_ref)
    三项分别对应"检测确信、U 分支异常强度、缺陷尺寸"，权重逐类可配。
  * 等级由逐类严重度门限给出（sev_crit/sev_maj）；F-Head 的"大缺陷必报"规则
    优先级更高，标定只影响非必报路径的分级与对外置信度数值。
拟合数据来自 scripts/eval_metrics.py 导出的验证集匹配记录（cls/conf/matched/area/anom）。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ClassCalib:
    temperature: float = 1.0                 # 独立温度系数（>1 收缩、<1 放大置信度）
    sev_w: tuple[float, float, float] = (0.5, 0.3, 0.2)
    sev_crit: float = 0.70                   # 该类 severity ≥ → critical
    sev_maj: float = 0.45                    # ≥ → major，否则 minor


class PerClassCalibrator:
    def __init__(self, table: dict[str, ClassCalib] | None = None):
        self.table: dict[str, ClassCalib] = table or {}

    def get(self, cls: str) -> ClassCalib:
        return self.table.setdefault(cls, ClassCalib())

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(float(p), 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def cal_conf(self, cls: str, conf: float) -> float:
        t = self.get(cls).temperature
        return 1.0 / (1.0 + math.exp(-self._logit(conf) / t))

    def severity(self, cls: str, conf_cal: float, anom_z: float, area: float) -> float:
        w = self.get(cls).sev_w
        return (w[0] * conf_cal
                + w[1] * min(max(anom_z, 0.0) / 6.0, 1.0)
                + w[2] * min(math.log1p(max(area, 0.0)) / math.log1p(25000.0), 1.0))

    def grade(self, cls: str, sev: float) -> str:
        c = self.get(cls)
        return "critical" if sev >= c.sev_crit else ("major" if sev >= c.sev_maj else "minor")

    # ---------- 拟合 ----------
    def fit_temperatures(self, records: list[dict], grid: tuple[float, float, float] = (0.3, 3.0, 0.1)) -> "PerClassCalibrator":
        """records: [{cls, conf, matched(bool)}]；逐类网格搜索 T_c 最小化 NLL。"""
        lo, hi, step = grid
        ts = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
        for cls in sorted({r["cls"] for r in records}):
            rs = [r for r in records if r["cls"] == cls]
            if len(rs) < 20:
                continue
            best_t, best_nll = None, math.inf
            for t in ts:
                nll = 0.0
                for r in rs:
                    p = 1.0 / (1.0 + math.exp(-self._logit(r["conf"]) / t))
                    y = 1.0 if r["matched"] else 0.0
                    nll -= y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
                nll /= len(rs)
                if nll < best_nll:
                    best_t, best_nll = t, nll
            if best_t:
                self.get(cls).temperature = round(best_t, 2)
        return self

    # ---------- 存取 ----------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(
            {k: asdict(v) for k, v in self.table.items()}, allow_unicode=True))

    @classmethod
    def load(cls, path: str | Path) -> "PerClassCalibrator":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text()) or {}
        return cls({k: ClassCalib(**{kk: (tuple(vv) if kk == "sev_w" else vv)
                                     for kk, vv in v.items()}) for k, v in raw.items()})
