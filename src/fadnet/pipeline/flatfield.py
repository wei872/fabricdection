"""平场校正（design §3.2-P2）

实测样本暗角 ≈ 中心/四角 2.8×，不校正会让 U 分支异常分被照明场主导。
用最近 N 帧滚动均值估计照明场（跟随灯管缓漂），逐帧除法归一。
部署时整体 fuse 进 TensorRT/ONNX 子图，估算 <0.5 ms（design §6.1）。
"""

from __future__ import annotations

import numpy as np


class FlatFieldCorrector:
    """滚动平场校正。

    Parameters
    ----------
    win_frames : int
        滚动窗口帧数（照明场估计历史）。
    downsample : int
        照明场估计的下采样倍率（平滑 + 提速），场再上采样回全分辨率。
    eps : float
        防除零。
    """

    def __init__(self, win_frames: int = 200, downsample: int = 16, eps: float = 1.0):
        self.win_frames = win_frames
        self.downsample = downsample
        self.eps = eps
        self._acc: np.ndarray | None = None
        self._n: int = 0

    @staticmethod
    def _pool(x: np.ndarray, k: int) -> np.ndarray:
        h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
        return x[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3), dtype=np.float32)

    def _upsample(self, field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        from PIL import Image

        im = Image.fromarray(field)
        im = im.resize((shape[1], shape[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)

    def observe(self, gray: np.ndarray) -> None:
        """把（假设为正常布面的）历史帧加入照明场估计。

        条带宽度可变：场统一重采样到首个观测帧的网格后再累加。
        """
        field = self._pool(gray, self.downsample)
        if self._acc is None:
            self._acc = field
        else:
            if field.shape != self._acc.shape:
                field = self._upsample(field, self._acc.shape)
            self._acc = self._acc + field
        self._n += 1
        if self._n > self.win_frames:  # 简单截断；生产用 EMA 更稳
            self._acc, self._n = self._acc * 0.95, int(self.win_frames * 0.95)

    def correct(self, gray: np.ndarray) -> np.ndarray:
        """返回平场校正后的灰度图（float32）。"""
        if self._acc is None or self._n == 0:
            return gray.astype(np.float32)
        field = self._upsample(self._acc / self._n, gray.shape)
        gain = float(np.median(field))
        return (gray.astype(np.float32) * gain) / (field + self.eps)
