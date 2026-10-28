"""Unicode-safe image / video I/O for JingWei.

Why this module exists
----------------------
The production dataset lives on Windows (``D:/tyhn/yolo/...``) and its image,
label and folder names frequently contain Chinese characters.  OpenCV's
``cv2.imread`` / ``cv2.imwrite`` / ``cv2.VideoCapture`` take a narrow C string
on some Windows builds and silently fail (return ``None`` / ``False`` or raise)
for non-ASCII paths.  ``cv2.putText`` also cannot reliably draw CJK glyphs with
the built-in Hershey fonts.

The whole codebase therefore goes through the helpers below:

* :func:`imread_unicode`  -- ``np.fromfile`` + ``cv2.imdecode``;
* :func:`imwrite_unicode` -- ``cv2.imencode`` + ``ndarray.tofile``;
* :class:`VideoCaptureUnicode` -- copies a non-ASCII video to a temporary
  ASCII path before handing it to ``cv2.VideoCapture``;
* :func:`ascii_safe` / :func:`console_safe` -- keep on-image labels and console
  output ASCII-only, while Chinese class names are still preserved in JSON.

No other module should call ``cv2.imread`` / ``cv2.imwrite`` directly.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

__all__ = [
    "imread_unicode",
    "imwrite_unicode",
    "VideoCaptureUnicode",
    "ascii_safe",
    "console_safe",
]


def _as_str(path: Any) -> str:
    """``os.PathLike`` / ``str`` / ``bytes`` -> Unicode ``str``."""
    return os.fsdecode(os.fspath(path))


def imread_unicode(path: Any, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image from a path that may contain Chinese / Unicode characters.

    Uses Python's file API (which understands Unicode paths) to load the bytes,
    then OpenCV's in-memory decoder.  Returns ``None`` on any failure, matching
    ``cv2.imread`` semantics.
    """
    p = _as_str(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    try:
        return cv2.imdecode(data, int(flags))
    except cv2.error:
        return None


def imwrite_unicode(path: Any, img: np.ndarray,
                    params: Sequence[int] | None = None) -> bool:
    """Write an image to a path that may contain Chinese / Unicode characters.

    ``cv2.imencode`` produces the encoded bytes in memory; ``ndarray.tofile``
    uses Python's Unicode-aware file API to write them.  The extension is taken
    from ``path`` (``.png`` if missing).  Returns ``True`` on success.
    """
    p = _as_str(path)
    ext = Path(p).suffix or ".png"
    if not ext.startswith("."):
        ext = "." + ext
    try:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        arr = np.ascontiguousarray(img)
        ok, buf = cv2.imencode(ext, arr, list(params) if params is not None else [])
        if not ok and ext != ext.lower():
            ok, buf = cv2.imencode(ext.lower(), arr,
                                   list(params) if params is not None else [])
        if not ok:
            return False
        buf.tofile(p)
        return True
    except (OSError, ValueError, cv2.error):
        return False


class VideoCaptureUnicode:
    """``cv2.VideoCapture`` wrapper that also accepts non-ASCII video paths.

    On Windows, ``cv2.VideoCapture`` cannot reliably open a Chinese-named file.
    For non-ASCII local paths we copy the file to a temporary ASCII filename
    (same suffix) and open that instead; the temp copy is removed by
    :meth:`release`.  Network URLs are passed through unchanged.

    Usage::

        with VideoCaptureUnicode("D:/布匹视频/测试.avi") as cap:
            ok, frame = cap.read()
    """

    def __init__(self, path: Any, *args: Any, **kwargs: Any):
        self.path = _as_str(path)
        self.temp_path: str | None = None
        open_path = self.path
        is_url = "://" in self.path
        if not self.path.isascii() and not is_url:
            suffix = Path(self.path).suffix
            if not suffix or not suffix.isascii():
                suffix = ".mp4"
            fd, tmp = tempfile.mkstemp(prefix="jingwei_video_", suffix=suffix)
            os.close(fd)
            try:
                shutil.copyfile(self.path, tmp)
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            self.temp_path = tmp
            open_path = tmp
        self.cap = cv2.VideoCapture(open_path, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        cap = self.__dict__.get("cap")
        if cap is None:
            raise AttributeError(name)
        return getattr(cap, name)

    def isOpened(self) -> bool:
        return bool(self.cap.isOpened())

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        try:
            self.cap.release()
        finally:
            if self.temp_path:
                try:
                    os.remove(self.temp_path)
                except OSError:
                    pass
                self.temp_path = None

    def __enter__(self) -> "VideoCaptureUnicode":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def ascii_safe(text: Any, fallback: str = "?") -> str:
    """Return an ASCII-only string suitable for ``cv2.putText``.

    Chinese class names become ``cls<id>`` (passed as ``fallback``); the full
    Chinese names are still written to ``results.json`` / evaluation reports.
    """
    s = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = " ".join(s.split())
    return s or fallback


def console_safe(text: Any) -> str:
    """Escape every non-ASCII character so stdout works on any Windows console."""
    s = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
    return s.encode("ascii", "backslashreplace").decode("ascii")
