"""Non-destructive output-directory checks, independent of torch/model loading."""
from pathlib import Path


def validate_output_dir(out, resume=None) -> Path:
    """Refuse accidental run reuse; continuation uses that run's last.pt.

    Paths are resolved before comparison (the CLI resolves relative paths from
    the repository root). No directories/files are created, moved or removed.
    This preflight is not a cross-process lock against concurrent trainers.
    """
    out = Path(out).resolve()
    checkpoint = Path(resume).resolve() if resume is not None else None
    if out.exists() and not out.is_dir():
        raise ValueError(f"--out is not a directory: {out}")
    if checkpoint is not None and not checkpoint.is_file():
        raise ValueError(f"--resume checkpoint does not exist: {checkpoint}")
    if out.is_dir() and next(out.iterdir(), None) is not None:
        if checkpoint != out / "last.pt":
            raise ValueError(
                f"--out is not empty: {out}. Refusing to overwrite existing run files. "
                "To continue this run, use --resume with this directory's last.pt. "
                "For a fresh run, --pretrained warm start, or a different/older checkpoint, "
                "choose a NEW --out directory. No files were changed."
            )
    return out
