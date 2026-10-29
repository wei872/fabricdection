"""Run reuse must not silently destroy checkpoints (no ML dependencies needed)."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from _run_dir import validate_output_dir


class TestRunDirectory(unittest.TestCase):
    def test_new_directory_is_allowed_without_creating_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            self.assertEqual(validate_output_dir(out), out.resolve())
            self.assertFalse(out.exists())

    def test_existing_empty_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(validate_output_dir(out), out.resolve())
            self.assertEqual(list(out.iterdir()), [])

    def test_fresh_or_weight_only_run_cannot_reuse_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            best, last = out / "best.pt", out / "last.pt"
            best.write_bytes(b"valuable previous best")
            last.write_bytes(b"previous optimizer state")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite") as error:
                validate_output_dir(out)
            self.assertIn("--resume", str(error.exception))
            self.assertIn("--pretrained", str(error.exception))
            self.assertEqual(best.read_bytes(), b"valuable previous best")
            self.assertEqual(last.read_bytes(), b"previous optimizer state")
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["best.pt", "last.pt"])

    def test_same_run_last_checkpoint_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "last.pt").write_bytes(b"state")
            (out / "best.pt").write_bytes(b"best")
            self.assertEqual(validate_output_dir(out, out / "last.pt"), out.resolve())
            self.assertEqual((out / "best.pt").read_bytes(), b"best")

    def test_resume_to_new_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "previous.pt"
            checkpoint.write_bytes(b"state")
            out = Path(tmp) / "new"
            self.assertEqual(validate_output_dir(out, checkpoint), out.resolve())
            self.assertFalse(out.exists())

    def test_foreign_checkpoint_cannot_overwrite_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "existing"
            out.mkdir()
            (out / "last.pt").write_bytes(b"existing state")
            source = root / "foreign.pt"
            source.write_bytes(b"foreign state")
            with self.assertRaisesRegex(ValueError, "NEW --out"):
                validate_output_dir(out, source)
            self.assertEqual((out / "last.pt").read_bytes(), b"existing state")

    def test_older_snapshot_requires_new_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "last.pt").write_bytes(b"epoch 10")
            snapshot = out / "epoch_0003.pt"
            snapshot.write_bytes(b"epoch 3")
            with self.assertRaisesRegex(ValueError, "older checkpoint"):
                validate_output_dir(out, snapshot)
            self.assertEqual((out / "last.pt").read_bytes(), b"epoch 10")

    def test_missing_resume_fails_without_creating_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                validate_output_dir(out, Path(tmp) / "missing.pt")
            self.assertFalse(out.exists())

    def test_output_file_is_not_treated_as_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "file"
            out.write_bytes(b"keep me")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                validate_output_dir(out)
            self.assertEqual(out.read_bytes(), b"keep me")

    def test_unicode_and_relative_paths_resolve_to_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "布匹实验"
            out.mkdir()
            checkpoint = out / "last.pt"
            checkpoint.write_bytes(b"state")
            previous = Path.cwd()
            try:
                os.chdir(tmp)  # avoid cross-drive relpath failures on Windows
                self.assertEqual(validate_output_dir(out, Path("布匹实验") / "last.pt"), out.resolve())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
