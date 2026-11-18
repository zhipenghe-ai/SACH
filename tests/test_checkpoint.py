"""Checkpoint I/O tests using the installed PyTorch and temporary paths."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sach import checkpoint


FILENAME = "MIRFlickr_16bits.pth"


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        # The Windows app sandbox allows atomic replacement inside the workspace.
        release_root = Path(__file__).resolve().parents[1]
        directory = tempfile.TemporaryDirectory(dir=release_root)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)
        self.image = nn.Linear(3, 2)
        self.text = nn.Linear(4, 2)

    def assert_state_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_creates_missing_directories_and_round_trips_both_heads(self):
        parent = self.root / "new" / "checkpoints"
        path = checkpoint.create_checkpoint_path(parent, FILENAME)
        self.assertFalse(path.exists())
        self.assertEqual(path.name, FILENAME)
        self.assertEqual(path.parent.parent, parent.resolve())
        expected_image = {key: value.clone() for key, value in self.image.state_dict().items()}
        expected_text = {key: value.clone() for key, value in self.text.state_dict().items()}
        torch_state = torch.get_rng_state()

        self.assertEqual(checkpoint.save_hash_heads(path, self.image, self.text), path)
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)
        self.assert_state_equal(self.image.state_dict(), expected_image)
        self.assert_state_equal(self.text.state_dict(), expected_text)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)

        image = nn.Linear(3, 2)
        text = nn.Linear(4, 2)
        self.assertEqual(checkpoint.load_hash_heads(path, image, text), path)
        self.assert_state_equal(image.state_dict(), expected_image)
        self.assert_state_equal(text.state_dict(), expected_text)

    def test_loads_legacy_format_from_original_flat_path(self):
        path = self.root / FILENAME
        torch.save({"ImgNet": self.image.state_dict(), "TxtNet": self.text.state_dict()}, path)

        image = nn.Linear(3, 2)
        text = nn.Linear(4, 2)
        checkpoint.load_hash_heads(path, image, text)
        self.assert_state_equal(image.state_dict(), self.image.state_dict())
        self.assert_state_equal(text.state_dict(), self.text.state_dict())

    def test_new_best_only_replaces_checkpoint_owned_by_the_run(self):
        legacy = self.root / FILENAME
        legacy.write_bytes(b"historical checkpoint sentinel")
        first_run = checkpoint.create_checkpoint_path(self.root, FILENAME)
        second_run = checkpoint.create_checkpoint_path(self.root, FILENAME)
        self.assertNotEqual(first_run.parent, second_run.parent)
        checkpoint.save_hash_heads(first_run, self.image, self.text)
        checkpoint.save_hash_heads(second_run, self.image, self.text)
        second_bytes = second_run.read_bytes()

        with torch.no_grad():
            self.image.weight.add_(1)
        checkpoint.save_hash_heads(first_run, self.image, self.text)

        image = nn.Linear(3, 2)
        checkpoint.load_hash_heads(first_run, image, nn.Linear(4, 2))
        self.assert_state_equal(image.state_dict(), self.image.state_dict())
        self.assertEqual(second_run.read_bytes(), second_bytes)
        self.assertEqual(legacy.read_bytes(), b"historical checkpoint sentinel")

    def test_rejects_invalid_parent_and_escaping_filename(self):
        invalid_parent = self.root / "regular-file"
        invalid_parent.touch()
        with self.assertRaisesRegex(OSError, "Could not create a checkpoint directory"):
            checkpoint.create_checkpoint_path(invalid_parent, FILENAME)
        with self.assertRaisesRegex(ValueError, "must be a basename"):
            checkpoint.create_checkpoint_path(self.root, "../outside.pth")

    def test_unwritable_target_does_not_publish_a_checkpoint(self):
        path = checkpoint.create_checkpoint_path(self.root, FILENAME)
        with (
            patch.object(
                checkpoint.tempfile, "NamedTemporaryFile", side_effect=PermissionError("denied")
            ),
            self.assertRaisesRegex(RuntimeError, "Could not save checkpoint.*denied"),
        ):
            checkpoint.save_hash_heads(path, self.image, self.text)
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.iterdir()), [])

    def test_serialization_failure_removes_temporary_and_preserves_old_best(self):
        path = checkpoint.create_checkpoint_path(self.root, FILENAME)
        checkpoint.save_hash_heads(path, self.image, self.text)
        expected_bytes = path.read_bytes()

        def fail_after_writing(_payload, stream):
            stream.write(b"incomplete checkpoint")
            raise OSError("serializer ran out of space")

        with (
            patch.object(checkpoint.torch, "save", side_effect=fail_after_writing),
            self.assertRaisesRegex(RuntimeError, "Could not save checkpoint.*out of space"),
        ):
            checkpoint.save_hash_heads(path, self.image, self.text)
        self.assertEqual(path.read_bytes(), expected_bytes)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_publication_failure_removes_temporary(self):
        path = checkpoint.create_checkpoint_path(self.root, FILENAME)
        with (
            patch.object(
                checkpoint.os, "replace", side_effect=PermissionError("publication denied")
            ),
            self.assertRaisesRegex(RuntimeError, "Could not save checkpoint.*publication denied"),
        ):
            checkpoint.save_hash_heads(path, self.image, self.text)
        self.assertEqual(list(path.parent.iterdir()), [])

    def test_invalid_second_head_leaves_first_head_unchanged(self):
        path = self.root / FILENAME
        torch.save(
            {"ImgNet": self.image.state_dict(), "TxtNet": nn.Linear(8, 2).state_dict()}, path
        )
        image = nn.Linear(3, 2)
        text = nn.Linear(4, 2)
        before = {key: value.clone() for key, value in image.state_dict().items()}

        with self.assertRaisesRegex(RuntimeError, r"Could not load checkpoint.*TxtNet\.weight"):
            checkpoint.load_hash_heads(path, image, text)
        self.assert_state_equal(image.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
