"""Exercise evaluation with real split files and two-head checkpoints."""

import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import warnings

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sach import data, evaluation, models
from test_regression import _cpu_adapter, _load_definitions, _Log, _trainer_namespace


RELEASE_ROOT = Path(__file__).resolve().parents[1]
BITS = 4
BATCH_SIZE = 3
DATASETS = {
    "MIRFlickr": ("MIRFlickr", "mir", "get_loader_flickr"),
    "NUSWIDE": ("NUS-WIDE", "nus", "get_loader_nus"),
    "COCO": ("MSCOCO", "COCO", "get_loader_coco"),
}


class StandaloneEvaluationTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory(dir=RELEASE_ROOT)
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.data_root = self.root / "precomputed feature splits"
        self.log_root = self.root / "output" / "evaluation logs"
        self.checkpoint = self.root / "hash heads.pth"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(731)
            image = models.Img_Net(BITS, 7)
            text = models.Txt_Net(BITS, 5)
        torch.save({"ImgNet": image.state_dict(), "TxtNet": text.state_dict()}, self.checkpoint)

    def _dataset(self, dataset="MIRFlickr", include_train=False):
        folder, prefix, _ = DATASETS[dataset]
        directory = self.data_root / folder
        directory.mkdir(parents=True)
        split_paths = {}
        for seed, (split, suffix, count) in enumerate(
            (("train", "tr", 8), ("query", "te", 5), ("database", "db", 7)), start=8
        ):
            if split == "train" and not include_train:
                continue
            rng = np.random.default_rng(seed)
            matrices = {
                "I_" + suffix: rng.normal(size=(count, 7)),
                "T_" + suffix: rng.normal(size=(count, 5)),
                "L_" + suffix: np.eye(3)[np.arange(count) % 3],
            }
            path = directory / f"{prefix}_{split}.mat"
            split_paths[split] = path
            if dataset == "COCO" and split == "database":
                with h5py.File(path, "w", libver="latest") as handle:
                    for key, values in matrices.items():
                        handle[key] = values.T
            else:
                sio.savemat(path, matrices)
        return split_paths

    def _command(self, dataset="MIRFlickr", legacy=False, **overrides):
        entry = ["main_mir.py", "--TRAIN", "False"] if legacy else ["-m", "sach.evaluate"]
        options = {
            "DATASET": dataset,
            "CHECKPOINT": self.checkpoint,
            "DATA_ROOT": self.data_root,
            "LOG_DIR": self.log_root,
            "DEVICE": "cpu",
            "HASH_BIT": BITS,
            "BATCH_SIZE": BATCH_SIZE,
        }
        options.update(overrides)
        arguments = [
            part
            for key, value in options.items()
            if value is not None
            for part in (f"--{key}", str(value))
        ]
        return subprocess.run(
            [sys.executable, "-B", *entry, *arguments],
            cwd=RELEASE_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _scores(self, output):
        self.assertNotIn("mAP@50", output)
        scores = []
        for direction in ("I->T", "T->I"):
            match = re.search(rf"MAP [(]full gallery[)] {direction}: *([0-9.eE+-]+)", output)
            self.assertIsNotNone(match, output)
            scores.append(float(match.group(1)))
        return tuple(scores)

    def _assert_failed(self, result, detail):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(detail.lower(), result.stderr.lower())
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("MAP (full gallery)", result.stdout)

    def _reference_scores(self, dataset):
        # Prefer the protected Phase 2C code. The in-training evaluator stays available
        # for a check when this release is used outside the audit workspace.
        archived = os.environ.get("SACH_PHASE2C")
        sibling = RELEASE_ROOT.parent / "SACH_release_phase2c" / "sach"
        if archived or sibling.is_dir():
            module_root = Path(archived) / "sach" if archived else sibling
            data_namespace = {
                "Path": Path,
                "np": np,
                "sio": sio,
                "h5py": h5py,
                "torch": torch,
                "DataLoader": DataLoader,
                "Dataset": Dataset,
            }
            baseline_data = _load_definitions(module_root / "data.py", data_namespace)
            baseline_eval = _load_definitions(
                module_root / "evaluation.py",
                {"np": np, "torch": torch, "F": F, "Variable": Variable},
            )
        else:
            baseline_data, baseline_eval = data, evaluation
            module_root = RELEASE_ROOT / "sach"
        trainer = _load_definitions(
            module_root / "trainer.py", _trainer_namespace(models, baseline_eval)
        ).PCIRH
        adapter = trainer.__new__(trainer)
        adapter.log = _Log()
        with torch.random.fork_rng(devices=[]):
            adapter.imgnet = models.Img_Net(BITS, 7)
            adapter.txtnet = models.Txt_Net(BITS, 5)
        payload = torch.load(self.checkpoint, map_location="cpu")
        adapter.imgnet.load_state_dict(payload["ImgNet"])
        adapter.txtnet.load_state_dict(payload["TxtNet"])
        _, _, loader = DATASETS[dataset]
        with warnings.catch_warnings(), _cpu_adapter():
            warnings.filterwarnings("ignore", message=".*np[.]float.*deprecated alias")
            splits, _ = getattr(baseline_data, loader)(BATCH_SIZE, self.data_root)
            adapter.test_loader = splits["query"]
            adapter.database_loader = splits["database"]
            return adapter.performance_eval()[:2]

    def test_fresh_cpu_process_matches_existing_evaluator_for_each_preset(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                paths = self._dataset(dataset, include_train=True)
                expected = self._reference_scores(dataset)
                paths["train"].unlink()
                checkpoint_bytes = self.checkpoint.read_bytes()
                result = self._command(dataset)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self._scores(result.stdout), expected)
                log_file = self.log_root / f"evaluation_{dataset}_{BITS}bits.txt"
                logged = log_file.read_text(encoding="utf-8")
                self.assertEqual(self._scores(logged), expected)
                self.assertIn(str(self.checkpoint), logged)
                self.assertIn("cpu", logged.lower())
                self.assertFalse(paths["train"].exists())
                self.assertEqual(self.checkpoint.read_bytes(), checkpoint_bytes)

    def test_legacy_false_invocation_appends_to_evaluation_log(self):
        self._dataset()
        first = self._command()
        self.assertEqual(first.returncode, 0, first.stderr)
        path = self.log_root / "evaluation_MIRFlickr_4bits.txt"
        first_log = path.read_text(encoding="utf-8")
        second = self._command(legacy=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self._scores(first.stdout), self._scores(second.stdout))
        appended = path.read_text(encoding="utf-8")
        self.assertTrue(appended.startswith(first_log))
        self.assertEqual(appended.count("MAP (full gallery) I->T:"), 2)
        self.assertEqual(set(self.log_root.iterdir()), {path})

    def test_cli_counts_relevant_items_beyond_rank_50_in_both_directions(self):
        bits = 53
        width = 2 * bits
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            image = models.Img_Net(bits, width)
            text = models.Txt_Net(bits, width)
            features = torch.arange(width)
            codes = torch.arange(bits)
            for head in (image, text):
                head.fc1.weight.zero_()
                head.fc1.bias.zero_()
                head.fc1.weight[features, features] = 1
                head.tohash.weight.zero_()
                head.tohash.bias.zero_()
                head.tohash.weight[codes, codes] = 1
                head.tohash.weight[codes, codes + bits] = -1
        torch.save({"ImgNet": image.state_dict(), "TxtNet": text.state_dict()}, self.checkpoint)

        def indicators(positive):
            return np.concatenate((positive, ~positive), axis=1).astype(np.float64)

        directory = self.data_root / "MIRFlickr"
        directory.mkdir(parents=True)
        all_positive = indicators(np.ones((1, bits), dtype=bool))
        text_features = indicators(np.arange(bits) >= np.arange(bits)[:, None])
        image_features = text_features.copy()
        image_features[[50, 52]] = image_features[[52, 50]]
        labels = np.tile([0, 1], (bits, 1))
        labels[-1] = [1, 0]
        sio.savemat(
            directory / "mir_query.mat",
            {"I_te": all_positive, "T_te": all_positive, "L_te": np.array([[1, 0]])},
        )
        sio.savemat(
            directory / "mir_database.mat",
            {"I_db": image_features, "T_db": text_features, "L_db": labels},
        )
        result = self._command(HASH_BIT=bits)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._scores(result.stdout), (1 / 53, 1 / 51))

    def test_missing_or_omitted_checkpoint_fails_without_metrics(self):
        self._dataset()
        absent = self.root / "missing.pth"
        self._assert_failed(self._command(CHECKPOINT=absent), absent.name)
        self._assert_failed(self._command(CHECKPOINT=None), "--CHECKPOINT")
        self.assertFalse((self.log_root / "evaluation_MIRFlickr_4bits.txt").exists())

    def test_corrupt_and_incompatible_checkpoints_report_context(self):
        self._dataset()
        corrupt = self.root / "corrupt.pth"
        corrupt.write_bytes(b"not a checkpoint")
        self._assert_failed(self._command(CHECKPOINT=corrupt), corrupt.name)
        self._assert_failed(self._command(HASH_BIT=8), self.checkpoint.name)

    def test_missing_dataset_reports_expected_split(self):
        self._assert_failed(self._command(), "mir_query.mat")
        paths = self._dataset()
        paths["database"].unlink()
        self._assert_failed(self._command(), "mir_database.mat")

    def test_existing_file_cannot_be_used_as_log_directory(self):
        self._dataset()
        target = self.root / "not a log directory"
        target.write_bytes(b"preserved file")
        self._assert_failed(self._command(LOG_DIR=target), target.name)
        self.assertEqual(target.read_bytes(), b"preserved file")

    def test_log_path_cannot_alias_checkpoint(self):
        self._dataset()
        self.log_root.mkdir(parents=True)
        collision = self.log_root / "evaluation_MIRFlickr_4bits.txt"
        checkpoint_bytes = self.checkpoint.read_bytes()
        collision.write_bytes(checkpoint_bytes)
        self._assert_failed(self._command(CHECKPOINT=collision), "points to the checkpoint")
        self.assertEqual(collision.read_bytes(), checkpoint_bytes)

    def test_unavailable_cuda_device_is_rejected(self):
        self._dataset()
        self._assert_failed(self._command(DEVICE="cuda:9999"), "cuda")


if __name__ == "__main__":
    unittest.main()
