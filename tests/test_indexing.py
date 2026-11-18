"""Verify that hash-head features, codes and graph targets refer to the same IDs."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from test_regression import _cpu_adapter, _load_definitions


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = os.environ.get("SACH_ORIGINAL")
PHASE2A_ROOT = os.environ.get("SACH_PHASE2A")


def _trainer_type(source_file):
    return _load_definitions(source_file, {"torch": torch, "F": F, "np": np}).PCIRH


class _RecordingHead(nn.Linear):
    def __init__(self, feature_width, code_width):
        super().__init__(feature_width, code_width)
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().clone())
        return super().forward(inputs)


class HashIndexingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corrected = _trainer_type(RELEASE_ROOT / "sach" / "trainer.py")
        fixture = np.random.default_rng(123)
        cls.image = fixture.uniform(0.1, 1.0, (8, 4))
        cls.text = fixture.uniform(0.1, 1.0, (8, 3))
        cls.codes_by_id = (2 * ((np.arange(8)[:, None] >> np.arange(3)) & 1) - 1).astype(np.float32)
        # Unique directed values distinguish both coordinates, even if swapped.
        cls.graph = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64

    def _run_head_pass(self, trainer_type, ids, batch_size):
        trainer = trainer_type.__new__(trainer_type)
        trainer.I_tr = self.image
        trainer.T_tr = self.text
        trainer.S_cmm = self.graph
        trainer.config = SimpleNamespace(BATCH_SIZE=batch_size, NUM_EPOCH=1, beta=0.01)
        trainer.record_Lshfunc = []
        trainer.log = SimpleNamespace(info=lambda *args: None)
        # This is the same representation produced by record_index.extend(index_).
        record_index = [torch.tensor(sample_id) for sample_id in ids]
        codes_in_collected_order = self.codes_by_id[ids]
        targets = []
        mse_loss = F.mse_loss

        def capture_target(prediction, target):
            targets.append(target.detach().clone())
            return mse_loss(prediction, target)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(456)
            trainer.imgnet = _RecordingHead(self.image.shape[1], self.codes_by_id.shape[1])
            trainer.txtnet = _RecordingHead(self.text.shape[1], self.codes_by_id.shape[1])
            trainer.opt_imgnet = torch.optim.Adam(trainer.imgnet.parameters(), lr=0.0001)
            trainer.opt_txtnet = torch.optim.Adam(trainer.txtnet.parameters(), lr=0.0001)
            with _cpu_adapter(), patch.object(F, "mse_loss", side_effect=capture_target):
                trainer.train_Hashfunc(codes_in_collected_order, record_index, 0)

        return SimpleNamespace(
            image=trainer.imgnet.inputs,
            text=trainer.txtnet.inputs,
            targets=targets,
            loss=trainer.record_Lshfunc[0],
            weights=(trainer.imgnet.state_dict(), trainer.txtnet.state_dict()),
        )

    def _assert_mapping(self, result, sample_ids, batch_size, *, graph_follows_ids):
        expected_batches = [
            sample_ids[start : start + batch_size]
            for start in range(0, len(sample_ids), batch_size)
        ]
        self.assertEqual(len(result.image), len(expected_batches))
        self.assertEqual(len(result.text), len(expected_batches))
        self.assertEqual(len(result.targets), 6 * len(expected_batches))
        for batch_number, ids in enumerate(expected_batches):
            torch.testing.assert_close(
                result.image[batch_number], F.normalize(torch.Tensor(self.image[ids]))
            )
            torch.testing.assert_close(
                result.text[batch_number], F.normalize(torch.Tensor(self.text[ids]))
            )
            target_index = 6 * batch_number
            expected_codes = torch.from_numpy(self.codes_by_id[ids])
            for actual in result.targets[target_index : target_index + 2]:
                torch.testing.assert_close(actual, expected_codes, rtol=0, atol=0)

            graph_ids = ids
            if not graph_follows_ids:
                start = batch_number * batch_size
                graph_ids = range(start, start + len(ids))
            # Check values against two independent coordinates rather than using
            # the production advanced-indexing expression to build the oracle.
            expected_graph = torch.tensor(
                [[self.graph[u, v].item() for v in graph_ids] for u in graph_ids]
            )
            for actual in result.targets[target_index + 3 : target_index + 6]:
                torch.testing.assert_close(actual, expected_graph, rtol=0, atol=0)

    def test_shuffled_features_codes_and_both_graph_axes_stay_aligned(self):
        ids = [6, 0, 3, 1, 7, 4, 5, 2]
        result = self._run_head_pass(self.corrected, ids, 3)
        self._assert_mapping(result, ids, 3, graph_follows_ids=True)
        self.assertEqual(tuple(result.targets[-1].shape), (2, 2))

    def test_non_contiguous_ids_with_single_row_tail(self):
        ids = [7, 1, 4, 0, 6]
        result = self._run_head_pass(self.corrected, ids, 2)
        self._assert_mapping(result, ids, 2, graph_follows_ids=True)
        torch.testing.assert_close(result.targets[-1], self.graph[6, 6].reshape(1, 1))

    def test_seeded_random_permutations_have_repeatable_mapping_and_results(self):
        generator = torch.Generator().manual_seed(789)
        for _ in range(4):
            ids = torch.randperm(8, generator=generator).tolist()
            with self.subTest(ids=ids):
                first = self._run_head_pass(self.corrected, ids, 3)
                repeated = self._run_head_pass(self.corrected, ids, 3)
                self._assert_mapping(first, ids, 3, graph_follows_ids=True)
                self.assertEqual(first.loss, repeated.loss)
                for expected, actual in zip(first.targets, repeated.targets):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for expected, actual in zip(first.weights, repeated.weights):
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_invalid_ids_are_rejected_before_starting_a_head_pass(self):
        trainer = self.corrected.__new__(self.corrected)
        trainer.I_tr = self.image
        for ids, error in (
            ([-1], IndexError),
            ([8], IndexError),
            ([1.5], TypeError),
            ([np.float64(1.0)], TypeError),
            ([True], TypeError),
            ([np.bool_(True)], TypeError),
            ([torch.tensor(True)], TypeError),
            ([torch.tensor([1])], TypeError),
            ([[1]], TypeError),
            (["1"], TypeError),
            ([], ValueError),
            ([0, 1], ValueError),
        ):
            with self.subTest(ids=ids), self.assertRaises(error):
                # No head objects are installed, so only pre-update validation can succeed.
                trainer.train_Hashfunc(self.codes_by_id[:1], ids, 0)

    @unittest.skipUnless(
        ORIGINAL_ROOT and PHASE2A_ROOT,
        "Set SACH_ORIGINAL and SACH_PHASE2A for a three-way comparison.",
    )
    def test_corrected_targets_and_loss_differ_from_both_archived_implementations(self):
        original = _trainer_type(Path(ORIGINAL_ROOT) / "my_opt.py")
        baseline = _trainer_type(Path(PHASE2A_ROOT) / "sach" / "trainer.py")
        ids = [6, 0, 3, 1, 7, 4, 5, 2]
        originals = [self._run_head_pass(trainer, ids, 3) for trainer in (original, baseline)]
        corrected = self._run_head_pass(self.corrected, ids, 3)
        for result in originals:
            self._assert_mapping(result, ids, 3, graph_follows_ids=False)
        self._assert_mapping(corrected, ids, 3, graph_follows_ids=True)
        self.assertEqual(originals[0].loss, originals[1].loss)
        self.assertNotEqual(originals[0].loss, corrected.loss)
        self.assertFalse(torch.equal(originals[0].targets[3], corrected.targets[3]))
        self.assertTrue(
            any(
                not torch.equal(old[name], new[name])
                for old, new in zip(originals[0].weights, corrected.weights)
                for name in old
            )
        )

        # Identity order already satisfies the positional assumption in the archive.
        ids = list(range(8))
        identity = [
            self._run_head_pass(trainer, ids, 3) for trainer in (original, baseline, self.corrected)
        ]
        self.assertEqual(len({result.loss for result in identity}), 1)
        for reference, candidate in zip(identity[0].weights, identity[-1].weights):
            for name in reference:
                torch.testing.assert_close(candidate[name], reference[name], rtol=0, atol=0)
