"""Bounded CPU comparisons with the archived implementation.

Set SACH_ORIGINAL to the read-only source directory to enable parity tests.
The CPU adapter and DCL probe below only apply inside this test suite.
"""

import ast
import importlib.util
import math
import os
import os.path as osp
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import h5py
import numpy as np
import scipy.io as sio
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sach import data, evaluation, models


ORIGINAL_ROOT = os.environ.get("SACH_ORIGINAL")
PHASE2A_ROOT = os.environ.get("SACH_PHASE2A")
RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _load_definitions(path, namespace):
    """Read functions and classes without importing the legacy modules."""
    module = ast.parse(path.read_bytes(), filename=str(path))
    module.body = [
        node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@contextmanager
def _cpu_adapter():
    """Redirect explicit CUDA transfers while keeping the math on real tensors."""
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0] == "cuda":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    def retain_on_cpu(value, *args, **kwargs):
        return value

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, "to", to))
        stack.enter_context(patch.object(torch.Tensor, "cuda", retain_on_cpu))
        stack.enter_context(patch.object(nn.Module, "cuda", retain_on_cpu))
        yield


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, message, *args):
        self.lines.append(message % args if args else message)


class _DCLProbe:
    """A differentiable stand-in for checking loss wiring without Lightly."""

    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, image, text):
        return self.temperature * F.mse_loss(image, text)


def _trainer_namespace(model_module, metric_module, loader=None):
    return {
        "F": F,
        "MYloss": SimpleNamespace(DCLLoss=_DCLProbe),
        "Variable": Variable,
        "np": np,
        "osp": osp,
        "torch": torch,
        "MyNet": model_module.MyNet,
        "Img_Net": model_module.Img_Net,
        "Txt_Net": model_module.Txt_Net,
        "TransformerEncoder": model_module.TransformerEncoder,
        "calculate_top_map": metric_module.calculate_top_map,
        "compress": metric_module.compress,
        "get_loader_flickr": loader,
        "get_loader_nus": loader,
        "get_loader_coco": loader,
        "p_topK": metric_module.p_topK,
        "pr_curve": metric_module.pr_curve,
    }


def _assert_states_equal(test, expected, actual):
    test.assertEqual(list(expected), list(actual))
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)


def _import_release_trainer_with_dcl_probe():
    lightly = ModuleType("lightly")
    lightly.loss = ModuleType("lightly.loss")
    lightly.loss.DCLLoss = _DCLProbe
    path = RELEASE_ROOT / "sach" / "trainer.py"
    spec = importlib.util.spec_from_file_location("sach._trainer_fixture", path)
    trainer_module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"lightly": lightly, "lightly.loss": lightly.loss}):
        spec.loader.exec_module(trainer_module)
    return trainer_module


def _feature_rows(length, seed):
    generator = np.random.default_rng(seed)
    return (
        generator.normal(size=(length, 7)),
        generator.normal(size=(length, 5)),
        (np.arange(length)[:, None] % 3 == np.arange(3)).astype(float),
    )


def _write_split_fixture(root, folder, prefix, is_coco):
    folder = root / folder
    folder.mkdir()
    for index, (split, suffix) in enumerate((("train", "tr"), ("query", "te"), ("database", "db"))):
        rows = _feature_rows(5 - index, index + 1)
        matrices = {f"{name}_{suffix}": values for name, values in zip("ITL", rows)}
        path = folder / f"{prefix}_{split}.mat"
        if is_coco and split == "database":
            with h5py.File(path, "w", libver="latest") as handle:
                for name, values in matrices.items():
                    handle.create_dataset(name, data=values.T)
        else:
            sio.savemat(path, matrices)


class ReleaseSmokeTests(unittest.TestCase):
    def test_trainer_import_and_configurable_loader_dispatch(self):
        # Load the actual release module. Only the unavailable loss dependency is stubbed.
        trainer_module = _import_release_trainer_with_dcl_probe()
        self.assertIs(trainer_module.MyNet, models.MyNet)
        self.assertIs(trainer_module.compress, evaluation.compress)

        # Stop at the loader boundary before allocation of training state or device use.
        class LoaderReached(Exception):
            pass

        with patch.object(trainer_module, "get_loader_nus", side_effect=LoaderReached) as load:
            config = SimpleNamespace(DATASET="NUSWIDE", BATCH_SIZE=1024, DATA_ROOT="chosen/splits")
            with self.assertRaises(LoaderReached):
                trainer_module.PCIRH(_Log(), config)
            load.assert_called_once_with(1024, "chosen/splits")

        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            trainer_module.PCIRH(_Log(), SimpleNamespace(DATASET="unrecognized"))

    def test_trainer_round_trip_leaves_existing_checkpoint_and_state_intact(self):
        trainer_class = _import_release_trainer_with_dcl_probe().PCIRH
        with TemporaryDirectory(dir=RELEASE_ROOT) as scratch:
            model_dir = Path(scratch) / "checkpoint"
            trainer = trainer_class.__new__(trainer_class)
            trainer.config = SimpleNamespace(DATASET="MIRFlickr", HASH_BIT=4, MODEL_DIR=model_dir)
            trainer.imgnet, trainer.txtnet = nn.Linear(2, 4), nn.Linear(3, 4)
            trainer.log = _Log()
            expected_image = {k: v.clone() for k, v in trainer.imgnet.state_dict().items()}
            expected_text = {k: v.clone() for k, v in trainer.txtnet.state_dict().items()}
            if ORIGINAL_ROOT and PHASE2A_ROOT:
                for source in (
                    Path(ORIGINAL_ROOT) / "my_opt.py",
                    Path(PHASE2A_ROOT) / "sach" / "trainer.py",
                ):
                    with self.subTest(implementation=source):
                        legacy = _load_definitions(source, {"osp": osp, "torch": torch}).PCIRH
                        old = legacy.__new__(legacy)
                        old.imgnet, old.txtnet = trainer.imgnet, trainer.txtnet
                        old.config, old.log = trainer.config, _Log()
                        old.save_checkpoints()
                        self.assertFalse(model_dir.exists())
                        self.assertEqual(
                            old.log.lines,
                            ["**********Save the trained model successfully.**********"],
                        )

            model_dir.mkdir()
            legacy_path = model_dir / "MIRFlickr_4bits.pth"
            torch.save({"ImgNet": expected_image, "TxtNet": expected_text}, legacy_path)
            legacy_bytes = legacy_path.read_bytes()

            saved = trainer.save_checkpoints()
            self.assertTrue(saved.is_file())
            self.assertEqual(saved.name, legacy_path.name)
            self.assertNotEqual(saved, legacy_path)
            self.assertEqual(trainer.save_checkpoints(), saved)
            self.assertEqual(legacy_path.read_bytes(), legacy_bytes)
            _assert_states_equal(self, expected_image, trainer.imgnet.state_dict())
            _assert_states_equal(self, expected_text, trainer.txtnet.state_dict())

            trainer.imgnet, trainer.txtnet = nn.Linear(2, 4), nn.Linear(3, 4)
            self.assertEqual(trainer.load_checkpoints(saved), saved)
            _assert_states_equal(self, expected_image, trainer.imgnet.state_dict())
            _assert_states_equal(self, expected_text, trainer.txtnet.state_dict())
            self.assertEqual(trainer.load_checkpoints(legacy_path.name), legacy_path)
            self.assertEqual(
                trainer.log.lines,
                [f"Saved hash-head checkpoint to {saved}"] * 2
                + [
                    f"Loaded hash-head checkpoint from {saved}",
                    f"Loaded hash-head checkpoint from {legacy_path}",
                ],
            )

    def test_failed_trainer_save_does_not_report_success(self):
        trainer_class = _import_release_trainer_with_dcl_probe().PCIRH
        with TemporaryDirectory(dir=RELEASE_ROOT) as scratch:
            invalid = Path(scratch) / "ordinary-file"
            invalid.write_text("unchanged", encoding="utf-8")
            trainer = trainer_class.__new__(trainer_class)
            trainer.config = SimpleNamespace(DATASET="MIRFlickr", HASH_BIT=4, MODEL_DIR=invalid)
            trainer.imgnet = trainer.txtnet = nn.Linear(2, 4)
            trainer.log = _Log()
            with self.assertRaises(OSError):
                trainer.save_checkpoints()
            self.assertEqual(trainer.log.lines, [])
            self.assertEqual(invalid.read_text(encoding="utf-8"), "unchanged")
            trainer.config.MODEL_DIR = Path(scratch) / "new-checkpoint-root"
            with (
                patch.object(torch, "save", side_effect=PermissionError("blocked for test")),
                self.assertRaisesRegex(RuntimeError, "Could not save checkpoint.*blocked for test"),
            ):
                trainer.save_checkpoints()
            self.assertEqual(trainer.log.lines, [])
            self.assertFalse(trainer._checkpoint_path.exists())
            self.assertEqual(list(trainer._checkpoint_path.parent.iterdir()), [])

    def test_full_gallery_metric_on_unambiguous_ranks(self):
        query_codes = np.array([[1, 1]])
        gallery_codes = np.array([[1, 1], [-1, 1], [-1, -1]])
        query_labels = np.array([[1, 0]])
        gallery_labels = np.array([[0, 1], [1, 0], [0, 1]])
        self.assertEqual(
            evaluation.calculate_map(query_codes, gallery_codes, query_labels, gallery_labels),
            0.5,
        )

    def test_full_gallery_both_directions_include_hits_past_rank_50(self):
        ranks = np.arange(53)
        gallery_text = np.where(ranks < ranks[:, None], -1, 1)
        gallery_image = gallery_text.copy()
        gallery_image[[50, 52]] = gallery_image[[52, 50]]
        gallery_labels = np.tile([0, 1], (53, 1))
        gallery_labels[52] = [1, 0]
        query = np.ones((1, 53))
        query_labels = np.array([[1, 0]])
        packed = gallery_image, gallery_text, gallery_labels, query, query, query_labels

        # Isolate actual ranking, return values and banner from network compression.
        namespace = _trainer_namespace(models, evaluation)
        namespace["compress"] = lambda *args: packed
        root = RELEASE_ROOT / "sach"
        trainer_class = _load_definitions(root / "trainer.py", namespace).PCIRH
        instance = trainer_class.__new__(trainer_class)
        instance.imgnet = instance.txtnet = nn.Identity()
        instance.database_loader = instance.test_loader = None
        instance.log = _Log()
        with _cpu_adapter():
            scores = instance.performance_eval()[:2]
            self.assertEqual(instance.eval2(), scores)
        self.assertEqual(scores, (1 / 53, 1 / 51))
        self.assertEqual(
            instance.log.lines,
            ["--------------------Evaluation: MAP (full gallery)-------------------"] * 2,
        )

        if ORIGINAL_ROOT and PHASE2A_ROOT:
            for legacy_path in (
                Path(ORIGINAL_ROOT) / "utils.py",
                Path(PHASE2A_ROOT) / "sach" / "evaluation.py",
            ):
                with self.subTest(implementation=legacy_path):
                    old = _load_definitions(legacy_path, {"np": np})
                    self.assertEqual(
                        (
                            old.calculate_top_map(
                                query, gallery_text, query_labels, gallery_labels, 50
                            ),
                            old.calculate_top_map(
                                query, gallery_image, query_labels, gallery_labels, 50
                            ),
                        ),
                        scores,
                    )


@unittest.skipUnless(ORIGINAL_ROOT, "Set SACH_ORIGINAL to run source comparisons.")
class ArchiveParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = Path(ORIGINAL_ROOT).resolve()
        cls.old_data = _load_definitions(
            cls.original / "load_data.py",
            {"DataLoader": DataLoader, "Dataset": Dataset, "np": np, "sio": sio, "h5py": h5py},
        )
        cls.old_models = _load_definitions(
            cls.original / "models.py",
            {
                "torch": torch,
                "nn": nn,
                "F": F,
                "math": math,
                "SpectralClustering": SpectralClustering,
                "OneHotEncoder": OneHotEncoder,
            },
        )
        cls.old_eval = _load_definitions(
            cls.original / "utils.py",
            {"torch": torch, "F": F, "np": np, "Variable": Variable},
        )
        threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)

    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def test_unchanged_training_and_evaluation_match_archived_source_structure(self):
        changed = {"__init__", "train_Hashfunc", "save_checkpoints", "load_checkpoints"}

        def method_bodies(path):
            tree = ast.parse(path.read_bytes())
            controller = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "PCIRH"
            )
            methods = {
                method.name: ast.dump(method).replace(
                    "Evaluation: mAP@50", "Evaluation: MAP (full gallery)"
                )
                for method in controller.body
                if isinstance(method, ast.FunctionDef) and method.name not in changed
            }
            methods["GEN_S_GPU"] = ast.dump(
                next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "GEN_S_GPU"
                )
            )
            return methods

        self.assertEqual(
            method_bodies(self.original / "my_opt.py"),
            method_bodies(RELEASE_ROOT / "sach" / "trainer.py"),
        )
        if PHASE2A_ROOT:
            self.assertEqual(
                method_bodies(Path(PHASE2A_ROOT) / "sach" / "trainer.py"),
                method_bodies(RELEASE_ROOT / "sach" / "trainer.py"),
            )

    def _trainer_variants(self, loader=None):
        paths = (
            (self.original / "my_opt.py", self.old_models, self.old_eval),
            (RELEASE_ROOT / "sach" / "trainer.py", models, evaluation),
        )
        for path, model_module, metric_module in paths:
            trainer = _load_definitions(
                path, _trainer_namespace(model_module, metric_module, loader)
            ).PCIRH
            yield trainer, model_module, metric_module

    def test_all_dataset_loaders_match_on_temporary_splits(self):
        datasets = (
            ("MIRFlickr", "mir", "get_loader_flickr", False),
            ("NUS-WIDE", "nus", "get_loader_nus", False),
            ("MSCOCO", "COCO", "get_loader_coco", True),
        )
        with TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`np[.]float` is a deprecated alias")
            root = Path(folder) / "datasets"
            root.mkdir()
            for dataset, prefix, loader_name, is_coco in datasets:
                with self.subTest(dataset=dataset):
                    _write_split_fixture(root, dataset, prefix, is_coco)
                    with _working_directory(folder):
                        old_loaders, old_train = getattr(self.old_data, loader_name)(2)
                    new_loaders, new_train = getattr(data, loader_name)(2, data_root=root)
                    for expected, actual in zip(old_train, new_train):
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(actual.dtype, expected.dtype)
                    for split in ("train", "query", "database"):
                        torch.manual_seed(11)
                        old_batches = list(old_loaders[split])
                        torch.manual_seed(11)
                        new_batches = list(new_loaders[split])
                        self.assertEqual(len(old_batches), len(new_batches))
                        for old_batch, new_batch in zip(old_batches, new_batches):
                            for expected, actual in zip(old_batch, new_batch):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_query_gallery_codes_and_both_metric_directions_match(self):
        results = []
        packed_results = []
        for trainer_class, model_module, metric_module in self._trainer_variants():
            trainer = trainer_class.__new__(trainer_class)
            trainer.log = _Log()
            trainer.database_loader = DataLoader(
                data.CustomDataSet(*_feature_rows(5, 30)), batch_size=2
            )
            trainer.test_loader = DataLoader(
                data.CustomDataSet(*_feature_rows(4, 31)), batch_size=2
            )
            torch.manual_seed(42)
            trainer.imgnet = model_module.Img_Net(4, 7)
            trainer.txtnet = model_module.Txt_Net(4, 5)
            with _cpu_adapter():
                results.append(trainer.performance_eval())
                packed_results.append(
                    metric_module.compress(
                        trainer.database_loader, trainer.test_loader, trainer.imgnet, trainer.txtnet
                    )
                )
        for expected, actual in zip(*results):
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                self.assertEqual(actual, expected)
        for expected, actual in zip(*packed_results):
            np.testing.assert_array_equal(actual, expected)

    def test_seeded_joint_model_forward_and_initial_state_match(self):
        image, text, _ = _feature_rows(12, 43)
        image = torch.tensor(image, dtype=torch.float32)
        text = torch.tensor(text, dtype=torch.float32)
        graph = F.normalize(image) @ F.normalize(image).T
        results = []
        states = []
        for model_module in (self.old_models, models):
            torch.manual_seed(44)
            np.random.seed(45)
            model = model_module.MyNet(4, 7, 5, 2).eval()
            states.append(model.state_dict())
            with _cpu_adapter(), torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Graph is not fully connected")
                results.append(model(image, text, graph))
        _assert_states_equal(self, *states)
        for expected, actual in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_teacher_pass_matches_with_differentiable_dcl_probe(self):
        rows = _feature_rows(12, 46)

        def loader(batch_size, data_root=None):
            train = DataLoader(data.CustomDataSet(*rows), batch_size=batch_size, shuffle=True)
            return {"train": train, "query": train, "database": train}, rows

        config = SimpleNamespace(
            DATASET="MIRFlickr",
            BATCH_SIZE=12,
            HASH_BIT=4,
            NUM_EPOCH=1,
            EPOCH_INTERVAL=1,
            K=2,
            a1=0.6,
            a2=0.6,
            num=2,
            anchor_nums=3,
            top_size=2,
            temperature=0.07,
            bt1=0.05,
            lambda1=10.0,
            lambda2=1.0,
            lr=0.001,
            LR_MyNet=0.001,
            LR_IMG=0.0001,
            LR_TXT=0.0001,
        )
        states = []
        for trainer_class, _, _ in self._trainer_variants(loader):
            torch.manual_seed(47)
            np.random.seed(48)
            with _cpu_adapter(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Graph is not fully connected")
                trainer = trainer_class(_Log(), config)
                codes, sample_ids = trainer.train_method(0)
            states.append((trainer, codes, sample_ids, torch.get_rng_state()))
        (old, old_codes, old_ids, old_rng), (new, new_codes, new_ids, new_rng) = states
        np.testing.assert_array_equal(new_codes, old_codes)
        self.assertEqual(new_ids, old_ids)
        self.assertEqual(new.record_Lsmodel, old.record_Lsmodel)
        self.assertEqual(new.log.lines, old.log.lines)
        for name in ("S_cmm", "S_I", "S_T"):
            torch.testing.assert_close(getattr(new, name), getattr(old, name), rtol=0, atol=0)
        torch.testing.assert_close(new_rng, old_rng, rtol=0, atol=0)
        for name in ("mynet", "img_TEs_enc", "txt_TEs_enc"):
            _assert_states_equal(
                self, getattr(old, name).state_dict(), getattr(new, name).state_dict()
            )

    def test_hash_pass_matches_archive_when_targets_align_with_partial_final_batch(self):
        image, text, _ = _feature_rows(5, 49)
        # Identity order makes the old positional target and the new ID gather agree.
        order = list(range(5))
        codes = np.where(_feature_rows(5, 50)[0][:, :4] >= 0, 1.0, -1.0).astype(np.float32)
        states = []
        for trainer_class, model_module, _ in self._trainer_variants():
            trainer = trainer_class.__new__(trainer_class)
            trainer.config = SimpleNamespace(BATCH_SIZE=3, NUM_EPOCH=1, beta=0.01)
            trainer.I_tr = image
            trainer.T_tr = text
            trainer.S_cmm = torch.arange(25, dtype=torch.float32).reshape(5, 5) / 25
            trainer.log = _Log()
            trainer.record_Lshfunc = []
            torch.manual_seed(51)
            trainer.imgnet = model_module.Img_Net(4, 7)
            trainer.txtnet = model_module.Txt_Net(4, 5)
            trainer.opt_imgnet = torch.optim.Adam(trainer.imgnet.parameters(), lr=0.0001)
            trainer.opt_txtnet = torch.optim.Adam(trainer.txtnet.parameters(), lr=0.0001)
            with _cpu_adapter():
                trainer.train_Hashfunc(codes, order, 0)
            states.append(trainer)
        old, new = states
        self.assertEqual(new.record_Lshfunc, old.record_Lshfunc)
        self.assertEqual(new.log.lines, old.log.lines)
        _assert_states_equal(self, old.imgnet.state_dict(), new.imgnet.state_dict())
        _assert_states_equal(self, old.txtnet.state_dict(), new.txtnet.state_dict())


class InstalledDependencyTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("lightly"), "Lightly is not installed")
    def test_real_lightly_dcl_forward_and_backward_on_cpu(self):
        from lightly.loss import DCLLoss

        image = torch.tensor([[1.0, 0.2], [-0.2, 1.0], [-1.0, -0.3]], requires_grad=True)
        text = torch.tensor([[0.7, 0.3], [-0.5, 0.8], [-0.8, -0.5]], requires_grad=True)
        loss = DCLLoss(0.07)(image, text)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for features in (image, text):
            self.assertTrue(torch.isfinite(features.grad).all().item())
            self.assertGreater(features.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
