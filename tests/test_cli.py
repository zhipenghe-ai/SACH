import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import warnings

from sach import cli
from sach.config import build_parser


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = os.environ.get("SACH_ORIGINAL")
ENTRY_POINTS = {
    "MIRFlickr": "main_mir.py",
    "NUSWIDE": "main_nus.py",
    "COCO": "main_coco.py",
}


def original_entry(filename, namespace, node_filter):
    path = Path(ORIGINAL_ROOT) / filename
    tree = ast.parse(path.read_bytes(), filename=str(path))
    nodes = node_filter(tree)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def original_args(filename, arguments):
    scope = original_entry(
        filename,
        {"argparse": argparse},
        lambda tree: next(node for node in tree.body if isinstance(node, ast.If)).body[:-2],
    )
    return scope["parser"].parse_args(arguments)


class LauncherTests(unittest.TestCase):
    def test_script_help_from_release_root(self):
        for filename in ENTRY_POINTS.values():
            with self.subTest(filename=filename):
                result = subprocess.run(
                    [sys.executable, "-B", filename, "--help"],
                    cwd=RELEASE_ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("--DATA_ROOT", result.stdout)
                self.assertIn("--LOG_DIR", result.stdout)
                self.assertIn("--HASH_BIT", result.stdout)

    @unittest.skipUnless(ORIGINAL_ROOT, "set SACH_ORIGINAL to compare with the archived launchers")
    def test_original_argument_names_types_and_presets(self):
        for preset, filename in ENTRY_POINTS.items():
            for arguments in (
                [],
                ["--HASH_BIT", "32", "--BATCH_SIZE", "512", "--lambda1", "7", "--TRAIN", "False"],
            ):
                with self.subTest(preset=preset, arguments=arguments):
                    expected = vars(original_args(filename, arguments))
                    actual = vars(build_parser(preset).parse_args(arguments))
                    self.assertEqual(actual.pop("DATA_ROOT"), "./datasets")
                    self.assertEqual(actual.pop("LOG_DIR"), "./logs")
                    self.assertIsNone(actual.pop("CHECKPOINT"))
                    self.assertEqual(actual.pop("DEVICE"), "cpu")
                    if "--TRAIN" in arguments:
                        # The original argv conversion treated any nonempty string as True.
                        self.assertIs(expected["TRAIN"], True)
                        expected["TRAIN"] = False
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        {k: type(v) for k, v in actual.items()},
                        {k: type(v) for k, v in expected.items()},
                    )

    @unittest.skipUnless(ORIGINAL_ROOT, "set SACH_ORIGINAL to compare with the archived launchers")
    def test_epoch_and_evaluation_dispatch_matches_original(self):
        arguments = ["--NUM_EPOCH", "4", "--EVAL_INTERVAL", "1"]
        for preset, filename in ENTRY_POINTS.items():
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as scratch:
                log_dir = Path(scratch) / "run" / "logs"
                original_config = original_args(filename, arguments)
                release_config = build_parser(preset).parse_args(
                    arguments + ["--LOG_DIR", str(log_dir)]
                )
                legacy = self.run_fixture(preset, filename, original_config, original=True)
                release = self.run_fixture(preset, filename, release_config, original=False)
                release_seeds, release_calls, release_logs = release
                legacy_seeds, legacy_calls, legacy_logs = legacy
                self.assertEqual(release_seeds, legacy_seeds)
                if preset != "MIRFlickr":
                    self.assertEqual(release_calls, legacy_calls)
                else:
                    self.assertEqual(
                        [call for call in release_calls if call[0] != "save_checkpoints"],
                        legacy_calls,
                    )
                # The same two strict improvements are selected in every preset.
                saves = [
                    index
                    for index, call in enumerate(release_calls)
                    if call[0] == "save_checkpoints"
                ]
                evaluations = [
                    index
                    for index, call in enumerate(release_calls)
                    if call[0] == "performance_eval"
                ]
                self.assertEqual(saves, [evaluations[0] + 1, evaluations[3] + 1])

                expected_logs = []
                for call in legacy_logs:
                    label, *log_arguments = call.args
                    label = (
                        label.replace("mAP@50 ", "MAP (full gallery) ")
                        .replace("Best MAP of ", "Best MAP (full gallery) of ")
                        .replace("Best mAP of ", "Best MAP (full gallery) of ")
                    )
                    expected_logs.append((label, *log_arguments))
                self.assertEqual([call.args for call in release_logs], expected_logs)
                metric_logs = [
                    call.args[0]
                    for call in release_logs
                    if call.args[0].startswith("MAP (full gallery)")
                ]
                self.assertEqual(len(metric_logs), 4)
                self.assertTrue(
                    all(
                        "MAP (full gallery) I->T:" in msg and "MAP (full gallery) T->I:" in msg
                        for msg in metric_logs
                    )
                )
                self.assertFalse(any("mAP@50" in call.args[0] for call in release_logs))
                self.assertTrue(log_dir.is_dir())

    def run_fixture(self, preset, filename, config, *, original):
        trainer = Mock()
        logger = Mock()
        torch = Mock()
        seed_calls = Mock()
        seed_calls.attach_mock(torch.manual_seed, "cpu")
        seed_calls.attach_mock(torch.cuda.manual_seed_all, "cuda")
        seed_calls.attach_mock(torch.cuda.set_device, "device")
        curves = (SimpleNamespace(shape=(2,)),) * 6
        trainer.train_method.side_effect = lambda epoch: (["codes", epoch], [epoch])
        trainer.performance_eval.side_effect = [
            (0.2, 0.3, *curves),
            (0.3, 0.2, *curves),
            (0.1, 0.2, *curves),
            (0.6, 0.4, *curves),
        ]
        if original:
            scope = original_entry(
                filename,
                {
                    "PCIRH": Mock(return_value=trainer),
                    "torch": torch,
                    "logger": Mock(return_value=logger),
                    "mylogger": Mock(),
                },
                lambda tree: [node for node in tree.body if isinstance(node, ast.FunctionDef)],
            )
            scope["main"](config)
        else:
            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": torch,
                        "sach.trainer": SimpleNamespace(PCIRH=Mock(return_value=trainer)),
                    },
                ),
                patch.object(cli, "logger", return_value=logger) as factory,
                warnings.catch_warnings(),
            ):
                cli.run(config, preset)
                factory.assert_called_once_with(
                    f"{config.DATASET}_{config.HASH_BIT}", log_dir=Path(config.LOG_DIR)
                )
        return seed_calls.mock_calls, trainer.mock_calls, logger.info.call_args_list
