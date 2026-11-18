"""Command-line defaults from the three archived dataset entry points."""

import argparse


DATASET_DEFAULTS = {
    "MIRFlickr": {
        "LR_IMG": 0.0001,
        "LR_TXT": 0.0001,
        "LR_MyNet": 0.001,
        "K": 3000,
        "a1": 0.6,
        "a2": 0.6,
        "HASH_BIT": 128,
        "BATCH_SIZE": 2048,
        "NUM_WORKERS": 8,
        "NUM_EPOCH": 60,
        "num": 30,
    },
    "NUSWIDE": {
        "LR_IMG": 0.0001,
        "LR_TXT": 0.0001,
        "LR_MyNet": 0.0001,
        "K": 3300,
        "a1": 0.5,
        "a2": 0.5,
        "HASH_BIT": 128,
        "BATCH_SIZE": 1024,
        "NUM_WORKERS": 0,
        "NUM_EPOCH": 60,
        "num": 15,
    },
    "COCO": {
        "LR_IMG": 0.0005,
        "LR_TXT": 0.0005,
        "LR_MyNet": 0.001,
        "K": 3000,
        "a1": 0.1,
        "a2": 0.3,
        "HASH_BIT": 16,
        "BATCH_SIZE": 2048,
        "NUM_WORKERS": 8,
        "NUM_EPOCH": 100,
        "num": 15,
    },
}


def _boolean(value):
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def build_parser(preset):
    defaults = DATASET_DEFAULTS[preset]
    parser = argparse.ArgumentParser(description=f"SACH archived baseline ({preset} defaults)")
    parser.add_argument("--TRAIN", default=True, type=_boolean, help="True to train, False to evaluate")
    parser.add_argument("--CHECKPOINT", help="saved hash-head checkpoint for evaluation")
    parser.add_argument("--DEVICE", default="cpu", help="evaluation device, cpu or cuda[:index]")
    parser.add_argument("--DATASET", default=preset, type=str, help="MIRFlickr, NUSWIDE or COCO")
    parser.add_argument(
        "--DATA_ROOT", default="./datasets", help="root of the precomputed split files"
    )
    parser.add_argument(
        "--LOG_DIR", default="./logs", help="directory for training or evaluation logs"
    )
    parser.add_argument(
        "--MODEL_DIR",
        default="./checkpoints",
        type=str,
        help="parent directory for checkpoint runs",
    )
    parser.add_argument("--lambda1", default=10, type=float)
    parser.add_argument("--lambda2", default=1, type=float)
    parser.add_argument("--beta", default=0.01, type=float)
    for name, default in defaults.items():
        parser.add_argument(f"--{name}", default=default, type=type(default))
    parser.add_argument("--GPU_ID", default=0, type=int)
    parser.add_argument("--SEED", default=1, type=int)
    parser.add_argument("--EPOCH_INTERVAL", default=2, type=int)
    parser.add_argument("--EVAL_INTERVAL", default=10, type=int)
    parser.add_argument("--anchor_nums", default=300, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--top_size", default=100, type=int)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--bt1", default=20, type=float)
    return parser
