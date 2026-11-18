"""Evaluate a saved SACH hash-head checkpoint on the query and gallery splits."""

import argparse
from pathlib import Path

from .config import DATASET_DEFAULTS


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a SACH checkpoint with full-gallery MAP")
    parser.add_argument("--DATASET", required=True, choices=DATASET_DEFAULTS)
    parser.add_argument("--CHECKPOINT", required=True, help="path to a saved hash-head checkpoint")
    parser.add_argument("--DATA_ROOT", default="./datasets", help="root of the precomputed splits")
    parser.add_argument("--LOG_DIR", default="./logs", help="directory for evaluation logs")
    parser.add_argument("--DEVICE", default="cpu", help="encoding device, cpu or cuda[:index]")
    parser.add_argument(
        "--HASH_BIT", type=int, help="checkpoint hash width, defaults to the dataset preset"
    )
    parser.add_argument(
        "--BATCH_SIZE", type=int, help="evaluation batch size, defaults to the dataset preset"
    )
    return parser


def _setting(config, name, defaults):
    value = getattr(config, name, None)
    value = defaults[name] if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"--{name} must be a positive integer")
    return value


def _resolve_device(torch, requested):
    if requested == "cpu":
        return torch.device("cpu")
    kind, separator, ordinal = str(requested).partition(":")
    if kind != "cuda" or (separator and not ordinal.isdecimal()):
        raise ValueError(f"Invalid --DEVICE {requested!r}. Use cpu, cuda or cuda:<index>.")
    # Validate the ordinal before torch converts it to a device index.
    index = int(ordinal) if separator else 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment. Use --DEVICE cpu.")
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {index} is unavailable. Choose a visible device or --DEVICE cpu."
        )
    return torch.device("cuda", index)


def run_evaluation(config):
    """Restore the two modality heads, append a result record and return directional MAP."""
    dataset = config.DATASET
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose MIRFlickr, NUSWIDE or COCO.")
    defaults = DATASET_DEFAULTS[dataset]
    bits = _setting(config, "HASH_BIT", defaults)
    batch_size = _setting(config, "BATCH_SIZE", defaults)
    filename = getattr(config, "CHECKPOINT", None)
    if not filename:
        raise ValueError("--CHECKPOINT is required for evaluation")
    checkpoint = Path(filename).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found or not a regular file: {checkpoint}")

    # Keep help and checkpoint path errors available without loading numeric dependencies.
    import torch

    from .checkpoint import load_hash_heads
    from .data import get_evaluation_loaders
    from .evaluation import calculate_top_map, compress
    from .models import Img_Net, Txt_Net

    device = _resolve_device(torch, getattr(config, "DEVICE", "cpu"))
    data_root = Path(getattr(config, "DATA_ROOT", "./datasets")).expanduser().resolve()
    query, gallery = get_evaluation_loaders(dataset, batch_size, data_root)
    image_head = Img_Net(bits, query.dataset.images.shape[1]).to(device).eval()
    text_head = Txt_Net(bits, query.dataset.texts.shape[1]).to(device).eval()
    try:
        load_hash_heads(checkpoint, image_head, text_head)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} Check that --HASH_BIT {bits} and the feature widths from --DATA_ROOT "
            "match the saved ImgNet and TxtNet heads."
        ) from exc

    re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(
        gallery, query, image_head, text_head, device=device
    )
    map_it = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=len(re_L))
    map_ti = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=len(re_L))
    scores = (
        f"MAP (full gallery) I->T: {float(map_it)!r}, "
        f"MAP (full gallery) T->I: {float(map_ti)!r}"
    )

    log_dir = Path(getattr(config, "LOG_DIR", "./logs")).expanduser().resolve()
    log_path = log_dir / f"evaluation_{dataset}_{bits}bits.txt"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_path.resolve() == checkpoint or (
            log_path.exists() and log_path.samefile(checkpoint)
        ):
            raise ValueError(f"Evaluation log path points to the checkpoint: {log_path}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"Dataset: {dataset}, hash bits: {bits}, device: {device}, "
                f"query rows: {len(qu_L)}, gallery rows: {len(re_L)}\n"
                f"Data root: {data_root}\nCheckpoint: {checkpoint}\n{scores}\n\n"
            )
    except OSError as exc:
        raise OSError(f"Could not write evaluation log {log_path}. {exc}") from exc

    print(scores)
    print(f"Evaluation log: {log_path}")
    return map_it, map_ti


def main(argv=None):
    parser = build_parser()
    config = parser.parse_args(argv)
    try:
        run_evaluation(config)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
