"""Persist the image and text hash heads in the original checkpoint format."""

from collections.abc import Mapping
from contextlib import suppress
import inspect
import os
from pathlib import Path
import tempfile

import torch


def create_checkpoint_path(model_dir, filename):
    """Allocate an absolute path in a directory owned by this training run."""
    filename = os.fspath(filename)
    if not filename or filename in (".", "..") or Path(filename).name != filename:
        raise ValueError(f"Checkpoint filename must be a basename, got {filename!r}")

    directory = Path(model_dir).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
    except OSError as exc:
        raise OSError(f"Could not create a checkpoint directory under {directory}. {exc}") from exc
    return run_dir / filename


def save_hash_heads(path, imgnet, txtnet):
    """Save to a run-owned path, keeping a prior best if serialization fails."""
    path = Path(path)
    temporary = None
    try:
        payload = {"ImgNet": imgnet.state_dict(), "TxtNet": txtnet.state_dict()}
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".checkpoint-", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except Exception as exc:
        raise RuntimeError(f"Could not save checkpoint {path}. {exc}") from exc
    finally:
        if temporary is not None:
            # Preserve the save exception if the filesystem also denies cleanup.
            with suppress(OSError):
                temporary.unlink()
    return path


def load_hash_heads(path, imgnet, txtnet):
    """Restore both heads from a trusted checkpoint, preserving their devices."""
    path = Path(path)
    load_options = {"map_location": "cpu"}
    # Older PyTorch versions lack weights_only and require trusted artifacts.
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    try:
        payload = torch.load(path, **load_options)
        if not isinstance(payload, Mapping):
            raise ValueError("Expected a checkpoint mapping")

        heads = {"ImgNet": imgnet, "TxtNet": txtnet}
        for name, model in heads.items():
            state = payload.get(name)
            if not isinstance(state, Mapping):
                raise ValueError(f"Missing or invalid {name} state dictionary")
            expected = model.state_dict()
            if state.keys() != expected.keys():
                missing = sorted(map(str, expected.keys() - state.keys()))
                unexpected = sorted(map(str, state.keys() - expected.keys()))
                raise ValueError(
                    f"{name} state keys differ. Missing {missing}, unexpected {unexpected}"
                )
            for key, tensor in expected.items():
                other = state[key]
                if (
                    not isinstance(other, torch.Tensor)
                    or other.shape != tensor.shape
                    or other.layout != tensor.layout
                    or other.is_quantized != tensor.is_quantized
                    or getattr(other, "is_meta", False)
                ):
                    raise ValueError(f"{name}.{key} is not a compatible tensor")

        # Validate both state dictionaries before either head is modified.
        for name, model in heads.items():
            model.load_state_dict(payload[name], strict=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load checkpoint {path}. {exc}") from exc
    return path
