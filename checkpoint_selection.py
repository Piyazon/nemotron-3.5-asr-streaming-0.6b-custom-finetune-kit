"""Select checkpoints within one training run without mixing validation scores."""

from pathlib import Path
import re


def run_directory(root: Path, run_name: str | None) -> Path:
    if run_name is None:
        return root
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name) or run_name in {".", ".."}:
        raise ValueError("Run name must be a directory name using letters, digits, _, -, or .")
    return root / run_name


def latest_checkpoint(root: Path) -> Path:
    epochs = list(root.rglob("nemotron-asr-finetuned-epoch=*.ckpt"))
    candidates = epochs or list(root.rglob("last.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in: {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def best_checkpoint_from_state(checkpoint: dict, source: Path) -> Path:
    """Recover Lightning's best WER checkpoint, including after moving a run.

    Only accept the selected run's copy. An old absolute training-server path
    can still exist after a run has been copied elsewhere.
    """
    matches = []
    for state in checkpoint.get("callbacks", {}).values():
        if isinstance(state, dict) and state.get("monitor") == "val_wer":
            recorded = state.get("best_model_path")
            if recorded:
                matches.append(source.parent / Path(recorded).name)
    if len(set(matches)) != 1:
        raise ValueError(
            f"Cannot determine the best validation-WER checkpoint from {source}. "
            "Pass --checkpoint explicitly or use --selection latest."
        )
    selected = matches[0]
    if not selected.is_file():
        raise FileNotFoundError(
            f"Best validation-WER checkpoint is missing: {selected}. "
            "Pass --checkpoint explicitly or use --selection latest."
        )
    return selected


def select_checkpoint(root: Path, selection: str = "best") -> Path:
    """Use the newest run's callback state to select its best retained weights."""
    latest = latest_checkpoint(root)
    if selection == "latest":
        return latest
    if selection != "best":
        raise ValueError(f"Unknown checkpoint selection: {selection}")
    import torch

    state = torch.load(latest, map_location="cpu", weights_only=False, mmap=True)
    return best_checkpoint_from_state(state, latest)


def best_nemo_checkpoint(root: Path) -> Path:
    """Find best1 in the most recently exported run, never its final-epoch file."""
    archives = list(root.rglob("*.nemo"))
    if not archives:
        raise FileNotFoundError(f"No exported .nemo checkpoints in: {root}")
    selected_run = max(archives, key=lambda path: path.stat().st_mtime_ns).parent
    best = list(selected_run.glob("nemotron-asr-best1-wer-*.nemo"))
    if not best:
        raise FileNotFoundError(
            f"No best1 validation-WER export in {selected_run}. "
            "Pass --checkpoint /path/to/model.nemo explicitly."
        )
    # Repeated exports may leave older best1 filenames in a run directory.
    return max(best, key=lambda path: path.stat().st_mtime_ns)
