"""Continue an extended-tokenizer run using its exact prepared assets."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf


def checkpoint_fit_kwargs(trainer, checkpoint: Path | None) -> dict:
    kwargs = {"ckpt_path": str(checkpoint) if checkpoint else None}
    # New Lightning versions expose torch.load's restriction on fit(). NeMo's
    # own training checkpoints contain OmegaConf metadata as well as tensors.
    if checkpoint and "weights_only" in inspect.signature(trainer.fit).parameters:
        kwargs["weights_only"] = False
    return kwargs


def verify_assets(assets: Path, metadata: dict) -> None:
    checksums = metadata["file_sha256"]
    required = {"train.jsonl", "validation.jsonl", "merged_tokenizer/tokenizer.model",
                "merged_tokenizer/vocab.txt"}
    if not required.issubset(checksums):
        raise ValueError("Resume metadata is missing required asset checksums")
    for name, expected in checksums.items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid resume asset: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Resume asset was modified: {path}")


def load_resume_recipe(checkpoint: Path, output_dir: Path, max_steps: int | None):
    """Find the source recipe beside the checkpoint; copy verified assets exactly.

    A .nemo export lacks optimizer/loop state and must not be passed here.
    Source files are kept intact; the continuation uses a fresh output directory.
    """
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        candidates = sorted(path for path in checkpoint.rglob("*.ckpt")
                            if path.is_file() and (path.name == "last.ckpt" or path.name.endswith("-last.ckpt")))
        if len(candidates) != 1:
            raise ValueError(f"Found {len(candidates)} last checkpoints under {checkpoint}. "
                             "Pass the exact .ckpt file to --resume-from.")
        checkpoint = candidates[0]
    if checkpoint.suffix != ".ckpt" or not checkpoint.is_file():
        raise ValueError("--resume-from requires a Lightning .ckpt file or a run directory containing one *-last.ckpt")
    source = next((parent for parent in checkpoint.parents
                   if (parent / "training_recipe.yaml").is_file()
                   and (parent / "preparation_metadata.json").is_file()), None)
    if source is None:
        raise ValueError("Cannot find training_recipe.yaml and preparation_metadata.json beside this checkpoint. "
                         "Keep the original extended-tokenizer run directory and its assets.")
    cfg = OmegaConf.load(source / "training_recipe.yaml")
    metadata = json.loads((source / "preparation_metadata.json").read_text(encoding="utf-8"))
    if cfg.model.optim.sched.name != "NoamAnnealing":
        raise ValueError("Resume currently supports this recipe's NoamAnnealing scheduler only")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    original_assets = Path(cfg.model.tokenizer.dir).parent.resolve()
    # The recipe and metadata must refer to one coherent preparation bundle.
    if original_assets.name != metadata["fingerprint"]:
        raise ValueError("Resume recipe and preparation fingerprint disagree")
    for split in ("train", "validation"):
        if Path(cfg.model[f"{split}_ds"].manifest_filepath).resolve() != original_assets / f"{split}.jsonl":
            raise ValueError(f"Resume {split} manifest does not match the prepared assets")
    verify_assets(original_assets, metadata)
    base_model = Path(cfg.init_from_nemo_model).expanduser().resolve()
    if not base_model.is_file():
        raise FileNotFoundError(f"The original base .nemo is needed to reconstruct the model: {base_model}")
    log_dir = output_dir / "checkpoints" / str(cfg.exp_manager.name) / str(cfg.exp_manager.version)
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"Training run already exists: {log_dir}. Resume into a new --output-dir.")
    assets = output_dir / "assets" / metadata["fingerprint"]
    if assets.exists():
        verify_assets(assets, metadata)
    else:
        assets.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(original_assets, assets)
    for split in ("train", "validation"):
        cfg.model[f"{split}_ds"].manifest_filepath = str(assets / f"{split}.jsonl")
    cfg.model.tokenizer.dir = str(assets / "merged_tokenizer")
    cfg.init_from_nemo_model = str(base_model)
    cfg.exp_manager.exp_dir = str(output_dir / "checkpoints")
    # Resume is explicit via Trainer.fit(ckpt_path=...), never auto-selected.
    cfg.exp_manager.resume_if_exists = False
    cfg.exp_manager.resume_from_checkpoint = None
    if max_steps is not None:
        cfg.trainer.max_steps = max_steps
        cfg.trainer.max_epochs = -1
    cfg.model.optim.sched.max_steps = cfg.trainer.max_steps
    cfg.resume_from_checkpoint = str(checkpoint)
    language = metadata["language"]
    prompt_index = int(cfg.model.model_defaults.prompt_dictionary[language])
    OmegaConf.resolve(cfg)
    return assets, cfg, metadata, prompt_index


def validate_resume_checkpoint(checkpoint: dict, cfg, metadata: dict, prompt_index: int) -> int:
    """Validate token IDs, prompt mapping, and training state before continuing."""
    saved_cfg = OmegaConf.create(checkpoint.get("hyper_parameters", {}).get("cfg", {}))
    provenance = saved_cfg.get("custom_finetune", {})
    expected = {
        "method": "nvidia_unigram_merge", "language": metadata["language"],
        "prompt_index": prompt_index, "tokenizer_vocab_size": metadata["merged_vocab_size"],
        "tokenizer_sha256": metadata["file_sha256"]["merged_tokenizer/tokenizer.model"],
        "base_tokenizer_sha256": metadata["base_tokenizer_sha256"],
        "assets_fingerprint": metadata["fingerprint"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match the saved recipe/assets; refusing incompatible resume")
    if saved_cfg.model_defaults.prompt_dictionary != cfg.model.model_defaults.prompt_dictionary:
        raise ValueError("Checkpoint language prompt dictionary differs from the resume recipe")
    step = checkpoint.get("global_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Checkpoint has no valid optimizer step counter")
    if cfg.trainer.max_steps <= step:
        raise ValueError(f"Checkpoint already has {step} optimizer updates. Set --max-steps greater than {step}; "
                         "it is the TOTAL target, not additional steps.")
    if not checkpoint.get("state_dict") or len(checkpoint.get("optimizer_states", [])) != 1:
        raise ValueError("Resume needs a full training checkpoint with model and optimizer state")
    states = checkpoint.get("lr_schedulers", [])
    if len(states) != 1:
        raise ValueError("Resume needs the saved Noam learning-rate scheduler state")
    schedule = cfg.model.optim.sched
    for key, value in {"warmup_steps": schedule.warmup_steps, "min_lr": schedule.min_lr,
                       "_normalize": schedule.d_model ** -0.5,
                       "base_lrs": [cfg.model.optim.lr]}.items():
        if states[0].get(key) != value:
            raise ValueError(f"Checkpoint Noam {key} differs from the saved training recipe")
    return step


def resume_callback(cfg, metadata: dict, prompt_index: int):
    # Lazy import preserves CPU preparation without Lightning/CUDA dependencies.
    from lightning.pytorch import Callback

    class ResumeTraining(Callback):
        restored_step = None

        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            self.restored_step = validate_resume_checkpoint(checkpoint, cfg, metadata, prompt_index)

        def on_train_start(self, trainer, pl_module):
            if self.restored_step is None:
                raise RuntimeError("Resume requested, but Lightning did not restore a training checkpoint")
            epoch_loop = trainer.fit_loop.epoch_loop
            batches = epoch_loop.batch_progress
            if batches.is_last_batch or batches.current.completed >= trainer.num_training_batches:
                # Some Lightning versions keep iteration-based loops in restart
                # mode even after a completed epoch. The stale batch count then
                # marks the first new batch as the last one and flushes gradients
                # early. Complete the epoch counters and let the normal epoch
                # reset run, preserving total optimizer/update counters.
                progress = trainer.fit_loop.epoch_progress
                for tracker in (progress.current, progress.total):
                    completed = max(tracker.ready, tracker.started, tracker.processed, tracker.completed)
                    tracker.ready = tracker.started = tracker.processed = tracker.completed = completed
                epoch_loop.restarting = False
            # Lightning restored last_epoch, warmup and optimizer moments. Only
            # update the scheduler's recorded horizon to match the new target.
            for entry in trainer.lr_scheduler_configs:
                if type(entry.scheduler).__name__ != "NoamAnnealing":
                    raise ValueError("Expected the saved NoamAnnealing scheduler for resume")
                entry.scheduler.max_steps = trainer.max_steps
            print(f"Resumed at optimizer step {trainer.global_step}; target={trainer.max_steps} total, "
                  f"remaining={trainer.max_steps - trainer.global_step}. Warmup is not restarted.", flush=True)

    return ResumeTraining()
