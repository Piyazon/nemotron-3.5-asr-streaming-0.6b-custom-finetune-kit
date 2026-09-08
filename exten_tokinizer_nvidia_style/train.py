#!/usr/bin/env python3
"""Extend Nemotron's tokenizer and train with NVIDIA's published settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

if __package__:
    from .recipe import build_recipe, configure_model
    from .tokenizer import prepare_assets
    from .tracking import track_training
    from .resume import checkpoint_fit_kwargs, load_resume_recipe, resume_callback
else:
    from recipe import build_recipe, configure_model
    from tokenizer import prepare_assets
    from tracking import track_training
    from resume import checkpoint_fit_kwargs, load_resume_recipe, resume_callback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_int(value: str) -> int:
    number = nonnegative_int(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  allow_abbrev=False)
    cli.add_argument("--base-model", type=Path, default=ROOT / "pretrained_model/nemotron-3.5-asr-streaming-0.6b.nemo")
    cli.add_argument("--train-manifest", type=Path, default=ROOT / "custom_asr_data/train_manifest.json")
    cli.add_argument("--validation-manifest", type=Path, default=ROOT / "custom_asr_data/test_manifest.json",
                     help="Held-out validation set; never used to train the tokenizer")
    cli.add_argument("--language", default="ug-CN")
    cli.add_argument("--tokenizer-vocab-size", type=int, default=2048,
                     help="Requested new-language Unigram size BEFORE merging")
    cli.add_argument("--prompt-index", type=int, help="Optional unused slot; otherwise allocate automatically")
    cli.add_argument("--output-dir", type=Path, default=HERE / "runs/ug-CN-nvidia")
    cli.add_argument("--prepare-only", action="store_true", help="Build and inspect assets/config on CPU; do not train")
    cli.add_argument("--max-steps", type=positive_int,
                     help="Total optimizer-update target; default 2000 for a fresh run, or the saved limit on resume")
    cli.add_argument("--resume-from", type=Path,
                     help="Resume an extended-tokenizer .ckpt or run folder with one *-last.ckpt; requires a fresh output directory")
    batching = cli.add_argument_group("Optional batching overrides; omitted flags preserve the recipe")
    batching.add_argument("--batch-duration", type=positive_float,
                          help="Dynamic training batch duration budget in seconds; changes the effective audio batch")
    batching.add_argument("--fused-batch-size", type=positive_int,
                          help="Clips per internal RNNT joint/loss batch; enables fused loss/WER batching")
    batching.add_argument("--train-workers", type=nonnegative_int, help="Training data-loader processes")
    batching.add_argument("--validation-workers", type=nonnegative_int, help="Validation data-loader processes")
    batching.add_argument("--validation-batch-size", type=positive_int, help="Clips per validation batch")
    tracking = cli.add_argument_group("Optional Weights & Biases tracking")
    tracking.add_argument("--wandb", action="store_true", help="Upload metrics, run settings, and aggregate tokenizer diagnostics")
    tracking.add_argument("--wandb-project", help="W&B project; defaults to WANDB_PROJECT or nemotron-asr-finetune")
    tracking.add_argument("--wandb-entity", help="W&B team/user; defaults to your W&B account settings")
    tracking.add_argument("--wandb-name", help="Run display name; defaults to the output directory name")
    tracking.add_argument("--wandb-offline", action="store_true", help="With --wandb, record locally for later wandb sync")
    return cli


def main(argv: list[str] | None = None) -> None:
    cli = parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = cli.parse_args(argv)
    if not args.wandb and any((args.wandb_project, args.wandb_entity, args.wandb_name, args.wandb_offline)):
        cli.error("W&B options require --wandb")
    for name in ("base_model", "train_manifest", "validation_manifest", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.resume_from:
        # The saved data/tokenizer/prompt/batching settings define the run being
        # continued. Reject conflicting CLI options instead of ignoring them.
        inherited = {"--base-model", "--train-manifest", "--validation-manifest", "--language",
                     "--tokenizer-vocab-size", "--prompt-index", "--batch-duration", "--fused-batch-size",
                     "--train-workers", "--validation-workers", "--validation-batch-size"}
        if supplied := sorted({token.split("=", 1)[0] for token in argv} & inherited):
            cli.error(f"--resume-from inherits these settings from the saved run; omit: {', '.join(supplied)}")
        args.resume_from = args.resume_from.expanduser().resolve()
        assets, cfg, metadata, prompt_index = load_resume_recipe(args.resume_from, args.output_dir, args.max_steps)
        args.resume_from = Path(cfg.resume_from_checkpoint)
        args.base_model = Path(cfg.init_from_nemo_model)
        args.language = metadata["language"]
    else:
        assets, base_cfg, metadata = prepare_assets(
            args.base_model, args.train_manifest, args.validation_manifest,
            args.output_dir, args.language, args.tokenizer_vocab_size)
        cfg, prompt_index = build_recipe(
            base_cfg, assets, args.output_dir, args.language, args.prompt_index,
            batch_duration=args.batch_duration, fused_batch_size=args.fused_batch_size,
            train_workers=args.train_workers, validation_workers=args.validation_workers,
            validation_batch_size=args.validation_batch_size, max_steps=args.max_steps)
    from omegaconf import OmegaConf

    cfg.init_from_nemo_model = str(args.base_model)
    cfg.wandb = {
        "enabled": args.wandb,
        "project": args.wandb_project or os.environ.get("WANDB_PROJECT") or "nemotron-asr-finetune",
        "entity": args.wandb_entity,
        "name": args.wandb_name or args.output_dir.name,
        "offline": args.wandb_offline,
    }
    # tracking.py owns W&B separately from exp_manager's fixed local version.
    cfg.exp_manager.create_wandb_logger = False
    log_dir = Path(cfg.exp_manager.exp_dir) / str(cfg.exp_manager.name) / str(cfg.exp_manager.version)
    # Do not overwrite weights or silently resume a run with changed token IDs.
    if not args.prepare_only and log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"Training run already exists: {log_dir}. Use a new --output-dir for a fresh run.")
    config_path = assets / f"training_prompt{prompt_index}.yaml"
    OmegaConf.save(cfg, config_path, resolve=True)
    stats = {key: metadata[key] for key in ("base_vocab_size", "new_vocab_size", "added_tokens", "merged_vocab_size")}
    print(json.dumps({"assets": str(assets), "vocabulary": stats, "language": args.language,
                      "prompt_index": prompt_index, "training_config": str(config_path),
                      "coverage": metadata["coverage"]}, ensure_ascii=False, indent=2), flush=True)
    if metadata["coverage"]["validation"]["unknown_tokens"]:
        print("WARNING: validation contains unknown tokens; inspect the coverage report. Held-out text was not used to build the tokenizer.", flush=True)
    if args.prepare_only:
        return

    import torch
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Training requires Linux and an NVIDIA CUDA GPU. CPU preparation succeeded; use --prepare-only on this machine.")
    from lightning.pytorch import Trainer
    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    except ImportError as error:
        raise ImportError("This NeMo installation lacks the Nemotron prompted RNNT class. Install the current NeMo Git dependency from requirements.txt on the training server.") from error
    from nemo.utils.exp_manager import exp_manager
    from nemo.utils.trainer_utils import resolve_trainer_cfg

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.exp_manager)
    if args.resume_from:
        trainer.callbacks.append(resume_callback(cfg, metadata, prompt_index))
    with track_training(trainer, cfg, metadata, prompt_index, log_dir) as wandb_run:
        model = EncDecRNNTBPEModelWithPrompt.restore_from(str(args.base_model), map_location="cpu", trainer=trainer)
        actual_base_hash = hashlib.sha256(model.tokenizer.tokenizer.serialized_model_proto()).hexdigest()
        if actual_base_hash != metadata["base_tokenizer_sha256"]:
            raise ValueError("Loaded tokenizer does not match the base tokenizer used for preparation")
        configure_model(model, cfg, metadata, prompt_index)
        if model.tokenizer.vocab_size != metadata["merged_vocab_size"] or len(model.joint.vocabulary) != metadata["merged_vocab_size"]:
            raise ValueError("Model output vocabulary differs from the merged tokenizer")
        OmegaConf.save(model.cfg, log_dir / "model_config_before_fit.yaml", resolve=True)
        OmegaConf.save(cfg, log_dir / "training_recipe.yaml", resolve=True)
        (log_dir / "preparation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Training: {cfg.trainer.max_steps} optimizer steps; Noam scale={cfg.model.optim.lr}, "
              f"warmup={cfg.model.optim.sched.warmup_steps}; {args.language} prompt={prompt_index}", flush=True)
        print(f"Batching: audio budget={cfg.model.train_ds.batch_duration}s, "
              f"RNNT internal={model.joint.fused_batch_size}, "
              f"accumulation={cfg.trainer.accumulate_grad_batches}, "
              f"validation clips={cfg.model.validation_ds.batch_size}, "
              f"workers={cfg.model.train_ds.num_workers}/{cfg.model.validation_ds.num_workers}", flush=True)
        trainer.fit(model, **checkpoint_fit_kwargs(trainer, args.resume_from))
        # exp_manager also saves validation-selected .nemo / Lightning checkpoints.
        # Export final weights explicitly even when a very small dataset never
        # reaches a scheduled validation before the step cap.
        final_path = log_dir / "nemotron-extended-final.nemo"
        model.save_to(str(final_path))
        if wandb_run is not None:
            wandb_run.summary["final_model_filename"] = final_path.name
        print(f"Final model: {final_path}", flush=True)


if __name__ == "__main__":
    main()
