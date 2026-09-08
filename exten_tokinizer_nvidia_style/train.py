#!/usr/bin/env python3
"""Extend Nemotron's tokenizer and train with NVIDIA's published settings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

if __package__:
    from .recipe import build_recipe, configure_model
    from .tokenizer import prepare_assets
else:
    from recipe import build_recipe, configure_model
    from tokenizer import prepare_assets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    return cli


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    for name in ("base_model", "train_manifest", "validation_manifest", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    assets, base_cfg, metadata = prepare_assets(
        args.base_model, args.train_manifest, args.validation_manifest,
        args.output_dir, args.language, args.tokenizer_vocab_size)
    cfg, prompt_index = build_recipe(base_cfg, assets, args.output_dir, args.language, args.prompt_index)
    from omegaconf import OmegaConf

    cfg.init_from_nemo_model = str(args.base_model)
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
    trainer.fit(model)
    # exp_manager also saves validation-selected .nemo / Lightning checkpoints.
    # Export final weights explicitly even when a very small dataset never
    # reaches a scheduled validation before the step cap.
    final_path = log_dir / "nemotron-extended-final.nemo"
    model.save_to(str(final_path))
    print(f"Final model: {final_path}", flush=True)


if __name__ == "__main__":
    main()
