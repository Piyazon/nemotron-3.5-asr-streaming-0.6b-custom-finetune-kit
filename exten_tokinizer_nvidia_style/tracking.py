"""Optional W&B tracking alongside NeMo's TensorBoard logger."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import shlex
from uuid import uuid4

from omegaconf import OmegaConf


def preparation_summary(metadata: dict, prompt_index: int) -> dict:
    """Upload aggregate diagnostics and provenance, without transcript examples."""
    keys = (
        "fingerprint", "language", "requested_vocab_size", "sentencepiece_version",
        "train_samples", "validation_samples", "base_vocab_size", "new_vocab_size",
        "added_tokens", "merged_vocab_size", "language_tag", "language_tag_id",
        "merged_normalizer", "byte_fallback", "base_tokenizer_sha256", "file_sha256",
    )
    summary = {key: metadata[key] for key in keys if key in metadata}
    summary["prompt_index"] = prompt_index
    summary["coverage"] = {
        split: {key: values[key] for key in ("tokens", "unknown_tokens", "unknown_rate") if key in values}
        for split, values in metadata.get("coverage", {}).items()
    }
    return summary


@contextmanager
def track_training(trainer, cfg, metadata: dict, prompt_index: int, log_dir: Path):
    """Own one fresh W&B run, including flushing after failure or interruption.

    Attach directly: exp_manager passes its local version (always ``test`` in
    NVIDIA's recipe) as the W&B ID, which would merge unrelated experiments.
    Imports stay lazy so CPU preparation and training without W&B still work.
    """
    if not cfg.wandb.enabled or not trainer.is_global_zero:
        yield None
        return
    try:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
    except ImportError as error:
        raise RuntimeError("Install W&B with: python -m pip install 'wandb>=0.19,<1'") from error
    if wandb.run is not None:
        raise RuntimeError("A W&B run is already active. Finish it before starting this training launcher.")

    save_dir = log_dir / "wandb_logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(
        project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.name,
        id=uuid4().hex, save_dir=str(save_dir), offline=cfg.wandb.offline,
        log_model=False, save_code=False,
        # NeMo can print reference/prediction text. Keep that console local.
        settings=wandb.Settings(console="off", disable_code=True),
    )
    run = None
    completed = False
    interrupted = False
    try:
        run = logger.experiment
        trainer.loggers = [*trainer.loggers, logger]
        run.define_metric("val_wer", summary="min")
        run.summary["training_status"] = "running"
        summary = preparation_summary(metadata, prompt_index)
        logger.log_hyperparams({"recipe": OmegaConf.to_container(cfg, resolve=True), "preparation": summary})

        # Small reproducibility files only; never attach audio, manifests,
        # transcript examples, model weights, or tokenizer binaries here.
        recipe_path = log_dir / "training_recipe.yaml"
        summary_path = log_dir / "preparation_summary.json"
        OmegaConf.save(cfg, recipe_path, resolve=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        artifact = wandb.Artifact(f"run-{run.id}-metadata", type="run-metadata")
        artifact.add_file(str(recipe_path))
        artifact.add_file(str(summary_path))
        run.log_artifact(artifact)

        offline = run.offline
        run_info = {"id": run.id, "project": run.project, "name": run.name,
                    "offline": offline, "directory": str(Path(run.dir).parent),
                    "url": None if offline else run.url}
        (log_dir / "wandb_run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
        if offline:
            print(f"W&B offline; upload later: wandb sync {shlex.quote(run_info['directory'])}", flush=True)
        else:
            print(f"W&B dashboard: {run.url}", flush=True)
        yield run
        completed = not trainer.interrupted
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if run is not None:
            try:
                run.summary["training_status"] = (
                    "completed" if completed else "interrupted" if interrupted or trainer.interrupted else "failed")
                run.summary["optimizer_steps_completed"] = trainer.global_step
                run.summary["final_epoch_index"] = trainer.current_epoch
                checkpoint = trainer.checkpoint_callback
                if checkpoint is not None and checkpoint.monitor == "val_wer" and checkpoint.best_model_score is not None:
                    best_wer = float(checkpoint.best_model_score)
                    if math.isfinite(best_wer):
                        run.summary["best_val_wer"] = best_wer
                        run.summary["best_checkpoint_filename"] = Path(checkpoint.best_model_path).name
            finally:
                # Lightning's WandbLogger.finalize does not finish the SDK run.
                run.finish(exit_code=0 if completed else 1)
