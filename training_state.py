"""Checkpoint/config safeguards for continuing a custom-vocabulary ASR run."""

import copy
import hashlib
import math
from pathlib import Path


def noam_scale_for_peak(peak_lr, d_model, warmup_steps):
    if not math.isfinite(peak_lr) or peak_lr <= 0:
        raise ValueError("--peak-lr must be positive and finite")
    if d_model <= 0 or warmup_steps <= 0:
        raise ValueError("Noam dimension and warmup steps must be positive")
    return peak_lr * math.sqrt(d_model * warmup_steps)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_config(checkpoint):
    from omegaconf import OmegaConf

    for key in ("cfg", "model_cfg"):
        value = checkpoint.get("hyper_parameters", {}).get(key)
        if value is not None:
            return OmegaConf.create(copy.deepcopy(value))
    raise ValueError("Checkpoint has no saved model config; use an exported .nemo for initialization")


def prepare_resume_config(checkpoint, tokenizer_dir=None):
    """Repair archive-only artifact paths without rebuilding the vocabulary."""
    from omegaconf import OmegaConf, open_dict
    import sentencepiece as spm

    if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
        raise ValueError("--resume-from requires optimizer and scheduler states in a Lightning .ckpt")
    groups = checkpoint["optimizer_states"][0].get("param_groups", [])
    if len(groups) != 2 or any(not group.get("param_names") for group in groups):
        raise ValueError("Legacy checkpoint has no optimizer parameter-name mapping. Export it to .nemo and use --init-from-nemo to avoid attaching Adam state to the wrong parameters")
    cfg = checkpoint_config(checkpoint)
    meta = cfg.get("custom_finetune", {})
    directory = tokenizer_dir or meta.get("tokenizer_dir") or OmegaConf.select(cfg, "tokenizer.dir")
    if not directory or str(directory).startswith("nemo:"):
        raise ValueError("Cannot locate the checkpoint tokenizer; supply its exact --tokenizer-dir")
    directory = Path(directory).expanduser().resolve()
    files = {"model_path": directory / "tokenizer.model",
             "vocab_path": directory / "vocab.txt",
             "spe_tokenizer_vocab": directory / "tokenizer.vocab"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint tokenizer missing: {path}; supply --tokenizer-dir")
    if OmegaConf.select(cfg, "tokenizer.type") != "bpe":
        raise ValueError("Checkpoint resume currently requires a monolingual BPE tokenizer; use --init-from-nemo otherwise")
    expected_hash = meta.get("tokenizer_sha256")
    if expected_hash and file_sha256(files["model_path"]) != expected_hash:
        raise ValueError("Tokenizer SHA256 differs from the checkpoint; refusing to change token IDs")
    processor = spm.SentencePieceProcessor(model_file=str(files["model_path"]))
    pieces = [processor.id_to_piece(index) for index in range(processor.get_piece_size())]
    vocabulary = OmegaConf.select(cfg, "joint.vocabulary")
    if vocabulary is None or list(vocabulary) != pieces:
        raise ValueError("Tokenizer vocabulary/order differs from the checkpoint joint vocabulary")
    with open_dict(cfg.tokenizer):
        cfg.tokenizer.dir = str(directory)
        for key, path in files.items():
            cfg.tokenizer[key] = str(path)
    return cfg, str(directory)


def resume_settings(cfg, checkpoint):
    """Use saved optimization/data settings; Lightning restores their state."""
    from omegaconf import OmegaConf

    meta = cfg.get("custom_finetune", {})
    if OmegaConf.select(cfg, "optim.sched.name") != "NoamAnnealing":
        raise ValueError("Resume supports this kit's NoamAnnealing optimizer configuration")
    required = ("language", "encoder_lr_scale", "max_duration", "batch_duration", "seed")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"Checkpoint lacks resume provenance {missing}; initialize from an exported .nemo")
    scheduler = checkpoint["lr_schedulers"][0]
    warmup = int(scheduler.get("warmup_steps", cfg.optim.sched.warmup_steps))
    # Current NeMo serializes _normalize rather than d_model in scheduler state.
    dimension = int(round(float(scheduler["_normalize"]) ** -2)) if "_normalize" in scheduler else int(cfg.optim.sched.d_model)
    settings = {key: meta[key] for key in required}
    settings.update(lr=float(cfg.optim.lr), warmup_steps=warmup, noam_d_model=dimension,
                    tokenizer_mode=meta.get("tokenizer_mode", "custom"),
                    tokenizer_vocab_size=int(meta.get("tokenizer_vocab_size", cfg.decoder.vocab_size)))
    expected_bases = [settings["lr"] * settings["encoder_lr_scale"], settings["lr"]]
    bases = scheduler.get("base_lrs", [])
    if len(bases) != 2 or any(not math.isclose(a, b, rel_tol=1e-7) for a, b in zip(bases, expected_bases)):
        raise ValueError("Saved optimizer parameter groups differ from this kit's encoder/decoder groups")
    return settings


def construct_resume_model(model_class, cfg):
    """Instantiate saved architecture while deferring data-loader construction."""
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    datasets = {key: copy.deepcopy(cfg.get(key)) for key in ("train_ds", "validation_ds", "test_ds")}
    with open_dict(cfg):
        for key in datasets:
            cfg[key] = None
    model = model_class(cfg=cfg)
    with open_dict(model.cfg):
        for key, value in datasets.items():
            model.cfg[key] = value
    return model


def validate_optimizer_group_names(current, saved):
    if len(current) != len(saved) or any(
        a.get("param_names") != b.get("param_names") for a, b in zip(current, saved)
    ):
        raise ValueError("Optimizer parameter ordering differs from the checkpoint; use --init-from-nemo for weights-only initialization")


def persist_tokenizer(tokenizer, directory):
    """Keep exact SentencePiece bytes from a .nemo's in-memory tokenizer."""
    processor = getattr(tokenizer, "tokenizer", None)
    serialize = getattr(processor, "serialized_model_proto", None)
    if not callable(serialize):
        raise ValueError("Continuation currently requires a SentencePiece tokenizer that can export its original model")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_bytes = serialize()
    target = directory / "tokenizer.model"
    if target.exists() and target.read_bytes() != model_bytes:
        raise ValueError("The run directory already contains a different tokenizer")
    target.write_bytes(model_bytes)
    pieces = [processor.id_to_piece(i) for i in range(processor.get_piece_size())]
    (directory / "tokenizer.vocab").write_text(
        "".join(f"{piece}\t{processor.get_score(i)}\n" for i, piece in enumerate(pieces)), encoding="utf-8")
    # NeMo's auxiliary vocabulary omits control symbols, maps leading word
    # boundaries to whole words, and prefixes within-word pieces with ##.
    words = [(piece[1:] or piece) if piece.startswith("▁") else "##" + piece
             for i, piece in enumerate(pieces) if not processor.is_control(i) and not processor.is_unknown(i)]
    (directory / "vocab.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    return str(directory.resolve())


def validate_resume_manifests(cfg, train_manifest, validation_manifest):
    meta = cfg.get("custom_finetune", {})
    for label, path in (("train", train_manifest), ("validation", validation_manifest)):
        expected = meta.get(f"{label}_manifest_sha256")
        if expected and file_sha256(path) != expected:
            raise ValueError(f"The {label} manifest changed since this checkpoint. Use --init-from-nemo for a new data stage")
