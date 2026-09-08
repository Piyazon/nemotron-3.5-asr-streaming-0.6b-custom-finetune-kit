"""Keep NVIDIA's training recipe while persisting the new language prompt."""

from __future__ import annotations

import copy
from pathlib import Path

from omegaconf import OmegaConf, open_dict

HERE = Path(__file__).resolve().parent


def allocate_prompt(base_cfg, language: str, requested_index: int | None = None) -> tuple[dict, int, int]:
    """Reserve a slot absent from every saved prompt dictionary, retaining aliases."""
    defaults = base_cfg.get("model_defaults") or {}
    num_prompts = int(base_cfg.get("num_prompts", defaults.get("num_prompts", 128)))
    mapping = {}
    for section in ("model_defaults", "train_ds", "validation_ds", "test_ds"):
        section_cfg = base_cfg.get(section) or {}
        for locale, value in (section_cfg.get("prompt_dictionary") or {}).items():
            index = int(value)
            if index != value or not 0 <= index < num_prompts:
                raise ValueError(f"Invalid saved prompt slot for {locale}: {value}")
            if locale in mapping and mapping[locale] != index:
                raise ValueError(f"Conflicting saved prompt slots for {locale}")
            mapping[locale] = index
    if not mapping:
        raise ValueError("The base model has no language prompt dictionary")
    if language in mapping:
        index = mapping[language]
        if requested_index is not None and requested_index != index:
            raise ValueError(f"{language} already uses slot {index}")
    else:
        used = set(mapping.values())
        index = requested_index
        if index is None:
            index = next((i for i in range(num_prompts) if i not in used), None)
        if index is None:
            raise ValueError(f"All {num_prompts} language prompt slots are occupied")
        if not 0 <= index < num_prompts or index in used:
            raise ValueError(f"Prompt slot {index} is out of range or already occupied")
        mapping[language] = index
    return mapping, index, num_prompts


def build_recipe(base_cfg, assets: Path, output_dir: Path, language: str,
                 requested_index: int | None = None):
    base_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg = OmegaConf.merge(
        OmegaConf.load(HERE / "upstream/fastconformer_transducer_bpe_streaming_prompt.yaml"),
        OmegaConf.load(HERE / "nvidia_recipe.yaml"),
    )
    mapping, index, num_prompts = allocate_prompt(base_cfg, language, requested_index)
    # The upstream finetune entry point restores architecture from .nemo. Do not
    # instantiate its example's 42-layer encoder instead of the 0.6B checkpoint.
    cfg.model.model_defaults = copy.deepcopy(base_cfg.model_defaults)
    cfg.model.encoder = copy.deepcopy(base_cfg.encoder)
    cfg.model.model_defaults.prompt_dictionary = OmegaConf.create(mapping)
    cfg.model.model_defaults.num_prompts = num_prompts
    cfg.model.train_ds.manifest_filepath = str(assets / "train.jsonl")
    cfg.model.validation_ds.manifest_filepath = str(assets / "validation.jsonl")
    cfg.model.tokenizer.dir = str(assets / "merged_tokenizer")
    # Our input is ordinary JSONL + audio, not tarred shards. Explicitly force
    # the selected language in validation too (the upstream default is unified).
    for split in ("train_ds", "validation_ds", "test_ds"):
        ds = cfg.model[split]
        ds.is_tarred = False
        ds.initialize_prompt_feature = True
        ds.default_prompt_mode = "langID"
        ds.default_lang = language
        ds.unified_auto_ratio = 0.0
        ds.prompt_dictionary = OmegaConf.create(mapping)
        ds.num_prompts = num_prompts
    cfg.exp_manager.exp_dir = str(output_dir / "checkpoints")
    # ModelPT persists child configs under model.cfg (without the recipe's
    # outer `model` key). Resolve root-relative references before handing off.
    OmegaConf.resolve(cfg)
    return cfg, index


def configure_model(model, cfg, metadata: dict, prompt_index: int) -> None:
    """Update vocabulary, model config, and inference mapping before optimization."""
    model.change_vocabulary(new_tokenizer_dir=cfg.model.tokenizer.dir, new_tokenizer_type="bpe")
    with open_dict(model.cfg):
        model.cfg.model_defaults.prompt_dictionary = copy.deepcopy(cfg.model.model_defaults.prompt_dictionary)
        model.cfg.model_defaults.num_prompts = cfg.model.model_defaults.num_prompts
        model.cfg.num_prompts = cfg.model.model_defaults.num_prompts
        model.cfg.compute_eval_loss = cfg.model.compute_eval_loss
        model.cfg.spec_augment = copy.deepcopy(cfg.model.spec_augment)
        model.cfg.custom_finetune = OmegaConf.create({
            "method": "nvidia_unigram_merge", "language": metadata["language"],
            "prompt_index": prompt_index, "tokenizer_mode": "merged",
            "tokenizer_dir": cfg.model.tokenizer.dir, "tokenizer_vocab_size": metadata["merged_vocab_size"],
            "base_tokenizer_sha256": metadata["base_tokenizer_sha256"],
            "tokenizer_sha256": metadata["file_sha256"]["merged_tokenizer/tokenizer.model"],
            "assets_fingerprint": metadata["fingerprint"],
        })
    model.compute_eval_loss = bool(cfg.model.compute_eval_loss)
    # Like speech_to_text_finetune.py, retain the restored architecture and
    # use the recipe's dataloaders, optimizer and SpecAugment.
    model.setup_training_data(cfg.model.train_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)
    # A saved test config can otherwise retain the old prompt dictionary.
    with open_dict(model.cfg):
        model.cfg.test_ds = copy.deepcopy(cfg.model.test_ds)
    model.setup_optimization(cfg.model.optim)
    model.spec_augment = model.from_config_dict(cfg.model.spec_augment)
