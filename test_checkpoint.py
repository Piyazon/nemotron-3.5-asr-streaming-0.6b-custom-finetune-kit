#!/usr/bin/env python3
"""Transcribe test_files/ audio with the best validation-WER checkpoint.

This reconstructs the custom tokenizer and language prompt before loading the
Lightning checkpoint.  Loading the checkpoint directly into the untouched base
model is invalid because custom-language training changes the RNNT decoder and
joint vocabulary dimensions.

Run on the Linux training server, for example:

    python test_checkpoint.py
    python test_checkpoint.py sample2.mp3
    python test_checkpoint.py --device cuda
    python test_checkpoint.py sample2.mp3 --checkpoint /path/to/model.ckpt
    python test_checkpoint.py sample2.mp3 --decoding-strategy beam --beam-size 4
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import json
import re
import sys
from pathlib import Path

from checkpoint_selection import latest_checkpoint, run_directory, select_checkpoint


# =============================================================================
# Defaults (all can also be overridden on the command line)
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

CHECKPOINT: str | None = None
TOKENIZER_DIR: str | None = None

# Keep this on CPU while training occupies GPU 0. Use --device cuda only when
# a GPU has enough free memory.
DEVICE = "cpu"
LANGUAGE = "ug-CN"

# Set this only if training used --prompt-index. None reproduces the training
# script's automatic first-unused-slot allocation.
PROMPT_INDEX: int | None = None

BASE_MODEL = ROOT_DIR / "pretrained_model" / "nemotron-3.5-asr-streaming-0.6b.nemo"
CHECKPOINT_DIR = (
    ROOT_DIR
    / "checkpoints"
    / "FastConformer-Transducer-BPE-Prompt-Streaming"
    / "test"
)
TOKENIZER_ROOT_DIR = ROOT_DIR / "custom_asr_data" / "tokenizers"
TEST_FILES_DIR = ROOT_DIR / "test_files"
SUPPORTED_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


def resolve_path(path: str | Path) -> Path:
    """Resolve a user path relative to the current working directory."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def find_audio_files(audio: str | None) -> list[Path]:
    """Resolve one requested file or discover all audio under test_files/."""
    if audio:
        requested = Path(audio).expanduser()
        if requested.is_absolute():
            audio_path = requested.resolve()
        else:
            test_files_candidate = TEST_FILES_DIR / requested
            audio_path = (
                test_files_candidate.resolve()
                if test_files_candidate.exists()
                else resolve_path(requested)
            )

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found directly or under {TEST_FILES_DIR}: {audio}"
            )
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio extension: {audio_path.suffix}")
        return [audio_path]

    if not TEST_FILES_DIR.is_dir():
        raise FileNotFoundError(f"Test-audio directory not found: {TEST_FILES_DIR}")

    audio_files = sorted(
        path.resolve()
        for path in TEST_FILES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    if not audio_files:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise FileNotFoundError(
            f"No supported audio files found under {TEST_FILES_DIR}. "
            f"Supported extensions: {supported}"
        )
    return audio_files


def find_latest_checkpoint() -> Path:
    """Compatibility helper for callers explicitly requesting the latest weights."""
    return latest_checkpoint(CHECKPOINT_DIR)


def find_latest_tokenizer(language: str) -> Path:
    """Find the newest complete generated tokenizer for ``language``."""
    safe_language = re.sub(r"[^A-Za-z0-9_.-]+", "_", language).strip("._") or "language"
    language_dir = TOKENIZER_ROOT_DIR / safe_language
    required = ("tokenizer.model", "tokenizer.vocab", "vocab.txt")

    candidates = []
    for path in language_dir.glob("bpe_v*"):
        if path.is_dir() and all((path / name).is_file() for name in required):
            metadata_path = path / "metadata.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if metadata.get("language") != language:
                    continue
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No complete generated tokenizer found for {language!r} in: {language_dir}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def prompt_index_from_checkpoint(checkpoint: dict, language: str) -> int | None:
    """Read the language prompt index from checkpoint config when available."""
    value = checkpoint_config_value(checkpoint, f"model_defaults.prompt_dictionary.{language}")
    return int(value) if value is not None else None


def tokenizer_dir_from_checkpoint(checkpoint: dict) -> Path | None:
    """Read the exact generated tokenizer path recorded during training."""
    value = checkpoint_config_value(checkpoint, "custom_finetune.tokenizer_dir")
    if not value:
        # NeMo also saves the directory installed by change_vocabulary here.
        # Older training scripts may not have written custom_finetune metadata.
        value = checkpoint_config_value(checkpoint, "tokenizer.dir")
    return resolve_path(str(value)) if value else None


def configure_language_prompt(model, language: str, requested_index: int | None) -> int:
    """Reproduce the prompt allocation used by the fine-tuning script."""
    from omegaconf import OmegaConf, open_dict

    prompt_cfg = model.cfg.model_defaults.get("prompt_dictionary", {})
    if not prompt_cfg:
        prompt_dictionary = {}
    elif OmegaConf.is_config(prompt_cfg):
        prompt_dictionary = OmegaConf.to_container(prompt_cfg, resolve=True)
    else:
        prompt_dictionary = dict(prompt_cfg)

    if language in prompt_dictionary:
        existing_index = int(prompt_dictionary[language])
        if requested_index is not None and requested_index != existing_index:
            raise ValueError(
                f"{language!r} already uses prompt index {existing_index}, "
                f"not requested index {requested_index}."
            )
        return existing_index

    num_prompts = int(
        model.cfg.get("num_prompts", model.cfg.model_defaults.get("num_prompts", 128))
    )
    used_indices = {int(value) for value in prompt_dictionary.values()}

    if requested_index is None:
        selected_index = next(
            (index for index in range(num_prompts) if index not in used_indices),
            None,
        )
        if selected_index is None:
            raise RuntimeError(f"All {num_prompts} language-prompt slots are already assigned.")
    else:
        selected_index = requested_index
        if not 0 <= selected_index < num_prompts:
            raise ValueError(
                f"Prompt index must be in [0, {num_prompts - 1}], got {selected_index}."
            )
        if selected_index in used_indices:
            raise ValueError(f"Prompt index {selected_index} is already assigned.")

    prompt_dictionary[language] = selected_index
    with open_dict(model.cfg.model_defaults):
        model.cfg.model_defaults.prompt_dictionary = OmegaConf.create(prompt_dictionary)

    return selected_index


def checkpoint_config_value(checkpoint: dict, key: str):
    """Read a saved setting without mutating or detaching unresolved references."""
    from omegaconf import OmegaConf

    hyper_parameters = checkpoint.get("hyper_parameters", {})
    # Lightning preserves OmegaConf containers when saving NeMo hparams.
    # DictConfig implements Mapping, but is not a Python dict. The nested cfg
    # can independently be either a plain dict or a DictConfig.
    if not isinstance(hyper_parameters, Mapping):
        return None
    for config_key in ("cfg", "model_cfg"):
        cfg = hyper_parameters.get(config_key)
        if not isinstance(cfg, Mapping):
            continue
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        value = OmegaConf.select(cfg, key)
        if value is not None:
            if OmegaConf.is_config(value):
                return OmegaConf.create(OmegaConf.to_container(value, resolve=True))
            return value
    return None


def configure_decoding(
    model, checkpoint: dict, strategy: str = "checkpoint", beam_size: int | None = None,
) -> str:
    """Restore saved RNNT decoding, then apply an explicitly requested experiment.

    Updating cfg alone does not rebuild NeMo's live decoder. Use its public
    change_decoding_strategy API after the custom vocabulary has been installed.
    """
    from omegaconf import OmegaConf, open_dict

    if strategy not in ("checkpoint", "greedy", "greedy_batch", "beam"):
        raise ValueError(f"Unsupported --decoding-strategy: {strategy}")
    if beam_size is not None:
        if not isinstance(beam_size, int) or isinstance(beam_size, bool) or beam_size < 1:
            raise ValueError("--beam-size must be a positive integer")
        if strategy != "beam":
            raise ValueError("--beam-size requires --decoding-strategy beam")

    saved_cfg = checkpoint_config_value(checkpoint, "decoding")
    if saved_cfg is None and strategy == "checkpoint":
        return str(model.cfg.decoding.strategy)

    source_cfg = saved_cfg if saved_cfg is not None else model.cfg.decoding
    decoding_cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=True))
    with open_dict(decoding_cfg):
        if strategy != "checkpoint":
            decoding_cfg.strategy = strategy
        if strategy == "beam":
            if decoding_cfg.get("beam") is None:
                decoding_cfg.beam = {}
            with open_dict(decoding_cfg.beam):
                decoding_cfg.beam.beam_size = 4 if beam_size is None else beam_size
                decoding_cfg.beam.return_best_hypothesis = True

    model.change_decoding_strategy(decoding_cfg)
    return str(model.cfg.decoding.strategy)


def best_transcription(decoded) -> str:
    """Accept NeMo's best-only, N-best, and older tuple return formats."""
    hypotheses = decoded[0] if isinstance(decoded, tuple) else decoded
    if not hypotheses:
        return ""
    hypothesis = hypotheses[0]
    if hasattr(hypothesis, "n_best_hypotheses"):
        hypothesis = hypothesis.n_best_hypotheses
    if isinstance(hypothesis, list):
        if not hypothesis:
            return ""
        hypothesis = hypothesis[0]
    return hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe test_files/ audio with the best custom-language checkpoint."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help=(
            "One audio path or filename under test_files/; when omitted, "
            "transcribe every supported file under test_files/"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT,
        help="Specific .ckpt file; overrides automatic selection",
    )
    parser.add_argument("--run-name", help="Select checkpoints within this training run")
    parser.add_argument("--selection", choices=("best", "latest"), default="best",
                        help="Automatic checkpoint selection (default: best validation WER)")
    parser.add_argument(
        "--tokenizer-dir",
        default=TOKENIZER_DIR,
        help=(
            "Generated tokenizer directory; default: exact path recorded in the "
            "checkpoint, with newest-tokenizer fallback for older checkpoints"
        ),
    )
    parser.add_argument("--language", default=LANGUAGE, help=f"Language locale (default: {LANGUAGE})")
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=PROMPT_INDEX,
        help="Prompt index used for training; normally recovered or allocated automatically",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=DEVICE,
        help=f"Inference device (default: {DEVICE})",
    )
    parser.add_argument(
        "--decoding-strategy",
        choices=("checkpoint", "greedy", "greedy_batch", "beam"),
        default="checkpoint",
        help="Decoding strategy (default: restore the checkpoint's saved settings)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        help="Number of hypotheses to explore with --decoding-strategy beam (default: 4)",
    )
    args = parser.parse_args(argv)
    if args.beam_size is not None:
        if args.beam_size < 1:
            parser.error("--beam-size must be a positive integer")
        if args.decoding_strategy != "beam":
            parser.error("--beam-size requires --decoding-strategy beam")
    return args


def main() -> None:
    args = parse_args()
    if sys.platform == "darwin":
        raise SystemExit(
            "This inference script is disabled on macOS. Run it on the Linux "
            "server with the NeMo/CUDA environment."
        )

    # Heavy ML/audio imports deliberately occur after the macOS guard.
    import librosa
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

    audio_paths = find_audio_files(args.audio)
    checkpoint_path = (
        resolve_path(args.checkpoint) if args.checkpoint else select_checkpoint(
            run_directory(CHECKPOINT_DIR, args.run_name), args.selection
        )
    )

    for label, path in (
        ("Base model", BASE_MODEL),
        ("Checkpoint", checkpoint_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    device = torch.device(args.device)

    print("Reading checkpoint metadata and weights...")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    recorded_tokenizer_path = tokenizer_dir_from_checkpoint(checkpoint)
    if args.tokenizer_dir:
        tokenizer_path = resolve_path(args.tokenizer_dir)
    elif recorded_tokenizer_path is not None:
        tokenizer_path = recorded_tokenizer_path
        print(f"Using tokenizer recorded in checkpoint: {tokenizer_path}")
    else:
        tokenizer_path = find_latest_tokenizer(args.language)
        print(
            "WARNING: this older checkpoint has no tokenizer provenance; "
            f"falling back to newest complete tokenizer: {tokenizer_path}"
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    print("=" * 80)
    print("Nemotron 3.5 custom-language checkpoint transcription")
    print("=" * 80)
    print(f"Audio files: {len(audio_paths)}")
    for audio_path in audio_paths:
        print(f"             {audio_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Tokenizer  : {tokenizer_path}")
    print(f"Language   : {args.language}")
    print(f"Device     : {device}")
    print(f"Checkpoint epoch       : {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint global step : {checkpoint.get('global_step', 'unknown')}")

    checkpoint_prompt_index = prompt_index_from_checkpoint(checkpoint, args.language)
    requested_prompt_index = (
        args.prompt_index if args.prompt_index is not None else checkpoint_prompt_index
    )

    print("\nLoading base model architecture...")
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=str(BASE_MODEL),
        map_location="cpu",
    )

    print("Installing generated tokenizer and rebuilding RNNT output layers...")
    model.change_vocabulary(
        new_tokenizer_dir=str(tokenizer_path),
        new_tokenizer_type="bpe",
    )

    prompt_id = configure_language_prompt(
        model=model,
        language=args.language,
        requested_index=requested_prompt_index,
    )
    print(f"Prompt ID              : {prompt_id}")

    print("Loading fine-tuned weights with strict shape/key validation...")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint has no state_dict: {checkpoint_path}")

    if state_dict and all(key.startswith("model.") for key in state_dict):
        state_dict = {
            key[len("model."):]: value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    strategy = configure_decoding(model, checkpoint, args.decoding_strategy, args.beam_size)
    print(f"Decoding strategy      : {strategy}")
    if strategy == "beam":
        print(f"Beam size              : {model.cfg.decoding.beam.beam_size}")
        print("Beam search considers more alternatives and can be much slower on long audio.")
    del state_dict, checkpoint
    gc.collect()

    model = model.to(device)
    model.eval()

    prompt_indices = torch.tensor([prompt_id], dtype=torch.long, device=device)

    for file_number, audio_path in enumerate(audio_paths, start=1):
        print(f"\n[{file_number}/{len(audio_paths)}] Loading: {audio_path}")
        waveform, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        if waveform.size == 0:
            print("Skipping empty audio file.")
            continue

        duration = waveform.shape[0] / 16000
        print(f"Duration: {duration:.2f} seconds")

        audio = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
        audio_length = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)

        print("Transcribing...")
        with torch.inference_mode():
            encoder_output, encoded_lengths = model(
                input_signal=audio,
                input_signal_length=audio_length,
                prompt_indices=prompt_indices,
            )
            decoded = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                return_hypotheses=True,
            )

        transcription = best_transcription(decoded)

        print("-" * 80)
        print(f"TRANSCRIPTION: {audio_path.name}")
        print("-" * 80)
        print(transcription)
        print("=" * 80)


if __name__ == "__main__":
    main()
