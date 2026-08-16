#!/usr/bin/env python3
"""Transcribe one audio file with the newest retained fine-tuning checkpoint.

This reconstructs the custom tokenizer and language prompt before loading the
Lightning checkpoint.  Loading the checkpoint directly into the untouched base
model is invalid because custom-language training changes the RNNT decoder and
joint vocabulary dimensions.

Run on the Linux training server, for example:

    python test_checkpoint.py sample2.mp3
    python test_checkpoint.py sample2.mp3 --device cuda
    python test_checkpoint.py sample2.mp3 --checkpoint /path/to/model.ckpt
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path


# =============================================================================
# Defaults (all can also be overridden on the command line)
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent

# AUDIO_FILE = "ref_arhip.wav"
# AUDIO_FILE = "sultan_20251224_0012_22050hz_1ch_segment_008.wav"

AUDIO_FILE = "sample2.mp3"
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


def resolve_path(path: str | Path) -> Path:
    """Resolve a user path relative to the current working directory."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def find_latest_checkpoint() -> Path:
    """Return the newest stable, epoch-named checkpoint.

    The epoch checkpoint is preferred over ``last.ckpt`` because ``last.ckpt``
    may be replaced while a concurrent training process finishes an epoch.
    """
    epoch_checkpoints = list(CHECKPOINT_DIR.glob("nemotron-asr-finetuned-epoch=*.ckpt"))

    if epoch_checkpoints:
        def checkpoint_order(path: Path) -> tuple[int, int]:
            match = re.search(r"epoch=(\d+)", path.name)
            epoch = int(match.group(1)) if match else -1
            return epoch, path.stat().st_mtime_ns

        return max(epoch_checkpoints, key=checkpoint_order)

    last_checkpoint = CHECKPOINT_DIR / "last.ckpt"
    if last_checkpoint.is_file():
        return last_checkpoint

    raise FileNotFoundError(f"No .ckpt files found in: {CHECKPOINT_DIR}")


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
    from omegaconf import OmegaConf

    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyper_parameters, dict):
        return None

    for key in ("cfg", "model_cfg"):
        cfg = hyper_parameters.get(key)
        if cfg is None:
            continue
        try:
            value = OmegaConf.select(cfg, f"model_defaults.prompt_dictionary.{language}")
        except (AttributeError, TypeError, ValueError):
            value = None
        if value is not None:
            return int(value)

    return None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one audio file with the latest custom-language checkpoint."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=AUDIO_FILE,
        help=f"WAV/MP3/FLAC file to transcribe (default: {AUDIO_FILE})",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT,
        help="Specific .ckpt file; default: newest retained epoch checkpoint",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=TOKENIZER_DIR,
        help="Generated tokenizer directory; default: newest tokenizer for --language",
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
    return parser.parse_args()


def main() -> None:
    if sys.platform == "darwin":
        raise SystemExit(
            "This inference script is disabled on macOS. Run it on the Linux "
            "server with the NeMo/CUDA environment."
        )

    args = parse_args()

    # Heavy ML/audio imports deliberately occur after the macOS guard.
    import librosa
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

    audio_path = resolve_path(args.audio)
    checkpoint_path = (
        resolve_path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    )
    tokenizer_path = (
        resolve_path(args.tokenizer_dir)
        if args.tokenizer_dir
        else find_latest_tokenizer(args.language)
    )

    for label, path in (
        ("Audio", audio_path),
        ("Base model", BASE_MODEL),
        ("Checkpoint", checkpoint_path),
        ("Tokenizer", tokenizer_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    device = torch.device(args.device)

    print("=" * 80)
    print("Nemotron 3.5 custom-language checkpoint transcription")
    print("=" * 80)
    print(f"Audio      : {audio_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Tokenizer  : {tokenizer_path}")
    print(f"Language   : {args.language}")
    print(f"Device     : {device}")

    print("\nReading checkpoint metadata and weights...")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
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
    del state_dict, checkpoint
    gc.collect()

    model = model.to(device)
    model.eval()

    print("\nLoading and resampling audio to mono 16 kHz...")
    waveform, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    duration = waveform.shape[0] / 16000
    print(f"Duration               : {duration:.2f} seconds")

    audio = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
    audio_length = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
    prompt_indices = torch.tensor([prompt_id], dtype=torch.long, device=device)

    print("\nTranscribing...\n")
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

    # Some NeMo versions return (best_hypotheses, all_hypotheses).
    hypotheses = decoded[0] if isinstance(decoded, tuple) else decoded
    if not hypotheses:
        transcription = ""
    else:
        hypothesis = hypotheses[0]
        transcription = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)

    print("=" * 80)
    print("TRANSCRIPTION")
    print("=" * 80)
    print(transcription)
    print("=" * 80)


if __name__ == "__main__":
    main()
