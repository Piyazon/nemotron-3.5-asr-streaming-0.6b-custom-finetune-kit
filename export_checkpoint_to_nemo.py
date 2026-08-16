#!/usr/bin/env python3
"""Export a custom-language Lightning checkpoint as a self-contained .nemo.

Run this on the Linux NeMo training server.  The resulting .nemo can be fed to
parakeet.cpp's ``scripts/convert_parakeet_to_gguf.py`` converter.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from test_checkpoint import (
    BASE_MODEL,
    CHECKPOINT_DIR,
    LANGUAGE,
    PROMPT_INDEX,
    configure_language_prompt,
    find_latest_checkpoint,
    find_latest_tokenizer,
    prompt_index_from_checkpoint,
    resolve_path,
    tokenizer_dir_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a fine-tuned .ckpt and generated tokenizer as a .nemo file."
    )
    parser.add_argument(
        "--checkpoint",
        help="Specific .ckpt; default: newest retained epoch checkpoint",
    )
    parser.add_argument(
        "--output",
        help="Output .nemo path; default: checkpoint directory with epoch in its name",
    )
    parser.add_argument(
        "--tokenizer-dir",
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
        help="Prompt index used during training; normally recovered automatically",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output .nemo file",
    )
    return parser.parse_args()


def default_output_path(
    checkpoint_path: Path,
    epoch: object,
    language: str,
) -> Path:
    try:
        epoch_label = f"{int(epoch):02d}"
    except (TypeError, ValueError):
        epoch_label = "unknown"
    safe_language = "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in language
    ).strip("._") or "language"
    return checkpoint_path.parent / (
        f"nemotron-asr-{safe_language}-epoch-{epoch_label}.nemo"
    )


def main() -> None:
    if sys.platform == "darwin":
        raise SystemExit(
            "Checkpoint export requires the Linux NeMo training environment. "
            "Export on Linux, then copy the resulting .nemo to macOS."
        )

    # Heavy dependencies load only after the platform guard.
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt

    args = parse_args()
    checkpoint_path = (
        resolve_path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    )

    for label, path in (
        ("Base model", BASE_MODEL),
        ("Checkpoint", checkpoint_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    print(f"Reading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
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

    output_path = (
        resolve_path(args.output)
        if args.output
        else default_output_path(checkpoint_path, epoch, args.language)
    )
    if output_path.suffix.lower() != ".nemo":
        raise ValueError(f"Output path must end in .nemo: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\nUse --overwrite to replace it."
        )

    checkpoint_prompt_index = prompt_index_from_checkpoint(checkpoint, args.language)
    requested_prompt_index = (
        args.prompt_index if args.prompt_index is not None else checkpoint_prompt_index
    )

    print(f"Checkpoint epoch: {epoch}; global step: {global_step}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Language: {args.language}")
    print(f"Loading base model: {BASE_MODEL}")
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
    print(f"Prompt ID: {prompt_id}")

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint has no state_dict: {checkpoint_path}")
    if state_dict and all(key.startswith("model.") for key in state_dict):
        state_dict = {
            key[len("model."):]: value
            for key, value in state_dict.items()
        }

    print("Loading fine-tuned weights with strict validation...")
    model.load_state_dict(state_dict, strict=True)
    del state_dict, checkpoint
    gc.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving self-contained NeMo archive: {output_path}")
    model.eval()
    model.save_to(str(output_path))
    print(f"Export complete: {output_path}")


if __name__ == "__main__":
    main()
