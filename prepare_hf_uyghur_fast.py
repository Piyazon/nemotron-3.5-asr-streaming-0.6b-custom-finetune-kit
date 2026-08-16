#!/usr/bin/env python3

# Prevent each worker from spawning lots of BLAS threads
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if sys.platform == "darwin":
    raise SystemExit(
        "Dataset preparation is intentionally disabled on macOS. Run it in "
        "the Linux training environment; use macOS only to edit the files."
    )

import numpy as np
import soundfile as sf

from datasets import load_dataset, Audio


DATASET_ID = "piyazon/cv-corpus-ug-24-augment"
TEXT_COLUMN = "sentence"

LANGUAGE = "ug-CN"
SAMPLE_RATE = 16000

MIN_DURATION = 0.2
MAX_DURATION = 60.0

DEFAULT_LONG_FORM_FRACTION = 0.25
DEFAULT_LONG_FORM_MIN_DURATION = 30.0
DEFAULT_LONG_FORM_TARGET_DURATION = 45.0
DEFAULT_LONG_FORM_MAX_DURATION = 55.0
DEFAULT_LONG_FORM_GAP = 0.25
RANDOM_SEED = 42


def normalize_text(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def decode_audio(audio):
    """
    Support current HuggingFace/torchcodec AudioDecoder
    and older datasets audio dictionaries.
    """

    # Older datasets
    if isinstance(audio, dict):
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        return arr, sr

    # Current datasets + torchcodec
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()

        arr = samples.data

        if hasattr(arr, "cpu"):
            arr = arr.cpu().numpy()
        else:
            arr = np.asarray(arr)

        sr = int(samples.sample_rate)

        return arr.astype(np.float32, copy=False), sr

    raise TypeError(f"Unknown audio type: {type(audio)}")


def make_mono(audio):
    audio = np.asarray(audio)

    if audio.ndim == 1:
        return audio

    # torchcodec normally returns [channels, samples]
    if audio.shape[0] <= 8:
        return audio.mean(axis=0)

    return audio.mean(axis=1)


def process_batch(
    batch,
    indices,
    split_name,
    audio_dir,
    output_format,
    source_id,
):
    """
    Runs inside Dataset.map() worker processes.
    """

    manifest_lines = []
    errors = []

    split_dir = Path(audio_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    extension = "wav" if output_format == "wav" else "flac"

    for audio_obj, text_raw, idx in zip(
        batch["audio"],
        batch[TEXT_COLUMN],
        indices,
    ):
        try:
            text = normalize_text(text_raw)

            if not text:
                manifest_lines.append("")
                errors.append("empty text")
                continue

            audio, sr = decode_audio(audio_obj)
            audio = make_mono(audio)

            # We cast Audio to 16 kHz before map(), so this should
            # already be 16000.
            if sr != SAMPLE_RATE:
                manifest_lines.append("")
                errors.append(f"bad sample rate: {sr}")
                continue

            duration = len(audio) / SAMPLE_RATE

            if duration < MIN_DURATION or duration > MAX_DURATION:
                manifest_lines.append("")
                errors.append(f"duration: {duration:.2f}")
                continue

            # Include the Hugging Face split fingerprint in every name. This
            # prevents a new dataset revision from silently reusing old audio
            # that happened to have the same row index.
            filename = f"{source_id}_{idx:08d}.{extension}"
            output_path = split_dir / filename

            # Resume only when the existing file is complete and matches the
            # source row. New files are installed atomically so a killed worker
            # cannot leave a truncated file that a later run silently accepts.
            if output_path.exists():
                audio_info = sf.info(output_path)
                existing_duration = audio_info.frames / audio_info.samplerate
                if (
                    audio_info.samplerate != SAMPLE_RATE
                    or abs(existing_duration - duration) > 0.1
                ):
                    raise RuntimeError(
                        f"Existing audio does not match the dataset row: {output_path}"
                    )
            else:
                temporary_path = output_path.with_name(output_path.name + ".tmp")
                if output_format == "wav":
                    # FAST: no compression
                    sf.write(
                        temporary_path,
                        audio,
                        SAMPLE_RATE,
                        format="WAV",
                        subtype="PCM_16",
                    )

                else:
                    # Smaller but slower
                    sf.write(
                        temporary_path,
                        audio,
                        SAMPLE_RATE,
                        format="FLAC",
                        subtype="PCM_16",
                    )
                os.replace(temporary_path, output_path)

            entry = {
                "audio_filepath": str(output_path.resolve()),
                "duration": round(duration, 4),

                # Native Uyghur transcript from the `sentence` column
                "text": text,

                # Custom Uyghur prompt allocated by the fine-tuning script
                "language": LANGUAGE,
                "lang": LANGUAGE,
                "target_lang": LANGUAGE,
            }

            manifest_lines.append(
                json.dumps(entry, ensure_ascii=False)
            )

            errors.append("")

        except Exception as e:
            manifest_lines.append("")
            errors.append(str(e)[:200])

    return {
        "manifest_line": manifest_lines,
        "error": errors,
    }


def export_split(
    ds,
    split_name,
    manifest_path,
    audio_dir,
    workers,
    batch_size,
    output_format,
    source_id,
):

    print()
    print("=" * 70)
    print(f"PARALLEL EXPORT: {split_name}")
    print("=" * 70)
    print(f"Samples : {len(ds):,}")
    print(f"Workers : {workers}")
    print(f"Batch   : {batch_size}")
    print(f"Format  : {output_format.upper()}")
    print(f"Text    : {TEXT_COLUMN}")
    print(f"Lang    : {LANGUAGE}")
    print()

    processed = ds.map(
        process_batch,

        batched=True,
        batch_size=batch_size,

        with_indices=True,

        num_proc=workers,

        fn_kwargs={
            "split_name": split_name,
            "audio_dir": str(audio_dir),
            "output_format": output_format,
            "source_id": source_id,
        },

        # Don't copy all original HF columns into temporary dataset
        remove_columns=ds.column_names,

        # Important because this map() has file-writing side effects
        load_from_cache_file=False,

        desc=f"Export {split_name}",
    )

    written = 0
    skipped = 0
    total_seconds = 0.0
    error_counts = Counter()

    with open(manifest_path, "w", encoding="utf-8") as fout:

        for line, error in zip(
            processed["manifest_line"],
            processed["error"],
        ):
            if not line:
                skipped += 1
                error_counts[error or "unknown error"] += 1
                continue

            fout.write(line + "\n")

            obj = json.loads(line)

            total_seconds += obj["duration"]
            written += 1

    print()
    print(f"{split_name} complete")
    print(f"  written : {written:,}")
    print(f"  skipped : {skipped:,}")
    print(f"  hours   : {total_seconds / 3600:.2f}")
    print(f"  manifest: {manifest_path}")
    if error_counts:
        print("  skip reasons:")
        for reason, count in error_counts.most_common(10):
            print(f"    {count:>8,}  {reason}")
    stale_errors = [
        reason for reason in error_counts
        if reason.startswith("Existing audio does not match")
    ]
    if stale_errors:
        raise RuntimeError(
            "Preparation found stale or incomplete audio and stopped instead "
            f"of silently reusing it. First error: {stale_errors[0]}"
        )

    return written, skipped, total_seconds


def read_manifest(path):
    entries = []
    with open(path, encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry["duration"] = float(entry["duration"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid manifest entry in {path} at line {line_number}"
                ) from exc
            entries.append(entry)
    return entries


def write_manifest_atomic(path, entries):
    """Write a complete manifest without leaving a partial file on failure."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as fout:
        for entry in entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def select_long_form_groups(
    entries,
    fraction,
    min_duration,
    target_duration,
    max_duration,
    gap_duration,
    seed,
):
    """Deterministically group short clips into realistic long utterances."""
    if fraction <= 0:
        return []

    candidates = [
        entry for entry in entries
        if 0 < float(entry["duration"]) < min_duration
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    target_source_seconds = sum(
        float(entry["duration"]) for entry in entries
    ) * fraction
    selected_source_seconds = 0.0
    groups = []
    current_group = []
    current_duration = 0.0

    def flush_group():
        nonlocal current_group, current_duration, selected_source_seconds
        if len(current_group) >= 2 and current_duration >= min_duration:
            groups.append(current_group)
            selected_source_seconds += sum(
                float(entry["duration"]) for entry in current_group
            )
        current_group = []
        current_duration = 0.0

    for entry in candidates:
        if selected_source_seconds >= target_source_seconds and not current_group:
            break

        duration = float(entry["duration"])
        added_duration = duration + (gap_duration if current_group else 0.0)
        if current_group and current_duration + added_duration > max_duration:
            flush_group()
            added_duration = duration

        current_group.append(entry)
        current_duration += added_duration

        if current_duration >= target_duration:
            flush_group()

    flush_group()
    return groups


def write_long_form_audio(group, output_path, output_format, gap_duration):
    """Concatenate one manifest group, validating every source audio file."""
    chunks = []
    gap = np.zeros(round(gap_duration * SAMPLE_RATE), dtype=np.float32)

    for position, entry in enumerate(group):
        audio, sample_rate = sf.read(
            entry["audio_filepath"],
            dtype="float32",
            always_2d=False,
        )
        audio = make_mono(audio).astype(np.float32, copy=False)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz, got {sample_rate}: "
                f"{entry['audio_filepath']}"
            )
        chunks.append(audio)
        if position + 1 < len(group) and gap.size:
            chunks.append(gap)

    combined = np.concatenate(chunks)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    format_name = "WAV" if output_format == "wav" else "FLAC"
    sf.write(
        temporary_path,
        combined,
        SAMPLE_RATE,
        format=format_name,
        subtype="PCM_16",
    )
    os.replace(temporary_path, output_path)
    return len(combined) / SAMPLE_RATE


def build_long_form_examples(
    base_manifest,
    long_manifest,
    long_audio_dir,
    split_name,
    output_format,
    fraction,
    min_duration,
    target_duration,
    max_duration,
    gap_duration,
    seed,
):
    """Create synthetic long-form examples from one already-isolated split."""
    entries = read_manifest(base_manifest)
    groups = select_long_form_groups(
        entries=entries,
        fraction=fraction,
        min_duration=min_duration,
        target_duration=target_duration,
        max_duration=max_duration,
        gap_duration=gap_duration,
        seed=seed,
    )
    extension = "wav" if output_format == "wav" else "flac"
    long_entries = []

    print()
    print(f"Building long-form {split_name} examples...")
    print(f"  selected groups: {len(groups):,}")

    for group_number, group in enumerate(groups, 1):
        signature = json.dumps(
            {
                "sources": [
                    {
                        "path": entry["audio_filepath"],
                        "duration": entry["duration"],
                        "text": entry["text"],
                    }
                    for entry in group
                ],
                "gap_duration": gap_duration,
                "sample_rate": SAMPLE_RATE,
                "format": output_format,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        content_id = hashlib.sha256(signature).hexdigest()[:16]
        output_path = Path(long_audio_dir) / split_name / (
            f"long_{content_id}.{extension}"
        )
        expected_duration = sum(float(entry["duration"]) for entry in group)
        expected_duration += gap_duration * (len(group) - 1)

        if output_path.exists():
            audio_info = sf.info(output_path)
            if audio_info.samplerate != SAMPLE_RATE:
                raise RuntimeError(
                    f"Existing long-form audio has sample rate "
                    f"{audio_info.samplerate}, expected {SAMPLE_RATE}: {output_path}"
                )
            actual_duration = audio_info.frames / audio_info.samplerate
            if abs(actual_duration - expected_duration) > 0.1:
                raise RuntimeError(
                    f"Existing long-form audio has the wrong duration: {output_path}. "
                    "Remove that file and rerun preparation."
                )
        else:
            actual_duration = write_long_form_audio(
                group,
                output_path,
                output_format,
                gap_duration,
            )

        long_entries.append(
            {
                "audio_filepath": str(output_path.resolve()),
                "duration": round(actual_duration, 4),
                "text": " ".join(entry["text"].strip() for entry in group),
                "language": LANGUAGE,
                "lang": LANGUAGE,
                "target_lang": LANGUAGE,
                "is_concatenated": True,
                "source_count": len(group),
            }
        )
        if group_number % 1000 == 0:
            print(f"  wrote/reused {group_number:,}/{len(groups):,}")

    write_manifest_atomic(long_manifest, long_entries)
    total_seconds = sum(entry["duration"] for entry in long_entries)
    print(
        f"  long-form output: {len(long_entries):,} samples, "
        f"{total_seconds / 3600:.2f} hours -> {long_manifest}"
    )
    return long_entries


def combine_manifests(base_manifest, extra_entries, output_manifest):
    base_entries = read_manifest(base_manifest)
    combined_entries = base_entries + extra_entries
    write_manifest_atomic(output_manifest, combined_entries)
    return combined_entries


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--format",
        choices=["wav", "flac"],
        default="wav",
        help="wav = fastest, flac = smaller",
    )

    parser.add_argument(
        "--long-form-only",
        action="store_true",
        help=(
            "Reuse existing train_manifest.json/test_manifest.json and only "
            "build/merge long-form examples; skips Hugging Face decoding"
        ),
    )

    parser.add_argument(
        "--long-form-fraction",
        type=float,
        default=DEFAULT_LONG_FORM_FRACTION,
        help=(
            "Fraction of each split's source duration additionally represented "
            "as concatenated long-form examples; 0 disables it "
            f"(default: {DEFAULT_LONG_FORM_FRACTION})"
        ),
    )

    parser.add_argument(
        "--long-form-min-duration",
        type=float,
        default=DEFAULT_LONG_FORM_MIN_DURATION,
        help=(
            "Minimum concatenated duration in seconds "
            f"(default: {DEFAULT_LONG_FORM_MIN_DURATION:g})"
        ),
    )

    parser.add_argument(
        "--long-form-target-duration",
        type=float,
        default=DEFAULT_LONG_FORM_TARGET_DURATION,
        help=(
            "Preferred concatenated duration in seconds "
            f"(default: {DEFAULT_LONG_FORM_TARGET_DURATION:g})"
        ),
    )

    parser.add_argument(
        "--long-form-max-duration",
        type=float,
        default=DEFAULT_LONG_FORM_MAX_DURATION,
        help=(
            "Maximum concatenated duration in seconds "
            f"(default: {DEFAULT_LONG_FORM_MAX_DURATION:g})"
        ),
    )

    parser.add_argument(
        "--long-form-gap",
        type=float,
        default=DEFAULT_LONG_FORM_GAP,
        help=(
            "Silence inserted between source clips in seconds "
            f"(default: {DEFAULT_LONG_FORM_GAP:g})"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Deterministic split/grouping seed (default: {RANDOM_SEED})",
    )

    args = parser.parse_args()

    if not 0.0 <= args.long_form_fraction <= 1.0:
        parser.error("--long-form-fraction must be between 0 and 1")
    if args.long_form_gap < 0:
        parser.error("--long-form-gap cannot be negative")
    if not (
        0 < args.long_form_min_duration
        <= args.long_form_target_duration
        <= args.long_form_max_duration
        <= MAX_DURATION
    ):
        parser.error(
            "Long-form durations must satisfy 0 < min <= target <= max "
            f"<= {MAX_DURATION:g}"
        )

    output_dir = Path("custom_asr_data")
    audio_dir = output_dir / "audio"

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    train_base_manifest = output_dir / "train_manifest_base.json"
    test_base_manifest = output_dir / "test_manifest_base.json"
    train_long_manifest = output_dir / "train_manifest_long.json"
    test_long_manifest = output_dir / "test_manifest_long.json"
    train_manifest = output_dir / "train_manifest.json"
    test_manifest = output_dir / "test_manifest.json"
    long_audio_dir = audio_dir / "long_form"

    print("=" * 70)
    print("Nemotron 3.5 Uyghur FAST Dataset Preparation")
    print("=" * 70)

    print(f"Dataset : {DATASET_ID}")
    print(f"Text    : {TEXT_COLUMN}")
    print(f"Prompt  : {LANGUAGE}")
    print(f"Workers : {args.workers}")
    print(f"Format  : {args.format}")
    print(
        f"Long    : +{args.long_form_fraction:.0%} at "
        f"{args.long_form_min_duration:g}-"
        f"{args.long_form_max_duration:g} seconds"
    )
    print()

    if args.long_form_only:
        for manifest_path in (train_manifest, test_manifest):
            if not manifest_path.is_file():
                parser.error(
                    f"--long-form-only requires an existing manifest: {manifest_path}"
                )

        # Strip examples from a previous augmentation run before rebuilding,
        # so repeated runs never concatenate already-concatenated audio.
        train_base_entries = [
            entry for entry in read_manifest(train_manifest)
            if not entry.get("is_concatenated", False)
        ]
        test_base_entries = [
            entry for entry in read_manifest(test_manifest)
            if not entry.get("is_concatenated", False)
        ]
        write_manifest_atomic(train_base_manifest, train_base_entries)
        write_manifest_atomic(test_base_manifest, test_base_entries)

        train_long_entries = build_long_form_examples(
            train_base_manifest,
            train_long_manifest,
            long_audio_dir,
            "train",
            args.format,
            args.long_form_fraction,
            args.long_form_min_duration,
            args.long_form_target_duration,
            args.long_form_max_duration,
            args.long_form_gap,
            args.seed,
        )
        test_long_entries = build_long_form_examples(
            test_base_manifest,
            test_long_manifest,
            long_audio_dir,
            "test",
            args.format,
            args.long_form_fraction,
            args.long_form_min_duration,
            args.long_form_target_duration,
            args.long_form_max_duration,
            args.long_form_gap,
            args.seed + 1,
        )
        combined_train = combine_manifests(
            train_base_manifest, train_long_entries, train_manifest
        )
        combined_test = combine_manifests(
            test_base_manifest, test_long_entries, test_manifest
        )
        print()
        print("LONG-FORM AUGMENTATION COMPLETE")
        print(
            f"train: {len(combined_train):,} entries "
            f"({len(train_long_entries):,} long-form)"
        )
        print(
            f"test : {len(combined_test):,} entries "
            f"({len(test_long_entries):,} long-form)"
        )
        print(train_manifest.resolve())
        print(test_manifest.resolve())
        return

    print("Loading dataset...")

    dataset = load_dataset(DATASET_ID)

    print()
    print("Available splits:")

    for name, split in dataset.items():
        print(f"  {name}: {len(split):,}")

    print()
    print("Casting audio to 16 kHz...")

    # Resampling happens lazily inside parallel map workers.
    for split_name in dataset.keys():
        dataset[split_name] = dataset[split_name].cast_column(
            "audio",
            Audio(
                sampling_rate=SAMPLE_RATE,
                num_channels=1,
            ),
        )

    if "train" not in dataset:
        raise RuntimeError("Dataset has no train split")

    train_ds = dataset["train"]

    if "validation" in dataset:
        test_ds = dataset["validation"]

    elif "test" in dataset:
        test_ds = dataset["test"]

    else:
        print("No validation/test split. Creating 98/2 split.")

        tmp = train_ds.train_test_split(
            test_size=0.02,
            seed=42,
        )

        train_ds = tmp["train"]
        test_ds = tmp["test"]

    def source_id(split_name, split):
        split_fingerprint = getattr(split, "_fingerprint", "unknown")
        source = f"{DATASET_ID}|{split_name}|{split_fingerprint}"
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]

    train_stats = export_split(
        train_ds,
        "train",
        train_base_manifest,
        audio_dir,
        args.workers,
        args.batch_size,
        args.format,
        source_id("train", train_ds),
    )

    test_stats = export_split(
        test_ds,
        "test",
        test_base_manifest,
        audio_dir,
        args.workers,
        args.batch_size,
        args.format,
        source_id("test", test_ds),
    )

    train_long_entries = build_long_form_examples(
        base_manifest=train_base_manifest,
        long_manifest=train_long_manifest,
        long_audio_dir=long_audio_dir,
        split_name="train",
        output_format=args.format,
        fraction=args.long_form_fraction,
        min_duration=args.long_form_min_duration,
        target_duration=args.long_form_target_duration,
        max_duration=args.long_form_max_duration,
        gap_duration=args.long_form_gap,
        seed=args.seed,
    )
    test_long_entries = build_long_form_examples(
        base_manifest=test_base_manifest,
        long_manifest=test_long_manifest,
        long_audio_dir=long_audio_dir,
        split_name="test",
        output_format=args.format,
        fraction=args.long_form_fraction,
        min_duration=args.long_form_min_duration,
        target_duration=args.long_form_target_duration,
        max_duration=args.long_form_max_duration,
        gap_duration=args.long_form_gap,
        seed=args.seed + 1,
    )
    combined_train = combine_manifests(
        train_base_manifest,
        train_long_entries,
        train_manifest,
    )
    combined_test = combine_manifests(
        test_base_manifest,
        test_long_entries,
        test_manifest,
    )

    print()
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)

    print(
        f"train: {len(combined_train):,} entries "
        f"({len(train_long_entries):,} long-form)"
    )

    print(
        f"test : {len(combined_test):,} entries "
        f"({len(test_long_entries):,} long-form)"
    )

    print()
    print(train_manifest.resolve())
    print(test_manifest.resolve())

    print()
    print("First training example:")

    with open(train_manifest, encoding="utf-8") as f:
        print(f.readline().strip())


if __name__ == "__main__":
    main()
