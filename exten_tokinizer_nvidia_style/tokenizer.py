"""CPU-only preparation: copy manifests, train Unigram, and extend SentencePiece.

Adapted from NVIDIA Riva's asr-extend-tokenizer-to-newlang-ft-acoustic-model
notebook. Existing piece IDs and special-piece types are preserved; duplicate
normal-piece scores use the maximum, as in the notebook.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tarfile
import tempfile

from omegaconf import OmegaConf
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def language_tag(language: str) -> str:
    if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language):
        raise ValueError("Use a language locale such as ug-CN")
    return f"<{language}>"


def tag_transcript(text: str, language: str) -> str:
    """Match NVIDIA's literal-period tagging, without duplicating existing tags."""
    tag = language_tag(language)
    foreign_tags = set(re.findall(r"<[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})+>", text)) - {tag}
    if foreign_tags:
        raise ValueError(f"Transcript has other language tags: {sorted(foreign_tags)}")
    return " ".join(re.sub(r"\.(?!\s*" + re.escape(tag) + r")", ". " + tag, text).split())


def read_manifest(path: Path, language: str) -> list[dict]:
    """Validate a single-language manifest and resolve relative audio paths."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            for key in ("language", "lang", "target_lang"):
                if row.get(key) and row[key] != language:
                    raise ValueError(f"{path}:{line_number}: {key}={row[key]!r}, expected {language!r}")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_number}: missing or empty text")
            duration = row.get("duration")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{path}:{line_number}: duration must be positive and finite")
            if not isinstance(row.get("audio_filepath"), str) or not row["audio_filepath"]:
                raise ValueError(f"{path}:{line_number}: missing audio_filepath")
            audio = Path(row["audio_filepath"]).expanduser()
            audio = (path.parent / audio).resolve() if not audio.is_absolute() else audio.resolve()
            if not audio.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: audio not found: {audio}")
            row.update(audio_filepath=str(audio), text=tag_transcript(row["text"], language),
                       language=language, lang=language, target_lang=language)
            # Prevent a pre-existing per-row mode from overriding fixed language prompting.
            row["prompt_mode"] = "langID"
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def read_base_archive(path: Path) -> tuple[object, bytes]:
    """Read only config/tokenizer bytes; never unpickle weights or extract paths."""
    with tarfile.open(path, "r:*") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        configs = [m for m in files if Path(m.name).name == "model_config.yaml"]
        if len(configs) != 1:
            raise ValueError("Expected one model_config.yaml in the .nemo archive")
        cfg = OmegaConf.create(archive.extractfile(configs[0]).read().decode("utf-8"))
        if cfg.get("tokenizer", {}).get("type") != "bpe":
            raise ValueError("This recipe requires a single SentencePiece tokenizer (NeMo type bpe)")
        configured = str(cfg.tokenizer.get("model_path", "")).removeprefix("nemo:")
        matches = [m for m in files if Path(m.name).name == Path(configured).name] if configured else []
        if not matches:
            matches = [m for m in files if Path(m.name).name.endswith("tokenizer.model")]
        if len(matches) != 1:
            raise ValueError("Cannot unambiguously locate the archive's SentencePiece model")
        return cfg, archive.extractfile(matches[0]).read()


def parse_model(data: bytes) -> pb.ModelProto:
    proto = pb.ModelProto()
    proto.ParseFromString(data)
    if proto.trainer_spec.model_type != pb.TrainerSpec.UNIGRAM:
        raise ValueError("NVIDIA-style Unigram merging requires a Unigram base tokenizer")
    return proto


def write_tokenizer(proto: pb.ModelProto, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer.model").write_bytes(proto.SerializeToString())
    # SentencePiece IDs, not alphabetical order, are authoritative. NeMo's BPE
    # wrapper reads the .model; vocab.txt is retained as an ordered audit artifact.
    (directory / "vocab.txt").write_text("".join(p.piece + "\n" for p in proto.pieces), encoding="utf-8")
    (directory / "tokenizer.vocab").write_text(
        "".join(f"{p.piece}\t{p.score}\n" for p in proto.pieces), encoding="utf-8")


def train_unigram(corpus: Path, directory: Path, vocab_size: int, language: str) -> None:
    if vocab_size < 2:
        raise ValueError("New-language vocabulary size must be at least 2 (and large enough for its alphabet)")
    directory.mkdir(parents=True, exist_ok=True)
    # These match process_asr_text_tokenizer.py / create_spt_model defaults,
    # including case-folding in the *new* tokenizer and no byte fallback.
    # The merged tokenizer retains the BASE model's normalizer, as NVIDIA does.
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(directory / "tokenizer"),
        model_type="unigram", vocab_size=vocab_size, character_coverage=1.0,
        user_defined_symbols=[language_tag(language)],
        bos_id=-1, eos_id=-1, pad_id=-1, hard_vocab_limit=False,
        normalization_rule_name="nmt_nfkc_cf", remove_extra_whitespaces=False,
        byte_fallback=False, shuffle_input_sentence=True,
    )
    write_tokenizer(parse_model((directory / "tokenizer.model").read_bytes()), directory)


def merge_tokenizers(base_data: bytes, new_data: bytes, directory: Path, language: str) -> dict:
    base, new = parse_model(base_data), parse_model(new_data)
    merged = pb.ModelProto()
    merged.CopyFrom(base)
    indices = {p.piece: i for i, p in enumerate(merged.pieces)}
    if len(indices) != len(merged.pieces):
        raise ValueError("Base tokenizer contains duplicate pieces")
    old_pieces = [p.piece for p in base.pieces]
    for piece in new.pieces:
        if piece.piece in indices:
            existing = merged.pieces[indices[piece.piece]]
            if existing.type == piece.type == pb.ModelProto.SentencePiece.NORMAL:
                existing.score = max(existing.score, piece.score)
        else:
            if piece.type not in (pb.ModelProto.SentencePiece.NORMAL, pb.ModelProto.SentencePiece.USER_DEFINED):
                raise ValueError(f"Cannot append a new control/unknown/byte piece: {piece.piece!r}")
            indices[piece.piece] = len(merged.pieces)
            merged.pieces.add().CopyFrom(piece)  # Preserve USER_DEFINED, not just text/score.
    tag = language_tag(language)
    if tag not in indices or merged.pieces[indices[tag]].type != pb.ModelProto.SentencePiece.USER_DEFINED:
        raise ValueError(f"{tag} must be a USER_DEFINED token in the merged model")
    merged.trainer_spec.vocab_size = len(merged.pieces)
    if tag not in merged.trainer_spec.user_defined_symbols:
        merged.trainer_spec.user_defined_symbols.append(tag)
    write_tokenizer(merged, directory)
    processor = spm.SentencePieceProcessor(model_file=str(directory / "tokenizer.model"))
    if [processor.id_to_piece(i) for i in range(len(old_pieces))] != old_pieces:
        raise ValueError("Merge changed existing token IDs")
    tag_id = processor.piece_to_id(tag)
    if processor.encode(tag).count(tag_id) != 1 or tag not in processor.decode(processor.encode(tag)):
        raise ValueError("Language tag failed the merged-tokenizer round trip")
    return {"base_vocab_size": len(old_pieces), "new_vocab_size": len(new.pieces),
            "added_tokens": len(merged.pieces) - len(old_pieces), "merged_vocab_size": len(merged.pieces),
            "language_tag": tag, "language_tag_id": tag_id,
            "merged_normalizer": merged.normalizer_spec.name,
            "byte_fallback": merged.trainer_spec.byte_fallback}


def coverage(model_path: Path, rows: list[dict]) -> dict:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    tokens = unknown = 0
    examples = []
    for row in rows:
        ids = processor.encode(row["text"])
        count = ids.count(processor.unk_id())
        tokens += len(ids)
        unknown += count
        if count and len(examples) < 5:
            examples.append(row["text"])
    return {"tokens": tokens, "unknown_tokens": unknown,
            "unknown_rate": unknown / tokens if tokens else 0.0, "unknown_examples": examples}


def prepare_assets(base_model: Path, train_manifest: Path, validation_manifest: Path,
                   output_dir: Path, language: str = "ug-CN", vocab_size: int = 2048) -> tuple[Path, object, dict]:
    """Build immutable, content-addressed assets; held-out text never trains SPM."""
    language_tag(language)
    train_rows = read_manifest(train_manifest, language)
    validation_rows = read_manifest(validation_manifest, language)
    overlap = {r["audio_filepath"] for r in train_rows} & {r["audio_filepath"] for r in validation_rows}
    if overlap:
        raise ValueError(f"Train and validation share {len(overlap)} audio path(s); use disjoint splits")
    base_cfg, base_data = read_base_archive(base_model)
    parse_model(base_data)
    signature = {"format_version": 1, "sentencepiece_version": spm.__version__,
                 "base_tokenizer_sha256": hashlib.sha256(base_data).hexdigest(),
                 "base_config": OmegaConf.to_container(base_cfg, resolve=True),
                 "language": language, "vocab_size": vocab_size,
                 "train": train_rows, "validation": validation_rows}
    fingerprint = hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
    assets_root = output_dir / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    target = assets_root / fingerprint
    if target.exists():
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        for name, expected in metadata["file_sha256"].items():
            if sha256(target / name) != expected:
                raise ValueError(f"Prepared asset was modified: {target / name}")
        return target, base_cfg, metadata
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=assets_root) as temp:
        stage = Path(temp)
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            (stage / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        corpus = stage / "train_text.txt"
        corpus.write_text("".join(row["text"] + "\n" for row in train_rows), encoding="utf-8")
        write_tokenizer(parse_model(base_data), stage / "base_tokenizer")
        train_unigram(corpus, stage / "new_tokenizer", vocab_size, language)
        stats = merge_tokenizers(base_data, (stage / "new_tokenizer/tokenizer.model").read_bytes(),
                                 stage / "merged_tokenizer", language)
        diagnostics = {split: coverage(stage / "merged_tokenizer/tokenizer.model", rows)
                       for split, rows in (("train", train_rows), ("validation", validation_rows))}
        if diagnostics["train"]["unknown_tokens"]:
            raise ValueError(f"Merged tokenizer still has training unknowns: {diagnostics['train']}")
        metadata = {"fingerprint": fingerprint, "language": language, "requested_vocab_size": vocab_size,
                    "base_model": str(base_model), "base_tokenizer_sha256": signature["base_tokenizer_sha256"],
                    "source_train_manifest": str(train_manifest), "source_validation_manifest": str(validation_manifest),
                    "train_samples": len(train_rows), "validation_samples": len(validation_rows),
                    "sentencepiece_version": spm.__version__, **stats, "coverage": diagnostics}
        metadata["file_sha256"] = {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(target)
    return target, base_cfg, metadata
