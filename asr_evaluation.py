"""Strict and punctuation-insensitive ASR diagnostics from JiWER alignments."""

import unicodedata


def normalize_for_scoring(text: str) -> str:
    """Normalize Unicode and whitespace, preserving spelling, case and punctuation."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_without_punctuation(text: str) -> str:
    """Replace Unicode punctuation with spaces without changing spelling or case."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(
        "".join(" " if unicodedata.category(char).startswith("P") else char for char in text).split()
    )


def _score_normalized_text(reference: str, hypothesis: str) -> dict:
    from jiwer import process_characters, process_words

    # A punctuation-only reference has no words after secondary normalization.
    # Keep its insertions for corpus totals, with undefined per-sample rates.
    # Handle explicitly for JiWER 3.x, which rejects an empty reference.
    if not reference:
        return {
            "wer": None,
            "cer": None,
            "reference_words": 0,
            "reference_characters": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hypothesis.split()),
            "character_errors": len(hypothesis),
            "deletion_rate": None,
            "deleted_spans": [],
        }
    words = process_words(reference, hypothesis)
    characters = process_characters(reference, hypothesis)
    reference_words = len(words.references[0])
    deleted_spans = [
        " ".join(words.references[0][chunk.ref_start_idx:chunk.ref_end_idx])
        for chunk in words.alignments[0]
        if chunk.type == "delete"
    ]
    return {
        "wer": words.wer,
        "cer": characters.cer,
        "reference_words": reference_words,
        "reference_characters": len(characters.references[0]),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "character_errors": characters.substitutions + characters.deletions + characters.insertions,
        "deletion_rate": words.deletions / reference_words,
        "deleted_spans": deleted_spans,
    }


def score_transcription(reference: str, hypothesis: str) -> dict:
    """Keep strict scores and add secondary scores that ignore punctuation only."""
    strict_reference = normalize_for_scoring(reference)
    strict_hypothesis = normalize_for_scoring(hypothesis)
    if not strict_reference:
        raise ValueError("Evaluation reference must contain transcript text")
    result = _score_normalized_text(strict_reference, strict_hypothesis)
    result["punctuation_insensitive"] = _score_normalized_text(
        normalize_without_punctuation(reference), normalize_without_punctuation(hypothesis),
    )
    return result


def _aggregate_counts(records: list[dict]) -> dict:
    fields = (
        "reference_words", "reference_characters", "substitutions",
        "deletions", "insertions", "character_errors",
    )
    totals = {field: sum(record[field] for record in records) for field in fields}
    word_count = totals["reference_words"]
    character_count = totals["reference_characters"]
    return {
        "samples": len(records),
        **totals,
        "wer": (totals["substitutions"] + totals["deletions"] + totals["insertions"]) / word_count if word_count else None,
        "cer": totals["character_errors"] / character_count if character_count else None,
        "deletion_rate": totals["deletions"] / word_count if word_count else None,
    }


def aggregate_scores(records: list[dict]) -> dict:
    """Compute corpus rates from total edit counts, not mean utterance WER."""
    result = _aggregate_counts(records)
    result["punctuation_insensitive"] = _aggregate_counts([
        record["punctuation_insensitive"]
        for record in records
        if isinstance(record.get("punctuation_insensitive"), dict)
    ])
    return result


def summarize_scores(records: list[dict]) -> dict:
    summary = aggregate_scores(records)
    summary["by_duration"] = {
        label: aggregate_scores([r for r in records if lower <= r["duration"] < upper])
        for label, lower, upper in (
            ("under_10s", 0, 10), ("10_to_20s", 10, 20),
            ("20_to_40s", 20, 40), ("40s_and_over", 40, float("inf")),
        )
    }
    if any("source_dataset" in record for record in records):
        sources = {}
        for record in records:
            source = str(record.get("source_dataset") or "unknown").strip() or "unknown"
            sources.setdefault(source, []).append(record)
        summary["by_source"] = {
            source: aggregate_scores(source_records)
            for source, source_records in sorted(sources.items())
        }
    return summary
