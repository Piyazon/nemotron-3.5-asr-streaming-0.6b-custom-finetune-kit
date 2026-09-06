"""Deletion diagnostics using word/character edit alignments from JiWER."""

import unicodedata


def normalize_for_scoring(text: str) -> str:
    """Normalize Unicode and whitespace, preserving spelling, case and punctuation."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def score_transcription(reference: str, hypothesis: str) -> dict:
    from jiwer import process_characters, process_words

    reference = normalize_for_scoring(reference)
    hypothesis = normalize_for_scoring(hypothesis)
    if not reference:
        raise ValueError("Evaluation reference must contain transcript text")
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


def aggregate_scores(records: list[dict]) -> dict:
    """Compute corpus rates from total edit counts, not mean utterance WER."""
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


def summarize_scores(records: list[dict]) -> dict:
    summary = aggregate_scores(records)
    summary["by_duration"] = {
        label: aggregate_scores([r for r in records if lower <= r["duration"] < upper])
        for label, lower, upper in (
            ("under_10s", 0, 10), ("10_to_20s", 10, 20),
            ("20_to_40s", 20, 40), ("40s_and_over", 40, float("inf")),
        )
    }
    return summary
