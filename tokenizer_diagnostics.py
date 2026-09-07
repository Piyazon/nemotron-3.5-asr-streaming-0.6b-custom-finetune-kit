"""Corpus tokenization diagnostics without importing NeMo or loading a model.

The token/word ratio uses whole utterances, because tokenizing each word on its
own can change SentencePiece segmentation. These statistics describe text
encoding, not recognition accuracy or an optimal vocabulary size.
"""

from collections import Counter
from collections.abc import Iterable
import math
from numbers import Integral
import re
import unicodedata


_BYTE_PIECE = re.compile(r"<0x[0-9A-Fa-f]{2}>\Z")


def _normalized_text(text: str) -> str:
    """Ignore only canonical Unicode representation and whitespace changes."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _integer_attribute(obj, name: str) -> int | None:
    try:
        value = getattr(obj, name, None)
        if callable(value):
            value = value()
    except NotImplementedError:
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return None


def _vocabulary_size(tokenizer) -> int | None:
    # NeMo's wrapper size includes any wrapper-level special tokens.
    size = _integer_attribute(tokenizer, "vocab_size")
    if size is not None and size > 0:
        return size
    for name in ("vocab", "get_vocab"):
        vocab = getattr(tokenizer, name, None)
        if callable(vocab):
            vocab = vocab()
        if isinstance(vocab, (dict, list, tuple)) and vocab:
            return len(vocab)
    return None


def _piece_lookup(tokenizer):
    lookup = getattr(tokenizer, "ids_to_tokens", None)
    if callable(lookup):
        return lookup
    lookup = getattr(tokenizer, "id_to_token", None)
    if callable(lookup):
        return lambda ids: [lookup(token_id) for token_id in ids]
    return None


def _distribution(histogram: Counter) -> dict:
    count = sum(histogram.values())
    ordered = sorted(histogram.items())

    def percentile(fraction: float) -> int | None:
        if not count:
            return None
        rank = max(1, math.ceil(fraction * count))
        seen = 0
        for length, occurrences in ordered:
            seen += occurrences
            if seen >= rank:
                return length
        raise AssertionError("Unreachable percentile rank")

    return {
        "min": ordered[0][0] if count else None,
        "max": ordered[-1][0] if count else None,
        "mean": sum(length * occurrences for length, occurrences in ordered) / count if count else None,
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "percentile_method": "nearest_rank",
        "histogram": {str(length): occurrences for length, occurrences in ordered},
    }


def diagnose_tokenizer(tokenizer, texts: Iterable[str], *, max_examples: int = 5) -> dict:
    """Measure a NeMo-style tokenizer on complete reference transcripts.

    Rates are fractions, with emitted tokens as the byte/unknown denominator.
    Words are whitespace-delimited, so punctuation and hyphens are retained.
    Byte fallback identifies SentencePiece pieces spelled ``<0xXX>``; it counts
    emitted byte tokens, not Unicode characters or words. Missing capabilities
    produce ``None`` plus a limitation, never a misleading zero error rate.

    Input text is not rewritten before encoding. NFC and collapsed whitespace
    apply only when comparing input with decoded output. Examples are indexed
    from zero in input order. Empty input is supported, but non-string entries
    are rejected. Exceptions from an implemented tokenizer method propagate so
    that a broken measurement cannot appear to be a successful audit.
    """
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 0:
        raise ValueError("max_examples must be a non-negative integer")
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be an iterable of transcript strings, not one string")
    encode = getattr(tokenizer, "text_to_ids", None)
    if not callable(encode):
        raise TypeError("Tokenizer must implement text_to_ids(text)")

    limitations = []
    size = _vocabulary_size(tokenizer)
    if size is None:
        limitations.append("Actual vocabulary size is unavailable from this tokenizer.")

    unk_id = _integer_attribute(tokenizer, "unk_id")
    if unk_id is None or unk_id < 0:
        unk_id = None
        limitations.append("Unknown-token counting is unavailable: tokenizer has no valid unk_id.")

    lookup = _piece_lookup(tokenizer)
    if lookup is None:
        limitations.append("Byte fallback counting is unavailable: tokenizer has no token-piece lookup.")
    decode = getattr(tokenizer, "ids_to_text", None)
    if not callable(decode):
        decode = None
        limitations.append("Roundtrip checking is unavailable: tokenizer has no ids_to_text method.")

    utterances = words = total_tokens = byte_tokens = unknown_tokens = 0
    checked = mismatches = 0
    examples = []
    lengths = Counter()
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"Transcript at index {index} must be a string")
        token_ids = list(encode(text))
        utterances += 1
        words += len(text.split())
        total_tokens += len(token_ids)
        lengths[len(token_ids)] += 1
        if unk_id is not None:
            unknown_tokens += sum(token_id == unk_id for token_id in token_ids)
        if lookup is not None:
            try:
                pieces = list(lookup(token_ids))
            except NotImplementedError:
                lookup = None
                limitations.append("Byte fallback counting is unavailable: token-piece lookup is not implemented.")
            else:
                if len(pieces) != len(token_ids) or any(not isinstance(piece, str) for piece in pieces):
                    raise ValueError("Token-piece lookup must return one string per emitted token ID")
                byte_tokens += sum(_BYTE_PIECE.fullmatch(piece) is not None for piece in pieces)
        if decode is not None:
            try:
                decoded = decode(token_ids)
            except NotImplementedError:
                decode = None
                limitations.append("Roundtrip checking is unavailable: ids_to_text is not implemented.")
            else:
                if not isinstance(decoded, str):
                    raise TypeError("ids_to_text must return a string")
                checked += 1
                if _normalized_text(text) != _normalized_text(decoded):
                    mismatches += 1
                    if len(examples) < max_examples:
                        examples.append({"index": index, "text": text, "decoded": decoded})

    return {
        "utterances": utterances,
        "words": words,
        "total_tokens": total_tokens,
        "tokens_per_word": total_tokens / words if words else None,
        "actual_vocab_size": size,
        "byte_fallback_tokens": byte_tokens if lookup is not None else None,
        "byte_fallback_rate": byte_tokens / total_tokens if lookup is not None and total_tokens else None,
        "unknown_tokens": unknown_tokens if unk_id is not None else None,
        "unknown_rate": unknown_tokens / total_tokens if unk_id is not None and total_tokens else None,
        "roundtrip_checked_utterances": checked,
        "roundtrip_mismatches": mismatches if decode is not None else None,
        "roundtrip_mismatch_rate": mismatches / checked if decode is not None and checked else None,
        "roundtrip_examples": examples,
        "tokens_per_utterance": _distribution(lengths),
        "limitations": limitations,
    }
