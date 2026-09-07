"""Tokenization measurement regression checks requiring no NeMo installation."""

import json
import unittest

from tokenizer_diagnostics import diagnose_tokenizer


class TableTokenizer:
    vocab_size = 512
    unk_id = 0

    def __init__(self, table, pieces=None):
        self.table = table
        self.pieces = pieces or {}
        self.encoded = []

    def text_to_ids(self, text):
        self.encoded.append(text)
        return self.table[text][0]

    def ids_to_text(self, ids):
        return next(decoded for token_ids, decoded in self.table.values() if token_ids == ids)

    def ids_to_tokens(self, ids):
        return [self.pieces.get(token_id, f"piece{token_id}") for token_id in ids]


class TokenizerDiagnosticsTests(unittest.TestCase):
    def test_corpus_ratio_encodes_whole_utterances_once(self):
        tokenizer = TableTokenizer({
            "مەن باردىم": ([1, 2, 3, 4], "مەن باردىم"),
            "ياخشى": ([5], "ياخشى"),
        })
        result = diagnose_tokenizer(tokenizer, iter(tokenizer.table))
        self.assertEqual(tokenizer.encoded, list(tokenizer.table))
        self.assertEqual((result["utterances"], result["words"], result["total_tokens"]), (2, 3, 5))
        self.assertAlmostEqual(result["tokens_per_word"], 5 / 3)
        self.assertEqual(result["actual_vocab_size"], 512)
        self.assertEqual(result["tokens_per_utterance"]["histogram"], {"1": 1, "4": 1})
        self.assertEqual(result["tokens_per_utterance"]["p50"], 1)
        self.assertEqual(result["tokens_per_utterance"]["p95"], 4)
        self.assertEqual(result["tokens_per_utterance"]["mean"], 2.5)
        self.assertEqual(result["limitations"], [])

    def test_byte_and_unknown_rates_count_emitted_tokens_not_characters(self):
        tokenizer = TableTokenizer({"ئ ؟": ([1, 2, 3, 0], "ئ ؟")}, {
            1: "<0xD8>", 2: "<0xA6>", 3: "▁؟", 0: "<unk>",
        })
        result = diagnose_tokenizer(tokenizer, tokenizer.table)
        self.assertEqual(result["byte_fallback_tokens"], 2)
        self.assertEqual(result["byte_fallback_rate"], 0.5)
        self.assertEqual(result["unknown_tokens"], 1)
        self.assertEqual(result["unknown_rate"], 0.25)

    def test_roundtrip_ignores_only_nfc_and_whitespace(self):
        tokenizer = TableTokenizer({
            "cafe\u0301  مەن\nباردىم": ([1], "café مەن باردىم"),
            "ئاتا-ئانىسى": ([2], "ئاتا ئانىسى"),
            "ياخشى؟": ([3], "ياخشى?"),
            "ئانا": ([4], "انا"),
        })
        result = diagnose_tokenizer(tokenizer, tokenizer.table, max_examples=2)
        self.assertEqual(result["roundtrip_checked_utterances"], 4)
        self.assertEqual(result["roundtrip_mismatches"], 3)
        self.assertEqual(result["roundtrip_mismatch_rate"], 0.75)
        self.assertEqual([example["index"] for example in result["roundtrip_examples"]], [1, 2])

    def test_alternative_wrapper_methods_and_callable_ids(self):
        class Tokenizer:
            def text_to_ids(self, text):
                return [2, 0]

            def ids_to_text(self, ids):
                return "ئ"

            def id_to_token(self, token_id):
                return {0: "<unk>", 2: "<0xd8>"}[token_id]

            def unk_id(self):
                return 0

            def get_vocab(self):
                return {"<unk>": 0, "ئ": 1, "<0xd8>": 2}

        result = diagnose_tokenizer(Tokenizer(), ["ئ"])
        self.assertEqual(result["actual_vocab_size"], 3)
        self.assertEqual(result["byte_fallback_tokens"], 1)
        self.assertEqual(result["unknown_tokens"], 1)
        self.assertEqual(result["limitations"], [])

    def test_unavailable_measurements_are_not_reported_as_zero_errors(self):
        class EncodeOnly:
            def text_to_ids(self, text):
                return [1, 2]

        result = diagnose_tokenizer(EncodeOnly(), ["ياخشى"])
        for key in ("actual_vocab_size", "byte_fallback_tokens", "byte_fallback_rate", "unknown_tokens",
                    "unknown_rate", "roundtrip_mismatches", "roundtrip_mismatch_rate"):
            self.assertIsNone(result[key], key)
        self.assertEqual(len(result["limitations"]), 4)
        self.assertEqual(result["total_tokens"], 2)

    def test_abstract_optional_methods_are_reported_as_unavailable(self):
        class Tokenizer(TableTokenizer):
            def ids_to_tokens(self, ids):
                raise NotImplementedError

            def ids_to_text(self, ids):
                raise NotImplementedError

        result = diagnose_tokenizer(Tokenizer({"one": ([1], "one")}), ["one"])
        self.assertIsNone(result["byte_fallback_tokens"])
        self.assertIsNone(result["roundtrip_mismatches"])
        self.assertEqual(result["roundtrip_checked_utterances"], 0)
        self.assertEqual(len(result["limitations"]), 2)

    def test_empty_corpus_and_empty_transcripts_are_finite_json(self):
        tokenizer = TableTokenizer({"": ([], "")})
        for texts in ([], [""]):
            result = diagnose_tokenizer(tokenizer, texts)
            self.assertIsNone(result["tokens_per_word"])
            self.assertIsNone(result["byte_fallback_rate"])
            self.assertIsNone(result["unknown_rate"])
            self.assertEqual(result["total_tokens"], 0)
            json.dumps(result, allow_nan=False)
        self.assertIsNone(diagnose_tokenizer(tokenizer, [])["tokens_per_utterance"]["p95"])

    def test_invalid_inputs_and_broken_lookup_are_not_silently_accepted(self):
        tokenizer = TableTokenizer({"one": ([1], "one")})
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                diagnose_tokenizer(tokenizer, ["one"], max_examples=value)
        with self.assertRaises(TypeError):
            diagnose_tokenizer(tokenizer, "one")
        with self.assertRaises(TypeError):
            diagnose_tokenizer(tokenizer, [None])
        tokenizer.ids_to_tokens = lambda ids: []
        with self.assertRaisesRegex(ValueError, "one string per emitted token"):
            diagnose_tokenizer(tokenizer, ["one"])


if __name__ == "__main__":
    unittest.main()
