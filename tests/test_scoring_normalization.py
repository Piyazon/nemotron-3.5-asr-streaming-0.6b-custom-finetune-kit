"""Checks for secondary scoring and source grouping without model inference."""

import copy
import json
import unittest

from jiwer import process_characters, process_words

from asr_evaluation import (
    aggregate_scores, normalize_without_punctuation, score_transcription, summarize_scores,
)


class PunctuationScoringTests(unittest.TestCase):
    def test_strict_scores_still_count_missing_punctuation(self):
        reference = "ياخشى، ياخشى. ۋاقتىڭىز بارمۇ؟"
        hypothesis = "ياخشى ياخشى ۋاقتىڭىز بارمۇ"
        result = score_transcription(reference, hypothesis)
        words = process_words(reference, hypothesis)
        chars = process_characters(reference, hypothesis)
        self.assertEqual(result["wer"], words.wer)
        self.assertEqual(result["cer"], chars.cer)
        self.assertEqual(result["substitutions"], words.substitutions)
        self.assertGreater(result["wer"], 0)
        self.assertEqual(result["punctuation_insensitive"]["wer"], 0)
        self.assertEqual(result["punctuation_insensitive"]["cer"], 0)

    def test_punctuation_inside_words_becomes_spaces(self):
        self.assertEqual(normalize_without_punctuation("ئاتا-ئانىسى،سېزىپ؟"), "ئاتا ئانىسى سېزىپ")
        result = score_transcription("ئاتا-ئانىسى سېزىپ", "ئاتا ئانىسى سېزىپ")
        self.assertEqual(result["punctuation_insensitive"]["wer"], 0)
        self.assertGreater(result["wer"], 0)
        joined = score_transcription("ئاتا-ئانىسى", "ئاتائانىسى")
        self.assertGreater(joined["punctuation_insensitive"]["wer"], 0)

    def test_all_unicode_punctuation_classes_are_separators(self):
        # Opening/closing brackets, quotes, dash, underscore, Arabic marks.
        self.assertEqual(
            normalize_without_punctuation("«a» ‘b’ (c) [d] e—f_g،h؛i؟"),
            "a b c d e f g h i",
        )

    def test_symbols_digits_spelling_and_case_are_preserved(self):
        self.assertEqual(normalize_without_punctuation("A+B $5 ١٢"), "A+B $5 ١٢")
        for reference, hypothesis in (
            ("ئاتا-ئانىسى سېزىپ", "ئاتا-ئانىسىز يېزىپ"),
            ("word", "Word"), ("١٢", "12"),
        ):
            with self.subTest(reference=reference):
                self.assertGreater(score_transcription(reference, hypothesis)["punctuation_insensitive"]["wer"], 0)

    def test_nfc_and_whitespace_normalization_applies_to_both_scores(self):
        result = score_transcription("café  مەن", "cafe\u0301\nمەن")
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["punctuation_insensitive"]["wer"], 0)
        self.assertEqual(result["punctuation_insensitive"]["cer"], 0)

    def test_punctuation_only_reference_has_no_invented_denominator(self):
        result = score_transcription("...", "word")
        secondary = result["punctuation_insensitive"]
        self.assertIsNone(secondary["wer"])
        self.assertIsNone(secondary["cer"])
        self.assertEqual(secondary["reference_words"], 0)
        self.assertEqual(secondary["insertions"], 1)
        self.assertEqual(secondary["character_errors"], 4)
        json.dumps(result, allow_nan=False)
        for empty in ("", " \n"):
            with self.subTest(empty=empty), self.assertRaises(ValueError):
                score_transcription(empty, "word")

    def test_secondary_deleted_spans_do_not_change_strict_diagnostics(self):
        result = score_transcription("one middle, two", "one two")
        self.assertEqual(result["deleted_spans"], ["middle,"])
        self.assertEqual(result["punctuation_insensitive"]["deleted_spans"], ["middle"])

    def test_secondary_corpus_scores_are_weighted_by_normalized_reference_counts(self):
        records = [
            score_transcription("one!", ""),
            score_transcription("two-three four", "two three four"),
        ]
        summary = aggregate_scores(records)
        self.assertEqual(summary["punctuation_insensitive"]["reference_words"], 4)
        self.assertEqual(summary["punctuation_insensitive"]["wer"], 0.25)
        self.assertEqual(summary["punctuation_insensitive"]["samples"], 2)

    def test_secondary_insertions_from_punctuation_only_samples_count_in_corpus(self):
        summary = aggregate_scores([
            score_transcription("?", "extra"), score_transcription("two words", "two words"),
        ])
        self.assertEqual(summary["punctuation_insensitive"]["wer"], 0.5)


class SourceSummaryTests(unittest.TestCase):
    def record(self, reference, hypothesis, duration, **extra):
        return {
            "reference": reference, "hypothesis": hypothesis, "duration": duration,
            **score_transcription(reference, hypothesis), **extra,
        }

    def test_sources_and_duration_buckets_include_both_score_variants(self):
        records = [
            self.record("one,", "one", 4, source_dataset="cv24"),
            self.record("two three four", "three four", 12, source_dataset="cv24"),
            self.record("مەن", "مەن", 30, source_dataset="thuyg20"),
        ]
        original = copy.deepcopy(records)
        summary = summarize_scores(records)
        self.assertEqual(summary["by_source"]["cv24"]["samples"], 2)
        self.assertEqual(summary["by_source"]["cv24"]["wer"], 0.5)
        self.assertEqual(summary["by_source"]["cv24"]["punctuation_insensitive"]["wer"], 0.25)
        self.assertEqual(summary["by_source"]["thuyg20"]["wer"], 0)
        self.assertEqual(summary["by_duration"]["under_10s"]["punctuation_insensitive"]["wer"], 0)
        self.assertEqual(records, original)

    def test_partially_missing_source_labels_are_counted_as_unknown(self):
        records = [
            self.record("one", "one", 2, source_dataset="cv24"),
            self.record("two", "", 3),
            self.record("three", "three", 5, source_dataset=None),
        ]
        summary = summarize_scores(records)
        self.assertEqual(summary["by_source"]["unknown"]["samples"], 2)
        self.assertEqual(sum(group["samples"] for group in summary["by_source"].values()), 3)

    def test_no_sources_and_empty_summaries_remain_json_safe(self):
        summary = summarize_scores([self.record("one", "one", 2)])
        self.assertNotIn("by_source", summary)
        empty = summarize_scores([])
        self.assertIsNone(empty["punctuation_insensitive"]["wer"])
        self.assertEqual(empty["punctuation_insensitive"]["samples"], 0)
        json.dumps(empty, allow_nan=False)

    def test_legacy_strict_records_still_aggregate(self):
        record = self.record("one", "", 2)
        del record["punctuation_insensitive"]
        summary = summarize_scores([record])
        self.assertEqual(summary["wer"], 1)
        self.assertEqual(summary["punctuation_insensitive"]["samples"], 0)
        self.assertIsNone(summary["punctuation_insensitive"]["wer"])


if __name__ == "__main__":
    unittest.main()
