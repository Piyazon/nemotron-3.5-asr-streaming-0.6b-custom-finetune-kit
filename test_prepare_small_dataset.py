import struct
import tempfile
from pathlib import Path
import unittest

from umsc import UgMultiScriptConverter

from prepare_small_dataset import (
    DatasetWriter, Unit, arabic_text, cut_pcm, group_units, overlap_refs, read_units,
    text_issues, timestamp_ms, verify_outputs,
)


class PreparationTests(unittest.TestCase):
    def test_timestamp_hours_and_milliseconds(self):
        self.assertEqual(timestamp_ms("01:02:03.004"), 3723004)
        for invalid in ("00:60:01.000", "00:00:61.000", "00:00:01.2", "-1"):
            with self.assertRaises(ValueError):
                timestamp_ms(invalid)

    def test_alignment_uses_eaf_and_skips_header_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            xml, eaf = Path(directory) / "source.xml", Path(directory) / "source.eaf"
            xml.write_text('''<c1root><header><annotation>Annotator</annotation></header>
              <transcript_body><c1_final><annotation who="Speaker A" ref="1">
              <timestamp>00:00:01.002-00:00:02.004</timestamp><iu> Yaxshi </iu>
              <seg>yaxshi</seg></annotation></c1_final></transcript_body></c1root>''')
            template = '''<ANNOTATION_DOCUMENT><HEADER TIME_UNITS="milliseconds"/>
              <TIME_ORDER><TIME_SLOT TIME_SLOT_ID="s" TIME_VALUE="{start}"/>
              <TIME_SLOT TIME_SLOT_ID="e" TIME_VALUE="2005"/></TIME_ORDER>
              <TIER PARTICIPANT="A" LINGUISTIC_TYPE_REF="Intonation Units">
              <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
              TIME_SLOT_REF1="s" TIME_SLOT_REF2="e"><ANNOTATION_VALUE>{text}</ANNOTATION_VALUE>
              </ALIGNABLE_ANNOTATION></ANNOTATION></TIER></ANNOTATION_DOCUMENT>'''
            eaf.write_text(template.format(start=1003, text="Yaxshi"))
            units, _ = read_units(xml, eaf)
            self.assertEqual(len(units), 1)
            self.assertEqual((units[0].start_ms, units[0].end_ms), (1003, 2005))
            for start, text in ((1004, "Yaxshi"), (1003, "Yaman")):
                eaf.write_text(template.format(start=start, text=text))
                with self.assertRaisesRegex(ValueError, "XML/EAF mismatch"):
                    read_units(xml, eaf)

    def test_overlap_is_positive_and_across_speakers(self):
        def unit(ref, speaker, start, end):
            return Unit(ref, speaker, "", "", start, end, ref, start, end)
        units = [unit("1", "A", 0, 1000), unit("2", "A", 500, 600),
                 unit("3", "B", 1000, 2000)]
        self.assertEqual(overlap_refs(units), set())
        units.append(unit("4", "B", 550, 560))
        self.assertEqual(overlap_refs(units), {"1", "2", "4"})

    def test_cut_selects_exact_samples_and_rejects_invalid_bounds(self):
        pcm = struct.pack("<1600h", *range(1600))
        self.assertEqual(cut_pcm(pcm, 20, 40), struct.pack("<320h", *range(320, 640)))
        for start, end in ((-1, 10), (20, 20), (30, 10), (0, 101)):
            with self.assertRaises(ValueError):
                cut_pcm(pcm, start, end)

    def test_conversion_hamza_digraphs_and_punctuation(self):
        converter = UgMultiScriptConverter("ULS", "UAS")
        for original, expected in {
            "Yaxshimusiz!": "ياخشىمۇسىز!", "uyghur": "ئۇيغۇر",
            "bügün": "بۈگۈن", "He’e": "ھەئە", "da'irisi": "دائىرىسى",
            "in'glizche": "ئىنگلىزچە", "sh ch gh zh": "ش چ غ ژ",
            "é e w": "ئې ئە ۋ", "  yaxshi... he? ": "ياخشى ھە؟",
        }.items():
            with self.subTest(original=original):
                self.assertEqual(arabic_text(original, converter), expected)

    def test_ambiguous_words_and_foreign_tags_are_flagged(self):
        self.assertIn("redacted_or_unclear_words", text_issues("[balining ismi] yaxshi"))
        self.assertIn("incomplete_speech", text_issues("ya-- yaxshi"))
        self.assertIn("incomplete_speech", text_issues("ya- yaxshi"))
        self.assertIn("foreign_words_need_arabic_transcription", text_issues("library", "EN:library"))
        self.assertIn("letters_outside_uyghur_latin_alphabet", text_issues("cuīhuàjì"))
        self.assertIn("laughter_annotation", text_issues("Haha, yaxshi"))
        self.assertIn("nonlexical_vocalization", text_issues("Hmm."))
        for text in ("Hm-mm.", "Mm-hmm?", "Hm-mm. Hm-mm.", "Mm, yaxshi."):
            self.assertIn("nonlexical_vocalization", text_issues(text))
        self.assertEqual(text_issues("ata-ana, he'e"), [])

    def test_compound_hyphens_are_kept_and_uncertain_spellings_flagged(self):
        converter = UgMultiScriptConverter("ULS", "UAS")
        self.assertEqual(arabic_text("méwe-chéwe", converter), "مېۋە-چېۋە")
        for text in ("méwe-chéwe", "ushshaq-chüshshek", "bir-ikki", "ata-ana"):
            self.assertEqual(text_issues(text), [])
        for text in ("Essalamu-aleykum", "we-eleykum-essalam", "online-da",
                     "zoom-da?", "rohiy-keypiyati", "seksen - yetmish"):
            self.assertIn("hyphen_spelling_needs_review", text_issues(text))

    @staticmethod
    def units(intervals):
        return [Unit(str(i), speaker, "yaxshi", "", start, end, str(i), start, end)
                for i, (speaker, start, end) in enumerate(intervals, start=1)]

    def grouped_refs(self, units, blocked=()):
        return [[u.ref for u in group] for group in group_units(
            units, {u.ref: ["review"] if u.ref in blocked else [] for u in units})]

    def test_merge_retains_short_phrases_and_the_pause_audio(self):
        units = self.units([("A", 0, 600), ("A", 850, 1400)])
        self.assertEqual(self.grouped_refs(units), [["1", "2"]])
        pcm = b"\x01\x00" * 9600 + b"\x02\x00" * 4000 + b"\x03\x00" * 8800
        self.assertEqual(cut_pcm(pcm, units[0].start_ms, units[-1].end_ms), pcm)

    def test_merge_stops_at_review_turns_long_gaps_and_overlaps(self):
        units = self.units([("A", 0, 500), ("A", 600, 800), ("A", 900, 1500)])
        self.assertEqual(self.grouped_refs(units, blocked={"2"}), [["1"], ["2"], ["3"]])
        units[1].speaker = "B"
        self.assertEqual(self.grouped_refs(units), [["1"], ["2"], ["3"]])
        for intervals in ([('A', 0, 500), ('A', 1001, 1500)],
                          [('A', 0, 500), ('A', 400, 900)]):
            self.assertEqual(self.grouped_refs(self.units(intervals)), [["1"], ["2"]])

    def test_merge_target_maximum_and_short_tail(self):
        units = self.units([("A", 0, 3000), ("A", 3100, 5500), ("A", 5600, 7600)])
        self.assertEqual(self.grouped_refs(units), [["1", "2"], ["3"]])
        units[-1].end_ms = 6000
        self.assertEqual(self.grouped_refs(units), [["1", "2", "3"]])
        units = self.units([("A", 0, 11900), ("A", 12000, 12400)])
        self.assertEqual(self.grouped_refs(units), [["1"], ["2"]])

    def test_outputs_separate_review_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = DatasetWriter(root / "stage", root / "final")
            writer.add("valid", b"\x01\x00" * 16000, "ياخشى", {})
            writer.add("unknown", b"\x01\x00" * 8000, None, {}, ["missing_transcript"])
            writer.add("short", b"\x01\x00" * 8000, "ياخشى", {})
            writer.add("foreign", b"\x01\x00" * 8000, "hello", {})
            writer.add("silent", b"\x00\x00" * 8000, "ياخشى", {})
            self.assertEqual(len(writer.accepted), 1)
            self.assertEqual(len(writer.review), 4)
            self.assertEqual(writer.accepted[0]["duration"], 1.0)
            self.assertIn("below_1_second_after_merging", writer.review[1]["review_reasons"])
            self.assertIsNone(writer.review[2]["text"])
            verify_outputs(writer)
            path = writer.staging / writer.accepted[0]["file_name"]
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "PCM mismatch"):
                verify_outputs(writer)


if __name__ == "__main__":
    unittest.main()
