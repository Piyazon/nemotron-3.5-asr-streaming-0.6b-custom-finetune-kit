"""Check Common Voice preparation and Arabic transcript selection without downloads."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


import prepare_hf_uyghur_fast as prepare

SCRIPT = Path(prepare.__file__)


class FakeSplit:
    column_names = ["audio", "sentence", "sentence_latn"]
    _fingerprint = "same-fingerprint"

    def __init__(self, name="train"):
        self.name = name
        self.map = Mock(return_value={"manifest_line": [json.dumps({"duration": 1})], "error": [""]})

    def __len__(self):
        return 1

    def cast_column(self, name, feature):
        return self


class CommonVoicePreparationTests(unittest.TestCase):
    def test_preserves_official_source_splits(self):
        train, test, validation = FakeSplit(), FakeSplit("test"), FakeSplit("validation")
        self.assertEqual(prepare.select_splits({"train": train, "test": test}, "source"),
                         {"train": train, "test": test})
        self.assertEqual(prepare.select_splits({"train": train, "test": test, "validation": validation}, "source"),
                         {"train": train, "validation": validation})

    def test_latin_only_source_is_rejected(self):
        train = FakeSplit()
        train.column_names = ["audio", "sentence_latn"]
        with self.assertRaisesRegex(ValueError, "sentence"):
            prepare.select_splits({"train": train, "test": FakeSplit()}, "source")

    def test_fallback_split_is_seeded_per_source(self):
        train, test = FakeSplit(), FakeSplit("test")
        original = FakeSplit()
        original.train_test_split = Mock(return_value={"train": train, "test": test})
        with patch("builtins.print"):
            selected = prepare.select_splits({"train": original}, "source")
        original.train_test_split.assert_called_once_with(test_size=0.02, seed=42)
        self.assertIs(selected["test"], test)

    def test_exports_from_different_sources_cannot_share_audio_paths(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            source_paths = []
            for index, source in enumerate(("example/source-one", "example/source-two")):
                ds = FakeSplit()
                prepare.export_split(ds, "train", Path(directory) / f"{index}.jsonl",
                                     directory, 1, 1, "flac", source)
                source_paths.append(ds.map.call_args.kwargs["fn_kwargs"]["audio_dir"])
            self.assertNotEqual(source_paths[0], source_paths[1])
            self.assertTrue(all("same-fingerprint" in path for path in source_paths))

    def test_arabic_text_and_uyghur_prompt_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = [0] * 16000
            writer = Mock(side_effect=lambda path, *args, **kwargs: Path(path).write_bytes(b"audio"))
            with patch.dict("sys.modules", {"soundfile": SimpleNamespace(write=writer)}), \
                    patch.object(prepare, "decode_audio", return_value=(audio, 16000)), \
                    patch.object(prepare, "make_mono", side_effect=lambda samples: samples):
                output = prepare.process_batch(
                    {"audio": [object()], "sentence": ["مەن باردىم."], "sentence_latn": ["men bardim"]},
                    [0], "train", directory, "flac", prepare.DATASET_IDS[0],
                )
            entry = json.loads(output["manifest_line"][0])
            self.assertEqual(entry["text"], "مەن باردىم.")
            self.assertEqual([entry[key] for key in ("language", "lang", "target_lang")], ["ug-CN"] * 3)
            self.assertEqual(entry["source_dataset"], prepare.DATASET_IDS[0])
            self.assertTrue(Path(entry["audio_filepath"]).is_file())
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])

    def test_failed_merge_preserves_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "train.json"
            destination.write_text("original\n")
            source = root / "source.jsonl"
            source.write_text("replacement\n")
            with self.assertRaises(FileNotFoundError):
                prepare.merge_manifests([source, root / "missing.jsonl"], destination)
            self.assertEqual(destination.read_text(), "original\n")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_main_replaces_mixed_manifests_with_common_voice_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "custom_asr_data"
            output_dir.mkdir()
            for split in ("train", "test"):
                (output_dir / f"{split}_manifest.json").write_text(
                    json.dumps({"source_dataset": "piyazon/thuyg20-datasets"}) + "\n")
            loader = Mock(side_effect=lambda source: {"train": FakeSplit(), "test": FakeSplit("test")})

            def export(ds, split, manifest, audio_dir, workers, batch_size, fmt, source, max_duration):
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(json.dumps({"source_dataset": source, "source_split": split}) + "\n")
                return 1, 0, 1.0

            with patch.dict("sys.modules", {"datasets": SimpleNamespace(load_dataset=loader, Audio=Mock())}), \
                    patch.object(prepare, "export_split", side_effect=export), \
                    patch.object(prepare, "Path", side_effect=lambda path: Path(directory) / path), \
                    patch("sys.argv", [str(SCRIPT), "--workers", "1"]), patch("builtins.print"):
                prepare.main()
            loader.assert_called_once_with("piyazon/cv-corpus-ug-24-latn")
            for split in ("train", "test"):
                rows = [json.loads(line) for line in (Path(directory) / f"custom_asr_data/{split}_manifest.json").read_text().splitlines()]
                self.assertEqual([row["source_dataset"] for row in rows], ["piyazon/cv-corpus-ug-24-latn"])
                self.assertTrue(all(row["source_split"] == split for row in rows))


if __name__ == "__main__":
    unittest.main()
