"""Regression checks runnable without NeMo, CUDA, or model downloads."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from asr_evaluation import aggregate_scores, score_transcription, summarize_scores
from asr_finetune_with_speechhints import (
    main, manifest_duration_summary, optimizer_parameter_groups, read_manifest_entries,
    run_evaluation,
)
from checkpoint_selection import (
    best_checkpoint_from_state, best_nemo_checkpoint, latest_checkpoint, run_directory,
)


class TranscriptionMetricsTests(unittest.TestCase):
    def test_missing_middle_uyghur_phrase(self):
        result = score_transcription("مەن بۈگۈن مەكتەپكە باردىم", "مەن باردىم")
        self.assertEqual(result["deletions"], 2)
        self.assertEqual(result["deleted_spans"], ["بۈگۈن مەكتەپكە"])
        self.assertEqual(result["deletion_rate"], 0.5)
        self.assertEqual(result["wer"], 0.5)

    def test_empty_output_is_all_deletions(self):
        result = score_transcription("مەن باردىم", "")
        self.assertEqual(result["wer"], 1)
        self.assertEqual(result["cer"], 1)
        self.assertEqual(result["deletions"], 2)

    def test_unicode_and_whitespace_do_not_create_errors(self):
        result = score_transcription("café  مەن", "cafe\u0301\nمەن")
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["cer"], 0)

    def test_substitution_and_insertion_are_not_deletions(self):
        result = score_transcription("one two three", "one four three five")
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["insertions"], 1)
        self.assertEqual(result["deletions"], 0)

    def test_corpus_wer_is_weighted_by_reference_length(self):
        records = [score_transcription("one", ""), score_transcription("two three four", "two three four")]
        result = aggregate_scores(records)
        self.assertEqual(result["wer"], 0.25)
        self.assertEqual(result["deletion_rate"], 0.25)

    def test_duration_buckets_and_empty_buckets_are_json_safe(self):
        records = [{"duration": duration, **score_transcription("one", "")} for duration in (9, 10, 20, 40)]
        result = summarize_scores(records)
        self.assertTrue(all(bucket["samples"] == 1 for bucket in result["by_duration"].values()))
        empty = summarize_scores([])
        self.assertIsNone(empty["wer"])
        json.dumps(empty, allow_nan=False)


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def file(self, relative, timestamp):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.utime(path, ns=(timestamp, timestamp))
        return path

    def test_evaluation_uses_best_from_newest_run_not_final_or_other_run(self):
        self.file("old/nemotron-asr-best1-wer-0.01.nemo", 1)
        expected = self.file("new/nemotron-asr-best1-wer-0.20.nemo", 2)
        self.file("new/nemotron-asr-finetuned.nemo", 3)
        self.assertEqual(best_nemo_checkpoint(self.root), expected)

    def test_missing_best_does_not_silently_evaluate_final(self):
        self.file("nemotron-asr-finetuned.nemo", 1)
        with self.assertRaisesRegex(FileNotFoundError, "--checkpoint"):
            best_nemo_checkpoint(self.root)

    def test_explicit_run_does_not_select_another_run(self):
        expected = self.file("wanted/nemotron-asr-best1-wer-0.2.nemo", 1)
        self.file("newest/nemotron-asr-best1-wer-0.1.nemo", 2)
        self.assertEqual(best_nemo_checkpoint(run_directory(self.root, "wanted")), expected)

    def test_best_weights_come_from_callback_and_support_moved_runs(self):
        best = self.file("copy/nemotron-asr-finetuned-epoch=01.ckpt", 1)
        newest = self.file("copy/nemotron-asr-finetuned-epoch=03.ckpt", 2)
        self.file("copy/last.ckpt", 3)
        state = {"callbacks": {"ModelCheckpoint": {
            "monitor": "val_wer", "best_model_path": f"/old/server/{best.name}",
        }}}
        self.assertEqual(latest_checkpoint(self.root), newest)
        self.assertEqual(best_checkpoint_from_state(state, newest), best)

    def test_missing_recorded_best_requires_explicit_selection(self):
        latest = self.file("nemotron-asr-finetuned-epoch=03.ckpt", 1)
        with self.assertRaisesRegex(ValueError, "--selection latest"):
            best_checkpoint_from_state({}, latest)
        state = {"callbacks": {"checkpoint": {"monitor": "val_wer", "best_model_path": "missing.ckpt"}}}
        with self.assertRaisesRegex(FileNotFoundError, "--checkpoint"):
            best_checkpoint_from_state(state, latest)

    def test_run_name_cannot_escape_checkpoint_root(self):
        for name in ("..", ".", "../other", "/absolute"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_directory(self.root, name)


class TrainingSetupTests(unittest.TestCase):
    def test_encoder_lr_reduction_does_not_affect_joint_or_prompt(self):
        encoder, decoder, prompt = [SimpleNamespace(requires_grad=True) for _ in range(3)]
        frozen = SimpleNamespace(requires_grad=False)
        model = SimpleNamespace(
            encoder=SimpleNamespace(parameters=lambda: iter([encoder])),
            parameters=lambda: iter([encoder, decoder, prompt, frozen]),
        )
        groups = optimizer_parameter_groups(model, 0.1, 0.1)
        self.assertAlmostEqual(groups[0]["lr"], 0.01)
        self.assertEqual(groups[1]["lr"], 0.1)
        self.assertEqual([id(p) for p in groups[0]["params"]], [id(encoder)])
        self.assertEqual([id(p) for p in groups[1]["params"]], [id(decoder), id(prompt)])
        baseline = optimizer_parameter_groups(model, 0.1, 1)
        self.assertEqual([group["lr"] for group in baseline], [0.1, 0.1])

    def test_invalid_lr_scale_rejected(self):
        for value in (0, -1, 2, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                optimizer_parameter_groups(None, 0.1, value)

    def test_invalid_transcripts_and_durations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            for text in (None, 123, " "):
                manifest.write_text(json.dumps({"text": text}), encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_manifest_entries(str(manifest))
            for duration in (-1, 0, float("nan"), float("inf")):
                manifest.write_text(json.dumps({"text": "مەن", "duration": duration}), encoding="utf-8")
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    manifest_duration_summary(str(manifest))


class EvaluationIntegrationTests(unittest.TestCase):
    def test_reports_include_chosen_checkpoint_and_missing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "chosen.nemo"
            checkpoint.touch()
            manifest = root / "references.jsonl"
            manifest.write_text(json.dumps({
                "audio_filepath": "/recordings/problem.wav", "duration": 22.0,
                "text": "مەن بۈگۈن مەكتەپكە باردىم", "language": "ug-CN",
            }), encoding="utf-8")
            model = Mock()
            model.cfg.model_defaults = {"prompt_dictionary": {"ug-CN": 40}}
            model.transcribe.return_value = ["مەن باردىم"]
            model_class = Mock()
            model_class.restore_from.return_value = model
            with patch.dict("sys.modules", {
                "nemo.collections.asr.models": SimpleNamespace(EncDecRNNTBPEModelWithPrompt=model_class),
                "lightning.pytorch": SimpleNamespace(Trainer=Mock()),
            }), patch("asr_finetune_with_speechhints.log"):
                result = run_evaluation(str(manifest), checkpoint=str(checkpoint), report_dir=str(root / "report"))
            saved = json.loads((root / "report/summary.json").read_text())
            recording = json.loads((root / "report/transcriptions.jsonl").read_text())
            self.assertEqual(saved, result)
            self.assertEqual(saved["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(saved["by_duration"]["20_to_40s"]["deletion_rate"], 0.5)
            self.assertEqual(recording["deleted_spans"], ["بۈگۈن مەكتەپكە"])
            self.assertEqual(model.transcribe.call_args.kwargs["target_lang"], "ug-CN")

    def test_full_pipeline_evaluates_the_best_export_it_just_trained(self):
        with patch("sys.argv", ["train", "--language", "ug-CN", "--encoder-lr-scale", "0.1"]), \
                patch("sys.platform", "linux"), \
                patch("asr_finetune_with_speechhints.log"), \
                patch("asr_finetune_with_speechhints.convert_audio"), \
                patch("asr_finetune_with_speechhints.build_manifests", return_value=("train.json", "valid.json")), \
                patch("asr_finetune_with_speechhints.run_training", return_value="this-run-best.nemo") as training, \
                patch("asr_finetune_with_speechhints.run_evaluation") as evaluation:
            main()
        self.assertEqual(training.call_args.kwargs["encoder_lr_scale"], 0.1)
        evaluation.assert_called_once_with(
            "valid.json", language="ug-CN", checkpoint="this-run-best.nemo", report_dir=None,
        )


if __name__ == "__main__":
    unittest.main()
