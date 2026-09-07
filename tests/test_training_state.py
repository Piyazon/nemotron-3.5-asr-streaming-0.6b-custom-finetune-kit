"""Checkpoint identity and command guards; no model download or GPU required."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import sentencepiece as spm

from asr_finetune_with_speechhints import main, run_training
from training_state import (
    construct_resume_model, file_sha256, noam_scale_for_peak, persist_tokenizer,
    prepare_resume_config, resume_settings, validate_optimizer_group_names,
    validate_resume_manifests,
)


class TrainingStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        writer = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(["مەن ئۇيغۇر تىلى ياخشى بۈگۈن سىز مەكتەپ"] * 5),
            model_writer=writer, model_type="bpe", vocab_size=40,
            hard_vocab_limit=False, bos_id=-1, eos_id=-1, minloglevel=2)
        cls.model_bytes = writer.getvalue()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.processor = spm.SentencePieceProcessor(model_proto=self.model_bytes)
        directory = persist_tokenizer(SimpleNamespace(tokenizer=self.processor), self.root / "tokens")
        pieces = [self.processor.id_to_piece(i) for i in range(self.processor.get_piece_size())]
        self.cfg = OmegaConf.create({
            "tokenizer": {"type": "bpe", "dir": "/old/path", "model_path": "nemo:tokenizer.model"},
            "joint": {"vocabulary": pieces}, "decoder": {"vocab_size": len(pieces)},
            "optim": {"lr": 0.1, "sched": {"name": "NoamAnnealing", "d_model": 1024, "warmup_steps": 100}},
            "custom_finetune": {"language": "ug-CN", "encoder_lr_scale": 0.3,
                "max_duration": 70, "batch_duration": 1920, "seed": 42,
                "tokenizer_dir": directory, "tokenizer_sha256": file_sha256(Path(directory) / "tokenizer.model")},
            "train_ds": {"sample_rate": 16000}, "validation_ds": {"sample_rate": 16000},
            "preprocessor": {"sample_rate": "${train_ds.sample_rate}"},
        })
        self.checkpoint = {"hyper_parameters": {"cfg": self.cfg},
            "optimizer_states": [{"param_groups": [{"param_names": ["encoder.weight"]}, {"param_names": ["decoder.weight"]}]}],
            "lr_schedulers": [{"base_lrs": [0.03, 0.1], "_normalize": 1024 ** -0.5, "warmup_steps": 100}],
            "epoch": 43, "global_step": 61992}

    def test_resume_resolves_exact_tokenizer_without_changing_original_config(self):
        cfg, directory = prepare_resume_config(self.checkpoint)
        self.assertEqual(Path(directory).name, "tokens")
        self.assertEqual(Path(cfg.tokenizer.model_path).read_bytes(), self.model_bytes)
        self.assertEqual(self.cfg.tokenizer.model_path, "nemo:tokenizer.model")
        self.assertEqual(list(cfg.joint.vocabulary), list(self.cfg.joint.vocabulary))

    def test_relocated_tokenizer_is_verified_by_hash_and_vocabulary_order(self):
        moved = self.root / "moved"
        (self.root / "tokens").rename(moved)
        prepare_resume_config(self.checkpoint, moved)
        self.cfg.custom_finetune.tokenizer_sha256 = "wrong"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare_resume_config(self.checkpoint, moved)
        self.cfg.custom_finetune.tokenizer_sha256 = None
        self.cfg.joint.vocabulary = list(reversed(self.cfg.joint.vocabulary))
        with self.assertRaisesRegex(ValueError, "vocabulary/order"):
            prepare_resume_config(self.checkpoint, moved)

    def test_weights_only_and_legacy_optimizer_checkpoints_require_new_stage(self):
        saved = copy.deepcopy(self.checkpoint)
        saved.pop("optimizer_states")
        with self.assertRaisesRegex(ValueError, "optimizer and scheduler"):
            prepare_resume_config(saved)
        saved = copy.deepcopy(self.checkpoint)
        saved["optimizer_states"][0]["param_groups"] = [{"params": [0]}, {"params": [1]}]
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            prepare_resume_config(saved)

    def test_constructing_model_defers_loaders_and_resolves_interpolations(self):
        def model_class(cfg):
            self.assertIsNone(cfg.train_ds)
            self.assertIsNone(cfg.validation_ds)
            self.assertEqual(cfg.preprocessor.sample_rate, 16000)
            return SimpleNamespace(cfg=copy.deepcopy(cfg))
        model = construct_resume_model(model_class, self.cfg)
        self.assertEqual(model.cfg.train_ds.sample_rate, 16000)
        self.assertEqual(self.cfg.train_ds.sample_rate, 16000)

    def test_saved_rates_and_sampler_settings_are_used(self):
        settings = resume_settings(self.cfg, self.checkpoint)
        self.assertEqual(settings["encoder_lr_scale"], 0.3)
        self.assertEqual(settings["batch_duration"], 1920)
        self.assertEqual(settings["lr"], 0.1)
        self.assertEqual(settings["noam_d_model"], 1024)
        self.checkpoint["lr_schedulers"][0]["base_lrs"] = [0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "parameter groups"):
            resume_settings(self.cfg, self.checkpoint)

    def test_optimizer_mapping_rejects_same_size_wrong_parameter_order(self):
        correct = [{"param_names": ["encoder.a", "encoder.b"]}, {"param_names": ["decoder.weight"]}]
        validate_optimizer_group_names(correct, copy.deepcopy(correct))
        wrong = copy.deepcopy(correct)
        wrong[0]["param_names"].reverse()
        with self.assertRaisesRegex(ValueError, "ordering"):
            validate_optimizer_group_names(correct, wrong)

    def test_manifest_changes_require_weights_only_new_stage(self):
        train, valid = self.root / "train", self.root / "valid"
        train.write_text("one")
        valid.write_text("two")
        self.cfg.custom_finetune.train_manifest_sha256 = file_sha256(train)
        self.cfg.custom_finetune.validation_manifest_sha256 = file_sha256(valid)
        validate_resume_manifests(self.cfg, train, valid)
        train.write_text("changed")
        with self.assertRaisesRegex(ValueError, "train manifest changed"):
            validate_resume_manifests(self.cfg, train, valid)

    def test_peak_lr_is_converted_without_changing_requested_peak(self):
        for warmup in (100, 1000):
            scale = noam_scale_for_peak(0.0003125, 1024, warmup)
            self.assertAlmostEqual(scale / (1024 * warmup) ** 0.5, 0.0003125)
        self.assertEqual(noam_scale_for_peak(0.0003125, 1024, 100), 0.1)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                noam_scale_for_peak(value, 1024, 100)


class ContinuationCommandTests(unittest.TestCase):
    def test_cli_rejects_options_that_would_be_ignored(self):
        for flags in (["--resume-from", "last.ckpt", "--lr=0.1"],
                      ["--resume-from", "last.ckpt", "--tokenizer-vocab-size", "4096"],
                      ["--init-from-nemo", "best.nemo"], ["--tokenizer-dir", "tokens"]):
            with self.subTest(flags=flags), patch("sys.argv", ["train", "--train-only", *flags]), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as exc:
                main()
            self.assertEqual(exc.exception.code, 2)

    def test_cli_forwards_weights_only_initialization_and_actual_peak(self):
        with patch("sys.argv", ["train", "--train-only", "--init-from-nemo", "best.nemo", "--peak-lr", "0.00002"]), \
                patch("sys.platform", "linux"), \
                patch("asr_finetune_with_speechhints.os.path.exists", return_value=True), \
                patch("asr_finetune_with_speechhints.resolve_manifest_language", return_value="ug-CN"), \
                patch("asr_finetune_with_speechhints.run_training") as training:
            main()
        self.assertEqual(training.call_args.kwargs["init_from_nemo"], "best.nemo")
        self.assertEqual(training.call_args.kwargs["peak_lr"], 0.00002)

    def test_invalid_stage_options_fail_before_loading_a_model(self):
        for options in ({"resume_from": "a", "init_from_nemo": "b"},
                        {"resume_from": "a", "peak_lr": 0.0001},
                        {"init_from_nemo": "a"}, {"epochs": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_training("unused", "unused", **options)


if __name__ == "__main__":
    unittest.main()
