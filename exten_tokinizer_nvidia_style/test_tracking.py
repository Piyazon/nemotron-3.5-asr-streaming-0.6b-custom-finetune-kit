"""W&B lifecycle and upload-scope checks; no network or GPU required."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from exten_tokinizer_nvidia_style.test_recipe import base_config
from exten_tokinizer_nvidia_style.train import main, parser
from exten_tokinizer_nvidia_style.tracking import preparation_summary, track_training


def sample_metadata():
    return {
        "language": "ug-CN", "fingerprint": "fixture", "base_vocab_size": 100,
        "new_vocab_size": 48, "added_tokens": 25, "merged_vocab_size": 125,
        "train_samples": 10, "validation_samples": 2,
        "source_train_manifest": "/data/private.jsonl",
        "coverage": {
            split: {"tokens": 100, "unknown_tokens": 0, "unknown_rate": 0.0,
                    "unknown_examples": [{"text": "PRIVATE TRANSCRIPT"}]}
            for split in ("train", "validation")
        },
    }


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = OmegaConf.create({
            "wandb": {"enabled": True, "project": "test-project", "entity": None,
                      "name": "uyghur", "offline": True},
            "trainer": {"max_steps": 2000}, "exp_manager": {"version": "test"},
        })
        self.tensorboard = object()
        self.trainer = SimpleNamespace(
            loggers=[self.tensorboard], is_global_zero=True, interrupted=False,
            global_step=0, current_epoch=0, checkpoint_callback=None,
        )
        self.run = MagicMock(id="unique-run", project="test-project", offline=True,
                             dir=str(self.root / "wandb_logs/wandb/offline-run/files"))
        self.run.name = "uyghur"
        self.run.summary = {}
        self.logger = MagicMock(experiment=self.run)
        self.logger_class = MagicMock(return_value=self.logger)
        self.wandb = SimpleNamespace(run=None, Settings=MagicMock(), Artifact=MagicMock())
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {
            "wandb": self.wandb,
            "lightning.pytorch.loggers": SimpleNamespace(WandbLogger=self.logger_class),
        }).start()
        patch("sys.stdout", new=io.StringIO()).start()

    def test_disabled_tracking_keeps_tensorboard_and_does_not_initialize(self):
        self.cfg.wandb.enabled = False
        with track_training(self.trainer, self.cfg, {}, 3, self.root) as run:
            self.assertIsNone(run)
        self.logger_class.assert_not_called()
        self.assertEqual(self.trainer.loggers, [self.tensorboard])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_nonzero_rank_does_not_start_duplicate_run(self):
        self.trainer.is_global_zero = False
        with track_training(self.trainer, self.cfg, {}, 3, self.root) as run:
            self.assertIsNone(run)
        self.logger_class.assert_not_called()

    def test_metrics_config_small_artifact_and_completion(self):
        with track_training(self.trainer, self.cfg, sample_metadata(), 3, self.root) as run:
            self.assertIs(run, self.run)
            self.assertEqual(self.trainer.loggers, [self.tensorboard, self.logger])
            self.trainer.global_step = 2000
            self.trainer.current_epoch = 39
            self.trainer.checkpoint_callback = SimpleNamespace(
                monitor="val_wer", best_model_score=0.27, best_model_path="/local/checkpoints/best.ckpt")
        self.run.define_metric.assert_called_once_with("val_wer", summary="min")
        logged_config = self.logger.log_hyperparams.call_args.args[0]
        self.assertEqual(logged_config["recipe"]["trainer"]["max_steps"], 2000)
        self.assertEqual(logged_config["preparation"]["prompt_index"], 3)
        self.assertNotIn("PRIVATE TRANSCRIPT", json.dumps(logged_config))
        artifact = self.wandb.Artifact.return_value
        files = [Path(call.args[0]) for call in artifact.add_file.call_args_list]
        self.assertEqual({file.name for file in files}, {"training_recipe.yaml", "preparation_summary.json"})
        self.assertTrue(all(file.is_file() for file in files))
        self.run.log_artifact.assert_called_once_with(artifact)
        self.assertEqual(self.run.summary["best_val_wer"], 0.27)
        self.assertEqual(self.run.summary["optimizer_steps_completed"], 2000)
        self.assertEqual(self.run.summary["training_status"], "completed")
        self.run.finish.assert_called_once_with(exit_code=0)
        run_info = json.loads((self.root / "wandb_run.json").read_text())
        self.assertTrue(run_info["offline"])
        self.assertIsNone(run_info["url"])

    def test_fresh_ids_do_not_use_fixed_nemo_version_or_upload_weights(self):
        for _ in range(2):
            with track_training(self.trainer, self.cfg, sample_metadata(), 3, self.root):
                pass
        first, second = [call.kwargs for call in self.logger_class.call_args_list]
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["id"], "test")
        self.assertNotIn("version", first)
        self.assertFalse(first["log_model"])
        self.assertFalse(first["save_code"])
        self.assertTrue(first["offline"])
        self.wandb.Settings.assert_called_with(console="off", disable_code=True)

    def test_exception_flushes_run_and_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with track_training(self.trainer, self.cfg, sample_metadata(), 3, self.root):
                raise RuntimeError("training failed")
        self.assertEqual(self.run.summary["training_status"], "failed")
        self.run.finish.assert_called_once_with(exit_code=1)

    def test_setup_failure_also_flushes_run(self):
        self.run.log_artifact.side_effect = RuntimeError("artifact failed")
        with self.assertRaisesRegex(RuntimeError, "artifact failed"):
            with track_training(self.trainer, self.cfg, sample_metadata(), 3, self.root):
                self.fail("Training should not start after setup failed")
        self.run.finish.assert_called_once_with(exit_code=1)

    def test_trainer_interruption_is_not_reported_as_completed(self):
        with track_training(self.trainer, self.cfg, sample_metadata(), 3, self.root):
            self.trainer.interrupted = True
        self.assertEqual(self.run.summary["training_status"], "interrupted")
        self.run.finish.assert_called_once_with(exit_code=1)

    def test_keyboard_interrupt_during_model_setup_flushes_run(self):
        with self.assertRaises(KeyboardInterrupt):
            with track_training(self.trainer, self.cfg, sample_metadata(), 3, self.root):
                raise KeyboardInterrupt
        self.assertEqual(self.run.summary["training_status"], "interrupted")
        self.run.finish.assert_called_once_with(exit_code=1)

    def test_active_run_is_not_reused(self):
        self.wandb.run = self.run
        with self.assertRaisesRegex(RuntimeError, "already active"):
            with track_training(self.trainer, self.cfg, sample_metadata(), 3, self.root):
                pass
        self.logger_class.assert_not_called()
        self.run.finish.assert_not_called()


class TrackingCliTests(unittest.TestCase):
    def test_defaults_and_options_require_enable_flag_before_preparation(self):
        self.assertFalse(parser().parse_args([]).wandb)
        for flag in (["--wandb-name", "demo"], ["--wandb-project", "demo"],
                     ["--wandb-entity", "demo"], ["--wandb-offline"]):
            with self.subTest(flag=flag), patch("exten_tokinizer_nvidia_style.train.prepare_assets") as prepare:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(flag)
                prepare.assert_not_called()

    def test_prepare_only_saves_tracking_settings_without_starting_wandb(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            assets.mkdir()
            with patch("exten_tokinizer_nvidia_style.train.prepare_assets",
                       return_value=(assets, base_config(), sample_metadata())), \
                 patch("exten_tokinizer_nvidia_style.train.track_training") as tracking, \
                 redirect_stdout(io.StringIO()):
                main(["--prepare-only", "--output-dir", str(root), "--wandb",
                      "--wandb-project", "uyghur-asr", "--wandb-name", "unigram-2048", "--wandb-offline"])
                tracking.assert_not_called()
            saved = OmegaConf.load(assets / "training_prompt3.yaml")
            self.assertTrue(saved.wandb.enabled)
            self.assertEqual(saved.wandb.project, "uyghur-asr")
            self.assertEqual(saved.wandb.name, "unigram-2048")
            self.assertTrue(saved.wandb.offline)
            self.assertFalse(saved.exp_manager.create_wandb_logger)
            self.assertEqual(saved.model.optim.lr, 2.0)
            self.assertEqual(saved.trainer.accumulate_grad_batches, 4)

    def test_preparation_summary_excludes_transcripts_and_preserves_unknown_rate(self):
        metadata = sample_metadata()
        metadata["coverage"]["validation"]["unknown_tokens"] = 3
        metadata["coverage"]["validation"]["unknown_rate"] = 0.03
        summary = preparation_summary(metadata, 3)
        self.assertEqual(summary["coverage"]["validation"]["unknown_rate"], 0.03)
        self.assertNotIn("unknown_examples", summary["coverage"]["validation"])
        self.assertNotIn("source_train_manifest", summary)


if __name__ == "__main__":
    unittest.main()
