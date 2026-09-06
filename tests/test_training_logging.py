"""Logging configuration and lifecycle checks without NeMo or network access."""

import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from asr_finetune_with_speechhints import main, run_training, training_loggers


class TrainingLoggingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.tb = Mock()
        self.wb = Mock()
        self.wb.return_value.experiment.url = None
        self.sdk = SimpleNamespace(Settings=Mock())
        self.modules = {
            "lightning.pytorch.loggers": SimpleNamespace(TensorBoardLogger=self.tb, WandbLogger=self.wb),
            "wandb": self.sdk,
        }
        self.enterContext(patch("asr_finetune_with_speechhints.DATA_DIR", str(self.root)))
        self.enterContext(patch("asr_finetune_with_speechhints.log"))

    def test_default_logging_works_without_wandb_installed(self):
        with patch.dict("sys.modules", {**self.modules, "wandb": None}):
            with training_loggers() as loggers:
                self.assertEqual(loggers, [self.tb.return_value])
        self.wb.assert_not_called()
        self.assertFalse((self.root / "checkpoints/wandb_logs").exists())

    def test_offline_logging_retains_tensorboard_and_finishes_run(self):
        config = {"training": {"language": "ug-CN", "batch_duration": 960}}
        with patch.dict("sys.modules", self.modules):
            with training_loggers(
                wandb_enabled=True, wandb_project="uyghur", wandb_entity="asr-team",
                wandb_offline=True, run_name="trial-1", hyperparameters=config,
            ) as loggers:
                self.assertEqual(loggers, [self.tb.return_value, self.wb.return_value])
                self.wb.return_value.experiment.finish.assert_not_called()
        options = self.wb.call_args.kwargs
        self.assertTrue(options["offline"])
        self.assertFalse(options["log_model"])
        self.assertEqual((options["project"], options["entity"], options["name"]),
                         ("uyghur", "asr-team", "trial-1"))
        self.assertTrue(Path(options["save_dir"]).is_dir())
        self.sdk.Settings.assert_called_once_with(console="off")
        self.wb.return_value.log_hyperparams.assert_called_once_with(config)
        self.wb.return_value.experiment.define_metric.assert_called_once_with("val_wer", summary="min")
        self.wb.return_value.experiment.finish.assert_called_once_with(exit_code=0)

    def test_failure_marks_run_failed_and_preserves_original_error(self):
        with patch.dict("sys.modules", self.modules):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with training_loggers(wandb_enabled=True):
                    raise RuntimeError("training failed")
        self.wb.return_value.experiment.finish.assert_called_once_with(exit_code=1)

    def test_setup_failure_also_closes_initialized_run(self):
        self.wb.return_value.log_hyperparams.side_effect = RuntimeError("config failed")
        with patch.dict("sys.modules", self.modules):
            with self.assertRaisesRegex(RuntimeError, "config failed"):
                with training_loggers(wandb_enabled=True):
                    self.fail("Logger setup should fail before training")
        self.wb.return_value.experiment.finish.assert_called_once_with(exit_code=1)

    def test_requested_missing_dependency_has_install_hint(self):
        with patch.dict("sys.modules", {**self.modules, "wandb": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install"):
                with training_loggers(wandb_enabled=True):
                    self.fail("Missing W&B should fail before training")

    def test_invalid_logging_options_fail_before_loading_model(self):
        for options in (
            {"wandb_offline": True},
            {"wandb_project": "uyghur"},
            {"wandb_enabled": True, "wandb_project": " "},
            {"wandb_enabled": True, "wandb_entity": ""},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_training("unused-train.json", "unused-val.json", **options)

    def test_cli_rejects_logging_flags_that_would_be_ignored(self):
        for arguments in (
            ["--train-only", "--wandb-offline"],
            ["--train-only", "--wandb-project", "uyghur"],
            ["--evaluate", "--wandb"],
            ["--manifest-only", "--wandb"],
        ):
            with self.subTest(arguments=arguments), \
                    patch("sys.argv", ["train", *arguments]), patch("sys.stderr", io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)


@unittest.skipUnless(os.environ.get("RUN_WANDB_OFFLINE_TEST") == "1",
                     "Set RUN_WANDB_OFFLINE_TEST=1 with torch, lightning, wandb, and tensorboard installed")
class OfflineLoggingIntegrationTests(unittest.TestCase):
    def test_training_and_every_validation_epoch_reach_both_loggers(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import wandb
        from lightning.pytorch import LightningModule, Trainer
        from lightning.pytorch.callbacks import LearningRateMonitor
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        class TinyModel(LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)

            def training_step(self, batch, batch_idx):
                loss = self.layer(batch[0]).square().mean()
                self.log("train_loss", loss, on_step=True, on_epoch=False)
                return loss

            def validation_step(self, batch, batch_idx):
                # Synthetic metric with a known value at each epoch.
                self.log("val_wer", 0.5 - 0.1 * self.current_epoch,
                         on_step=False, on_epoch=True, batch_size=len(batch[0]))

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as directory, \
                patch("asr_finetune_with_speechhints.DATA_DIR", directory), \
                patch.dict(os.environ, {
                    "WANDB_MODE": "offline", "WANDB_SILENT": "true",
                    "WANDB_CONFIG_DIR": str(Path(directory) / "config"),
                    "WANDB_CACHE_DIR": str(Path(directory) / "cache"),
                    "WANDB_DATA_DIR": str(Path(directory) / "staging"),
                }):
            loader = DataLoader(TensorDataset(torch.ones(16, 2)), batch_size=4)
            with training_loggers(
                wandb_enabled=True, wandb_offline=True, run_name="offline-smoke",
                hyperparameters={"training": {"language": "ug-CN", "epochs": 2}},
            ) as loggers:
                with patch.object(loggers[1], "log_metrics", wraps=loggers[1].log_metrics) as metrics:
                    trainer = Trainer(
                        accelerator="cpu", devices=1, max_epochs=2,
                        logger=loggers, callbacks=[LearningRateMonitor(logging_interval="step")],
                        log_every_n_steps=2, num_sanity_val_steps=0,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                    )
                    trainer.fit(TinyModel(), train_dataloaders=loader, val_dataloaders=loader)
                logged = [call.kwargs["metrics"] if "metrics" in call.kwargs else call.args[0]
                          for call in metrics.call_args_list]
                validation = [record["val_wer"] for record in logged if "val_wer" in record]
                self.assertEqual(len(validation), 2)
                self.assertAlmostEqual(validation[0], 0.5)
                self.assertAlmostEqual(validation[1], 0.4)
                self.assertTrue(any("train_loss" in record for record in logged))
                self.assertTrue(any("lr-SGD" in record for record in logged))
                self.assertEqual(loggers[1].experiment.config["training"]["language"], "ug-CN")
                self.assertAlmostEqual(loggers[1].experiment.summary["val_wer"]["min"], 0.4)
                tensorboard_dir = loggers[0].log_dir

            self.assertIsNone(wandb.run)
            events = EventAccumulator(tensorboard_dir).Reload()
            self.assertEqual(len(events.Scalars("val_wer")), 2)
            run_files = list(Path(directory).glob("checkpoints/wandb_logs/wandb/offline-run-*/run-*.wandb"))
            self.assertEqual(len(run_files), 1)
            self.assertGreater(run_files[0].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
