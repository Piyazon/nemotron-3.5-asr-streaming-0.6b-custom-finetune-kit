"""CPU integration: a resumed AdamW/NeMo Noam run must match uninterrupted training.

Requires the full training dependencies; no GPU, real audio, or network needed.
"""

import copy
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf

from exten_tokinizer_nvidia_style.recipe import build_recipe
from exten_tokinizer_nvidia_style.resume import checkpoint_fit_kwargs, resume_callback
from exten_tokinizer_nvidia_style.test_recipe import base_config
from exten_tokinizer_nvidia_style.test_resume import checkpoint_fixture


class ResumeIntegrationTests(unittest.TestCase):
    def test_resume_preserves_weights_adamw_moments_step_counter_and_noam_warmup(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from lightning.pytorch import Callback, LightningModule, Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        from nemo.core.optim.lr_scheduler import NoamAnnealing

        cfg, _ = build_recipe(base_config(), Path("/assets"), Path("/run"), "ug-CN")
        cfg.trainer.max_steps = 2
        cfg.model.optim.sched.warmup_steps = 2  # Exercise continuation past warmup in a tiny run.
        metadata = {"language": "ug-CN", "fingerprint": "fixture", "merged_vocab_size": 125,
                    "base_tokenizer_sha256": "base", "file_sha256": {"merged_tokenizer/tokenizer.model": "merged"}}
        saved_model_cfg = checkpoint_fixture(cfg, metadata)["hyper_parameters"]["cfg"]

        class TinyModel(LightningModule):
            def __init__(self, training_cfg):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)
                self.save_hyperparameters({"cfg": copy.deepcopy(saved_model_cfg)})
                self.training_cfg = training_cfg
                self.learning_rates = []

            def training_step(self, batch, batch_idx):
                self.learning_rates.append(self.optimizers().param_groups[0]["lr"])
                return self.layer(batch[0]).square().mean()

            def validation_step(self, batch, batch_idx):
                self.log("val_wer", 0.5, logger=False)

            def configure_optimizers(self):
                optim = self.training_cfg.model.optim
                optimizer = torch.optim.AdamW(self.parameters(), lr=optim.lr,
                                              betas=tuple(optim.betas), weight_decay=optim.weight_decay)
                scheduler = NoamAnnealing(optimizer, d_model=optim.sched.d_model,
                                         warmup_steps=optim.sched.warmup_steps, min_lr=optim.sched.min_lr,
                                         max_steps=self.training_cfg.trainer.max_steps)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        class CaptureRestoredState(Callback):
            def on_train_start(self, trainer, model):
                self.step = trainer.global_step
                self.weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                self.optimizer = copy.deepcopy(trainer.optimizers[0].state_dict())
                self.scheduler = copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict())

        def make_trainer(max_steps, callbacks=()):
            return Trainer(accelerator="cpu", devices=1, max_steps=max_steps, max_epochs=-1,
                           limit_train_batches=8, accumulate_grad_batches=4, logger=False,
                           enable_checkpointing=any(isinstance(cb, ModelCheckpoint) for cb in callbacks),
                           enable_progress_bar=False, enable_model_summary=False,
                           num_sanity_val_steps=0, callbacks=list(callbacks))

        loader = DataLoader(TensorDataset(torch.ones(8, 2)), batch_size=1)
        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(123)
            first_model = TinyModel(cfg)
            checkpoint_callback = ModelCheckpoint(dirpath=temp, save_top_k=1, save_last=True,
                                                  monitor="val_wer", save_on_train_epoch_end=False)
            first = make_trainer(2, [checkpoint_callback])
            first.fit(first_model, loader, loader)
            checkpoint_path = (Path(temp) / "last.ckpt").resolve()
            self.assertEqual(Path(checkpoint_callback.last_model_path), checkpoint_path)
            before = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            longer = OmegaConf.create(OmegaConf.to_container(cfg))
            longer.trainer.max_steps = 4
            capture = CaptureRestoredState()
            continued_checkpoint = ModelCheckpoint(dirpath=str(Path(temp) / "continued"), save_top_k=1, save_last=True,
                                                    monitor="val_wer", save_on_train_epoch_end=False)
            resumed = make_trainer(4, [resume_callback(longer, metadata, 3), capture, continued_checkpoint])
            torch.manual_seed(999)  # Verify that constructor weights really get replaced.
            resumed_model = TinyModel(longer)
            resumed.fit(resumed_model, loader, loader, **checkpoint_fit_kwargs(resumed, checkpoint_path))
            self.assertEqual(capture.step, 2)
            self.assertEqual(capture.scheduler["last_epoch"], 2)
            self.assertEqual(capture.scheduler["warmup_steps"], 2)
            self.assertEqual(capture.scheduler["max_steps"], 4)
            for name, tensor in before["state_dict"].items():
                torch.testing.assert_close(capture.weights[name], tensor, rtol=0, atol=0)
            for param_id, state in before["optimizer_states"][0]["state"].items():
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(capture.optimizer["state"][param_id][key], state[key], rtol=0, atol=0)
            self.assertEqual(resumed.global_step, 4)

            torch.manual_seed(123)
            continuous_model = TinyModel(longer)
            continuous = make_trainer(4)
            continuous.fit(continuous_model, loader, loader)
            self.assertEqual(first_model.learning_rates + resumed_model.learning_rates, continuous_model.learning_rates)
            for name, tensor in continuous_model.state_dict().items():
                torch.testing.assert_close(resumed_model.state_dict()[name], tensor, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
