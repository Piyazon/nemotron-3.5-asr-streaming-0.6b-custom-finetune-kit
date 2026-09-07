"""Exercise Adam/Noam checkpoint continuity across NeMo module reordering."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from asr_finetune_with_speechhints import optimizer_parameter_groups


def initialize_layers(model, torch, *, replaced):
    model.encoder = torch.nn.Linear(4, 4)
    model.decoder = torch.nn.Linear(4, 3)
    model.joint = torch.nn.Linear(3, 2)
    model.prompt_kernel = torch.nn.Linear(4, 4)
    if replaced:
        # Match change_vocabulary(): deleting/reassigning modules changes
        # module registration order despite preserving the parameter names.
        del model.joint
        model.joint = torch.nn.Linear(3, 2)
        del model.decoder
        model.decoder = torch.nn.Linear(4, 3)
    model.double()


def forward_layers(model, inputs):
    features = model.encoder(inputs) + model.prompt_kernel(inputs)
    return model.joint(model.decoder(features))


def create_optimizer(model, torch):
    return torch.optim.AdamW(
        optimizer_parameter_groups(model, lr=0.1, encoder_lr_scale=0.3),
        lr=0.1,
        betas=(0.9, 0.98),
        weight_decay=0.001,
    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires torch")
class OptimizerResumeIntegrationTests(unittest.TestCase):
    def assert_matching_moments(self, first, first_optimizer, second, second_optimizer):
        import torch

        first_parameters = dict(first.named_parameters())
        second_parameters = dict(second.named_parameters())
        self.assertEqual(first_parameters.keys(), second_parameters.keys())
        for name, first_parameter in first_parameters.items():
            second_parameter = second_parameters[name]
            self.assertTrue(torch.equal(first_parameter, second_parameter), name)
            first_state = first_optimizer.state[first_parameter]
            second_state = second_optimizer.state[second_parameter]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(first_state[key], second_state[key]), f"{name}: {key}")

    def test_changed_module_order_retains_names_moments_and_next_adam_step(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self, *, replaced):
                super().__init__()
                initialize_layers(self, torch, replaced=replaced)

            def forward(self, inputs):
                return forward_layers(self, inputs)

        torch.manual_seed(42)
        trained = TinyModel(replaced=True)
        restored = TinyModel(replaced=False)
        self.assertNotEqual(list(trained._modules), list(restored._modules))
        trained_optimizer = create_optimizer(trained, torch)
        restored_optimizer = create_optimizer(restored, torch)
        trained_names = [group["param_names"] for group in trained_optimizer.param_groups]
        self.assertEqual(trained_names, [group["param_names"] for group in restored_optimizer.param_groups])
        self.assertEqual(trained_names, [sorted(names) for names in trained_names])

        inputs = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 16

        def step(model, optimizer):
            optimizer.zero_grad()
            model(inputs).square().mean().backward()
            optimizer.step()

        step(trained, trained_optimizer)
        step(trained, trained_optimizer)
        restored.load_state_dict(copy.deepcopy(trained.state_dict()))
        restored_optimizer.load_state_dict(copy.deepcopy(trained_optimizer.state_dict()))
        self.assert_matching_moments(trained, trained_optimizer, restored, restored_optimizer)

        step(trained, trained_optimizer)
        step(restored, restored_optimizer)
        self.assert_matching_moments(trained, trained_optimizer, restored, restored_optimizer)

    def test_lightning_restores_adam_noam_and_training_progress(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if not importlib.util.find_spec("lightning") or not importlib.util.find_spec("nemo"):
            self.skipTest("Requires Lightning and NeMo for their actual checkpoint/scheduler implementation")
        from lightning.pytorch import LightningModule, Trainer
        from nemo.core.optim.lr_scheduler import NoamAnnealing

        class TinyLightningModel(LightningModule):
            def __init__(self, *, replaced):
                super().__init__()
                initialize_layers(self, torch, replaced=replaced)

            def training_step(self, batch, batch_idx):
                return forward_layers(self, batch[0]).square().mean()

            def configure_optimizers(self):
                optimizer = create_optimizer(self, torch)
                scheduler = NoamAnnealing(optimizer, d_model=16, warmup_steps=3)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
                }

        def trainer(epochs, directory):
            return Trainer(
                accelerator="cpu", devices=1, max_epochs=epochs,
                default_root_dir=directory,
                logger=False, enable_checkpointing=False,
                enable_model_summary=False, enable_progress_bar=False,
                num_sanity_val_steps=0,
            )

        inputs = torch.arange(32, dtype=torch.float64).reshape(8, 4) / 32
        loader = DataLoader(TensorDataset(inputs), batch_size=4, shuffle=False)
        torch.manual_seed(42)
        uninterrupted = TinyLightningModel(replaced=True)
        initial_state = copy.deepcopy(uninterrupted.state_dict())
        first_stage = TinyLightningModel(replaced=True)
        first_stage.load_state_dict(initial_state)

        with tempfile.TemporaryDirectory() as directory:
            full_trainer = trainer(2, directory)
            full_trainer.fit(uninterrupted, train_dataloaders=loader)
            first_trainer = trainer(1, directory)
            first_trainer.fit(first_stage, train_dataloaders=loader)
            checkpoint = str(Path(directory) / "resume.ckpt")
            first_trainer.save_checkpoint(checkpoint)

            resumed = TinyLightningModel(replaced=False)
            resumed_trainer = trainer(2, directory)
            resumed_trainer.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint)

            self.assertEqual(full_trainer.global_step, 4)
            self.assertEqual(resumed_trainer.global_step, full_trainer.global_step)
            self.assertEqual(resumed_trainer.current_epoch, full_trainer.current_epoch)
            first_scheduler = full_trainer.lr_scheduler_configs[0].scheduler
            second_scheduler = resumed_trainer.lr_scheduler_configs[0].scheduler
            self.assertEqual(first_scheduler.state_dict(), second_scheduler.state_dict())
            self.assertEqual(first_scheduler.get_last_lr(), second_scheduler.get_last_lr())
            self.assertAlmostEqual(second_scheduler.get_last_lr()[0] / second_scheduler.get_last_lr()[1], 0.3)
            self.assert_matching_moments(
                uninterrupted, full_trainer.optimizers[0], resumed, resumed_trainer.optimizers[0],
            )


if __name__ == "__main__":
    unittest.main()
