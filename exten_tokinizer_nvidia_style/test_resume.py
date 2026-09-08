"""Resume configuration and checkpoint compatibility checks without CUDA."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from exten_tokinizer_nvidia_style.recipe import build_recipe
from exten_tokinizer_nvidia_style.resume import checkpoint_fit_kwargs, load_resume_recipe, validate_resume_checkpoint
from exten_tokinizer_nvidia_style.test_recipe import base_config
from exten_tokinizer_nvidia_style.train import main, parser


def checkpoint_fixture(cfg, metadata, step=2000):
    return {
        "global_step": step, "state_dict": {"weight": [1]}, "optimizer_states": [{"state": {}}],
        "lr_schedulers": [{"warmup_steps": cfg.model.optim.sched.warmup_steps,
                           "min_lr": cfg.model.optim.sched.min_lr,
                           "_normalize": cfg.model.optim.sched.d_model ** -0.5,
                           "base_lrs": [cfg.model.optim.lr]}],
        "hyper_parameters": {"cfg": {"model_defaults": OmegaConf.to_container(cfg.model.model_defaults),
            "custom_finetune": {
                "method": "nvidia_unigram_merge", "language": "ug-CN", "prompt_index": 3,
                "tokenizer_vocab_size": metadata["merged_vocab_size"],
                "tokenizer_sha256": metadata["file_sha256"]["merged_tokenizer/tokenizer.model"],
                "base_tokenizer_sha256": metadata["base_tokenizer_sha256"],
                "assets_fingerprint": metadata["fingerprint"],
            }}},
    }


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.source = self.root / "original"
        self.assets = self.source / "assets/fixture123"
        (self.assets / "merged_tokenizer").mkdir(parents=True)
        payloads = {"train.jsonl": "original training data", "validation.jsonl": "original validation data",
                    "merged_tokenizer/tokenizer.model": "EXACT TOKEN IDS", "merged_tokenizer/vocab.txt": "vocab"}
        for name, value in payloads.items():
            (self.assets / name).write_text(value)
        self.metadata = {
            "fingerprint": "fixture123", "language": "ug-CN", "base_tokenizer_sha256": "base-hash",
            "base_vocab_size": 100, "new_vocab_size": 48, "added_tokens": 25, "merged_vocab_size": 125,
            "file_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in payloads.items()},
            "coverage": {"validation": {"unknown_tokens": 0}},
        }
        self.cfg, _ = build_recipe(base_config(), self.assets, self.source, "ug-CN",
                                   fused_batch_size=8, validation_batch_size=8)
        base = self.root / "base.nemo"
        base.touch()
        self.cfg.init_from_nemo_model = str(base)
        self.logs = Path(self.cfg.exp_manager.exp_dir) / self.cfg.exp_manager.name / self.cfg.exp_manager.version
        (self.logs / "checkpoints").mkdir(parents=True)
        self.ckpt = self.logs / "checkpoints/model-last.ckpt"
        self.ckpt.touch()
        OmegaConf.save(self.cfg, self.logs / "training_recipe.yaml")
        (self.logs / "preparation_metadata.json").write_text(json.dumps(self.metadata))
        self.destination = self.root / "continued"

    def test_resume_reuses_identical_assets_and_saved_settings(self):
        original_recipe = (self.logs / "training_recipe.yaml").read_bytes()
        assets, cfg, metadata, prompt = load_resume_recipe(self.ckpt, self.destination, 10000)
        self.assertEqual(prompt, 3)
        self.assertEqual(metadata, self.metadata)
        for name in metadata["file_sha256"]:
            self.assertEqual((assets / name).read_bytes(), (self.assets / name).read_bytes())
        self.assertEqual(cfg.trainer.max_steps, 10000)
        self.assertEqual(cfg.model.optim.sched.max_steps, 10000)
        self.assertEqual(cfg.model.optim.sched.warmup_steps, 2000)
        self.assertEqual(cfg.model.joint.fused_batch_size, 8)
        self.assertEqual(cfg.model.validation_ds.batch_size, 8)
        self.assertEqual(cfg.model.model_defaults.prompt_dictionary, self.cfg.model.model_defaults.prompt_dictionary)
        self.assertEqual(cfg.model.train_ds.manifest_filepath, str(assets / "train.jsonl"))
        self.assertEqual(cfg.resume_from_checkpoint, str(self.ckpt))
        self.assertEqual((self.logs / "training_recipe.yaml").read_bytes(), original_recipe)
        self.assertEqual(load_resume_recipe(self.ckpt, self.destination, 10000)[0], assets)

    def test_modified_source_or_destination_assets_are_rejected(self):
        assets, _, _, _ = load_resume_recipe(self.ckpt, self.destination, 10000)
        (assets / "merged_tokenizer/tokenizer.model").write_text("changed token IDs")
        with self.assertRaisesRegex(ValueError, "modified"):
            load_resume_recipe(self.ckpt, self.destination, 10000)
        (self.assets / "train.jsonl").write_text("changed training data")
        with self.assertRaisesRegex(ValueError, "modified"):
            load_resume_recipe(self.ckpt, self.root / "another", 10000)

    def test_resume_refuses_overwriting_source_run_and_nemo_exports(self):
        with self.assertRaisesRegex(FileExistsError, "new --output-dir"):
            load_resume_recipe(self.ckpt, self.source, 10000)
        with self.assertRaisesRegex(ValueError, "Lightning .ckpt"):
            load_resume_recipe(Path(self.cfg.init_from_nemo_model), self.destination, 10000)
        (self.logs / "training_recipe.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "Cannot find"):
            load_resume_recipe(self.ckpt, self.destination, 10000)

    def test_run_directory_requires_one_unambiguous_last_checkpoint(self):
        _, cfg, _, _ = load_resume_recipe(self.source, self.destination, 10000)
        self.assertEqual(cfg.resume_from_checkpoint, str(self.ckpt))
        (self.logs / "checkpoints/another-last.ckpt").touch()
        with self.assertRaisesRegex(ValueError, "2 last checkpoints"):
            load_resume_recipe(self.source, self.root / "another", 10000)

    def test_prepare_only_resume_never_retrains_tokenizer_or_loads_model(self):
        with patch("exten_tokinizer_nvidia_style.train.prepare_assets") as prepare, \
             patch("exten_tokinizer_nvidia_style.train.track_training") as tracking, \
             redirect_stdout(io.StringIO()):
            main(["--resume-from", str(self.ckpt), "--max-steps", "10000", "--output-dir", str(self.destination),
                  "--prepare-only", "--wandb", "--wandb-name", "continued"])
            prepare.assert_not_called()
            tracking.assert_not_called()
        saved = OmegaConf.load(self.destination / "assets/fixture123/training_prompt3.yaml")
        self.assertEqual(saved.trainer.max_steps, 10000)
        self.assertEqual(saved.wandb.name, "continued")
        self.assertEqual(saved.model.joint.fused_batch_size, 8)

    def test_cli_rejects_incompatible_resume_overrides_and_invalid_step_counts(self):
        for flags in (["--base-model", "base.nemo"], ["--fused-batch-size", "16"],
                      ["--language", "ms-MY"], ["--batch-duration=400"]):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--resume-from", str(self.ckpt), *flags])
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(["--max-steps", value])

    def test_checkpoint_tokenizer_prompt_and_optimizer_state_must_match(self):
        self.cfg.trainer.max_steps = 10000
        checkpoint = checkpoint_fixture(self.cfg, self.metadata)
        self.assertEqual(validate_resume_checkpoint(checkpoint, self.cfg, self.metadata, 3), 2000)
        for key in ("language", "prompt_index", "tokenizer_sha256", "assets_fingerprint", "base_tokenizer_sha256"):
            wrong = copy.deepcopy(checkpoint)
            wrong["hyper_parameters"]["cfg"]["custom_finetune"][key] = "WRONG"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not match"):
                validate_resume_checkpoint(wrong, self.cfg, self.metadata, 3)
        for key in ("optimizer_states", "lr_schedulers", "state_dict"):
            wrong = copy.deepcopy(checkpoint)
            del wrong[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume_checkpoint(wrong, self.cfg, self.metadata, 3)
        wrong = copy.deepcopy(checkpoint)
        wrong["lr_schedulers"][0]["warmup_steps"] = 10000
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            validate_resume_checkpoint(wrong, self.cfg, self.metadata, 3)
        self.cfg.trainer.max_steps = 2000
        with self.assertRaisesRegex(ValueError, "TOTAL target"):
            validate_resume_checkpoint(checkpoint, self.cfg, self.metadata, 3)

    def test_resume_loading_handles_both_lightning_fit_apis(self):
        class Older:
            def fit(self, model, ckpt_path=None):
                pass

        class Newer:
            def fit(self, model, ckpt_path=None, weights_only=None):
                pass

        self.assertEqual(checkpoint_fit_kwargs(Older(), self.ckpt), {"ckpt_path": str(self.ckpt)})
        self.assertEqual(checkpoint_fit_kwargs(Newer(), self.ckpt),
                         {"ckpt_path": str(self.ckpt), "weights_only": False})
        self.assertEqual(checkpoint_fit_kwargs(Newer(), None), {"ckpt_path": None})


if __name__ == "__main__":
    unittest.main()
