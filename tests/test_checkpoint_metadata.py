"""Recover the exact tokenizer/prompt from Lightning's NeMo metadata formats."""

from pathlib import Path
import tempfile
import unittest

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from test_checkpoint import (
    checkpoint_config_value,
    prompt_index_from_checkpoint,
    tokenizer_dir_from_checkpoint,
)


@unittest.skipIf(OmegaConf is None, "OmegaConf is installed with the NeMo environment")
class CheckpointMetadataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.tokenizer = self.root / "assets" / "fingerprint" / "merged_tokenizer"

    def config(self):
        return {
            "assets": str(self.tokenizer.parent),
            "custom_finetune": {
                "method": "nvidia_unigram_merge",
                "tokenizer_dir": str(self.tokenizer),
            },
            "model_defaults": {"prompt_dictionary": {"en-US": 0, "ug-CN": 37}},
            "decoding": {"strategy": "greedy_batch", "strip_lang_tags": True},
        }

    def assert_metadata(self, checkpoint):
        self.assertEqual(tokenizer_dir_from_checkpoint(checkpoint), self.tokenizer)
        self.assertEqual(prompt_index_from_checkpoint(checkpoint, "ug-CN"), 37)
        self.assertEqual(prompt_index_from_checkpoint(checkpoint, "en-US"), 0)
        self.assertIsNone(prompt_index_from_checkpoint(checkpoint, "zz-ZZ"))
        decoding = checkpoint_config_value(checkpoint, "decoding")
        self.assertEqual(decoding.strategy, "greedy_batch")
        self.assertTrue(decoding.strip_lang_tags)

    def test_plain_and_omegaconf_containers_at_both_levels(self):
        for key in ("cfg", "model_cfg"):
            for config_container in (dict, OmegaConf.create):
                for hparams_container in (dict, OmegaConf.create):
                    with self.subTest(key=key, cfg=config_container, hparams=hparams_container):
                        hparams = hparams_container({key: config_container(self.config())})
                        self.assert_metadata({"hyper_parameters": hparams})

    def test_interpolations_keep_their_original_config_scope(self):
        # A standalone cfg and a cfg inside an OmegaConf hparams container have
        # different roots. Preserve each format's valid references on lookup.
        cfg = self.config()
        cfg["custom_finetune"]["tokenizer_dir"] = "${assets}/merged_tokenizer"
        checkpoint = {"hyper_parameters": {"cfg": OmegaConf.create(cfg)}}
        self.assert_metadata(checkpoint)
        raw = OmegaConf.to_container(checkpoint["hyper_parameters"]["cfg"], resolve=False)
        self.assertEqual(raw["custom_finetune"]["tokenizer_dir"], "${assets}/merged_tokenizer")

        cfg["custom_finetune"]["tokenizer_dir"] = "${cfg.assets}/merged_tokenizer"
        checkpoint = {"hyper_parameters": OmegaConf.create({"cfg": cfg})}
        self.assert_metadata(checkpoint)
        raw = OmegaConf.to_container(checkpoint["hyper_parameters"], resolve=False)
        self.assertEqual(raw["cfg"]["custom_finetune"]["tokenizer_dir"], "${cfg.assets}/merged_tokenizer")

    def test_saved_nemo_tokenizer_directory_without_custom_metadata(self):
        cfg = self.config()
        del cfg["custom_finetune"]
        cfg["tokenizer"] = {"dir": str(self.tokenizer), "type": "bpe"}
        self.assert_metadata({"hyper_parameters": OmegaConf.create({"cfg": cfg})})

    def test_custom_provenance_takes_priority_over_generic_directory(self):
        cfg = self.config()
        cfg["tokenizer"] = {"dir": "/old/base/tokenizer"}
        self.assert_metadata({"hyper_parameters": {"cfg": cfg}})

    def test_saved_missing_directory_is_not_replaced_by_another_tokenizer(self):
        checkpoint = {"hyper_parameters": OmegaConf.create({"cfg": self.config()})}
        self.assertFalse(self.tokenizer.exists())
        self.assertEqual(tokenizer_dir_from_checkpoint(checkpoint), self.tokenizer)

    def test_missing_or_malformed_metadata(self):
        for hparams in (None, [], "invalid", {}, {"cfg": None}, {"cfg": []},
                        {"cfg": "invalid"}, OmegaConf.create({"cfg": {}})):
            with self.subTest(hparams=hparams):
                checkpoint = {"hyper_parameters": hparams}
                self.assertIsNone(tokenizer_dir_from_checkpoint(checkpoint))
                self.assertIsNone(prompt_index_from_checkpoint(checkpoint, "ug-CN"))
                self.assertIsNone(checkpoint_config_value(checkpoint, "decoding"))

    def test_model_cfg_fallback_when_cfg_is_missing_a_setting(self):
        checkpoint = {"hyper_parameters": OmegaConf.create({"cfg": {}, "model_cfg": self.config()})}
        self.assert_metadata(checkpoint)

    def test_real_torch_checkpoint_round_trip_preserves_metadata(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is installed with the NeMo environment")
        for container in (dict, OmegaConf.create):
            with self.subTest(hparams=container):
                path = self.root / "model.ckpt"
                checkpoint = {
                    "hyper_parameters": container({"cfg": self.config()}),
                    "state_dict": {"weight": torch.tensor([1.0, 2.0])},
                    "epoch": 99,
                    "global_step": 5000,
                }
                torch.save(checkpoint, path)
                restored = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                self.assert_metadata(restored)
                self.assertTrue(torch.equal(restored["state_dict"]["weight"], checkpoint["state_dict"]["weight"]))


if __name__ == "__main__":
    unittest.main()
