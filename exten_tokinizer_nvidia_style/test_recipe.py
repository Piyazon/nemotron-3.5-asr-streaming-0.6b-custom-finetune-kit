"""CPU regression checks for tokenizer IDs, language slots, and training settings.

Run from the repository root:
    python -m unittest exten_tokinizer_nvidia_style.test_recipe -v
"""

import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from exten_tokinizer_nvidia_style.recipe import allocate_prompt, build_recipe, configure_model
from exten_tokinizer_nvidia_style.train import parser
from exten_tokinizer_nvidia_style.tokenizer import (
    merge_tokenizers, parse_model, prepare_assets, read_manifest, tag_transcript,
)


def base_config():
    return OmegaConf.create({
        "sample_rate": 16000, "num_prompts": 5,
        "model_defaults": {"prompt_dictionary": {"en-US": 0, "en": 0, "ms-MY": 2},
                           "num_prompts": 5, "enc_hidden": 1024, "pred_hidden": 640, "joint_hidden": 640},
        "encoder": {"d_model": 1024, "n_layers": 24, "subsampling_factor": 8},
        "joint": {"fused_batch_size": 2, "fuse_loss_wer": True},
        "tokenizer": {"type": "bpe", "model_path": "nemo:abc_tokenizer.model"},
        "train_ds": {"prompt_dictionary": {"en-US": 0, "fr-FR": 1}},
        "validation_ds": {"prompt_dictionary": {"en-US": 0}},
        "test_ds": {"prompt_dictionary": {"en-US": 0}},
    })


class RecipeTests(unittest.TestCase):
    def test_slot_allocation_preserves_aliases_and_all_saved_reservations(self):
        cfg = base_config()
        before = OmegaConf.to_container(cfg)
        mapping, index, count = allocate_prompt(cfg, "ug-CN")
        self.assertEqual((index, count), (3, 5))
        self.assertEqual(mapping, {"en-US": 0, "en": 0, "ms-MY": 2, "fr-FR": 1, "ug-CN": 3})
        self.assertEqual(OmegaConf.to_container(cfg), before)
        self.assertEqual(allocate_prompt(cfg, "ms-MY")[1], 2)
        for index in (0, 1, 2, -1, 5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                allocate_prompt(cfg, "ug-CN", index)
        self.assertEqual(allocate_prompt(cfg, "ug-CN", 4)[1], 4)

    def test_prompt_conflicts_and_exhaustion_fail(self):
        cfg = base_config()
        cfg.train_ds.prompt_dictionary["en-US"] = 2
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            allocate_prompt(cfg, "ug-CN")
        cfg = base_config()
        cfg.num_prompts = 3
        with self.assertRaisesRegex(ValueError, "occupied"):
            allocate_prompt(cfg, "ug-CN")
        with self.assertRaisesRegex(ValueError, "already uses"):
            allocate_prompt(base_config(), "ms-MY", 3)

    def test_official_hyperparameters_and_checkpoint_architecture(self):
        cfg, index = build_recipe(base_config(), Path("/assets"), Path("/run"), "ug-CN")
        trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
        expected = {"devices": 1, "max_steps": 2000, "max_epochs": -1, "limit_train_batches": 200,
                    "check_val_every_n_epoch": 20, "val_check_interval": 0.5,
                    "accumulate_grad_batches": 4, "log_every_n_steps": 100,
                    "gradient_clip_val": 0.5, "precision": "bf16"}
        self.assertEqual({key: trainer[key] for key in expected}, expected)
        self.assertEqual(cfg.model.optim.name, "adamw")
        self.assertEqual(cfg.model.optim.lr, 2.0)
        self.assertEqual(list(cfg.model.optim.betas), [0.9, 0.98])
        self.assertEqual(cfg.model.optim.weight_decay, 0.001)
        self.assertEqual(cfg.model.optim.sched.name, "NoamAnnealing")
        self.assertEqual(cfg.model.optim.sched.warmup_steps, 2000)
        self.assertEqual(cfg.model.optim.sched.d_model, 1024)
        self.assertEqual(cfg.model.optim.sched.min_lr, 1e-6)
        self.assertEqual(cfg.model.train_ds.batch_duration, 200)
        self.assertEqual(cfg.model.train_ds.max_duration, 39.99)
        self.assertEqual(cfg.model.train_ds.num_workers, 8)
        self.assertEqual(cfg.model.validation_ds.batch_size, 2)
        self.assertEqual(cfg.model.encoder.n_layers, 24)
        self.assertEqual(index, 3)
        for split in ("train_ds", "validation_ds", "test_ds"):
            self.assertEqual(cfg.model[split].prompt_dictionary["ug-CN"], 3)
            self.assertEqual(cfg.model[split].default_prompt_mode, "langID")
            self.assertFalse(cfg.model[split].is_tarred)
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    def test_prompt_mapping_and_provenance_survive_configuration_serialization(self):
        class Joint:
            fused_batch_size = 2
            fuse_loss_wer = True

            def set_fused_batch_size(self, value):
                self.fused_batch_size = value

            def set_fuse_loss_wer(self, value, loss, metric):
                self.fuse_loss_wer = value
                self.loss, self.metric = loss, metric

        class Model:
            def __init__(self):
                self.cfg = base_config()
                self.calls = []

            def change_vocabulary(self, **kwargs):
                self.calls.append(("vocabulary", kwargs))
                self.joint = Joint()
                self.loss, self.wer = object(), object()

            def setup_training_data(self, cfg):
                self.cfg.train_ds = copy.deepcopy(cfg)

            def setup_multiple_validation_data(self, cfg):
                self.cfg.validation_ds = copy.deepcopy(cfg)

            def setup_optimization(self, cfg):
                self.cfg.optim = copy.deepcopy(cfg)

            def from_config_dict(self, cfg):
                return cfg

        cfg, index = build_recipe(base_config(), Path("/assets"), Path("/run"), "ug-CN", fused_batch_size=8)
        model = Model()
        configure_model(model, cfg, {
            "language": "ug-CN", "merged_vocab_size": 123,
            "base_tokenizer_sha256": "original", "fingerprint": "assets123",
            "file_sha256": {"merged_tokenizer/tokenizer.model": "merged"},
        }, index)
        saved = OmegaConf.create(OmegaConf.to_yaml(model.cfg, resolve=True))
        for section in ("model_defaults", "train_ds", "validation_ds", "test_ds"):
            self.assertEqual(saved[section].prompt_dictionary["ug-CN"], 3)
            self.assertEqual(saved[section].prompt_dictionary["ms-MY"], 2)
        self.assertEqual(saved.custom_finetune.tokenizer_sha256, "merged")
        self.assertEqual(saved.custom_finetune.prompt_index, 3)
        self.assertEqual(model.joint.fused_batch_size, 8)
        self.assertEqual(saved.joint.fused_batch_size, 8)
        self.assertEqual(saved.custom_finetune.fused_batch_size, 8)
        self.assertEqual(saved.custom_finetune.batch_duration, 200)
        self.assertTrue(model.joint.fuse_loss_wer)
        self.assertIs(model.joint.loss, model.loss)
        self.assertIs(model.joint.metric, model.wer)
        self.assertFalse(model.compute_eval_loss)
        self.assertEqual(model.calls[0][1]["new_tokenizer_type"], "bpe")

    def test_cli_batch_overrides_preserve_optimizer_schedule_and_accumulation(self):
        args = parser().parse_args([
            "--batch-duration", "400", "--fused-batch-size", "8", "--train-workers", "16",
            "--validation-workers", "4", "--validation-batch-size", "8",
        ])
        original, _ = build_recipe(base_config(), Path("/assets"), Path("/run"), "ug-CN")
        cfg, _ = build_recipe(
            base_config(), Path("/assets"), Path("/run"), "ug-CN",
            **{key: getattr(args, key) for key in (
                "batch_duration", "fused_batch_size", "train_workers", "validation_workers", "validation_batch_size")},
        )
        self.assertEqual(cfg.model.train_ds.batch_duration, 400)
        self.assertEqual(cfg.model.joint.fused_batch_size, 8)
        self.assertEqual(cfg.model.train_ds.num_workers, 16)
        self.assertEqual(cfg.model.validation_ds.num_workers, 4)
        self.assertEqual(cfg.model.validation_ds.batch_size, 8)
        self.assertEqual(cfg.model.optim, original.model.optim)
        self.assertEqual(cfg.trainer, original.trainer)
        self.assertEqual(cfg.model.train_ds.max_duration, original.model.train_ds.max_duration)

    def test_no_batch_overrides_preserves_base_joint_settings(self):
        base = base_config()
        base.joint.fused_batch_size = 4
        base.joint.fuse_loss_wer = False
        cfg, _ = build_recipe(base, Path("/assets"), Path("/run"), "ug-CN")
        self.assertEqual(cfg.model.joint.fused_batch_size, 4)
        self.assertFalse(cfg.model.joint.fuse_loss_wer)
        cfg, _ = build_recipe(base, Path("/assets"), Path("/run"), "ug-CN", fused_batch_size=8)
        self.assertTrue(cfg.model.joint.fuse_loss_wer)

    def test_invalid_batch_overrides_fail_before_preparation(self):
        for flag, value in (("--batch-duration", "nan"), ("--batch-duration", "inf"),
                            ("--batch-duration", "0"), ("--fused-batch-size", "0"),
                            ("--fused-batch-size", "2.5"), ("--train-workers", "-1"),
                            ("--validation-workers", "-1"), ("--validation-batch-size", "0")):
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args([flag, value])
        self.assertEqual(parser().parse_args(["--train-workers", "0"]).train_workers, 0)


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        corpus = cls.root / "base.txt"
        corpus.write_text("Hello world. The boy is running.\nAnother English sentence.\n" * 10, encoding="utf-8")
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(cls.root / "base"),
            model_type="unigram", vocab_size=48, bos_id=-1, eos_id=-1, hard_vocab_limit=False,
            user_defined_symbols=["<en-US>"], minloglevel=2)
        cls.base = (cls.root / "base.model").read_bytes()
        corpus.write_text("مەن ئۇيغۇرچە سۆزلەيمەن. <ug-CN>\nئۇيغۇر تىلى گۈزەل. <ug-CN>\n" * 10, encoding="utf-8")
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(cls.root / "ug"),
            model_type="unigram", vocab_size=48, bos_id=-1, eos_id=-1, hard_vocab_limit=False,
            user_defined_symbols=["<ug-CN>"], minloglevel=2)
        cls.new = (cls.root / "ug.model").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_merge_retains_base_ids_normalizer_and_special_piece_types(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            stats = merge_tokenizers(self.base, self.new, path, "ug-CN")
            merged = parse_model((path / "tokenizer.model").read_bytes())
            base = parse_model(self.base)
            self.assertEqual([p.piece for p in merged.pieces[:len(base.pieces)]], [p.piece for p in base.pieces])
            self.assertEqual(merged.normalizer_spec, base.normalizer_spec)
            self.assertEqual(merged.pieces[stats["language_tag_id"]].type, pb.ModelProto.SentencePiece.USER_DEFINED)
            self.assertEqual((path / "vocab.txt").read_text().splitlines(), [p.piece for p in merged.pieces])
            sp = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
            for text in ("Hello world.", "مەن ئۇيغۇرچە سۆزلەيمەن. <ug-CN>"):
                ids = sp.encode(text)
                self.assertNotIn(sp.unk_id(), ids)
                self.assertEqual(sp.decode(ids), text)
            self.assertEqual(stats["merged_vocab_size"], stats["base_vocab_size"] + stats["added_tokens"])
            self.assertLess(stats["added_tokens"], stats["new_vocab_size"])

    def test_tags_are_idempotent_and_preserve_uyghur_letters(self):
        text = "مەن ئۇيغۇر. ئىككىنچى جۈملە."
        tagged = tag_transcript(text, "ug-CN")
        self.assertEqual(tagged, "مەن ئۇيغۇر. <ug-CN> ئىككىنچى جۈملە. <ug-CN>")
        self.assertEqual(tag_transcript(tagged, "ug-CN"), tagged)
        self.assertEqual(tag_transcript("جۈملە", "ug-CN"), "جۈملە")
        with self.assertRaisesRegex(ValueError, "other language tags"):
            tag_transcript("Hello. <en-US>", "ug-CN")

    def test_prepare_uses_train_text_only_and_detects_asset_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "base.nemo"
            with tarfile.open(archive_path, "w") as archive:
                for name, data in (("model_config.yaml", OmegaConf.to_yaml(base_config()).encode()),
                                   ("abc_tokenizer.model", self.base)):
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            for name in ("train.wav", "val.wav"):
                (root / name).touch()
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            train.write_text(json.dumps({"audio_filepath": "train.wav", "duration": 1,
                                         "text": "مەن ئۇيغۇرچە سۆزلەيمەن.", "language": "ug-CN"}) + "\n")
            val.write_text(json.dumps({"audio_filepath": "val.wav", "duration": 1,
                                       "text": "VALIDATION_ONLY_Ж", "language": "ug-CN"}) + "\n")
            original = train.read_bytes()
            assets, _, metadata = prepare_assets(archive_path, train, val, root / "run", vocab_size=48)
            self.assertNotIn("VALIDATION_ONLY", (assets / "train_text.txt").read_text())
            self.assertEqual(train.read_bytes(), original)
            self.assertEqual(metadata["coverage"]["train"]["unknown_tokens"], 0)
            self.assertGreater(metadata["coverage"]["validation"]["unknown_tokens"], 0)
            with patch("exten_tokinizer_nvidia_style.tokenizer.train_unigram", side_effect=AssertionError("should reuse")):
                self.assertEqual(prepare_assets(archive_path, train, val, root / "run", vocab_size=48)[0], assets)
            (assets / "merged_tokenizer/vocab.txt").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "was modified"):
                prepare_assets(archive_path, train, val, root / "run", vocab_size=48)

    def test_manifest_rejects_missing_audio_and_conflicting_languages(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.jsonl"
            row = {"audio_filepath": "absent.wav", "duration": 1, "text": "مەن"}
            manifest.write_text(json.dumps(row))
            with self.assertRaises(FileNotFoundError):
                read_manifest(manifest, "ug-CN")
            row.update(language="ug-CN", target_lang="en-US")
            manifest.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "target_lang"):
                read_manifest(manifest, "ug-CN")


if __name__ == "__main__":
    unittest.main()
