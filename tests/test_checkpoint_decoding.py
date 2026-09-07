"""Inference configuration checks without importing NeMo or running a model."""

from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from test_checkpoint import (
    best_transcription, checkpoint_config_value, configure_decoding, parse_args,
)


class DecodingArgumentsTests(unittest.TestCase):
    def test_default_preserves_checkpoint_strategy(self):
        args = parse_args([])
        self.assertEqual(args.decoding_strategy, "checkpoint")
        self.assertIsNone(args.beam_size)

    def test_explicit_beam_comparison(self):
        args = parse_args(["sample.mp3", "--decoding-strategy", "beam", "--beam-size", "8"])
        self.assertEqual(args.audio, "sample.mp3")
        self.assertEqual(args.beam_size, 8)

    def test_invalid_or_unused_beam_size_fails_before_model_load(self):
        for args in (
            ["--beam-size", "8"],
            ["--decoding-strategy", "greedy_batch", "--beam-size", "8"],
            ["--decoding-strategy", "beam", "--beam-size", "0"],
            ["--decoding-strategy", "beam", "--beam-size", "-1"],
            ["--decoding-strategy", "beam", "--beam-size", "2.5"],
        ):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)


@unittest.skipIf(OmegaConf is None, "OmegaConf is installed with the NeMo environment")
class DecodingConfigurationTests(unittest.TestCase):
    def model(self):
        model = SimpleNamespace(cfg=OmegaConf.create({
            "decoding": {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}},
        }))

        def update(cfg):
            # The real NeMo method also reconstructs RNNTBPEDecoding and WER.
            model.cfg.decoding = cfg

        model.change_decoding_strategy = Mock(side_effect=update)
        return model

    def test_saved_settings_rebuild_live_decoder_and_preserve_original(self):
        checkpoint = {"hyper_parameters": {"cfg": OmegaConf.create({
            "token_limit": 17,
            "decoding": {
                "strategy": "greedy",
                "greedy": {"max_symbols": "${token_limit}"},
                "temperature": 0.8,
            },
        })}}
        model = self.model()
        self.assertEqual(configure_decoding(model, checkpoint), "greedy")
        model.change_decoding_strategy.assert_called_once()
        self.assertEqual(model.cfg.decoding.greedy.max_symbols, 17)
        self.assertEqual(model.cfg.decoding.temperature, 0.8)
        model.cfg.decoding.temperature = 0.5
        self.assertEqual(checkpoint["hyper_parameters"]["cfg"].decoding.temperature, 0.8)

    def test_older_checkpoint_without_config_keeps_live_decoder(self):
        model = self.model()
        self.assertEqual(configure_decoding(model, {}), "greedy_batch")
        model.change_decoding_strategy.assert_not_called()

    def test_plain_dict_model_cfg_and_beam_override(self):
        checkpoint = {"hyper_parameters": {"model_cfg": {"decoding": {
            "strategy": "greedy_batch", "temperature": 0.7,
            "beam": {"beam_size": 2, "score_norm": False, "return_best_hypothesis": False},
        }}}}
        model = self.model()
        configure_decoding(model, checkpoint, "beam", 8)
        model.change_decoding_strategy.assert_called_once()
        self.assertEqual(model.cfg.decoding.strategy, "beam")
        self.assertEqual(model.cfg.decoding.beam.beam_size, 8)
        self.assertFalse(model.cfg.decoding.beam.score_norm)
        self.assertTrue(model.cfg.decoding.beam.return_best_hypothesis)
        self.assertEqual(model.cfg.decoding.temperature, 0.7)
        self.assertEqual(checkpoint["hyper_parameters"]["model_cfg"]["decoding"]["beam"]["beam_size"], 2)

    def test_default_beam_width_and_no_saved_beam_settings(self):
        model = self.model()
        configure_decoding(model, {}, "beam")
        self.assertEqual(model.cfg.decoding.beam.beam_size, 4)
        self.assertEqual(model.cfg.decoding.greedy.max_symbols, 10)

    def test_saved_beam_width_is_unchanged_without_override(self):
        model = self.model()
        checkpoint = {"hyper_parameters": {"cfg": {"decoding": {
            "strategy": "beam", "beam": {"beam_size": 12},
        }}}}
        configure_decoding(model, checkpoint)
        self.assertEqual(model.cfg.decoding.beam.beam_size, 12)

    def test_invalid_programmatic_options_fail_without_rebuilding(self):
        for strategy, size in (("beam", 0), ("beam", True), ("beam", 1.5), ("greedy", 8), ("invalid", None)):
            with self.subTest(strategy=strategy, size=size), self.assertRaises(ValueError):
                configure_decoding(self.model(), {}, strategy, size)

    def test_missing_or_malformed_hyperparameters_return_none(self):
        for checkpoint in ({}, {"hyper_parameters": None}, {"hyper_parameters": {"cfg": {}}}):
            with self.subTest(checkpoint=checkpoint):
                self.assertIsNone(checkpoint_config_value(checkpoint, "decoding"))


class HypothesisOutputTests(unittest.TestCase):
    def test_best_hypothesis_across_supported_nemo_formats(self):
        best = SimpleNamespace(text="ئاتا-ئانىسى سېزىپ")
        other = SimpleNamespace(text="ئاتا-ئانىسىز يېزىپ")
        outputs = (
            [best], ([best], [[best, other]]), [[best, other]],
            [SimpleNamespace(n_best_hypotheses=[best, other])],
        )
        for output in outputs:
            with self.subTest(output=output):
                self.assertEqual(best_transcription(output), best.text)

    def test_empty_hypotheses(self):
        for output in ([], ([], []), [[]], [SimpleNamespace(n_best_hypotheses=[])]):
            with self.subTest(output=output):
                self.assertEqual(best_transcription(output), "")


if __name__ == "__main__":
    unittest.main()
