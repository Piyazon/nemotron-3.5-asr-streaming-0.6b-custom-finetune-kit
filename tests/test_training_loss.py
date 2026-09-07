"""Regression coverage for loss options discarded by NeMo vocabulary changes."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from training_loss import restore_configured_rnnt_loss


class TrainingLossTests(unittest.TestCase):
    def setUp(self):
        self.factory = Mock()
        self.enterContext(patch.dict("sys.modules", {
            "nemo.collections.asr.losses.rnnt": SimpleNamespace(RNNTLoss=self.factory),
        }))
        self.original_loss = object()
        self.options = {"fastemit_lambda": 0.005, "clamp": -1.0}
        self.loss_cfg = {
            "loss_name": "warprnnt_numba",
            "warprnnt_numba_kwargs": self.options,
        }
        self.model = SimpleNamespace(
            cfg={"loss": self.loss_cfg, "rnnt_reduction": "mean_volume"},
            joint=SimpleNamespace(
                num_classes_with_blank=2049,
                num_extra_outputs=0,
                fuse_loss_wer=True,
                set_loss=Mock(),
            ),
            loss=self.original_loss,
            extract_rnnt_loss_cfg=Mock(return_value=("warprnnt_numba", self.options)),
        )

    def test_restores_backend_fastemit_reduction_and_fused_reference(self):
        restore_configured_rnnt_loss(self.model)
        self.model.extract_rnnt_loss_cfg.assert_called_once_with(self.loss_cfg)
        self.factory.assert_called_once_with(
            num_classes=2048,
            loss_name="warprnnt_numba",
            loss_kwargs={"fastemit_lambda": 0.005, "clamp": -1.0},
            reduction="mean_volume",
        )
        self.assertIs(self.model.loss, self.factory.return_value)
        self.model.joint.set_loss.assert_called_once_with(self.model.loss)

    def test_uses_current_vocabulary_and_default_reduction(self):
        self.model.cfg.pop("rnnt_reduction")
        self.model.joint.num_classes_with_blank = 4097
        restore_configured_rnnt_loss(self.model)
        self.assertEqual(self.factory.call_args.kwargs["num_classes"], 4096)
        self.assertEqual(self.factory.call_args.kwargs["reduction"], "mean_batch")

    def test_nonfused_joint_keeps_its_execution_mode(self):
        self.model.joint.fuse_loss_wer = False
        restore_configured_rnnt_loss(self.model)
        self.assertIs(self.model.loss, self.factory.return_value)
        self.model.joint.set_loss.assert_not_called()
        self.assertFalse(self.model.joint.fuse_loss_wer)

    def test_absent_loss_config_is_resolved_by_nemo(self):
        self.model.cfg.pop("loss")
        self.model.extract_rnnt_loss_cfg.return_value = ("warprnnt_numba", None)
        restore_configured_rnnt_loss(self.model)
        self.model.extract_rnnt_loss_cfg.assert_called_once_with(None)
        self.assertIsNone(self.factory.call_args.kwargs["loss_kwargs"])

    def test_failed_backend_does_not_replace_existing_loss(self):
        self.factory.side_effect = ImportError("backend is unavailable")
        with self.assertRaisesRegex(ImportError, "backend is unavailable"):
            restore_configured_rnnt_loss(self.model)
        self.assertIs(self.model.loss, self.original_loss)
        self.model.joint.set_loss.assert_not_called()

    def test_tdt_duration_outputs_are_not_counted_as_token_classes(self):
        self.model.extract_rnnt_loss_cfg.return_value = ("tdt", {"durations": [0, 1, 2]})
        self.model.joint.num_classes_with_blank = 2052
        self.model.joint.num_extra_outputs = 3
        restore_configured_rnnt_loss(self.model)
        self.assertEqual(self.factory.call_args.kwargs["num_classes"], 2048)


if __name__ == "__main__":
    unittest.main()
