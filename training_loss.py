"""Keep NeMo's configured transducer objective after replacing a vocabulary."""


def restore_configured_rnnt_loss(model) -> None:
    """Rebuild the loss for the live joint without losing configured options.

    Some NeMo releases implement ``change_vocabulary()`` by constructing a
    default RNNTLoss, even when ``model.cfg.loss`` specifies another backend,
    FastEmit options, or reduction. Match EncDecRNNTModel's initialization and
    refresh the fused joint's reference so both execution paths use that loss.
    Call this after vocabulary replacement, before training/device placement.
    """
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss"))
    num_classes = model.joint.num_classes_with_blank - 1
    if loss_name == "tdt":
        num_classes -= model.joint.num_extra_outputs

    # Construct first: an unavailable backend should raise without replacing
    # the model's existing loss or leaving its joint with a stale new reference.
    loss = RNNTLoss(
        num_classes=num_classes,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        reduction=model.cfg.get("rnnt_reduction", "mean_batch"),
    )
    model.loss = loss
    if model.joint.fuse_loss_wer:
        model.joint.set_loss(loss)
