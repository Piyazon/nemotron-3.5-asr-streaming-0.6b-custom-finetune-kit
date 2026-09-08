#!/usr/bin/env python3
"""Transcribe files with an exported model and its saved language prompt."""

import argparse


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True)
    cli.add_argument("--language", default="ug-CN")
    cli.add_argument("--keep-language-tags", action="store_true")
    cli.add_argument("audio", nargs="+")
    args = cli.parse_args()

    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    from omegaconf import open_dict

    model = EncDecRNNTBPEModelWithPrompt.restore_from(args.model, map_location="cpu")
    if args.language not in model.cfg.model_defaults.prompt_dictionary:
        raise ValueError(f"No saved language prompt for {args.language}")
    with open_dict(model.cfg.decoding):
        model.cfg.decoding.strip_lang_tags = not args.keep_language_tags
    model.change_decoding_strategy(model.cfg.decoding)
    model.eval().cuda()
    results = model.transcribe(audio=args.audio, batch_size=1, target_lang=args.language, verbose=False)
    for path, result in zip(args.audio, results):
        print(f"{path}\t{result if isinstance(result, str) else result.text}")


if __name__ == "__main__":
    main()
