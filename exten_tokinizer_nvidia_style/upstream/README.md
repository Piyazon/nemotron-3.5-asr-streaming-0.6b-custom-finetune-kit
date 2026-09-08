# NVIDIA source configuration

`fastconformer_transducer_bpe_streaming_prompt.yaml` is an unmodified copy from
[NVIDIA NeMo, commit 6ad981f26ec1e7156a73fad78b1b9c9c7ec0f463](https://github.com/NVIDIA-NeMo/NeMo/blob/6ad981f26ec1e7156a73fad78b1b9c9c7ec0f463/examples/asr/conf/fastconformer/cache_aware_streaming/fastconformer_transducer_bpe_streaming_prompt.yaml),
retrieved on 2026-09-08. Its Apache-2.0 license is included in `LICENSE`.

The model is restored from the user's `.nemo`; the example encoder architecture
in this YAML is not instantiated. `../nvidia_recipe.yaml` applies the training
overrides from the [NVIDIA Riva new-language tutorial](https://github.com/nvidia-riva/tutorials/blob/main/asr-extend-tokenizer-to-newlang-ft-acoustic-model.ipynb)
as inspected on 2026-09-08. Runtime integration, paths and the new prompt mapping
are supplied by `../recipe.py`. The NeMo package itself follows the parent
repository's dependency specification; the vendored YAML does not pin the
installed package version.
