# Gemma4 shared full-logit fixture

This directory contains reusable Google FP64 and llama.cpp CPU arrays for the
exact GGUF and numerical contract in `manifest.json`. Follow the
[comparison and reproduction guide](../README.md) and
[corpus attributions](../ATTRIBUTION.md). Model weights are not included.

| Input | Tokens including BOS | Observed positions (zero based) |
| --- | ---: | --- |
| wiki-du-fu | 5348 | 511, 1023, 2047, 4095, 5347 |
| wiki-frank-headlam | 2048 | 511, 1023, 2047 |
| instruction-json | 42 | 20, 41 |
| python-close-elements | 132 | 65, 131 |
| italian-accordion | 206 | 102, 205 |

There are 14 full-vocabulary rows per backend: 28 MiB Google FP64 and
14 MiB llama FP32, plus headers and metadata. All cases use cached prefill,
explicit chunks of at most 1,024 tokens and initially empty state.
Frank Headlam retains the full source text for attribution; its numerical
input is a 2,048-token prefix. Du Fu is complete, with no artificial repetition.

Google is the trusted operational reference. This corpus does not claim new
independent HF qualification. The public finite-suite verdict does not establish
population equivalence or task quality. Historical versions are retained by Git;
compare the manifest content hash when exchanging results.
