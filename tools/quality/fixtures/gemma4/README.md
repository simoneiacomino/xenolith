# Gemma4 shared full-logit fixture

This directory contains reusable Google FP64 arrays and two llama.cpp Vulkan
baselines (`panther-vulkan` and `raptor-vulkan`) for the
exact GGUF and numerical contract in `manifest.json`. Follow the
[comparison and reproduction guide](../README.md) and
[corpus attributions](../ATTRIBUTION.md). Model weights are not included.

Each profile was generated on its named GPU platform. Additional hosts require
their own verified captures. See the [producer source record](../producers/README.md)
for the exact wrappers and acquisition details.

| Input | Tokens including BOS | Observed positions (zero based) |
| --- | ---: | --- |
| wiki-du-fu | 5348 | 511, 1023, 2047, 4095, 5347 |
| wiki-frank-headlam | 2048 | 511, 1023, 2047 |
| instruction-json | 42 | 20, 41 |
| python-close-elements | 132 | 65, 131 |
| italian-accordion | 206 | 102, 205 |

There are 14 full-vocabulary rows per backend: 28 MiB Google FP64 and
14 MiB FP32 per Vulkan baseline, plus headers and metadata. All cases use cached prefill,
explicit chunks of at most 1,024 tokens and initially empty state.
Frank Headlam retains the full source text for attribution; its numerical
input is a 2,048-token prefix. Du Fu is complete, with no artificial repetition.

Google is the trusted operational reference. Follow the linked comparison guide
for baseline selection, acceptance and coverage limits. Compare the manifest
content hash when exchanging results; Git retains earlier fixture versions.
