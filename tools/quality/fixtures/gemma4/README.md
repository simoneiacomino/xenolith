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
| manzoni-promessi-sposi-8192 | 8192 | 1023, 2047, 4095, 8191 |
| cpython-base-events-8192 | 8192 | 1023, 2047, 4095, 8191 |
| manzoni-promessi-sposi-16384 | 16384 | 16383 |

The validation report groups inputs up to 5,348 tokens, 8,192 tokens and
16,384 tokens. Panther covers all eight cases and 23 positions. Raptor covers
the first five cases and 14 positions. There are 46 MiB of Google FP64 logits,
23 MiB of Panther FP32 logits and 14 MiB of Raptor FP32 logits, plus headers
and metadata. All cases use cached prefill, explicit chunks of at most 1,024
tokens and initially empty state. The two Manzoni cases share one source text.
Frank Headlam retains the full source text for attribution; its numerical
input is a 2,048-token prefix. Du Fu is complete, with no artificial repetition.

Google is the trusted operational reference. Follow the linked comparison guide
for baseline selection, acceptance and coverage limits. Compare the manifest
content hash when exchanging results; Git retains earlier fixture versions.
