# Sources used to produce the committed fixtures

These immutable source snapshots let contributors inspect the exact wrappers
that produced the shared arrays. Their SHA256 values match `wrapper_sha256` in
the fixture manifest:

| Baseline | Source | SHA256 |
| --- | --- | --- |
| `panther-vulkan` | [llama_driver-5de74d8e.cpp](llama_driver-5de74d8e.cpp) | `5de74d8e67b864050fbd9fcfeae18adeb2239b287e0d687453b33a9d024ff8d6` |
| `raptor-vulkan` | [llama_driver-b39a78bc.cpp](llama_driver-b39a78bc.cpp) | `b39a78bcaf88f54ab6ccc998c98f73a427992d73c088c0132374fa3194dc1c50` |

The Raptor acquisition used GPU-synchronized cooling pauses between unchanged
chunks, enabled by `XE_QUALITY_CHUNK_COOLDOWN` in its historical wrapper. That
snapshot is retained for provenance; temperature management is not a feature of
the maintained generator. Acquisition elapsed times are not throughput results.

For new captures use the maintained wrapper and [generation guide](../README.md).
The manifest records the pinned llama.cpp revision and numerical configuration.
These snapshots identify source bytes; reproducing the original binary also
requires its compiler and linked libraries. Keep snapshot bytes unchanged.
