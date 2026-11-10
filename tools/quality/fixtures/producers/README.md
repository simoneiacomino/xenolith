# Fixture producer wrappers

`llama_driver-5de74d8e.cpp` is the exact native wrapper used by the
`panther-vulkan` fixture. Its SHA256 is
`5de74d8e67b864050fbd9fcfeae18adeb2239b287e0d687453b33a9d024ff8d6`.
`llama_driver-b39a78bc.cpp` is the exact wrapper used by `raptor-vulkan`,
compiled with `XE_QUALITY_CHUNK_COOLDOWN`. Its SHA256 is
`b39a78bcaf88f54ab6ccc998c98f73a427992d73c088c0132374fa3194dc1c50`.

Keep these bytes unchanged; a new producer receives a new identity.

Use this source with the pinned llama.cpp revision and numerical configuration
in the fixture manifest. The current build tool can build this wrapper. The
snapshot provides the wrapper source; it does not archive the original build
tool or guarantee recreation of the producer binary hash. Compiler, linked
libraries and build environment also contribute to that identity.

See the [generation guide](../README.md) for capture and verification.
