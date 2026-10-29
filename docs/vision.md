# Vision

Xenolith accepts PNG and JPEG images with the Gemma 4 26B-A4B QAT language
model and its F16 vision projector (`mmproj`). The reference files, revisions
and SHA-256 hashes are recorded in
[tests/vision/manifest.json](../tests/vision/manifest.json). Vision uses the
same [supported GPU](../README.md#gpu-compatibility) and runtime as text inference.
The engine has no Python or llama.cpp runtime dependency.

The loader checks architecture, required metadata and tensor types/shapes.
Compatible weight variants can load with a notice; reference hashes are
diagnostic. These checks do not establish training compatibility between a
model and projector or assess output quality. Weight files must remain
unchanged while an engine uses them.

## Use an image

Run from the repository root, with the language model and matching projector:

```sh
./xenolith run /path/to/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj /path/to/mmproj-F16.gguf --image screenshot.png \
  -p "What is shown in this screenshot?" -n 128
```

The CLI and services default image inference to 65536 context elements and also
support 131072.
`chat` accepts the same model, projector and image options. It attaches one
image to the first message, then reuses the visual prefix for follow-up
questions. The image encoder runs once at startup. Use `/exit` to leave chat.

The image budget is 70, 140, 280, 560 or 1120, with a default of 280. Actual
projected rows depend on aspect ratio. EXIF orientation is applied, alpha is
composited on white, and pixels are resized with the Gemma budget geometry
and Pillow-compatible bicubic filtering. Images are limited to 33,554,432
decoded pixels. CLI image loading accepts up to 64 MiB of encoded media;
persistent services accept 48 MiB per image and 256 MiB per active context.

Encoding runs on the GPU. The CLI releases encoder allocations before loading
the language model; larger budgets still increase encoder work. Startup hashes
the complete language model and projector. Loading, hashing, image encoding and
language-model prefill have separate timing fields.

See the [CLI reference](cli.md) for image options, text prefill batch sizes and
numerical dumps. `vision-encode` produces embeddings without language-model
inference and takes no context option. Validation commands and independent
reference tools are in [testing](testing.md#vision-checks).

## Conversations and caches

`wire` and `serve` enable protocol v2 when opened with a projector. This
supports images in later user turns and tool results, retained media, retrieval,
rewind, rebuild and ephemeral requests. Each message carries at most one image;
multiple images use separate messages. See the [wire protocol](protocol.md)
for request formats, limits and capability negotiation.

Durable conversations retain original encoded media. Source files are no
longer needed after append. Projected embeddings and KV checkpoints are derived
caches: missing or incompatible entries are rebuilt from retained media and
the current compatible model artifacts. Source integrity or preprocessing
mismatches fail explicitly; missing media cannot become an empty image.

The checkpoint budget defaults to 16 GiB. Projected embeddings have a separate
2 GiB disk limit. A zero cache budget disables reads and writes for both disk
caches, preserving existing files and transcript media. Vision services prepare
weight and numerical identities at startup even when disk caching is disabled.

Visual replay is synchronous. The runtime's asynchronous shadow prefill
optimization is disabled for visual prompts. The internal session implementation
supports shadows, copy-on-write (COW) and promotion with owned image data; these
operations are declared in [xenolith_internal.h](../xenolith_internal.h).

## Mixed input and session contract

Logical input is a sequence of text token IDs and visual spans. Each visual
span has a logical start, 1..1120 valid projected F32 rows of width 2816, and a
32-byte identity. Visual positions use the model's image placeholder token;
the token alone does not identify the image. Image start/end delimiters are
ordinary tokens outside the span.

Span order is meaningful and spans cannot overlap. Partial spans cannot be
submitted, reused, checkpointed or rewound into. Visual spans are processed
whole, independently of the text prefill batch size.

Compute padding occupies no logical positions. It must not affect RoPE,
attention keys, routing, KV commits, usage, prefix hashes or persisted token
counts. Kernels may round launch dimensions but must bound all reads and
writes by the valid row count.

The session owns a copy of visual data; shadows share it read-only. Appending
images can preserve an identical computed prefix. Changed images or edited
continuations use conservative full replay. Rewind accepts complete-image
boundaries. Token-only sync/common APIs reject visual sessions because token
IDs cannot establish image equivalence. See [xenolith.h](../xenolith.h) for
mixed-input declarations, [vision.h](../vision.h) for embedding buffers, and
the [C API guide](api.md) for engine/session lifetimes.

## Content and numerical identity

Image preprocessing identity is SHA-256 over a version tag, the encoded bytes'
SHA-256, oriented RGB pixels with width/height, requested budget, resized-pixel
SHA-256, resized dimensions and valid output count. The version tag binds EXIF
orientation, white alpha compositing, budget geometry and bicubic filtering.
Visual identity additionally binds the full language-model and projector
SHA-256 hashes, encoder numerical revision and projected F32 embedding bytes.
Integers and F32 bytes use little-endian encoding; strings are length-prefixed.

The computed-input hash also includes media, budget and embedding bytes.
Keeping a caller-supplied identity unchanged cannot conceal altered data.
Paths, timestamps and memory addresses do not identify content.

Prefix comparison checks text IDs and visual spans at their logical offsets.
A changed image invalidates computation from its start; a prefix ending inside
a span is invalid. Encoder variants producing different embedding bytes cannot
silently share computed caches. Encoder revision changes require re-encoding
and a new identity.

Full GGUF hashes use the in-tree SHA-256 implementation in `format.c`, with
Intel SHA acceleration when available and an equivalent portable C fallback.
Weight changes select distinct derived caches and replay retained media.

The runtime numerical fingerprint uses domain `xenolith-visual-numeric-v4` and
binds encoder revision, source, compiler/flags, SPIR-V, device UUID, driver
version and the compiled native GPU module. It excludes the driver UUID,
which contains a per-initialization timestamp on Intel, as well as file
timestamps/inodes. Recorded embedding identity is compared only when both
weight identities and numerical identity match. Media integrity and
preprocessing checks always apply.

## Persistence formats

### Portable input archives

`xe_input_save/load` serialize mixed input, original media and projected rows
as `XEVINPUT` version 1. Integers and F32 values are little-endian; the header
contains the computed-input SHA-256. Limits are 64 MiB per encoded image,
256 MiB total media, 2 GiB on archive load and 131072 logical elements.
`xe_image_load_memory` reconstructs preprocessing from retained media. Projected
rows remain derived data and need re-encoding after an encoder revision change.

### Conversation media and embeddings

The persistent runtime stores one `CONVERSATION_BLOCK_IMAGE` (type 4) alongside
the text block in an image-bearing user message or tool result. `XEVIMG1\n`
records contain a 288-byte little-endian header followed by original encoded
media. The header stores the span offset within that message's rendered tokens,
rows, budget, original/resized geometry, media length, source/RGB/resized hashes,
visual identity, projector/model fingerprints and numerical fingerprint.
Reserved bytes must be zero.

These blocks are authoritative media. Projection recomputes absolute span
offsets from visible message boundaries, including after reasoning is discarded,
rewind and rebuild.

Runtime embedding files (`XEVEMB01`) have a 160-byte header plus F32 rows. Their
key binds preprocessing identity, both weight hashes and numerical fingerprint.
Header CRC64, row SHA-256, finite values and computed visual identity are checked
before reuse. Publication is atomic and eviction uses last access. Missing or
corrupt embeddings are re-encoded from media.

### KV snapshots and grouped checkpoints

Visual `XENXKV` snapshots use version 3 with the same six KV sections as text
snapshots. Header flag at byte 24 is 1; bytes 320..351 contain the complete
computed-input hash, bytes 352..355 contain the text numerical ABI, and bytes
356..375 are reserved zero. MODEL binds the full language-model GGUF SHA-256.
Restore requires `xe_session_snapshot_load_input` and matching mixed input;
token-only load rejects visual snapshots. Saving a COW shadow is rejected;
promote it first.

Text snapshots use version 2 with a stable numerical ABI and diagnostic fields.
Visual version 2 is recognized by flag 1 and rejected for numerical compatibility;
the runtime re-encodes and replays retained media without changing history.
Legacy text version 1 requires its audited compatibility check; changed kernels
do not qualify for that bridge.

Grouped visual checkpoints use the `XEKVSEG` disk envelope and content-addressed
`.xkc` segments for global KV streams. Prefix checks bind the complete mixed
input, omit images after the checkpoint boundary and reject boundaries inside
images. Reserve pruning occurs only after the transcript durably references
the new checkpoint. Restore tries older compatible checkpoints when the latest
one is missing or rejected.
