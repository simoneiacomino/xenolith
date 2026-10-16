# Multimodal input contract, version 1

Logical input is a sequence of text token IDs and visual spans. Token IDs for
visual elements use the model's image placeholder, but a placeholder alone
does not identify its content. A visual span has its logical start, valid
length (1..1120), projected F32 rows of width 2816, and a 32-byte identity.
Image start/end delimiters are ordinary tokens outside the visual span.
Span order is meaningful. Spans cannot overlap; partial spans cannot be
submitted, reused, checkpointed, or rewound into.

Compute padding is not part of this sequence. It must not affect positions,
RoPE, attention keys, routing, KV commits, usage, prefix hashes or persisted
token counts. Kernels may round launch dimensions but must bound all reads
and writes by the valid row count.

The canonical image identity is SHA-256 over a domain/version tag, SHA-256 of
original encoded bytes, SHA-256 of decoded RGB pixels plus width/height,
preprocessing revision (EXIF transpose, alpha composited on white, HF budget geometry, Pillow bicubic), requested budget, resized-pixel SHA-256, actual resized dimensions, valid
output count, text-model and mmproj fingerprints, and numeric encoder revision.
Integers use unsigned little-endian encoding and strings use a length prefix.
The visual identity hashes projected F32 embedding bytes in little-endian order.
The computed-input hash also hashes the media, budget and embedding bytes,
so an unchanged caller-supplied identity cannot conceal altered data. Paths, timestamps and memory addresses do not identify content.

Prefix comparison checks text IDs and visual span identities at their
logical offsets. Any changed visual span invalidates computation from its
start. A prefix ending inside a visual span is invalid. CPU/GPU encoder
variants producing different embedding bytes cannot silently share computed
cache entries. An encoder revision change invalidates the affected cache.

Persistence stores original encoded media, decoded/preprocessing metadata,
identities and the ordered span map. Snapshots additionally bind the complete
computed-input identity and numerical fingerprint. Embedding caches are
disposable derived data; replay must succeed from retained media and compatible
model artifacts. Missing media is an explicit error, never an empty image.

The CLI uses an immutable visual prompt followed by text-only decode. The C
session owns a copy of its visual data; shadow sessions share it read-only.
Appending images can preserve an identical previously computed prefix.
Changed images or edited continuations use conservative full replay. Rewind
accepts complete-image boundaries. Legacy token-only sync/common APIs reject
visual sessions, because token IDs alone cannot establish image equivalence.

`xe_input_save/load` serialize the input, original media and projected rows in
an archive (`XEVINPUT`, version 1). Archive integers and F32 values use little
endian. The header contains the computed-input SHA-256. Limits are 64 MiB per
encoded image, 256 MiB total media, 2 GiB archive and 131072 logical elements.
`xe_image_load_memory` can reconstruct preprocessing from retained media. The
archive's projected rows are a derived cache; an encoder revision change
requires re-encoding and a new identity.

Visual KV snapshots use version 3 of `XENXKV`: the six existing KV sections
are unchanged. Header flag at byte 24 is 1, bytes 320..351 contain the complete
computed-input hash, bytes 352..355 contain the text numerical ABI, and
bytes 356..375 are reserved zero. MODEL binds the full language-model GGUF SHA-256. Loading requires
`xe_session_snapshot_load_input` and the matching mixed input. Token-only
load rejects these snapshots. Text snapshots use upstream version 2 with its
stable numerical ABI and diagnostic fields. The older visual version 2 is
recognized by flag 1 but conservatively rejected for numerical compatibility;
the runtime re-encodes/replays retained media without changing history. Legacy
text version 1 retains upstream's audited compatibility check; changed kernels
do not qualify for that bridge. Checkpointing
a COW shadow is rejected; promote it before saving.

Grouped visual checkpoints use upstream's `XEKVSEG` disk envelope and
content-addressed `.xkc` segments for the global KV streams. Prefix checks bind
the complete mixed input, omit images following the checkpoint boundary and
reject boundaries inside an image. Reserve pruning occurs only after the
transcript durably references the new checkpoint. Restore tries older compatible
checkpoints when the latest one is missing or rejected.

`--cache-budget` bounds checkpoint files/shared segments (default 16 GiB).
Projected embedding files remain a separate rebuildable cache capped at 2 GiB.
Setting the budget to zero disables disk reads/writes for both caches while
preserving existing files and durable transcript media.

The persistent runtime stores one `CONVERSATION_BLOCK_IMAGE` (type 4) alongside
the text block in user messages and tool results. `XEVIMG1\n` records contain a
288-byte little-endian header followed by original encoded media. The header
stores the span offset within that message's rendered tokens, rows, budget,
original/resized geometry, media length, source/RGB/resized hashes, visual
identity, mmproj/model fingerprints and numerical fingerprint. Reserved bytes must
be zero. Blocks are authoritative media; embeddings are separate derived data.
Projection recomputes absolute span offsets from visible message boundaries,
including after reasoning is discarded, rewind and rebuild.

Runtime embedding files (`XEVEMB01`) have a 160-byte header plus F32 rows.
Their key binds preprocessing identity, both weight hashes and numerical
fingerprint. Header CRC64, row SHA-256, finite values and computed visual
identity are checked before reuse. Publication is atomic and the 2 GiB cache
evicts by last use. Missing or corrupt derived data is re-encoded from media;
source integrity and preprocessing mismatches fail explicitly. Weight changes
select distinct derived caches and replay the retained media. Runtime media limits
are 48 MiB per image and 256 MiB per active context; the low-level C archive
retains its 64 MiB per-image limit. Source files are never needed after append.

The numerical fingerprint binds source, compiler/flags, SPIR-V, device UUID,
driver version and the actual compiled native GPU module. The driver UUID is
deliberately excluded: Intel includes an initialization timestamp, which changes
across processes and would prevent persistent reuse. The native module binds
compiler output while remaining stable across validated restarts.

The opt-in wire protocol v2 exposes images, media references and retrieval;
see [PROTOCOL.md](PROTOCOL.md). Protocol v1 remains text-only. Runtime visual
replay is synchronous; its asynchronous shadow prefill optimization is disabled
for visual prompts. Core shadow/COW and promotion still support owned images.

Language-model and projector identities are SHA-256 of the entire original
GGUF, computed with the in-tree implementation from `format.c`. Intel SHA
instructions accelerate the same digest; the portable C fallback preserves
identical semantics. OpenSSL and `XENOLITH_WEIGHTS_VERIFY` are no longer used.

The engine validates architecture, metadata required by the implementation, and
all tensor types/shapes before enabling vision. Reference hashes in the manifest
are diagnostic: compatible variants load with a notice, and their full hashes
isolate embeddings and KV. This does not validate model/projector training
compatibility or output quality. The weight files must remain unchanged while
an engine uses them.

Visual numerical identity uses domain `xenolith-visual-numeric-v4`; verification
mode and file timestamps/inodes are no longer inputs. Old computed state is
invalidated, while original media can be replayed under the current weights.
Recorded embedding identity is compared only when both weight identities and
the numerical identity match. Preprocessing and media integrity remain mandatory.
For services, vision identities are prepared at startup even with a zero disk
cache budget; text-only services follow `--cache-budget` as described in
[PROTOCOL.md](PROTOCOL.md).

## Validation

Run commands from the repository root. The engine has no Python or llama.cpp
runtime dependency. Python 3 is used by the checks; Pillow and NumPy are optional
dependencies for independent reference generation. The recorded versions,
source hashes and artifact identities are in [the manifest](tests/vision/manifest.json).

`tests/vision/` holds checks, fixtures and reference metadata. `tools/` holds
artifact verification, fixture generation, preprocessing and numerical oracles.
`tests/vision/results/` contains historical reports for the revisions recorded
inside them; a report named `final` does not certify the current checkout.
Use a new output directory for each validation run. The Python integration
checks refuse existing directories so they cannot reuse old success reports or
modify an existing session/cache.

### Host and GPU checks

```sh
make check-unit check-context
make check-gpu
```

The first command needs no model or GPU device. The second needs a supported
GPU, but no weights. `make check-vision` is a focused shortcut for the image
input and GPU attention/ring checks already included in those suites.

For preprocessing comparison, build `tools/vision_prepare`, obtain the pinned
Transformers reference files with `python3 tools/vision_artifacts.py --references`,
and install the manifest's Pillow version in a development environment. The
artifact utility verifies local files; add `--download` to download missing
projector/reference files into its `--directory` (default `.vision-cache`).

```sh
make tools/vision_prepare
python3 tests/vision/check_preprocess.py --output /tmp/xe-preprocess-new
```

This compares EXIF orientation, alpha compositing, resizing and all five image
budgets, with a separate JPEG pixel tolerance. `python3 tools/vision_fixtures.py`
regenerates reference fixtures; do not regenerate committed fixtures merely to
make a candidate pass.

### Model and session checks

Set `MODEL` and `MMPROJ` to the pinned GGUF files. The scripts create their own
state/cache under the supplied fresh output directory:

```sh
python3 tests/vision/check_wire.py --engine ./xenolith \
  --model "$MODEL" --mmproj "$MMPROJ" --output /tmp/xe-wire-new
python3 tests/vision/check_wire.py --engine ./xenolith \
  --model "$MODEL" --mmproj "$MMPROJ" --large --output /tmp/xe-wire-large-new
python3 tests/vision/check_chat.py --engine ./xenolith \
  --model "$MODEL" --mmproj "$MMPROJ" --output /tmp/xe-chat-new
make check-text-batch-model MODEL="$MODEL" TEST_CONTEXT=65536
```

The wire suite covers image transport, retained media, multiple images, rewind,
rebuild, ephemeral requests, checkpoints and restart. `--large` selects the
separate 131072-context/1120-row checkpoint and restart scenario. Chat uses the
included quadrant image and checks both answers, plus two text-only turns.
The text batch check compares prefill sizes 512 and 1024; repeat it with
`TEST_CONTEXT=131072` for the larger capacity.

For C session lifecycle checks, produce a matching `prompt.ids` and
`embedding.f32` using `xenolith run` with `--image`, `--mmproj` and
`--vision-dump DIR` (create `DIR` first). Then run:

```sh
make check-vision-session-model MODEL="$MODEL" TEST_CONTEXT=65536 \
  VISION_PROMPT_IDS=/path/to/dump/prompt.ids \
  VISION_EMBEDDING=/path/to/dump/embedding.f32
```

The prompt must contain the matching image span. This check exercises identity,
snapshot restore, shadow ownership, promotion, replay and rewind without
rerunning the encoder. It is distinct from encoder numerical validation.

`make check-weights-identity` takes `MODEL`, `MMPROJ`, `MMPROJ_VARIANT`
(a compatible projector with changed tensor bytes) and `MMPROJ_INVALID`
(invalid projector metadata). It needs the pinned originals and separate test
variants, but no GPU device. Never mutate the original weight files.

`python3 tests/vision/check_cache.py --help` describes the migration check. Its
`--previous-engine` must be a pre-integration build that writes the older
visual-v2 snapshots; an arbitrary second build is not a valid substitute.
The script checks replay into the current format, predecessor recovery and
disabled caches using disposable state.

### Independent numerical references

The C++ encoder and decoder oracles require two separate llama.cpp checkouts
at `references.llama_cpp_commit` from the manifest. Apply
`python3 tools/vision_reference_patch.py /path/to/encoder-checkout` only to the
encoder copy: its F32 GELU adapter is necessary for the pinned vision model.
Keep the decoder copy unmodified. Build shared CPU libraries in each checkout's
`build-cpu`, including `mtmd` for the encoder. Use the manifest's
`ggml_cpu_options` for reproducible reference builds. Then compile the wrappers:

```sh
make vision-reference-tools \
  LLAMA_VISION_DIR=/path/to/encoder-checkout \
  LLAMA_DECODE_DIR=/path/to/decoder-checkout
```

Override `LLAMA_VISION_BUILD` and `LLAMA_DECODE_BUILD` for other build directories.
The targets check the pinned revisions, GELU source variant and source/build
association. These checks do not replace rebuilding libraries after source
changes. Neither target downloads dependencies nor changes their sources.

`tools/vision_prepare IMAGE BUDGET prepared.rgb` produces the resized pixels
consumed by `tools/vision_reference MMPROJ prepared.rgb OUTPUT_DIR` and
`python3 tools/vision_numpy_reference.py MMPROJ prepared.rgb OUTPUT_DIR`. Compare encoder
dumps with `python3 tools/vision_compare.py REFERENCE_DIR CANDIDATE_DIR --output REPORT`;
its default checks all 30 selected stages with fixed RMS/cosine limits.
The decoder oracle accepts `MODEL IDS EMBEDDING START ROWS CONTEXT OUTPUT_PREFIX`;
use the same prompt/span and projected rows as the candidate.

For timing, use `python3 bench/text_regression.py --help`. It records the text
matrix or paired ABBA runs, executable identities and commands. Successful
measurement collection is not a numerical or performance regression verdict.
