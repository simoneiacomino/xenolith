# Command-line reference

```text
xenolith <command> <model.gguf> [options]
```

The model path must follow the command; options follow the model path. Run
`./xenolith --help` to print the usage summary (it exits with status 1).

| Command | Purpose | Shared options |
|---|---|---|
| `info` | Print model metadata | Context |
| `tokenize` | Encode text or decode token IDs | None |
| `run` | Generate from one prompt | Context, CPU, sampling |
| `chat` | Interactive conversation | Context, CPU, sampling |
| `oracle` | Run the CPU reference forward pass | Context, CPU |
| `bench` | Measure synthetic prefill and decode | Context, CPU |
| `vision-encode` | Encode an image and write numerical dumps | CPU |
| `wire` | Serve the wire protocol over stdin/stdout | Context, CPU |
| `serve` | Serve the wire protocol over a Unix socket | Context, CPU |

## Context and CPU options

| Option | Values | Default |
|---|---|---|
| `--ctx N` | Decimal integer, 64–262144 | 262144 for text; 65536 with image input or `--mmproj` |
| `--workers N\|auto` | Positive decimal integer or `auto` | Automatic selection, subject to environment fallback below |
| `--cpu-list id,id,...` | Ordered decimal logical CPU IDs; no spaces or ranges | Selected by the engine |

Each of these options may appear only once. A CPU list sets the worker count;
an accompanying numeric `--workers` must equal its length. `--workers auto`
cannot be combined with `--cpu-list`.

With neither CPU option present, the CLI reads `XE_WORKERS` (a positive integer,
without `auto`) and `XE_CPU_LIST` (the same list syntax). If both are set, the
count must equal the list length. With **either CPU option present, both
environment variables are ignored**, including malformed values.

```sh
./xenolith run /path/to/model.gguf --ctx 16384 --workers 4 -p "Hello" -n 128
./xenolith run /path/to/model.gguf --workers auto -p "Hello"
```

Explicit CPU IDs must be available to the engine and belong to distinct
physical cores; repeated IDs, SMT siblings and unknown topology are rejected.
A numeric worker count bypasses the catalog and selects distinct physical cores
in ascending CPU ID order; it must fit the available cores. Use a CPU list to
control core membership and order. CPU affinity must be permitted by the host
or container. The engine prints its chosen
worker count and ordered CPU map at startup; the selection lasts until it
closes. See [tuning](tuning.md) for automatic profiles and selecting a manual
map, and the [C API](api.md) for embedded configuration.

`info` and `tokenize` ignore the CPU environment variables and reject the CPU
options. `tokenize` and `vision-encode` reject `--ctx`.

## Sampling options

`run` and `chat` accept these options:

| Option | Meaning and accepted range | Default |
|---|---|---|
| `-n N` | Maximum generated tokens; nonnegative integer, per reply in `chat` | 512 |
| `--temp F` | Finite, nonnegative temperature; `0` selects greedily | 1.0 |
| `--top-k N` | Integer from 0 to vocabulary size; `0` removes the top-k limit | 64 |
| `--top-p F` | Finite nucleus probability, greater than 0 and at most 1 | 0.95 |
| `--seed S` | Unsigned 64-bit integer seed | Current time in seconds |

Generation also stops at an end token or the context limit. Prompts and chat
formatting must fit the selected context; the CLI does not truncate them.

## `info`

```sh
./xenolith info /path/to/model.gguf --ctx 16384
```

Prints model metadata to stdout. This command opens the full engine, including
weights and GPU initialization; use `tokenize` for vocabulary-only work.

## `tokenize`

```text
xenolith tokenize model.gguf (-p TEXT | -f FILE) [--pieces]
xenolith tokenize model.gguf --detok
```

Use exactly one of `-p` and `-f`. Encoding prepends BOS and writes comma-separated
token IDs to stdout, with `n_tokens N` on stderr. `--pieces` instead writes one
token per line as its decimal ID, a tab and the token piece.

`--detok` reads IDs from stdin, separated by commas or whitespace, and writes
decoded bytes to stdout. IDs must be in 0–262143. It enters decoding immediately
when encountered, so place it directly after the model path. Both modes open
the vocabulary without loading model weights.

```sh
./xenolith tokenize /path/to/model.gguf -p "Hello" --pieces
printf '2,9259\n' | ./xenolith tokenize /path/to/model.gguf --detok
```

## `run`

```sh
./xenolith run /path/to/model.gguf -p "Hello" -n 64 --temp 0
```

Requires `-p TEXT`. Text-only input is BOS followed by the supplied text, without
a chat template. Generated text goes to stdout. Use `chat` for conversation
formatting.

`--prefill 512|1024` selects the text prefill batch size (default 512). Image
input uses `--mmproj FILE --image FILE`, optionally `--image-tokens N` and
`--vision-dump DIR`; see [image options](#image-options). `--vision-dump` requires
an image. A zero generation limit is accepted; with an image it still performs
visual prefill and can produce dumps.

## `chat`

```sh
./xenolith chat /path/to/model.gguf -n 256
```

Reads one user message per line and retains the conversation in memory. Type
`/exit` or send EOF to leave. Sampling options apply to every reply. This command
does not persist the conversation; use `wire` or `serve` for durable sessions.

For an initial image, pass `--mmproj FILE --image FILE` and optionally
`--image-tokens N`. The image attaches to the first user message and remains in
the conversation for follow-up questions. `chat` accepts one initial image;
additional images require the [wire protocol](protocol.md). It does not accept
`--prefill` or `--vision-dump`.

## Image options

| Option | Accepted by | Meaning |
|---|---|---|
| `--mmproj FILE` | `run`, `chat`, `vision-encode`, `wire`, `serve` | Vision projector GGUF |
| `--image FILE` | `run`, `chat`, `vision-encode` | PNG or JPEG input |
| `--image-tokens N` | `run`, `chat`, `vision-encode` | Budget: 70, 140, 280, 560 or 1120; default 280 |
| `--vision-dump DIR` | `run`, `vision-encode` | Directory for numerical outputs |
| `--final-only` | `vision-encode` | Write only `embedding.f32` |

`run` and `chat` require `--image` and `--mmproj` together. With an image, or with
`--mmproj` on `wire`/`serve`, context defaults to 65536 and only 65536 or 131072
is accepted. `vision-encode` has no context option. Create the dump directory
before invoking a dump command.

```sh
./xenolith run /path/to/model.gguf --mmproj /path/to/mmproj-F16.gguf \
  --image photo.jpg -p "What is shown?" -n 128
```

See [vision](vision.md) for supported artifacts, image preparation, limits and
visual session behavior.

## `vision-encode`

```sh
mkdir -p vision-dump
./xenolith vision-encode /path/to/model.gguf \
  --mmproj /path/to/mmproj-F16.gguf --image photo.jpg \
  --vision-dump vision-dump --final-only
```

Requires `--mmproj`, `--image` and `--vision-dump`. It accepts the CPU options and
`--image-tokens`, encodes the image without loading language-model weights, and
prints `vision rows N width W` to stdout. By default it also writes intermediate
encoder dumps. `--final-only` writes only the projected float32 embeddings to
`DIR/embedding.f32`. Encoding timings go to stderr.

## `oracle`

```text
xenolith oracle model.gguf -t "id,id,..." [-o logits.bin] [--layers DIR] [--q8]
```

Requires 1–8192 token IDs in 0–262143, also bounded by `--ctx`. Supply BOS yourself
if needed. The CPU reference pass prints the eight highest final-token logits.
`-o FILE` writes the full final logits as raw floats; `--layers DIR` writes
intermediate arrays for the final token and creates the directory if needed
(its parent must exist). `--q8` enables the reference path's Q8 activations and
FP16 rounding; it is off by default.

## `bench`

```sh
./xenolith bench /path/to/model.gguf -p 512 -n 128 -r 3 -o jsonl
```

| Option | Meaning | Default |
|---|---|---|
| `-p N`, `--n-prompt N` | Synthetic prefill tokens; `0` skips this test | 512 |
| `-n N`, `--n-gen N` | Synthetic decode tokens; `0` skips this test | 128 |
| `-d N`, `--n-depth N` | Untimed context prefix before each measured run | 0 |
| `-r N`, `--repetitions N` | Measured repetitions, 1–1000 | 5 |
| `--prefill 512\|1024` | Text prefill batch size | 512 |
| `--delay S` | Integer seconds before each test's warmup, 0–3600 | 0 |
| `--no-warmup` | Skip warmup | Warmup enabled |
| `--progress` | Print progress to stderr | Off |
| `-o md\|jsonl`, `--output md\|jsonl` | Output format on stdout | `md` |

Prompt, generation and depth counts must be nonnegative integers. At least one
of prompt or generation must be nonzero; depth plus either count must fit the
context. Prefill and decode are separate tests, each starting from a reset
session and the selected depth. Decode uses synthetic token IDs rather than
sampled text. See [benchmark comparisons](../bench/compare_pp_tg.md) for a
reproducible comparison protocol.

## `wire` and `serve`

```sh
./xenolith wire /path/to/model.gguf
./xenolith serve /path/to/model.gguf --idle-shutdown 15
```

`wire` exchanges newline-delimited JSON over stdin/stdout. `serve` exposes the
same protocol on a Unix socket. See [protocol](protocol.md) for requests,
events, connection handling and persistence semantics.

| Option | Accepted by | Meaning / default |
|---|---|---|
| `--state DIR` | Both | Durable conversation storage; default below |
| `--cache DIR` | Both | Disk KV cache directory; default below |
| `--cache-budget SIZE` | Both | KV checkpoint and shared-segment budget; default `16GiB`; `0` disables disk caches |
| `--mmproj FILE` | Both | Enable image protocol v2; otherwise text protocol v1 |
| `--socket PATH` | `serve` | Unix socket path; default below |
| `--idle-shutdown MIN` | `serve` | Finite positive idle timeout in minutes; disabled by default |

`SIZE` is an unsigned integer byte count, optionally followed by the exact
suffix `B`, `kB`, `MB`, `GB`, `KiB`, `MiB` or `GiB`. Decimal units use powers of
1000; binary units use powers of 1024. Fractions, signs and whitespace are
rejected; the result must fit an unsigned 64-bit integer.

| Path | Default | Fallback |
|---|---|---|
| State | `$XDG_STATE_HOME/xenolith` | `$HOME/.local/state/xenolith` |
| Cache | `$XDG_CACHE_HOME/xenolith/kv` | `$HOME/.cache/xenolith/kv` |
| Socket | `$XDG_RUNTIME_DIR/xenolith/wire.sock` | `wire.sock` inside the selected state directory |

`XDG_STATE_HOME` and `XDG_CACHE_HOME` must be absolute; relative values use the
`HOME` fallbacks. `XDG_RUNTIME_DIR` must be absolute and its socket directory
usable, otherwise the state fallback is used. Pass an absolute `--state` path;
absolute cache and socket paths also avoid dependence on the launch directory.
The parent of an explicit socket path must already exist.

A positive cache budget computes the complete model GGUF SHA-256 before
repacking, even if the cache directory later fails to open. With `--mmproj`,
startup also prepares the full model and projector hashes with a zero budget;
the model hash is computed before repacking and reused by vision.

For text-only services, `--cache-budget 0` skips model hashing. A zero budget
disables KV and projected-embedding disk cache reads and writes, while preserving
durable history, retained image media and existing cache files. Projected
embeddings have a separate 2 GiB budget when disk caching is enabled.
Standalone `chat`, `run`, `info`, `oracle` and `bench` skip model hashing when
used without images.

Commands that load weights or encode images share an exclusive `engine.lock`
in the default state directory. A custom `--state` or `--socket` does not move
that lock. Close the current engine before starting another such command.

## Launching through XenoPi

When XenoPi starts a new Xenolith process, it inherits the environment fallback
described above. With the binary and model already configured, launch `xenopi`
without CPU overrides for automatic selection, or set a count:

```sh
XE_WORKERS=6 xenopi
```

Use `XE_CPU_LIST` instead to supply CPU IDs verified on that machine. XenoPi
currently has no worker-count or CPU-map field in `xenolith.json`. If a service
is already running on the socket, XenoPi uses its existing CPU plan. Restart
Xenolith to apply a new plan; restarting only XenoPi does not change it.

## Experimental activation steering

`run` and `chat` accept `--steering FILE --steering-strength F`, alongside
`--workers` and `--cpus`. Chat supports `/steer F` and `/steer off`.
`wire` and `serve` accept `--steering-dir DIR` for a local artifact registry.
See [the steering guide](../STEERING.md) for `steering-capture`,
`steering-import`, supported modes/sites, and cache semantics.
