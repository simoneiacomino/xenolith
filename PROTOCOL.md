# Xenolith wire protocol

This document describes protocol version 1 as implemented by [serve.c](serve.c).
It is a local, text-only protocol for one served model. A client can maintain a
durable conversation by appending turns, or send a complete transcript for a
temporary generation. Xenolith generates tool calls; the client executes tools.

## Connection and framing

Start a Unix socket service:

```sh
./xenolith serve /path/to/model.gguf --socket /path/to/wire.sock
```

Or use stdin for requests and stdout for replies:

```sh
./xenolith wire /path/to/model.gguf
```

Both commands accept `--state DIR` and `--cache DIR`. `serve` also accepts
`--idle-shutdown MINUTES`, a positive number; by default it stays running.
Use absolute directory paths; the parent of an explicit socket path must exist.
Without `--socket`, the socket is `$XDG_RUNTIME_DIR/xenolith/wire.sock` when
available, otherwise `wire.sock` inside the state directory. Default state and
cache directories are `$XDG_STATE_HOME/xenolith` and `$XDG_CACHE_HOME/xenolith/kv`,
falling back to `~/.local/state/xenolith` and `~/.cache/xenolith/kv`.

Each frame is one JSON object followed by a newline. Escape newlines inside
strings; do not pretty-print a frame across lines. There are no request IDs or
unsolicited greetings. Diagnostics go to stderr. Read continuously while a
generation is running, and keep the connection's write side open if you intend
to send more commands or cancel.
Avoid duplicate object keys. The current parser rejects embedded NUL characters
(including escaped `\u0000`) and nesting deeper than 64 containers. An unterminated
final line is not a frame.

`describe.max_frame` gives the byte limit for a frame. Keep the serialized JSON
plus its terminating newline within that limit. An oversized request can return
`invalid_request` and close the connection; an oversized response or a client
that does not drain output can also cause a disconnect. There is no chunked
request upload or pagination for `history` and `list`.

Use one outstanding operation per connection, except for `cancel` during a
generation. The service runs one generation at a time and queues work from
other connections. `describe`, `list` and `stat` can be handled between generation
steps on other connections. On the generating connection, commands other than
`cancel` receive `busy`. Cancellation is cooperative, not an interrupt of a
running kernel.

Requests use an `op` string. Ordinary replies have `ok: true` and operation
fields, or `ok: false`, `code` and `error`. Successful `generate` and `ephemeral`
requests enter the event stream directly: **there is no initial `ok: true` ACK**.
A rejected generation instead returns an ordinary `ok: false` reply.

## Discover the service

Send `{"op":"describe"}` first. The reply contains:

| Field | Meaning |
|---|---|
| `ok` | `true`. |
| `protocol` | `1`. Reject versions your client does not support. |
| `model` | Served model identifier; there is no model-selection field on requests. |
| `context_window` | Maximum context size in tokens. |
| `max_output` | Advertised output limit. Available context can impose a lower limit. |
| `max_frame` | Maximum frame size in bytes. |
| `kvstore` | Whether the snapshot store is available in this process. |
| `reasoning` | Supported `efforts`, `history` modes and boolean `budget_tokens` support. |

The current reasoning capabilities are efforts `low`, `medium`, `high`, `max`;
history modes `discard`, `preserve_tool_calls`; and support for explicit budgets.
Read these capabilities rather than deriving them from the model name.

## A conversation

Send these requests in order, waiting for each reply or terminal event:

```json
{"op":"create","system":"Answer concisely."}
{"op":"append","role":"user","text":"Hello"}
{"op":"generate","max_tokens":128}
```

`create` returns a session ID and binds that session to the connection. `append`
returns a marker. `generate` streams events until `done` or `error`. For another
turn, append only the new message and generate again. After reconnecting, use
`open` with the saved session ID before session operations.

A session ID is 32 lowercase hexadecimal characters. Markers are opaque,
session-local positions in the persisted record, not token counts or indexes
into the returned history array. Reuse returned markers rather than calculating
them. A connection has at most one bound session; `create` or `open` replaces
that binding. Multiple connections can bind the same session: binding is not an
exclusive edit lock. Clients sharing a session must coordinate their changes.

## Requests

Every request below includes `op`. Optional fields may be omitted. Reply fields
in the table are in addition to `ok: true`; “empty” means `{"ok":true}`.

| Operation | Request fields | Successful reply / effect |
|---|---|---|
| `describe` | None | Service capabilities above. |
| `create` | Optional `system` string, `tools` array | `session`, `marker`; creates and binds a durable conversation. |
| `open` | `session` | `tokens`, `marker` or `null`, `turn_open`, `zero_prefill`, `resume`, `pending`; binds an existing conversation. |
| `stat` | `session` | `session`, `title`, `tokens`, `resumable`, `created`, `updated`. Does not change the binding. |
| `list` | None | `sessions`: array of `{session,title,tokens,resumable}`. |
| `delete` | `session` | Empty; deletes the conversation and clears the caller's binding if it refers to it. Returns `busy` if another connection has it bound. |
| `append` | `role: "user"`, `text` | `marker`; adds a user message to the bound session. |
| `append` | `role: "tool"`, `call_id`, `text`, optional `status: "ok"` or `"error"` | `marker`; records a tool result. Status defaults to `ok`. |
| `generate` | Optional generation settings below | Event stream using the bound conversation. |
| `ephemeral` | Optional `system`, `tools`, `messages`, generation settings | Event stream for a temporary context; does not append to a durable session. `done.marker` is `null`. |
| `rebuild` | Optional `system`, `tools`, `messages` | `marker`; replaces the bound session's active transcript with the supplied content and invalidates its snapshot epoch. Does not generate. |
| `rewind` | `marker` | Empty; restores the bound conversation through that boundary. |
| `rewind_cost` | `marker` | `prefill`: estimated tokens to compute to restore that boundary, given current state/snapshot references. |
| `checkpoint` | None | `saved`, `tokens`, and `reason` when not saved; attempts a snapshot of the bound session. |
| `pending` | None | `calls`: unresolved tool-call IDs in the bound session. |
| `history` | None | `entries`: the bound session's visible history. |
| `cancel` | None | Empty ACK for an active or queued generation on this connection; see cancellation below. |

`append`, `generate`, `rebuild`, `rewind`, `rewind_cost`, `checkpoint`, `pending`
and `history` require a bound session. There is no separate `close` operation;
disconnecting releases the binding without deleting the conversation.

For `open`, `tokens` is the projected transcript size and `turn_open` says whether
the model turn is still open. `resume` is `snapshot`, `stale` or `none`.
`zero_prefill` and `resume` describe snapshot references, not a guarantee that the
file will load successfully. The generation's optional `resume` report gives the
actual load result. `resumable` in summaries describes whether the persisted
record can be resumed, independently of cached KV; it does not promise zero work.
`created` and `updated` in `stat` are Unix timestamps in nanoseconds.

The current `open.pending` array is capped at 64 IDs and `pending.calls` at 256;
neither includes a truncation flag. Pending means an outcome has not been
recorded, not proof that the tool has never run. Reconcile with the client's tool
execution record before repeating an operation after a disconnect.

### Tools and imported messages

A tool declaration has a required `name` and `description`, plus an optional
`parameters` JSON object. Schemas must be supported by the served model's
renderer; support for arbitrary JSON Schema is not implied.

```json
{"name":"read_file","description":"Read a text file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}
```

`messages` in `rebuild` and `ephemeral` is an ordered array with these shapes:

```json
{"role":"user","text":"Read notes.txt"}
{"role":"assistant","text":"","calls":[{"name":"read_file","arguments":{"path":"notes.txt"}}]}
{"role":"tool","tool":"read_file","text":"Contents of the file","status":"ok"}
```

Assistant messages may also contain a `reasoning` string. Put the system message
in the top-level `system` field, not in this array. Tool results need an open tool
turn. For `ephemeral`, the tool name and text are required. For `rebuild`, tool
results are matched to pending imported calls by tool name, or the first pending
call if no name is supplied. Preserve result order when several calls use the
same tool name. Imported `call_id` values are not preserved:
rebuild creates new IDs. Read `history` afterward to recover the new markers and
call IDs. Use ordinary `append` with the server-issued `call_id` for live results.

### Generation settings

Settings are top-level fields of `generate` or `ephemeral`:

| Field | Meaning | Fresh defaults |
|---|---|---|
| `temperature` | Nonnegative number; zero selects greedy sampling. | `1.0` |
| `top_k` | Integer from zero through the model's vocabulary size; zero disables top-k, one selects greedy sampling. | `64` |
| `top_p` | Sampling probability threshold; use a value in `(0,1]`. | `0.95` |
| `max_tokens` | Nonnegative integer; zero means no explicit output cap beyond available context. | `0` |
| `seed` | Positive integer reseeds the sampler; zero or omission leaves its state unchanged. | Initial seed `1` |
| `reasoning` | `false`, or the object below. `true` is not accepted. | Off |

```json
{"op":"generate","reasoning":{"effort":"high","budget_tokens":1024,"history":"preserve_tool_calls"}}
```

When reasoning is an object, `effort` is required. `budget_tokens` is an optional
nonnegative integer up to `2147483647`; `history` is optional. The initial history
mode is `discard`. When no explicit budget is stored, the runtime derives one
from the effort and available output/context budget. Omitting `budget_tokens`
with unchanged effort preserves any stored override. Changing effort resets an old
explicit budget unless a new budget is supplied. Reasoning consumes output
tokens; it is not an additional allowance beyond `max_tokens`.

For durable generations, omitted settings reuse the conversation's current
settings and sampler state. For `ephemeral`, every request starts with fresh
defaults. Resending a nonzero seed reseeds each time. While reasoning is enabled,
the current open turn retains its reasoning regardless of history mode.
`discard` omits past reasoning from the projected context; `preserve_tool_calls`
also retains reasoning from historical assistant entries containing tool calls.
This is projection policy, not deletion of the persisted reasoning text.

Send correctly typed, finite numbers. Use nonnegative integers for counts, IDs
and markers; keep `top_k` and `max_tokens` within signed 32-bit range. The JSON
parser uses binary64 numbers, so integer request values above `2^53-1` cannot be
relied upon to round-trip exactly. Current parsing is permissive for some invalid
types or ranges; clients must not depend on every malformed value being rejected.
Temperature must remain finite when converted to a 32-bit float. `top_k` must
also be no greater than the model's vocabulary size, which `describe` does not
advertise; preserve the default unless the client knows the model's limits.
Invalid sampler values can terminate the engine rather than return a JSON error.

## Generation events

Events have an `event` field instead of `ok`. Text and reasoning arrive as
fragments; concatenate each channel independently. A fragment is not a reliable
token count. Tool argument text may be buffered without producing text deltas.

| Event | Fields and meaning |
|---|---|
| `start` | Generation stream has begun; preparation can already have taken place. |
| `inference_progress` | `phase`, `state`, `tokens`, `elapsed_ms`, and `total` for prefill. See below. |
| `text_delta` | `text`: append to visible assistant text. |
| `reasoning_delta` | `text`: append to the reasoning channel. |
| `toolcall_start` | `id`, `name`: a tool call is being generated. |
| `toolcall_end` | `id`, `name`, `arguments`: complete call arguments as a JSON value, not an encoded JSON string. |
| `done` | `stop`, `usage`, `marker`, `inference`; optional `reasoning_close`, `checkpoint`, `resume`. Ends the request. |
| `error` | `code`, `error`, `inference`; optional context-overflow details. Ends the request. |

`done.stop` is `stop` (normal end), `tool_use`, `length` or `aborted`.
`reasoning_close`, when present, is `natural`, `soft`, `hard`, `length`, `aborted`
or `eos`. It describes how the reasoning segment ended, independently of the
request's stop reason.

Wait for `done` before treating a durable generation as committed or executing
its tool calls. `toolcall_end` means arguments are complete, not that persistence
has succeeded. For `done.stop: "tool_use"`, execute the calls on the client,
append their results by ID, then send another `generate`. An error can occur
after content or calls have already been streamed.

### Inference phases

```json
{"event":"inference_progress","phase":"prefill","state":"running","tokens":0,"total":1024,"elapsed_ms":0}
{"event":"inference_progress","phase":"prefill","state":"running","tokens":512,"total":1024,"elapsed_ms":400}
{"event":"inference_progress","phase":"prefill","state":"finished","tokens":1024,"total":1024,"elapsed_ms":800}
{"event":"inference_progress","phase":"decode","state":"running","tokens":0,"elapsed_ms":0}
{"event":"inference_progress","phase":"decode","state":"finished","tokens":40,"elapsed_ms":2000}
```

The first `running` opens a phase before computation, with zero tokens and time.
Later `running` events update cumulative measurements. `finished` carries the
final sample: that phase will do no more work for this generation. It does not
mean the request succeeded. No later progress event for that phase follows.
Only one inference phase is open at a time.

Prefill opens after preparatory waits and finishes before decode setup. A fully
cached prefill still sends `running` and `finished` with `tokens: 0, total: 0`.
Decode opens explicitly; never infer it from a prefill finish. Cancelling during
prefill produces a partial prefill finish and no decode events. A phase never
entered emits nothing. `done` or `error` can close an open phase directly;
disconnects also require the client to stop its display.

Decode finishes before finalization, which can include reconciliation,
additional computation and persistence. A UI can show preparation between
prefill and decode, and finalization after decode, while waiting for the terminal
event. Those labels are client presentation, not additional protocol phases.

| Measurement | Definition |
|---|---|
| Prefill `tokens` | Token computations performed by the initial prompt synchronization, excluding cache reuse and terminal replay. |
| Prefill `total` | Work done plus remaining prompt tokens. The initial cache estimate may be corrected by synchronization. It is always at least `tokens`; normal completion reaches equality. |
| Decode `tokens` | Output count consistent with `usage.output`, including reasoning, tool arguments and accounted control tokens. No decode total is known in advance. |
| `elapsed_ms` | Cumulative active runtime wall time for that phase, measured with a monotonic clock and truncated to whole milliseconds. |

Prefill time measures synchronization, excluding shadow waits and setup.
Decode time includes sampling, parsing and in-phase shadow work. Both exclude
client/transport waits, snapshot loading and finalization. These are runtime
throughput measurements, not isolated CPU/GPU kernel timings.

The average rate is `1000 * tokens / elapsed_ms`; a recent rate uses differences
between samples instead. Do not divide by zero or invent a speed for zero work.
Prefill percentage is `100 * tokens / total`: it measures tokens, not elapsed-time
fraction. Treat `total: 0` as no prefill work required. A local clock started and
stopped by phase events measures the wait observed by the client; use the
engine's time for tok/s, not network arrival intervals.

Prefill updates follow synchronization chunks, normally 512 tokens. Decode
updates are checked between steps after at least 250 ms of accumulated active
time, plus opening and closing samples. A slow step may delay delivery: 250 ms
is not a wall-clock heartbeat guarantee.

Terminal events carry frozen measurements in this shape:

```json
{"prefill":{"tokens":1024,"total":1024,"elapsed_ms":800},"decode":{"tokens":40,"elapsed_ms":2000}}
```

This is the value of `inference`, not a separate event. Counts and times are zero
for work never performed; an unstarted prefill can still have a nonzero planned
`total`. Finalization does not increase these measurements.

### Usage, checkpoints and resume

`done.usage` contains:

| Field | Meaning |
|---|---|
| `input` | Token computations charged to input, including replay and shadow work. |
| `cache_read` | Tokens reused in the first prompt synchronization. |
| `output` | Accounted generated/control tokens. |
| `reasoning` | Tokens counted in the reasoning output channel. |
| `replayed` | Input computations attributed to replay, including shadow work. |
| `total` | Final projected conversation size; for ephemeral requests, prompt plus output count. |
| `shadow_prefilled` | Tokens computed for the auxiliary conversation state used to reconcile the final transcript. |
| `shadow_background` | Subset of shadow work accounted as background work. |
| `shadow_remaining` | Remaining projected tokens at the start of terminal shadow reconciliation. |
| `shadow_kv_bytes` | Peak tracked KV storage for the shadow state. |
| `shadow_wait_us` | Terminal shadow reconciliation duration in microseconds when promotion succeeds. |

These counters have different scopes: do not assume `total` equals
`input + cache_read + output`, or `input` equals `inference.prefill.tokens`.
Shadow counters can cover work carried across several generations in a tool turn;
do not sum them as independent per-request totals.

An explicit checkpoint reply, or optional `done.checkpoint`, contains `saved`,
`tokens`, and `reason` when `saved` is false. Reasons are `no_kvstore`, `empty`,
`nothing_new`, `kv_diverged`, `budget`, `io`, `rejected`, `record_failed`.
`saved: false` is a checkpoint outcome, not a failed generation. The durable
transcript remains the source of truth; a later request may need to recompute KV.

Optional `done.resume` contains `loaded`, `tokens`, and `reason` when loading
failed: `evicted`, `model_mismatch`, `token_mismatch`, `io` or `rejected`.
Its token count is the snapshot boundary loaded or attempted.

## History and rewinding

`history.entries` contains visible records, each with `kind`, `marker` and `text`.
Kinds are `system`, `user`, `assistant`, `tool_result`. Optional fields include
`reasoning` and `extra`. Assistant tool calls are stored in `extra` as an array of
`{id,name,arguments}`; system tool declarations are also carried as extra JSON.
Tool results have `call_id`, `status` and, when known, `tool`.

An assistant entry's optional `stop` uses record vocabulary: `eot`,
`eot_synthetic`, `eos`, `limit`, `cancelled`, `tool_calls`. These are not the same
strings as `done.stop`.

Rewind keeps the chosen boundary and discards later active history. A marker
can become unavailable after transcript changes; handle `marker_unavailable`
rather than assuming old markers remain valid. `rewind_cost` is an estimate,
not a reservation of a snapshot. Rebuilding a transcript also replaces tool-call
identities: refresh the client mapping from history before continuing.

## Cancellation and connection loss

Send `{"op":"cancel"}` on the same connection as the generation. Its ordinary
`{"ok":true}` ACK can be interleaved with generation events; it does not terminate
the stream. Keep reading until `done` or `error`.

If a generation is queued and has not started, cancellation removes it and
returns the ACK followed by `done` with `stop: "aborted"`, `marker: null` and zero
usage/measurements. There need not be a `start` or any phase event. If no active
or queued generation exists on that connection, cancellation returns
`invalid_request`.

For an active generation, cancellation takes effect at a runtime control point.
An open phase emits its final, possibly partial sample before request
finalization; no unstarted phase is invented. Errors can end the stream directly.
Cancellation does not roll back already produced content or skip the work needed
to leave a consistent conversation.

**A cancel ACK confirms acceptance, not a guaranteed `aborted` result.** Once the
runtime has decided the terminal result, including after `decode finished`, a
late cancel leaves that result unchanged. Finalization continues, with the
original stop reason on `done` unless finalization itself fails.

Loss of the generating connection asks the runtime to cancel, but the client
cannot assume it received the final state. Reconnect and inspect the session's
history and pending calls. There is no deduplication ID: blindly retrying an
`append` or `generate` after an unknown outcome can duplicate work or content.
On sockets and interactive stdin, EOF also requests
cancellation of an active generation; a half-closed connection may still receive
the terminal output. With a regular file as stdin, requests are processed
sequentially and EOF is not read during generation. Do not use EOF as a substitute
for the explicit cancel/terminal exchange.

`open` rejects a conversation whose current token projection exceeds the engine
capacity with `context_length_exceeded`, `tokens` and `context`. The stored record
and the connection's previous session binding are unchanged. No inference or
snapshot restoration is attempted. Clients should show the token count and limit
and suggest restarting the service with a sufficient context capacity. This is
not a missing session: clients must not silently recreate or truncate it.

A record exactly at capacity can be opened, but another turn may not fit:
`append` and `generate` also account for framing and output space and reject
overflowing prompts with the same structured error.

## Errors and compatibility

Before streaming, errors are replies such as:

```json
{"ok":false,"code":"context_length_exceeded","error":"context length exceeded","tokens":40000,"context":32768}
```

During streaming, they use `event: "error"` instead of `ok: false`. Classify by
`code`; `error` is human-readable text. Context overflow can include numeric
`tokens` and `context`. Other error codes are `session_not_found`,
`marker_unavailable`, `invalid_request`, `busy`, `io_error`, `out_of_memory`.
A disconnected stream without a terminal event has an unknown outcome, not an
implicit successful completion.

Clients should tolerate extra fields and unknown nonterminal events. Older v1
engines may emit `{"event":"progress","prefilled":512,"total":1024}` instead
of measured inference events, or `inference_progress` without `state`.
The current server emits `inference_progress`, not the old `progress` event.
Legacy progress has no engine timing and no reliable decode token counter.
Without `state`, a client cannot reliably identify the end of decode before the
terminal event. Optional reports may also be absent on older engines.

The external protocol is implemented in [serve.c](serve.c); [runtime.h](runtime.h)
defines the C orchestration API behind it. The `wire` CLI command and socket
name remain unchanged by the internal rename to `runtime`.
