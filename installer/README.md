# Xenolith installation

The installer checks the runtime, prepares the selected Xenolith components,
and creates native configuration. Shell profiles and system packages change
only with the specific consent described below.

## Use a prepared distribution

```sh
sh /absolute/distribution/install.sh
```

A prepared installer can embed its distribution source and manifest checksum.
To use a source checkout with a local distribution, run
`sh installer/install.sh --source /absolute/distribution` from the repository
root. `--source` is an advanced override, not required by a prepared installer.
See [distribution preparation](RELEASING.md) for builds, versions and packages.
Commands below run from this `installer/` directory unless stated otherwise.

The interactive path checks the device and runtime, offers chat, XenoPi or CodeXe, reuses
compatible installed components or installs missing ones, accepts an
existing model or downloads the recommended one, creates native configuration
and opens the chosen interface in the terminal.
Choosing `all` installs both clients and asks which interface to open. Without
a terminal, setup prints the launch commands instead. Images are optional.
Prompts and the launched interface use `/dev/tty`, including
when a locally served copy is invoked through a pipe:

```sh
curl -fsSL http://127.0.0.1:8000/install.sh | sh -s -- \
  --source http://127.0.0.1:8000
```

If the installer was prepared with that URL as its default source, the command is
simply `curl -fsSL http://127.0.0.1:8000/install.sh | sh`.

For unattended setup:

```sh
sh /absolute/distribution/install.sh --yes \
  --model /absolute/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --interface xenopi
```

Use `--download-model` instead of `--model` to download the pinned recommendation.
`--vision` downloads the matching F16 projector, or `--mmproj /absolute/file`
reuses an existing one. The model is approximately 14.25 GB, the projector 1.19 GB.
URLs, revisions, sizes and SHA-256 values come from the engine's
`tests/vision/manifest.json`. Existing weights are not copied. GGUF magic is
checked before setup; model loading and generation happen in the selected
interface when used.

New configurations start with 65536 context positions, a 4 GiB disk cache budget
and 30-minute service idle shutdown. These are starting settings, not a guarantee
that every enabled GPU or 32 GB machine can run them. Weights, cache, packages
and runtime memory have separate resource costs.

Model and projector downloads display sizes and destinations. Interruptions
retain `.part` files; rerunning resumes HTTP Range transfers and verifies the
complete size and hash.
A server without resume support produces an error; explicitly remove only that
partial file to restart. Mismatching complete files are left untouched.

## Subsequent sessions and reruns

The normal config is `$XDG_CONFIG_HOME/xenolith/config.json`, falling back to
`~/.config/xenolith/config.json`. It is also available to a source build:

```sh
xenolith config init --model /absolute/model.gguf --ctx 65536
xenolith chat
xenopi
codexe
```

Each command requires its corresponding component. XenoPi and CodeXe ask the
engine for resolved startup paths and use the actual service's `describe` reply;
they do not parse inference configuration. The engine must be discoverable on
PATH for automatic startup. The installer prints exact quoted commands with the selected engine path and
configuration. Reused npm clients retain their existing Node resolution; setup
does not silently prepend another runtime directory to their PATH.

Interactive setup can add a managed PATH block to `.bashrc` for Bash,
`$ZDOTDIR/.zshrc` (or `~/.zshrc`) for Zsh, or `.profile` for sh/dash. It preserves
other content and creates an initial `.xenolith.bak` backup. Reruns do not
duplicate the block. Symlinked or non-writable profiles remain untouched; other
shells receive manual guidance. `--add-to-path` explicitly requests setup and
`--no-modify-path` disables it. `--yes` alone never authorizes profile changes.
Open a shell that reads the selected profile, or use the printed export in the
current terminal; a child installer cannot alter the parent shell environment.

XenoPi's package entrypoint only locates its bundled Node runtime; it contains no
model, socket or configuration settings. A source/npm installation uses the same
integration. CodeXe's `codexe` entrypoint is the native fork, not upstream Codex.

For source installations, Xenolith supports `make install` (default `~/.local`),
an optional `PREFIX` override and DESTDIR staging. XenoPi documents standard
`npm link` and offline installation
of a local archive with locked dependencies bundled; no npm publication is
required. CodeXe's `just install-codexe` builds its canonical package and delegates to the
same upstream-derived shell activation used for prebuilt packages. The build
helper uses Python; installing a prepared package needs no Python or npm. These installation
steps do not create inference configuration.

Advanced options include `--config`, `--prefix`, `--bin-dir`, `--models-dir`,
`--context`, `--cache-budget`, `--allow-unverified`, `--source`, and `--manifest-sha256`.
A custom config needs the same `XENOLITH_CONFIG` for clients (or their explicit
selector); the printed commands include it. Normal shared configuration requires
no environment exports. Explicit client/provider selections retain
precedence and are never rewritten by the installer.

Reruns preserve native config and can add clients or resume downloads. Model and
vision options apply only when first creating the config; on rerun they are
visibly ignored. To change the model or context, explicitly edit native config.
Managed package updates switch command symlinks, retaining earlier package
directories. Existing external commands are checked and reused without copying,
replacing, or adopting them into the installer's update mechanism. Conversations
and user settings are never reset.

An update activates components individually. If a later component fails, an
earlier component may already have been updated; there is no automatic rollback
of the whole distribution. Resolve the reported download/runtime issue and
rerun the same pinned installer. For a checksum failure, remove only the named
failed `.part` download as instructed before retrying. User configuration and
conversation storage are separate from the retained package directories.
Reverting to a previous distribution is appropriate only when its compatibility
with the current configuration and stored sessions has been verified. Manage
reused external installations through their own installation method.

Existing commands are found in `--bin-dir` first, then on PATH. Interactive setup
asks before reusing them; `--yes` accepts reuse. `--engine-bin`, `--xenopi-bin` and
`--codexe-bin` select explicit absolute command paths (`XENOLITH_BIN` also selects
the engine). Engine reuse requires native config inspection and `check-device`.
XenoPi/CodeXe packages must declare native configuration compatibility; arbitrary
commands with matching names or version strings are not sufficient. The normal
runtime and helper checks still run before model setup. Package-only CPU/glibc
requirements are not imposed on an existing build.

To install the distribution's packages separately, use `--use-packages` and a
free `--bin-dir`; unrelated existing targets are never overwritten. Explicit
executable selections conflict with `--use-packages`. For custom paths outside
PATH, use the exact printed commands, which select the same engine and config
for the client. Ordinary PATH installations need no duplicated client settings.

Interactive setup opens the chosen interface directly after its package, runtime
and configuration checks. For setup without opening an interface, use
`--skip-verify`.
An active engine may prevent standalone chat from acquiring the weight lock;
close that engine normally before opening chat.
Client trust and tool permissions retain their normal behavior.

## Hardware boundaries

Linux x86-64 only. GPU tested/untested status comes from the assembler's
selected engine metadata source; release review must match it to the engine archive. Untested enabled GPUs need `--allow-unverified`; unlisted GPUs
are rejected. Device detection alone is not an inference test or a certification
of every driver/distribution. RAM availability is reported separately.

The packager audits ELF requirements for every component, including bundled Node
and client helpers. The installer checks glibc for the engine and selected clients.
`elf-requirements.json` records dynamic dependencies; `--max-glibc` rejects builds
above a declared baseline.

The package declares CPU features and minimum glibc. The isolated build below
uses x86-64-v3 plus AVX-VNNI; GPU-compatible CPUs without those features need a
different package. There is no automatic source-build fallback. The local
artifact's glibc requirement follows its build host. A broad Linux release still
requires a qualified build environment and cross-device tests.

Level Zero and Intel GPU compute runtime are system dependencies. Before model
downloads, `xenolith check-device` initializes Level Zero, enumerates the same
GPU inference selects and checks device status, without loading weights/kernels.
PCI detection distinguishes visible hardware from a missing DRM driver node;
absent visibility is reported as unverifiable rather than incompatible.

If this check fails and runtime packages are missing, interactive setup offers
their installation through the system package manager:

| Distribution | Runtime packages |
| --- | --- |
| Arch Linux | `level-zero-loader`, `intel-compute-runtime` |
| Ubuntu 24.04 | `libze1`, `libze-intel-gpu1` |

When XenoPi is selected, setup checks its runtime before activating the command
or downloading weights. If Node cannot load `libatomic.so.1`, setup offers
`libatomic1` on Ubuntu 24.04 or `libatomic` on Arch, with the same explicit
package-manager consent. Other startup errors retain their diagnostic and do
not trigger this installation. Chat-only and CodeXe-only setup do not install
XenoPi's dependencies.

When CodeXe is selected, setup also checks its CLI and all bundled helpers before
downloading weights. Missing shared libraries can be installed with the same
explicit consent: `openssl libcap ncurses` on Arch, or
`libssl3t64 libcap2 libtinfo6` on Ubuntu 24.04. Choosing chat or XenoPi does not
install CodeXe's dependencies. A successful `--version` for the CLI alone would
not establish that its shell, sandbox and code execution helpers can start.

Unattended installation requires both `--install-runtime --yes`. Only the package
manager uses sudo when needed; the installer runs as the ordinary user. Existing
package-manager settings are preserved. On Ubuntu, the displayed `apt-get update`
also runs only after this consent, before installing the missing packages.
If packages are already installed, a
driver/permission failure is reported without assuming reinstallation fixes it.
No kernel, group or device-permission changes are performed. Other distributions
need their documented runtime setup. Package availability does not certify every
GPU; newer GPUs may require a newer vendor runtime. Actual inference remains the
final functional check.

## Prepare a distribution (maintainer workflow)

Each repository owns its component build and GitHub Release: Xenolith builds the
engine, XenoPi builds its Node bundle, and CodeXe builds its canonical native
package. The distribution assembler accepts prebuilt archives with independent
versions and pinned official Release URLs. It checks safe layouts and actual ELF
requirements, then emits installer metadata. It never builds clients or copies
component archives into the metadata bundle.

See [the release guide](RELEASING.md) for the archive/version/URL arguments,
engine-only container build, local development smoke tests and promotion gates.
The public manifest selects exact component hashes; S3 hosts installer metadata
while each component's archive stays on its owning GitHub Release.

The rendered installer embeds its metadata source and manifest checksum. To
serve local metadata, use an exact loopback default source and run:

```sh
python3 -m http.server --bind 127.0.0.1 --directory /absolute/distribution
```

Pre-publication smoke tests can select loopback HTTP component URLs using the
assembler's `--development-components` option. Serve the archives from a separate
local directory; absolute component locations are rejected by the installer. Those
candidates are marked for development and cannot be promoted. The publisher
re-downloads official pinned component assets and verifies sizes and hashes
before any S3 mutation; stable installer promotion occurs last.

## Tests

To exercise real packages without loading model weights, use a fresh work directory:

```sh
python3 tests/smoke_install.py --distribution /absolute/distribution \
  --work-dir /absolute/new-install-check --model /absolute/model.gguf
```

This checks loopback `curl | sh`, native GPU access, selected package/helper startup,
PATH setup, reruns, and reuse without duplicate downloads or command replacement.
Add `--mmproj /absolute/mmproj-F16.gguf` for existing vision configuration and
`--existing-xenopi /absolute/npm-prefix/bin/xenopi` to also check an npm installation.
It uses isolated HOME and `--skip-verify` to prepare without launching an interface;
success does not establish inference.
Add `--use-default-source` to test a rendered installer without passing a source
or manifest checksum on its command line. The embedded source must be reachable.

Fixture suite:

```sh
python3 -m unittest discover -s tests -v
```

The suite uses labelled simulated hardware/engine fixtures but real archives,
checksums, filesystem changes, curl and loopback HTTP interruption/Range resume.
Node 26+ must be on PATH for the existing-XenoPi integration fixtures; no real
XenoPi or CodeXe installation is needed. The repository CI runs this suite in a
separate installer job. From the repository root, `make check-engine-package`
runs only engine build, packaging and release-tool fixtures without Node;
`make check-installer` runs the full suite.
It covers pipe invocation, exact printed commands, reruns, preservation and invalid
artifacts. Loopback binding must be permitted. `XENOLITH_INSTALL_TEST_ROOT` is a
fixture override; leave it unset for real installation. These tests cannot prove
GPU/inference correctness; real-model client smoke must be reported separately.

For an opt-in real-model check, prepare all three packages, stop your own existing
Xenolith session normally, and use an empty output directory:

```sh
python3 tests/smoke_model.py --distribution /absolute/distribution \
  --model /absolute/model.gguf --mmproj /absolute/mmproj-F16.gguf \
  --work-dir /absolute/new-smoke-results --check-tools
```

This Linux test needs at least 21 GiB available RAM and permission to bind a
loopback HTTP server and access the GPU. It installs through actual `curl | sh`
and reruns setup, both in prepare-only mode with `--skip-verify`. The smoke runner
then explicitly tests generated replies in chat and both clients, including
persisted conversations after restarting the engine. With a
projector it also sends a small test image. It isolates configuration and client
homes, retains the user's global engine lock, and only stops engines it started
on its private endpoint. Logs and `report.json` preserve both successes and
failures. These are smoke checks, not model quality or broad hardware benchmarks.

`--check-tools` additionally asks CodeXe to read a newly generated local nonce
through a tool. The check requires a completed tool event and the exact nonce
in the final answer, exercising the packaged execution helpers with real inference.
