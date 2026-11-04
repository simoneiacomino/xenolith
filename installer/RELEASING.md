# Engine releases and Linux distributions

This guide is for maintainers producing a Xenolith engine release or publishing
an installer distribution. For installation and everyday use, see the
[installation guide](README.md).

Each project builds and releases its own package. Xenolith's distribution tools
consume those archives, inspect their payloads and assemble installer metadata.
They do not build XenoPi or CodeXe, create component releases or publish npm
packages. The commands below prepare local candidates; remote publication is an
explicit, separate operation.

## Independent identities

| Identity | Purpose |
| --- | --- |
| Component version and SHA-256 | Identify the exact package being installed |
| Distribution version | Identify selected packages and model recommendations |
| Wire protocol and capabilities | Establish client compatibility with a running engine |
| Native discovery schema | Discover startup settings before connecting |

Component release labels use SemVer core and optional prerelease identifiers,
for example `0.1.0` or `0.1.0-rc.1`. The installer filename contract excludes
optional `+build` metadata; source and build identity belong in the accompanying
provenance. Distribution identifiers are separate from component SemVer.

Versions remain independent. A distribution can select a new engine while
retaining an existing client release. Clients check `describe` protocol and
capabilities; the installer also checks native discovery and executable startup.
Matching version strings alone do not establish compatibility.

For Xenolith engine versions, compatible fixes increment PATCH. During `0.x`,
features and deliberate breaking changes increment MINOR. From `1.0` onward,
compatible features increment MINOR and incompatible public interface changes
increment MAJOR. Document breaking changes and migration even during `0.x`.
Public interfaces include CLI/configuration, the C API, wire protocol and stored
data behavior. Each client owns its version policy in its repository.

Change wire or discovery schema versions only when their contracts change.
Document supported client combinations and preserve user configuration and
sessions; migrations need validation, failure handling and recovery instructions.
A distribution identifier selects exact component packages and model metadata;
a label such as `2026.11.04-1` is sufficient. Installer-only fixes can publish a
new distribution while retaining the same component releases. If creating Git
tags for distributions, use `dist-<LABEL>` to distinguish them from engine tags.

## Build and qualify an engine release

Choose an exact committed revision, version and target platform, and build from
a clean checkout. A merge into `main` does not publish an engine release. If
qualification needs to continue while `main` advances, use a temporary
`release/<version>` branch, bring its fixes back to `main` and qualify the final
revision. Xenolith's engine-only container builder is:

```sh
python3 installer/tools/build_linux.py --source "$ENGINE_SOURCE" \
  --output "$OUTPUT_ROOT/engine-build" --build-image
```

It snapshots the C engine sources and records source hashes, commit, image
identity, flags and recipe. Ubuntu 24.04 supplies Level Zero development tools
and `ocloc`; no GPU, host home or Docker socket is mounted inside the build
container. `ocloc` emits SPIR-V and the user's Intel runtime compiles GPU code
at startup. The CPU baseline remains x86-64-v3 plus AVX-VNNI. The image contains
no client or Rust toolchain. This userspace baseline does not certify every
Linux distribution, kernel, GPU or driver.

Create the engine release archive with `installer/tools/package_engine.py` as
described by its `--help`. For a candidate that can be promoted by the release
workflow, run [`linux-candidate.yml`](../.github/workflows/linux-candidate.yml).
It runs `make check-engine-package`, builds the engine, checks the resulting CLI
and packages that binary, without GPU inference or publication. The separate CI
installer job uses Node 26+ for client integration fixtures.

Qualify the exact candidate's installation, configuration preservation and
applicable model, image and session behavior on supported hardware. Record its
source revision, artifact hashes, results and compatibility limits. A successful
build or device probe alone does not establish inference correctness. Reuse
prior results only for demonstrably unchanged artifacts and scope.

After approval, [`release-engine.yml`](../.github/workflows/release-engine.yml)
verifies the selected successful candidate run, version and approved archive
hash, then uploads its exact files to a draft release and publishes it under
`v<VERSION>`. Enable repository release immutability before publication. Do not
move a published tag or replace its artifacts with different bytes.

A candidate may embed its intended stable version before publication; public
prereleases use suffixes such as `-rc.1`. Changing an embedded version or
rebuilding changes artifact identity and requires qualification of the resulting
artifact. Builds are not claimed to be byte-for-byte reproducible: toolchain
inputs and archive metadata require further pinning for that guarantee.

## Component packages consumed by the installer

The engine belongs to `simoneiacomino/xenolith`, XenoPi to
`simoneiacomino/xenopi`, and CodeXe to `crrisci/codexe`. Client build and release
instructions live in their repositories:
[XenoPi packages](https://github.com/simoneiacomino/xenopi/blob/main/docs/releases.md)
and [CodeXe packages](https://github.com/crrisci/codexe/blob/main/packaging/README.md).
The contracts below describe the archives accepted by Xenolith's assembler.
Preserve component archives byte-for-byte, including their resources, notices
and canonical checksums.

Release assets use these exact tag forms:

| Component | Repository | Tag | Required archive layout |
| --- | --- | --- | --- |
| Engine | `simoneiacomino/xenolith` | `vVERSION` | `bin/xenolith` |
| XenoPi | `simoneiacomino/xenopi` | `vVERSION` | `bin/xenopi`, `node/bin/node`, `lib/xenopi/` |
| CodeXe | `crrisci/codexe` | `codexe-vVERSION` | `bin/codex`, `install/install.sh`, canonical metadata/checksums |

Archives must contain regular files and directories without a wrapper directory,
links, special files, duplicate paths, traversal or privileged modes. Executable
bits must be present. The assembler audits actual ELF files, including bundled
Node and client helpers, without running them. Production entrypoints must be
Linux x86-64 ELF binaries (XenoPi's launcher uses its bundled ELF Node runtime).
XenoPi's `lib/xenopi/package.json` must be an object named `xenopi` with the
selected version; CodeXe's `codex-package.json` version must match its selected
version.

## Assemble metadata from prebuilt archives

Run from the Xenolith repository. Use absolute input/output paths and a fresh
output directory outside inputs. `ENGINE_SOURCE` supplies GPU and model metadata;
it is recorded as an observation, not proof that a prebuilt archive came from
that checkout. Review that source revision against the chosen engine release.
Retain each component's original build provenance alongside its release.

```sh
python3 installer/tools/prepare_distribution.py \
  --engine-repo "$ENGINE_SOURCE" \
  --engine-archive "$ENGINE_ARCHIVE" --engine-version "$ENGINE_VERSION" \
  --engine-url "https://github.com/simoneiacomino/xenolith/releases/download/v$ENGINE_VERSION/$(basename "$ENGINE_ARCHIVE")" \
  --cpu-flags "$ENGINE_CPU_FLAGS" --max-glibc 2.39 \
  --xenopi-archive "$XENOPI_ARCHIVE" --xenopi-version "$XENOPI_VERSION" \
  --xenopi-url "https://github.com/simoneiacomino/xenopi/releases/download/v$XENOPI_VERSION/$(basename "$XENOPI_ARCHIVE")" \
  --codexe-archive "$CODEXE_ARCHIVE" --codexe-version "$CODEXE_VERSION" \
  --codexe-url "https://github.com/crrisci/codexe/releases/download/codexe-v$CODEXE_VERSION/$(basename "$CODEXE_ARCHIVE")" \
  --distribution-version "$DISTRIBUTION_VERSION" \
  --default-source "https://xenolith.it/distributions/$DISTRIBUTION_VERSION" \
  --output "$OUTPUT_ROOT/distribution"

(cd "$OUTPUT_ROOT/distribution" && sha256sum --check SHA256SUMS)
```

Supply each optional client's archive, version and URL together, or omit all
three. URLs must name the exact official repository, matching tag and matching
archive basename. Mutable `latest` links, query strings, encoded paths and
unrelated repositories are rejected. CPU requirements are an explicit engine
build declaration; glibc requirements are measured from every supplied ELF.
`--max-glibc` rejects a component above the selected baseline.

The output contains only `install.sh`, `manifest.tsv`, `requirements.tsv`,
`gpu-matrix.tsv`, `elf-requirements.json`, `provenance.json` and `SHA256SUMS`.
Component archives stay in their original locations. The manifest records their
pinned Release URLs, exact sizes and SHA-256 values. Local archive paths are
never inferred as public download URLs. Models are referenced by pinned source
metadata; weights are neither loaded nor copied.

The rendered installer embeds its immutable metadata source and manifest hash.
Users do not need `--source`. The advanced `--source` and
`XENOLITH_DISTRIBUTION_SOURCE` overrides do not inherit a pin for a different
source; pass `--manifest-sha256` to pin an override independently. Checksums detect
changed bytes; trust in the initial script and approved release remains separate.

For local smoke tests before component publication, serve component archives
from a separate local directory and pass explicit loopback HTTP component URLs
with `--development-components`. For example, in a separate terminal:

```sh
mkdir "$OUTPUT_ROOT/component-downloads"
cp "$ENGINE_ARCHIVE" "$XENOPI_ARCHIVE" "$CODEXE_ARCHIVE" "$OUTPUT_ROOT/component-downloads/"
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory "$OUTPUT_ROOT/component-downloads"
```

Use `--engine-url "http://127.0.0.1:8765/$(basename "$ENGINE_ARCHIVE")"` and
equivalent client URL arguments when assembling fresh metadata. Keep this
component server running throughout the smoke test; its metadata HTTP server
may use a different port. Set `--default-source` to the local metadata directory
or exact metadata loopback URL being served. Absolute component paths are
rejected early because the installer accepts component URLs or flat relative
artifact names. Archives remain outside the seven-file metadata bundle.

The development flag also permits simulated entrypoints for fixtures. Such
candidates record their development status and are rejected by the publisher.
Do not use them as public release candidates.

## Qualify and publish the complete distribution

1. Run `make check-installer` for shell, package, update and promotion fixtures.
2. Run `installer/tests/smoke_install.py` against the prepared metadata, optionally
   with `--use-default-source`, using an isolated home/prefix. Referenced component
   URLs or local development locations must be reachable.
3. On supported hardware, run `installer/tests/smoke_model.py` for generation,
   vision and resumed client conversations. The runner prepares the installation
   without launching an interface, then explicitly runs its inference checks;
   normal installer completion provides no generation evidence.
   See [the installer guide](README.md#tests).
4. Review the exact component build provenance, selected hashes, ABI report,
   source metadata and model/inference evidence before approving a distribution.

A successful build, device probe or protocol handshake does not prove inference.
A successful engine candidate does not replace this full installer and
distribution qualification. Review the release decision separately from code
merges and retain the previous qualified distribution for recovery.

Publish approved component assets first under their owning repositories and
version tags. Then approve the metadata bundle, manifest SHA-256 and installer
SHA-256. The manual
[`publish-distribution.yml`](../.github/workflows/publish-distribution.yml)
workflow downloads a checksum-approved, flat metadata bundle and validates it
without AWS credentials. Its gated publication job revalidates the same artifact.

The local helper defaults to a dry run:

```sh
python3 installer/tools/publish_distribution.py \
  --distribution-dir "$OUTPUT_ROOT/distribution" \
  --distribution-version "$DISTRIBUTION_VERSION" \
  --source-base https://xenolith.it/distributions \
  --approved-manifest-sha256 "$MANIFEST_SHA256" \
  --approved-installer-sha256 "$INSTALLER_SHA256" --bucket "$BUCKET"
```

Both default validation and `--execute` re-download every selected pinned
component asset over HTTPS, with bounded redirects, streaming size limits and
SHA-256 verification. A mismatch or network failure stops before any AWS call.
`--local-validation` performs offline metadata validation only and cannot be
combined with `--execute`.

With explicit `--execute`, only the seven metadata files are uploaded to
`distributions/VERSION/` with conditional immutable writes. An existing key is
accepted only when S3 reports the same SHA-256 checksum. Component archives are
never uploaded to S3. The stable `install.sh` is updated last, after every
immutable object succeeds. Existing versioned installers keep selecting their
original manifest and component hashes. The deployment workflow additionally
requires its repository gate, main branch and production environment approval.

No credentials, live endpoint or approval are established by editing these tools.
Publishing the documentation site is separate from distribution promotion.
