#!/bin/sh
# Xenolith installer. No compilation or background service setup.
set -eu
umask 077

# Filled by the distribution packager. Source checkouts assume no live endpoint.
DEFAULT_SOURCE=''
DEFAULT_MANIFEST_SHA256=''

die() { printf 'xenolith installer: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || die "Required command missing: $1"; }
absolute() { case "$2" in /*) ;; *) die "$1 must be an absolute path";; esac; }
quote() { printf "'"; printf '%s' "$1" | sed "s/'/'\\\\''/g"; printf "'"; }
prompt() {
    [ "$interactive" = 1 ] || die "$1; supply the corresponding option (see --help)."
    printf '%s [%s]: ' "$1" "$2" >&3
    IFS= read -r answer <&3 || die 'Input interrupted; rerun to continue.'
    [ -n "$answer" ] || answer=$2
}
confirm() {
    [ "$yes" = 1 ] && return 0
    prompt "$1 (yes/no)" no
    [ "$answer" = yes ] || die 'Cancelled. Existing files and partial downloads are preserved.'
}
usage() {
    cat <<'EOF'
Usage: sh install.sh [options]
  --source SOURCE         Advanced: override the prepared distribution source
  --prefix ABS            Component storage (default ~/.local/share/xenolith/install)
  --bin-dir ABS           Commands (default ~/.local/bin)
  --config ABS            Native engine config (default XDG config path)
  --model ABS             Use an existing GGUF without copying it
  --download-model        Download the pinned recommended GGUF (resumable)
  --models-dir ABS        Download destination (default ~/.local/share/xenolith/models)
  --mmproj ABS            Existing matching F16 projector
  --vision                Download projector if --mmproj was not supplied
  --engine-bin ABS        Reuse this existing Xenolith executable
  --xenopi-bin ABS        Reuse this existing XenoPi command
  --codexe-bin ABS        Reuse this existing CodeXe command
  --use-packages          Install managed packages instead of searching PATH
  --interface NAME        chat (default), xenopi, codexe, or all
  --context N             Context for newly created config (default 65536)
  --cache-budget SIZE     Disk cache budget for new config (default 4GiB)
  --yes                   Accept displayed downloads and installation
  --allow-unverified      Accept GPU enabled by engine but not yet tested
  --check-only            Report device/package compatibility, then exit
  --skip-verify           Prepare only; do not launch an interface
  --add-to-path           Add a managed PATH block to this user's shell profile
  --no-modify-path        Never offer or perform shell profile changes
  --install-runtime      Permit guided installation of missing runtime packages
  --manifest-sha256 HASH  Require this exact distribution manifest digest
  --help

Existing configuration is preserved. Rerun to resume downloads or add clients.
Prepared installers include their distribution source and manifest checksum.
Without a default distribution source, supply --source.
EOF
}

source=${XENOLITH_DISTRIBUTION_SOURCE:-$DEFAULT_SOURCE}
case "${XDG_CONFIG_HOME:-}" in
    /*) config_home=$XDG_CONFIG_HOME;;
    *) config_home=$HOME/.config;;
esac
case "${XDG_DATA_HOME:-}" in
    /*) data_home=$XDG_DATA_HOME;;
    *) data_home=$HOME/.local/share;;
esac
prefix=$data_home/xenolith/install
bindir=$HOME/.local/bin
config=${XENOLITH_CONFIG:-$config_home/xenolith/config.json}
models=$data_home/xenolith/models
model= mmproj= interface= context=65536 budget=4GiB
engine_option=${XENOLITH_BIN:-} xenopi_option= codexe_option= use_packages=0
original_path=$PATH
download=0 vision=0 yes=0 unverified=0 check_only=0 prepare_only=0
manifest_hash= interactive=0 tmp= lock_owned=0
path_mode=ask install_runtime=0
while [ "$#" -gt 0 ]; do
    key=$1; shift
    case "$key" in
        --help|-h) usage; exit 0;;
        --yes) yes=1;; --vision) vision=1;; --download-model) download=1;;
        --allow-unverified) unverified=1;; --check-only) check_only=1;;
        --skip-verify) prepare_only=1;;
        --add-to-path) path_mode=add;; --no-modify-path) path_mode=never;;
        --install-runtime) install_runtime=1;;
        --use-packages) use_packages=1;;
        --engine-bin|--xenopi-bin|--codexe-bin|--source|--prefix|--bin-dir|--config|--models-dir|--model|--mmproj|--interface|--context|--cache-budget|--manifest-sha256)
            [ "$#" -gt 0 ] || die "Missing value for $key"
            value=$1; shift
            case "$key" in
                --engine-bin) engine_option=$value;; --xenopi-bin) xenopi_option=$value;; --codexe-bin) codexe_option=$value;;
                --source) source=$value;; --prefix) prefix=$value;; --bin-dir) bindir=$value;;
                --config) config=$value;; --models-dir) models=$value;; --model) model=$value;;
                --mmproj) mmproj=$value; vision=1;; --interface) interface=$value;;
                --context) context=$value;; --cache-budget) budget=$value;;
                --manifest-sha256) manifest_hash=$value;;
            esac;;
        *) die "Unknown option: $key";;
    esac
done
[ -n "$source" ] || die 'This source installer has no default distribution. Use a prepared installer or --source with a local directory or URL.'
# An explicit source override may select a different candidate. Only inherit the
# embedded pin when it still describes the selected source; an explicit pin wins.
if [ -z "$manifest_hash" ] && [ "$source" = "$DEFAULT_SOURCE" ]; then
    manifest_hash=$DEFAULT_MANIFEST_SHA256
fi
for cmd in curl tar sha256sum awk sed wc df uname mktemp readlink timeout tr grep dd getconf cmp id env; do need "$cmd"; done
absolute prefix "$prefix"; absolute bin-dir "$bindir"; absolute config "$config"; absolute models-dir "$models"
for executable in "$engine_option" "$xenopi_option" "$codexe_option"; do [ -z "$executable" ] || absolute executable "$executable"; done
[ "$use_packages" = 0 ] || [ -z "$engine_option$xenopi_option$codexe_option" ] || die '--use-packages conflicts with explicit executable selections (including XENOLITH_BIN).'
[ -z "$model" ] || absolute model "$model"
[ -z "$mmproj" ] || absolute mmproj "$mmproj"
[ -z "$model" ] || [ "$download" = 0 ] || die 'Choose --model or --download-model, not both.'
case "$source" in https://*|http://127.0.0.1:*|http://localhost:*|/*) ;; *) die 'Source must be an absolute directory, HTTPS URL, or loopback HTTP URL.';; esac
case "$context" in ''|*[!0-9]*) die 'Context must be a positive integer';; esac
if ( : </dev/tty ) 2>/dev/null; then exec 3<>/dev/tty; interactive=1; fi
# A short, plain-text orientation on the prompt channel only. No ANSI,
# terminal-width assumptions, external commands, or extra unattended output.
if [ "$interactive" = 1 ] && [ "$yes" = 0 ] && [ "$check_only" = 0 ]; then
    printf '\n%s\n%s\n%s\n\n' \
        'Xenolith | Local inference setup' \
        'Check device, choose a client,' \
        'configure a model and open the selected client.' >&3
fi
cleanup() {
    # Let stage and lock removal finish even if cancellation is repeated.
    trap '' INT TERM
    [ -z "$tmp" ] || rm -rf -- "$tmp"
    [ "$lock_owned" = 0 ] || rmdir -- "$prefix/.install-lock" 2>/dev/null || :
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p -- "$prefix/downloads" "$prefix/releases" "$bindir"
mkdir -- "$prefix/.install-lock" 2>/dev/null || die "Another installer holds $prefix/.install-lock. After a crash, verify no installer is running before removing that empty directory."
lock_owned=1
tmp=$(mktemp -d "$prefix/.stage.XXXXXX")

# All downloaded objects are checksum addressed. Failed downloads retain .part.
fetch_raw() {
    case "$1" in
        https://*|http://127.0.0.1:*|http://localhost:*)
            curl --silent --show-error --fail --location --proto '=https,http' --proto-redir '=https,http' \
                --connect-timeout 20 --retry 2 --retry-delay 1 --output "$2" "$1";;
        /*) cp -- "$1" "$2";;
        *) die "Unsupported artifact location: $1";;
    esac
}
fetch_raw "${source%/}/manifest.tsv" "$tmp/manifest.tsv" || die 'Cannot fetch distribution manifest.'
if [ -n "$manifest_hash" ]; then
    got=$(sha256sum "$tmp/manifest.tsv"); got=${got%% *}
    [ "$got" = "$manifest_hash" ] || die 'Manifest checksum mismatch.'
fi
manifest=$tmp/manifest.tsv
awk -F '\t' '
 NR==1 {if ($0!="# xenolith-distribution-v1") exit 1; next}
 /^#/ || /^$/ {next}
 NF!=5 || $1!~/^[a-z-]+$/ || $2!~/^[A-Za-z0-9._-]+$/ || $4!~/^[0-9]+$/ || length($5)!=64 || $5!~/^[0-9a-f]+$/ {exit 1}
 {if(seen[$1]++) exit 1}
' "$manifest" || die 'Invalid distribution manifest.'
record() {
    row=$(awk -F '\t' -v name="$1" '$1==name {print}' "$manifest")
    [ -n "$row" ] || die "This distribution has no $1 package. Choose a distribution that includes it."
    tab=$(printf '\t')
    IFS="$tab" read -r component version location bytes digest <<EOF
$row
EOF
    case "$location" in
        https://*|http://127.0.0.1:*|http://localhost:*) url=$location;;
        *[!A-Za-z0-9._-]*|.*|'') die 'Invalid relative artifact name';;
        *) url=${source%/}/$location;;
    esac
}
valid_file() {
    [ -f "$1" ] && [ "$(wc -c < "$1" | tr -d ' ')" = "$bytes" ] || return 1
    got=$(sha256sum "$1"); got=${got%% *}
    [ "$got" = "$digest" ]
}
download_record() {
    destination=$1
    if valid_file "$destination"; then
        case "$component" in
            gpu-matrix|requirements) ;;
            engine|xenopi|codexe) say "Reusing verified $component $version";;
            *) say "Reusing verified $component: $destination";;
        esac
        return
    fi
    [ ! -e "$destination" ] || die "Existing file differs from the pinned artifact: $destination (left untouched)."
    mkdir -p -- "$(dirname -- "$destination")"
    part=$destination.part
    case "$component" in
        gpu-matrix|requirements) ;;
        engine|xenopi|codexe) say "$component $version: $bytes bytes";;
        *) say "$component: $bytes bytes -> $destination";;
    esac
    have=0; [ ! -f "$part" ] || have=$(wc -c < "$part" | tr -d ' ')
    [ "$have" -le "$bytes" ] || die "Oversized partial download: $part (remove explicitly to retry)."
    free=$(df -Pk "$(dirname -- "$destination")" | awk 'END{print $4}')
    [ "$free" -ge "$(( (bytes - have + 1048575) / 1024 ))" ] || die 'Insufficient free space for download.'
    if [ "$have" != "$bytes" ]; then
        case "$url" in
            https://*|http://127.0.0.1:*|http://localhost:*)
                set --
                case "$component" in gpu-matrix|requirements) set -- --silent --show-error;; esac
                curl "$@" --fail --location --proto '=https,http' --proto-redir '=https,http' \
                    --connect-timeout 20 --retry 2 --retry-delay 1 --continue-at - \
                    --output "$part" "$url" || die "Download interrupted; rerun to resume $part. If the server cannot resume, remove only that partial file to restart.";;
            /*) cp -- "$url" "$part";;
        esac
    fi
    valid_file "$part" || die "Checksum/size mismatch in $part; no installation performed from this file. Remove that partial file to retry."
    mv -- "$part" "$destination"
}

configure_path() {
    if [ "$path_mode" != add ]; then
        case ":$original_path:" in *":$bindir:"*) return;; esac
    fi
    [ "$path_mode" != never ] || return 0
    single_line=$(printf '%s' "$bindir" | tr -d '\r\n')
    [ "$single_line" = "$bindir" ] || { say 'PATH directory contains line breaks; shell profile left unchanged.'; return; }
    user_shell=${SHELL:-/bin/sh}
    case "${user_shell##*/}" in
        bash) profile=$HOME/.bashrc;;
        zsh) profile=${ZDOTDIR:-$HOME}/.zshrc;;
        sh|dash) profile=$HOME/.profile;;
        *) say 'Automatic PATH setup is available for bash, zsh and sh; use the printed command or configure your shell manually.'; return;;
    esac
    case "$profile" in /*) ;; *) say 'Relative shell profile location: PATH left unchanged.'; return;; esac
    if [ -L "$profile" ] || { [ -e "$profile" ] && [ ! -f "$profile" ]; }; then
        say "Shell profile is a symlink or non-regular file, left untouched: $profile"; return
    fi
    if [ -f "$profile" ] && [ ! -w "$profile" ]; then
        say "Shell profile is not writable, left untouched: $profile"; return
    fi
    if [ "$path_mode" = ask ]; then
        [ "$interactive" = 1 ] && [ "$yes" = 0 ] || return 0
        prompt "Add $bindir to PATH in $profile for future shells? yes/no" yes
        case "$answer" in yes) ;; no) return;; *) die 'Answer yes or no';; esac
    fi
    begin='# >>> xenolith PATH >>>'
    end='# <<< xenolith PATH <<<'
    # Quote the directory as data, even when HOME contains spaces/apostrophes.
    {
        say "$begin"
        say 'case ":$PATH:" in'
        say "    *$(quote ":$bindir:")*) ;;"
        say "    *) export PATH=$(quote "$bindir"):\"\$PATH\" ;;"
        say 'esac'
        say "$end"
    } > "$tmp/path-block"
    old_profile=$tmp/old-profile
    if [ -f "$profile" ]; then cp -p -- "$profile" "$old_profile"; else : > "$old_profile"; fi
    # A malformed existing block is user data: do not guess what to replace.
    awk -v b="$begin" -v e="$end" '
      $0==b {if(active || starts++) exit 1; active=1; next}
      $0==e {if(!active) exit 1; active=0; ends++}
      END {if(active || starts!=ends) exit 1}
    ' "$old_profile" || die "Malformed managed PATH block in $profile; file left unchanged."
    awk -v b="$begin" -v e="$end" '
      $0==b {skip=1; next} $0==e {skip=0; next} !skip {print}
    ' "$old_profile" > "$tmp/profile-body"
    cat "$tmp/path-block" >> "$tmp/profile-body"
    if cmp -s "$old_profile" "$tmp/profile-body"; then return; fi
    mkdir -p -- "$(dirname -- "$profile")"
    next_profile=$(mktemp "$profile.xenolith.XXXXXX")
    if [ -f "$profile" ]; then
        cp -p -- "$profile" "$next_profile"
        cmp -s "$profile" "$old_profile" || { rm -f -- "$next_profile"; die 'Shell profile changed during installation; rerun PATH setup.'; }
        if [ ! -e "$profile.xenolith.bak" ] && [ ! -L "$profile.xenolith.bak" ]; then cp -p -- "$profile" "$profile.xenolith.bak"; fi
    fi
    cat "$tmp/profile-body" > "$next_profile"
    mv -f -- "$next_profile" "$profile"
    say "PATH configured in $profile. Open a new shell that reads this file, or run:"
    say "  export PATH=$(quote "$bindir"):\"\$PATH\""
}

run_package_command() {
    if [ "$(id -u)" != 0 ]; then need sudo; set -- sudo -- "$@"; fi
    if [ "$interactive" = 1 ]; then "$@" <&3; else "$@" </dev/null; fi
}

install_missing_runtime() {
    runtime_kind=$1
    distro=$(awk -F= '$1=="ID" {gsub(/"/,"",$2);print $2}' "$probe_root/etc/os-release")
    distro_version=$(awk -F= '$1=="VERSION_ID" {gsub(/"/,"",$2);print $2}' "$probe_root/etc/os-release")
    missing=
    case "$distro:$distro_version" in
        arch:*)
            need pacman
            packages="level-zero-loader intel-compute-runtime"
            [ "$runtime_kind" != codexe ] || packages="openssl libcap ncurses"
            [ "$runtime_kind" != xenopi ] || packages="libatomic"
            for package in $packages; do
                pacman -Q "$package" >/dev/null 2>&1 || missing="$missing $package"
            done
            set -- pacman -S --needed
            [ "$yes" = 0 ] || set -- "$@" --noconfirm
            ;;
        ubuntu:24.04)
            need dpkg-query; need apt-get
            packages="libze1 libze-intel-gpu1"
            [ "$runtime_kind" != codexe ] || packages="libssl3t64 libcap2 libtinfo6"
            [ "$runtime_kind" != xenopi ] || packages="libatomic1"
            for package in $packages; do
                state=$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || :)
                [ "$state" = 'install ok installed' ] || missing="$missing $package"
            done
            set -- apt-get install --no-install-recommends
            [ "$yes" = 0 ] || set -- "$@" -y
            ;;
        *) die 'Runtime check failed. Automatic package installation is available for Arch and Ubuntu 24.04 only; use your distribution instructions and rerun.';;
    esac
    [ -n "$missing" ] || die 'Runtime packages are installed, but the component check failed. Resolve the diagnostic above and rerun.'
    # Package names come only from the fixed distro mapping above.
    for package in $missing; do set -- "$@" "$package"; done
    say "Missing runtime packages:$missing"
    [ "$distro" != ubuntu ] || say 'Package index command: apt-get update'
    say "Package command: $*"
    if [ "$install_runtime" = 0 ]; then
        [ "$interactive" = 1 ] && [ "$yes" = 0 ] || die 'Rerun with --install-runtime --yes to install these system packages.'
        prompt 'Install these runtime packages with the system package manager? yes/no' no
        [ "$answer" = yes ] || die 'Runtime installation declined; files and config are preserved.'
    fi
    [ "$interactive" = 1 ] || [ "$yes" = 1 ] || die 'Unattended runtime installation also requires --yes.'
    if [ "$distro" = ubuntu ]; then
        run_package_command apt-get update || die 'Package index update failed; no runtime installation attempted.'
    fi
    run_package_command "$@" || die 'Runtime package installation failed; rerun after resolving the package manager error.'
}

check_runtime() {
    if timeout --signal=TERM --kill-after=5 30 "$engine" check-device > "$tmp/runtime-check" 2>&1; then
        say 'GPU runtime ready.'; return
    else
        runtime_status=$?
    fi
    cat "$tmp/runtime-check" >&2
    case "$runtime_status" in 124|137) die 'GPU runtime check timed out; investigate the driver/device state before retrying.';; esac
    install_missing_runtime engine
    timeout --signal=TERM --kill-after=5 30 "$engine" check-device > "$tmp/runtime-check" 2>&1 || {
        cat "$tmp/runtime-check" >&2
        die 'Runtime packages installed, but the device is still unavailable. See the diagnostic above.'
    }
    say 'GPU runtime ready.'
}

select_command() {
    command_name=$1; explicit_command=$2
    chosen_command=$bindir/$command_name; reuse=0
    if [ -n "$explicit_command" ]; then
        candidate=$explicit_command
    elif [ "$use_packages" = 1 ]; then
        return
    elif [ -e "$chosen_command" ] || [ -L "$chosen_command" ]; then
        candidate=$chosen_command
    else
        candidate=$(command -v "$command_name" || :)
        [ -n "$candidate" ] || return 0
    fi
    if [ -z "$explicit_command" ]; then
        managed_target=$(readlink "$candidate" 2>/dev/null || :)
        case "$managed_target" in
            "$prefix"/releases/*/bin/"$command_name") return;;
            "$prefix/lib/codexe/standalone/current/bin/codex")
                if [ "$command_name" = codexe ] && [ "$(cat "$prefix/lib/codexe/standalone/.codexe-local" 2>/dev/null || :)" = codexe-local-v1 ]; then return; fi;;
        esac
    fi
    [ -f "$candidate" ] && [ -x "$candidate" ] || die "Existing command is not executable, left untouched: $candidate"
    case "$candidate" in /*) ;; *) candidate=$(CDPATH= cd -- "$(dirname -- "$candidate")" && pwd)/$(basename -- "$candidate");; esac
    chosen_command=$candidate; reuse=1
    say "Found $command_name: $candidate"
    if [ "$interactive" = 1 ] && [ "$yes" = 0 ]; then
        prompt 'Reuse this installation? yes/no' yes
        [ "$answer" = yes ] || die 'Existing installation left untouched. Use --use-packages with a different --bin-dir to install separately.'
    fi
}

check_existing_xenopi() {
    entry=$(readlink -f -- "$xenopi_command")
    case "$entry" in
        */dist/src/bin/xenopi.js) app=${entry%/dist/src/bin/xenopi.js}; runtime=$(command -v node || :);;
        */bin/xenopi) client_root=${entry%/bin/xenopi}; app=$client_root/lib/xenopi
            if [ -x "$client_root/node/bin/node" ]; then runtime=$client_root/node/bin/node; else runtime=$(command -v node || :); fi;;
        *) die 'Existing XenoPi layout is unrecognized; use a compatible npm/source package or a separate --bin-dir with --use-packages.';;
    esac
    [ -n "$runtime" ] || die 'Existing XenoPi requires Node 26 or newer.'
    check_xenopi_runtime "$runtime"
    timeout 15 "$runtime" --input-type=module - "$app" <<'JS' || die 'Existing XenoPi lacks the required native configuration integration; left untouched.'
import {readFileSync, statSync} from 'node:fs';
import {join} from 'node:path';
const root=process.argv[2], p=JSON.parse(readFileSync(join(root,'package.json'),'utf8'));
if (Number(process.versions.node.split('.')[0]) < 26 || p.name!=='xenopi' ||
    p.bin?.xenopi!=='dist/src/bin/xenopi.js' || p.piConfig?.name!=='xenopi' ||
    p.piConfig?.configDir!=='.xenopi' || p.xenolith?.nativeConfig!==1) process.exit(1);
for (const file of ['bin/xenopi.js','extensions/provider.js','wire/discovery.js','wire/client.js'])
    if (!statSync(join(root,'dist/src',file)).isFile()) process.exit(1);
JS
}

check_existing_codexe() {
    existing_root=$(dirname -- "$(dirname -- "$(readlink -f -- "$codexe_command")")")
    [ -f "$existing_root/codex-package.json" ] &&
        grep -Eq '"nativeConfig"[[:space:]]*:[[:space:]]*1[[:space:]]*([,}]|$)' "$existing_root/codex-package.json" ||
        die 'Existing CodeXe package lacks native configuration compatibility metadata; left untouched. Install a current CodeXe package into a separate prefix.'
}

probe_version() {
    mkdir -p -- "$tmp/probe-home"
    timeout 30 env -u XENOPI_DIR -u XENOPI_CODING_AGENT_DIR -u PI_CODING_AGENT_DIR -u PI_PACKAGE_DIR \
        HOME="$tmp/probe-home" CODEX_HOME="$tmp/probe-home/.codexe" \
        XDG_CONFIG_HOME="$tmp/probe-home/config" XDG_DATA_HOME="$tmp/probe-home/data" \
        XDG_CACHE_HOME="$tmp/probe-home/cache" XDG_STATE_HOME="$tmp/probe-home/state" \
        "$1" --version
}

probe_codexe() {
    probe_version "$codexe_command" &&
    timeout 15 "$codexe_root/bin/codex-code-mode-host" --help &&
    timeout 15 "$codexe_root/codex-path/rg" --version &&
    timeout 15 "$codexe_root/codex-resources/bwrap" --version &&
    timeout 15 "$codexe_root/bin/codex-exec-server" --help
}

check_xenopi_runtime() {
    if probe_version "$1" > "$tmp/xenopi-runtime-check" 2>&1; then return; else xenopi_status=$?; fi
    cat "$tmp/xenopi-runtime-check" >&2
    # Only a missing libatomic loader dependency is repairable by this mapping.
    [ "$xenopi_status" = 127 ] &&
        grep -Fq 'error while loading shared libraries: libatomic.so.1:' "$tmp/xenopi-runtime-check" ||
        die 'XenoPi runtime check failed; resolve the diagnostic above and rerun.'
    install_missing_runtime xenopi
    probe_version "$1" > "$tmp/xenopi-runtime-check" 2>&1 || {
        cat "$tmp/xenopi-runtime-check" >&2
        die 'XenoPi runtime remains incomplete; see the diagnostic above.'
    }
}

check_codexe_runtime() {
    codexe_root=$(dirname -- "$(dirname -- "$(readlink -f -- "$codexe_command")")")
    # A missing bundled helper is a package defect, not a host library problem.
    # Reject it before offering package-manager remediation or activation.
    for helper in bin/codex-code-mode-host bin/codex-exec-server codex-path/rg codex-resources/bwrap; do
        [ -x "$codexe_root/$helper" ] ||
            die "CodeXe package is missing required helper: $helper."
    done
    if ! probe_codexe > "$tmp/codexe-runtime-check" 2>&1; then
        cat "$tmp/codexe-runtime-check" >&2
        install_missing_runtime codexe
        probe_codexe > "$tmp/codexe-runtime-check" 2>&1 || {
            cat "$tmp/codexe-runtime-check" >&2
            die 'CodeXe runtime remains incomplete; see the component diagnostic above.'
        }
    fi
    say 'CodeXe runtime ready.'
}

record gpu-matrix; download_record "$prefix/downloads/$digest.gpus"
gpu_matrix=$destination
record requirements; download_record "$prefix/downloads/$digest.requirements"
requirements=$destination
[ "$(uname -s)" = Linux ] || die 'Unsupported OS: Linux is required.'
[ "$(uname -m)" = x86_64 ] || die 'No binary package for this architecture (Linux x86_64 required).'
probe_root=${XENOLITH_INSTALL_TEST_ROOT:-}
if [ -n "$probe_root" ]; then
    absolute test-root "$probe_root"
    say "SIMULATED DEVICE EVIDENCE from test fixture: $probe_root"
fi
say "Device: Linux x86_64; $(awk -F= '$1=="PRETTY_NAME" {gsub(/"/,"",$2);print $2}' "$probe_root/etc/os-release")"
gpu_status=none
# Prefer PCI enumeration: a missing DRM node can mean a missing kernel driver,
# not incompatible hardware. DRM remains a fallback for restricted environments.
devices="$probe_root/sys/bus/pci/devices"
[ -d "$devices" ] || devices="$probe_root/sys/class/drm"
detected_intel=0
for entry in "$devices"/*; do
    if [ -d "$entry/device" ]; then dev=$entry/device; else dev=$entry; fi
    if [ -f "$dev/class" ]; then case "$(cat "$dev/class")" in 0x03*) ;; *) continue;; esac; fi
    [ -f "$dev/vendor" ] && [ -f "$dev/device" ] || continue
    [ "$(cat "$dev/vendor")" = 0x8086 ] || continue
    detected_intel=1
    id=$(cat "$dev/device"); id=${id#0x}
    status=$(awk -F '\t' -v id="$id" '$1==id {print $2}' "$gpu_matrix")
    say "Intel GPU 8086:$id: ${status:-not enabled by Xenolith}"
    case "$status" in tested) gpu_status=tested;; untested) [ "$gpu_status" = tested ] || gpu_status=untested;; esac
done
[ "$detected_intel" = 1 ] || die 'No Intel GPU is visible. Check PCI/device access and the kernel driver, then rerun.'
[ "$gpu_status" != none ] || die 'Incompatible device: no enabled Intel integrated GPU found.'
if [ "$gpu_status" = untested ] && [ "$unverified" = 0 ]; then
    say 'This GPU is enabled by Xenolith but untested.'
    [ "$check_only" = 1 ] || die 'Use --allow-unverified if you want to try this configuration.'
fi
if [ -z "$interface" ]; then
    if [ "$interactive" = 1 ] && [ "$yes" = 0 ] && [ "$check_only" = 0 ]; then
        say 'chat: simple conversation; xenopi: Pi coding agent with images; codexe: Codex coding agent.'
        prompt 'Interface: chat, xenopi, codexe, all' chat; interface=$answer
    else interface=chat; fi
fi
case "$interface" in chat|xenopi|codexe|all) ;; *) die 'Unknown interface';; esac
engine=$bindir/xenolith; engine_reused=0
xenopi_command=$bindir/xenopi; xenopi_reused=0
codexe_command=$bindir/codexe; codexe_reused=0
if [ "$check_only" = 0 ]; then
    select_command xenolith "$engine_option"; engine=$chosen_command; engine_reused=$reuse
    case "$interface" in xenopi|all) select_command xenopi "$xenopi_option"; xenopi_command=$chosen_command; xenopi_reused=$reuse;; esac
    case "$interface" in codexe|all) select_command codexe "$codexe_option"; codexe_command=$chosen_command; codexe_reused=$reuse;; esac
fi
if [ "$engine_reused" = 0 ]; then
flags=$(awk -F '\t' '$1=="cpu_flags" {print $2}' "$requirements")
for flag in $flags; do
    awk -v flag="$flag" '/^flags/ {for(i=3;i<=NF;i++) if($i==flag) found=1; exit} END{exit !found}' "$probe_root/proc/cpuinfo" || die "No compatible package: this artifact requires CPU feature $flag."
done
required_glibc=$(awk -F '\t' '$1=="glibc" {print $2}' "$requirements")
host_glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}') || die 'This package requires glibc.'
awk -v have="$host_glibc" -v want="$required_glibc" 'BEGIN{split(have,h,".");split(want,w,".");exit !(h[1]>w[1] || (h[1]==w[1] && h[2]>=w[2]))}' || die "Package requires glibc $required_glibc; host has $host_glibc."
else
    host_glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
fi
ram=$(awk '/^MemTotal:/ {printf "%.1f",$2/1048576}' "$probe_root/proc/meminfo")
available=$(awk '/^MemAvailable:/ {printf "%.1f",$2/1048576}' "$probe_root/proc/meminfo")
say "Memory: $ram GiB total, $available GiB available."

for selected in xenopi codexe; do
    case "$selected" in xenopi) [ "$xenopi_reused" = 0 ] || continue;; codexe) [ "$codexe_reused" = 0 ] || continue;; esac
    case "$interface:$selected" in all:*|xenopi:xenopi|codexe:codexe)
        minimum=$(awk -F '\t' -v name="${selected}_glibc" '$1==name {print $2}' "$requirements")
        if [ -n "$minimum" ]; then
            awk -v have="$host_glibc" -v want="$minimum" 'BEGIN{split(have,h,".");split(want,w,".");exit !(h[1]>w[1] || (h[1]==w[1] && h[2]>=w[2]))}' || die "$selected package requires glibc $minimum; host has $host_glibc. Choose a distribution compatible with this glibc version."
        fi;;
    esac
done
[ "$check_only" = 0 ] || exit 0
say "Set up $interface; components: $prefix; commands: $bindir"
confirm 'Continue installation?'

install_component() {
    record "$1"
    archive=$prefix/downloads/$digest.tar.gz
    download_record "$archive"
    if [ "$component" = codexe ]; then
        # The canonical package owns CodeXe activation for both source installs
        # and onboarding. Keep no second extracted copy under our releases/.
        stage=$tmp/codexe; mkdir -- "$stage"
        tar -tzf "$archive" > "$tmp/archive-list"
        awk '/^\// || /(^|\/)\.\.(\/|$)/ {exit 1}' "$tmp/archive-list" || die 'Unsafe archive paths.'
        tar -tvzf "$archive" > "$tmp/archive-types"
        awk 'substr($0,1,1)!="-" && substr($0,1,1)!="d" {exit 1}' "$tmp/archive-types" || die 'CodeXe canonical package must contain only regular files and directories.'
        tar -xzf "$archive" --no-same-owner --no-same-permissions -C "$stage"
        [ -f "$stage/install/install.sh" ] && [ -x "$stage/bin/codex" ] || die 'CodeXe package lacks its canonical installer or executable.'
        # Probe before activation so missing runtime libraries can take the same
        # explicit-consent repair path; no command is replaced on probe failure.
        codexe_command=$stage/bin/codex
        check_codexe_runtime
        sh "$stage/install/install.sh" --codexe --package-dir "$stage" --prefix "$prefix" --bin-dir "$bindir" || die 'CodeXe package activation failed.'
        codexe_command=$bindir/codexe
        say "Installed codexe: $codexe_command"
        return
    fi
    target=$prefix/releases/$component-$version-${digest%????????????????????????????????????????????????}
    if [ ! -d "$target" ]; then
        stage=$tmp/$component; mkdir -- "$stage"
        tar -tzf "$archive" > "$tmp/archive-list"
        awk '/^\// || /(^|\/)\.\.(\/|$)/ {exit 1}' "$tmp/archive-list" || die 'Unsafe archive paths.'
        tar -xzf "$archive" --no-same-owner --no-same-permissions -C "$stage"
        [ -x "$stage/bin/$2" ] || die "Package lacks executable bin/$2"
        mv -- "$stage" "$target"
    fi
    [ "$component" != xenopi ] || check_xenopi_runtime "$target/bin/$2"
    command_path=$bindir/$2
    if [ -e "$command_path" ] || [ -L "$command_path" ]; then
        old=$(readlink "$command_path" 2>/dev/null || :)
        case "$old" in "$prefix"/releases/*/bin/"$2") ;;
            *) die "Existing command left untouched: $command_path. Choose another --bin-dir or use your existing installation.";;
        esac
    fi
    ln -s "$target/bin/$2" "$tmp/link"
    mv -Tf -- "$tmp/link" "$command_path"
    say "Installed $2: $command_path"
}
if [ "$engine_reused" = 0 ]; then install_component engine xenolith; fi
if [ "$engine_reused" = 1 ]; then
    if timeout 15 "$engine" --help > "$tmp/engine-help" 2>&1; then help_status=0; else help_status=$?; fi
    # A missing shared loader prevents even --help from running. Give the same
    # explicit runtime repair path as a newly installed engine before deciding
    # whether this executable supports the required CLI. Other failures are not
    # assumed to be repairable by installing Intel packages.
    if [ "$help_status" = 127 ] && grep -q 'libze_loader\.so' "$tmp/engine-help"; then
        check_runtime
        timeout 15 "$engine" --help > "$tmp/engine-help" 2>&1 || :
    fi
    grep -q 'check-device' "$tmp/engine-help" || die 'Existing Xenolith lacks check-device; left untouched. Install a current Xenolith version.'
    timeout 30 "$engine" config inspect --json --config "$config" > "$tmp/inspect.json" || die 'Existing Xenolith cannot read configuration; left untouched. Install a compatible Xenolith version.'
    grep -Eq '"schema_version"[[:space:]]*:[[:space:]]*1([[:space:]]*[,}])' "$tmp/inspect.json" || die 'Existing Xenolith returned an unsupported native configuration schema.'
fi
check_runtime
# Check dynamic loader/config API before any multi-GB model download.
"$engine" config inspect --json --config "$config" > "$tmp/inspect.json" || die 'Engine configuration/runtime check failed. Install the required Level Zero/Intel runtime or correct the reported configuration error.'
case "$interface" in xenopi|all)
    if [ "$xenopi_reused" = 1 ]; then check_existing_xenopi; else install_component xenopi xenopi; fi
    check_xenopi_runtime "$xenopi_command";;
esac
case "$interface" in codexe|all)
    if [ "$codexe_reused" = 1 ]; then check_existing_codexe; else install_component codexe codexe; fi
    check_codexe_runtime;;
esac

if [ -e "$config" ]; then
    say "Keeping existing configuration: $config"
    if [ -n "$model$mmproj" ] || [ "$download" = 1 ] || [ "$vision" = 1 ]; then
        say "Model/vision options ignored for this existing config. To change models or image support, edit $config."
    fi
else
    if [ -z "$model" ] && [ "$download" = 0 ]; then
        prompt 'Existing GGUF absolute path, or download' download
        if [ "$answer" = download ]; then download=1; else model=$answer; absolute model "$model"; fi
    fi
    if [ "$interactive" = 1 ] && [ "$yes" = 0 ] && [ "$vision" = 0 ]; then
        prompt 'Enable images (downloads the matching 1.19 GB projector)? yes/no' no
        case "$answer" in yes) vision=1;; no) ;; *) die 'Answer yes or no';; esac
    fi
    if [ "$download" = 1 ]; then
        record model; say "Recommended model: $location ($bytes bytes). Destination: $models"
        confirm 'Download the recommended model?'
        download_record "$models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"; model=$destination
    fi
    [ -r "$model" ] && [ -f "$model" ] || die "Cannot read model: $model"
    [ "$(dd if="$model" bs=4 count=1 2>/dev/null)" = GGUF ] || die 'Selected model is not a GGUF file.'
    if [ "$vision" = 1 ] && [ -z "$mmproj" ]; then
        record projector; say "Matching F16 projector: $bytes bytes. Destination: $models"
        confirm 'Download the vision projector?'
        download_record "$models/mmproj-F16.gguf"; mmproj=$destination
    fi
    [ -z "$mmproj" ] || { [ -r "$mmproj" ] && [ -f "$mmproj" ]; } || die 'Cannot read projector.'
    say "Configuration: $config; context: $context; disk cache budget: $budget."
    set -- config init --config "$config" --model "$model" --ctx "$context" --cache-budget "$budget" --idle-shutdown 30
    [ -z "$mmproj" ] || set -- "$@" --mmproj "$mmproj"
    "$engine" "$@" || die 'Configuration was not created.'
fi

say "Configuration: $config"
say 'Commands for this and later sessions:'
if [ "$config" = "$config_home/xenolith/config.json" ] && [ "${XENOLITH_CONFIG+x}" != x ]; then
    config_arg=
else config_arg=" --config $(quote "$config")"; fi
say "  $(quote "$engine") chat$config_arg"
client_env="XENOLITH_BIN=$(quote "$engine") "
if [ -n "$config_arg" ]; then client_env="${client_env}XENOLITH_CONFIG=$(quote "$config") "; fi
case "$interface" in xenopi|all) say "  ${client_env}$(quote "$xenopi_command")";; esac
case "$interface" in codexe|all) say "  ${client_env}$(quote "$codexe_command")";; esac
configure_path
if [ "$interactive" = 1 ] && [ "$prepare_only" = 0 ]; then
    launch=$interface
    if [ "$launch" = all ]; then
        prompt 'Start now: chat, xenopi, codexe' chat
        launch=$answer
        case "$launch" in chat|xenopi|codexe) ;; *) die 'Unknown interface; use one of the printed commands.';; esac
    fi
    say "Starting $launch."
    # Release installer resources before handing over the terminal. In a
    # curl|sh invocation stdin is script source, so use the controlling TTY.
    cleanup
    trap - 0 INT TERM
    case "$launch" in
        chat)
            if [ -n "${TERM:-}" ] && [ "$TERM" != dumb ]; then
                printf '\033[2J\033[H' >&3
            fi
            exec "$engine" chat --config "$config" <&3 >&3 2>&3 3>&-;;
        xenopi|codexe)
            XENOLITH_CONFIG=$config; XENOLITH_BIN=$engine; export XENOLITH_CONFIG XENOLITH_BIN
            case "$launch" in xenopi) launch_command=$xenopi_command;; codexe) launch_command=$codexe_command;; esac
            exec "$launch_command" <&3 >&3 2>&3 3>&-;;
    esac
fi
