#!/bin/sh
# Run test commands in a private mount namespace; do not edit host configuration.
set -eu
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi
if ! command -v bwrap >/dev/null 2>&1; then
    echo 'bubblewrap (bwrap) is required for isolated test execution.' >&2
    exit 1
fi
if [ ! -f /etc/ld.so.preload ]; then
    echo 'No system preload file exists; running the command directly.' >&2
    exec "$@"
fi
workdir=$(pwd -P)
# Keep the checkout and test scratch space writable. System files remain
# read-only; device access is retained for optional native/GPU checks.
exec bwrap --ro-bind / / --dev-bind /dev /dev --proc /proc \
    --bind "$workdir" "$workdir" --bind /tmp /tmp \
    --ro-bind /dev/null /etc/ld.so.preload -- "$@"
