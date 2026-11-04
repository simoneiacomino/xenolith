#!/usr/bin/env python3
"""Validate a prepared Xenolith distribution and optionally promote it to S3.

Dry-run is the default. --execute uses the AWS CLI without a shell, uploads
versioned objects conditionally, and updates the stable installer only last.
"""
import argparse
import base64
import hashlib
import http.client
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from distribution_metadata import component_url


HEX = re.compile(r"[0-9a-f]{64}\Z")
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SUM_LINE = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)\Z")


class PublishError(ValueError):
    pass


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_b64(hex_digest):
    return base64.b64encode(bytes.fromhex(hex_digest)).decode("ascii")


def expected_source(base, version):
    parts = urlsplit(base)
    try:
        parts.port
    except ValueError as error:
        raise PublishError("source base has an invalid port") from error
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise PublishError("source base must be an HTTPS URL without credentials, query or fragment")
    if parts.path.rstrip("/") != "/distributions":
        raise PublishError("source base must use the fixed /distributions route")
    return base.rstrip("/") + "/" + version


def shell_assignment(text, name):
    """Read one standalone shell assignment using shell tokenization, never eval."""
    matches = []
    for line in text.splitlines():
        if not line.startswith(name + "="):
            continue
        try:
            words = shlex.split(line, comments=False, posix=True)
        except ValueError as error:
            raise PublishError(f"malformed {name} assignment in install.sh") from error
        if len(words) != 1 or not words[0].startswith(name + "="):
            raise PublishError(f"{name} must be a standalone assignment in install.sh")
        matches.append(words[0][len(name) + 1:])
    if len(matches) != 1:
        raise PublishError(f"install.sh must contain exactly one standalone {name} assignment")
    return matches[0]


def load_bundle(directory, version, approved_manifest, approved_installer, source_base):
    if not VERSION.fullmatch(version):
        raise PublishError("distribution version is not a safe path segment")
    if not HEX.fullmatch(approved_manifest) or not HEX.fullmatch(approved_installer):
        raise PublishError("approved manifest and installer digests must be lowercase SHA-256")
    supplied_root = Path(directory)
    if supplied_root.is_symlink():
        raise PublishError("distribution path must not be a symlink")
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise PublishError("distribution path must be a directory")
    entries = list(root.iterdir())
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise PublishError(f"distribution must be flat regular files only: {path.name}")
    names = {path.name for path in entries}
    if len(names) != len(entries):
        raise PublishError("duplicate distribution filenames")
    allowed = {"SHA256SUMS", "manifest.tsv", "provenance.json", "install.sh",
               "gpu-matrix.tsv", "requirements.tsv", "elf-requirements.json"}
    if names - allowed:
        raise PublishError("distribution may contain only installer metadata, never component archives: " + ", ".join(sorted(names - allowed)))
    required = allowed
    if not required <= names:
        raise PublishError("distribution is missing required files: " + ", ".join(sorted(required - names)))

    sums = {}
    raw_sums = (root / "SHA256SUMS").read_text(encoding="utf-8")
    if not raw_sums.endswith("\n"):
        raise PublishError("SHA256SUMS must end with a newline")
    for line in raw_sums.splitlines():
        match = SUM_LINE.fullmatch(line)
        if not match:
            raise PublishError("SHA256SUMS has a malformed record")
        file_name = match.group(2)
        if file_name in (".", "..", "SHA256SUMS") or "/" in file_name or "\\" in file_name or any(ord(c) < 32 for c in file_name):
            raise PublishError(f"unsafe or unsupported SHA256SUMS path: {file_name!r}")
        if file_name in sums:
            raise PublishError(f"duplicate SHA256SUMS record: {file_name}")
        sums[file_name] = match.group(1)
    if set(sums) != names - {"SHA256SUMS"}:
        missing = sorted((names - {"SHA256SUMS"}) - set(sums))
        extra = sorted(set(sums) - names)
        raise PublishError(f"SHA256SUMS file set mismatch (unlisted={missing}, absent={extra})")
    for name, expected in sums.items():
        path = root / name
        if digest(path) != expected:
            raise PublishError(f"SHA-256 mismatch for {name}")

    manifest = root / "manifest.tsv"
    manifest_sha = digest(manifest)
    if manifest_sha != approved_manifest:
        raise PublishError("manifest digest does not match the approved SHA-256")
    installer = root / "install.sh"
    installer_sha = digest(installer)
    if installer_sha != approved_installer:
        raise PublishError("installer digest does not match the approved SHA-256")

    try:
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublishError(f"invalid provenance.json: {error}") from error
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1 or provenance.get("distribution_version") != version:
        raise PublishError("provenance schema or distribution_version does not match")
    if provenance.get('development_components') is not False:
        raise PublishError('development or legacy distributions cannot be promoted')
    components = provenance.get('components')
    if not isinstance(components, dict):
        raise PublishError('provenance components must be an object')
    installer_metadata = provenance.get("installer")
    if not isinstance(installer_metadata, dict):
        raise PublishError("provenance installer metadata is missing or malformed")
    if installer_metadata.get("default_source") != expected_source(source_base, version):
        raise PublishError("installer embedded DEFAULT_SOURCE does not match the immutable HTTPS version URL")
    if installer_metadata.get("pinned_manifest_sha256") != manifest_sha:
        raise PublishError("provenance installer manifest pin does not match manifest.tsv")
    installer_text = installer.read_text(encoding="utf-8")
    if shell_assignment(installer_text, "DEFAULT_SOURCE") != expected_source(source_base, version) or shell_assignment(installer_text, "DEFAULT_MANIFEST_SHA256") != manifest_sha:
        raise PublishError("install.sh embedded source or manifest pin is inconsistent")

    # Manifest rows have name, version, location, byte size, SHA-256. Relative
    # local artifacts must be flat files with exact recorded sizes and digests.
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise PublishError("manifest.tsv is not UTF-8") from error
    if not lines or lines[0] != "# xenolith-distribution-v1":
        raise PublishError("manifest.tsv has an unsupported header")
    seen_locations = set()
    seen_names = set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 5:
            raise PublishError("manifest.tsv contains a malformed record")
        name, item_version, location, size_text, file_sha = fields
        if (not re.fullmatch(r"[a-z-]+", name) or name not in {"engine", "xenopi", "codexe", "model", "projector", "gpu-matrix", "requirements"} or
                not re.fullmatch(r"[A-Za-z0-9._-]+", item_version) or name in seen_names or location in seen_locations or
                not re.fullmatch(r"[0-9]+", size_text) or not HEX.fullmatch(file_sha)):
            raise PublishError("manifest.tsv contains an invalid or duplicate record")
        seen_names.add(name)
        seen_locations.add(location)
        parsed = urlsplit(location)
        if name in ("engine", "xenopi", "codexe"):
            try:
                component_url(name, item_version, location)
            except ValueError as error:
                raise PublishError(str(error)) from error
            if not 0 < int(size_text) <= 2 * 1024 * 1024 * 1024:
                raise PublishError('component archive size must be between 1 byte and 2 GiB')
            component = components.get(name)
            if not isinstance(component, dict) or not isinstance(component.get('archive'), dict):
                raise PublishError(f'component provenance is missing or malformed: {name}')
            archive = component['archive']
            if (component.get('version') != item_version or archive.get('url') != location or
                    archive.get('size_bytes') != int(size_text) or archive.get('sha256') != file_sha or
                    archive.get('file') != parsed.path.rsplit('/', 1)[-1]):
                raise PublishError(f'component provenance does not match manifest: {name}')
            continue
        if name in ("model", "projector"):
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise PublishError('model artifacts must use HTTPS URLs without credentials/query/fragment')
            continue
        if parsed.scheme or location.startswith('/') or '/' in location or "\\" in location or location in ('.', '..'):
            raise PublishError(f'manifest metadata path is not flat: {location}')
        path = root / location
        if location not in sums or not path.is_file() or path.stat().st_size != int(size_text) or digest(path) != file_sha:
            raise PublishError(f"manifest artifact record does not match local file: {location}")
    required_records = {"engine", "model", "gpu-matrix", "requirements"}
    if not required_records <= seen_names:
        raise PublishError("manifest.tsv is missing required records: " + ", ".join(sorted(required_records - seen_names)))
    return root, sorted((root / name for name in names), key=lambda p: p.name), installer


class HttpsRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, request, response, code, message, headers, new_url):
        parsed = urlsplit(new_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise PublishError('component download redirect must remain HTTPS without credentials or fragment')
        return super().redirect_request(request, response, code, message, headers, new_url)


def verify_remote_components(root):
    """Read every selected pinned Release asset before the first AWS mutation."""
    opener = urllib.request.build_opener(HttpsRedirect())
    for line in (root / 'manifest.tsv').read_text().splitlines()[1:]:
        name, version, url, size, expected = line.split('\t')
        if name not in ('engine', 'xenopi', 'codexe'):
            continue
        component_url(name, version, url)
        expected_size = int(size)
        request = urllib.request.Request(url, headers={'User-Agent': 'xenolith-distribution-promotion',
                                                       'Accept-Encoding': 'identity'})
        total, hasher, deadline = 0, hashlib.sha256(), time.monotonic() + 300
        with opener.open(request, timeout=30) as response:
            if urlsplit(response.geturl()).scheme != 'https':
                raise PublishError('component download redirected away from HTTPS')
            if response.status != 200:
                raise PublishError(f'component download did not return HTTP 200: {name}')
            length = response.headers.get('Content-Length')
            if length is not None and (not length.isdigit() or int(length) != expected_size):
                raise PublishError(f'remote component size does not match approved manifest: {name}')
            while True:
                chunk = response.read(min(1024 * 1024, expected_size - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_size or time.monotonic() > deadline:
                    raise PublishError(f'remote component exceeds approved size or time limit: {name}')
                hasher.update(chunk)
        if total != expected_size or hasher.hexdigest() != expected:
            raise PublishError(f'remote component bytes do not match approved manifest: {name}')
        print(f'Verified pinned component {name} {version} ({expected})')


def aws_call(aws, args):
    return subprocess.run([aws, "s3api", *args], check=False, capture_output=True, text=True)


def head_checksum(aws, bucket, key):
    result = aws_call(aws, ["head-object", "--bucket", bucket, "--key", key, "--checksum-mode", "ENABLED", "--output", "json"])
    if result.returncode:
        return None, result.stderr.strip() or f"head-object exited {result.returncode}"
    try:
        value = json.loads(result.stdout).get("ChecksumSHA256")
    except (json.JSONDecodeError, AttributeError):
        return None, "head-object returned malformed JSON"
    return value, None


def put_immutable(aws, bucket, key, path):
    file_sha = digest(path)
    args = ["put-object", "--bucket", bucket, "--key", key, "--body", str(path),
            "--if-none-match", "*", "--checksum-algorithm", "SHA256", "--checksum-sha256", sha_b64(file_sha),
            "--cache-control", "public,max-age=31536000,immutable", "--output", "json"]
    result = aws_call(aws, args)
    if result.returncode == 0:
        return "uploaded"
    diagnostic = result.stderr.strip()
    if not re.search(r"(?:\b412\b|PreconditionFailed|ConditionalRequestConflict)", diagnostic):
        raise PublishError(f"immutable upload failed for s3://{bucket}/{key}: {diagnostic or f'exit {result.returncode}'}")
    remote, error = head_checksum(aws, bucket, key)
    if error:
        raise PublishError(f"immutable upload conflicted for s3://{bucket}/{key}; cannot verify existing object: {error}; put: {diagnostic}")
    if remote != sha_b64(file_sha):
        raise PublishError(f"immutable key already exists with a different or missing SHA-256 checksum: s3://{bucket}/{key}")
    return "already-present-identical"


def put_stable(aws, bucket, key, path):
    file_sha = digest(path)
    result = aws_call(aws, ["put-object", "--bucket", bucket, "--key", key, "--body", str(path),
                            "--checksum-algorithm", "SHA256", "--checksum-sha256", sha_b64(file_sha),
                            "--cache-control", "no-cache,max-age=0,must-revalidate", "--output", "json"])
    if result.returncode:
        raise PublishError(f"stable installer upload failed for s3://{bucket}/{key}: {result.stderr.strip()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-dir", required=True, type=Path)
    parser.add_argument("--distribution-version", required=True)
    parser.add_argument("--source-base", required=True, help="HTTPS base at the fixed /distributions route; embedded source is BASE/VERSION")
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument("--approved-installer-sha256", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--aws", default="aws", help="AWS CLI executable")
    parser.add_argument("--local-validation", action="store_true",
                        help="Offline metadata checks only; incompatible with --execute")
    parser.add_argument("--execute", action="store_true", help="Upload after all local validation succeeds")
    args = parser.parse_args(argv)
    try:
        if args.execute and args.local_validation:
            raise PublishError('--local-validation cannot be combined with --execute')
        root, immutable_files, installer = load_bundle(args.distribution_dir, args.distribution_version,
            args.approved_manifest_sha256, args.approved_installer_sha256, args.source_base)
        print(f"Validated {len(immutable_files)} immutable files for distributions/{args.distribution_version}/")
        print(f"Stable installer: install.sh ({digest(installer)})")
        if not args.local_validation:
            verify_remote_components(root)
        if not args.execute:
            print("Dry run: no AWS command executed. Pass --execute to publish.")
            return 0
        for path in immutable_files:
            key = f"distributions/{args.distribution_version}/{path.name}"
            print(f"{put_immutable(args.aws, args.bucket, key, path)} s3://{args.bucket}/{key}")
        # This is intentionally the only mutable key and is reached only after
        # every versioned immutable object is present and checksum-verified.
        put_stable(args.aws, args.bucket, "install.sh", installer)
        print(f"updated stable installer s3://{args.bucket}/install.sh")
        return 0
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException, subprocess.SubprocessError) as error:
        print(f"publish_distribution: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
