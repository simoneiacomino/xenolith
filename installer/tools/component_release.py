#!/usr/bin/env python3
"""Validate an exact engine candidate artifact and its originating workflow run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
from engine_version import validate as validate_engine_version


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_sums(root):
    sums = root / 'SHA256SUMS'
    seen = set()
    for line in sums.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  ([A-Za-z0-9._-]+)', line)
        if not match:
            raise ValueError('malformed SHA256SUMS entry')
        expected, filename = match.groups()
        if filename in seen or filename == 'SHA256SUMS':
            raise ValueError('duplicate or recursive SHA256SUMS entry')
        seen.add(filename)
        path = root / filename
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f'candidate checksum mismatch: {filename}')
    return seen


def verify(args):
    root = args.artifact_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('artifact directory must exist')
    version = validate_engine_version(args.version)
    run = json.loads(args.run_metadata.read_text())
    if (str(run.get('id')) != str(args.expected_run_id) or run.get('status') != 'completed' or
            run.get('conclusion') != 'success' or run.get('event') != 'workflow_dispatch' or
            run.get('path') != '.github/workflows/linux-candidate.yml' or
            not (run.get('head_branch') == 'main' or str(run.get('head_branch', '')).startswith('release/')) or
            run.get('repository', {}).get('full_name') != args.expected_repository or
            run.get('head_repository', {}).get('full_name') != args.expected_repository or
            run.get('head_sha') != args.expected_source_sha):
        raise ValueError('candidate run metadata does not match the approved repository, workflow, run, or source commit')
    required = verify_sums(root)
    archive_name = f'engine-{version}-linux-x86_64.tar.gz'
    expected_files = {archive_name, 'provenance.json', 'build-provenance.json', 'source-manifest.json'}
    if required != expected_files:
        raise ValueError('candidate artifact file set is incomplete or contains unexpected files')
    archive = root / archive_name
    if not re.fullmatch(r'[0-9a-f]{64}', args.sha256) or sha_file(archive) != args.sha256:
        raise ValueError('archive digest does not match approved SHA-256')
    provenance = json.loads((root / 'provenance.json').read_text())
    build = json.loads((root / 'build-provenance.json').read_text())
    source_manifest = json.loads((root / 'source-manifest.json').read_text())
    if (provenance.get('schema_version') != 1 or provenance.get('component') != 'engine' or
            provenance.get('version') != version or provenance.get('source_commit') != args.expected_source_sha or
            provenance.get('archive', {}).get('file') != archive_name or
            provenance.get('archive', {}).get('sha256') != args.sha256 or
            provenance.get('build_provenance_sha256') != sha_file(root / 'build-provenance.json') or
            provenance.get('source_manifest_sha256') != sha_file(root / 'source-manifest.json') or
            build.get('component') != 'engine' or build.get('base_commit') != args.expected_source_sha or
            build.get('source_manifest_sha256') != sha_file(root / 'source-manifest.json') or
            not isinstance(source_manifest, dict) or not source_manifest):
        raise ValueError('component provenance does not bind the requested version and workflow source')
    payload = {}
    try:
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar.getmembers():
                name = member.name
                if name in payload or name not in ('bin', 'bin/xenolith', 'LICENSE.md'):
                    raise ValueError(f'unexpected or duplicate archive member: {name}')
                if member.isdir() and name == 'bin':
                    payload[name] = None
                elif member.isreg() and name in ('bin/xenolith', 'LICENSE.md'):
                    if name == 'bin/xenolith' and member.mode & 0o111 == 0:
                        raise ValueError('packaged engine binary is not executable')
                    stream = tar.extractfile(member)
                    payload[name] = stream.read() if stream else None
                else:
                    raise ValueError(f'unsafe archive member: {name}')
    except (tarfile.TarError, EOFError) as exc:
        raise ValueError(f'invalid engine package archive: {exc}') from exc
    if set(payload) != {'bin', 'bin/xenolith', 'LICENSE.md'}:
        raise ValueError('engine package is missing required files')
    if not payload['bin/xenolith'] or not payload['bin/xenolith'].startswith(b'\x7fELF'):
        raise ValueError('packaged engine binary is empty or not ELF')
    if (sha_bytes(payload['bin/xenolith']) != provenance.get('binary_sha256') or
            provenance.get('package_files') != {
                'bin/xenolith': sha_bytes(payload['bin/xenolith']),
                'LICENSE.md': sha_bytes(payload['LICENSE.md']),
            }):
        raise ValueError('archive payload does not match component provenance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--run-metadata', type=Path, required=True)
    parser.add_argument('--expected-run-id', required=True)
    parser.add_argument('--expected-repository', required=True)
    parser.add_argument('--expected-source-sha', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    try:
        verify(args)
        print('Candidate archive and source workflow provenance verified.')
    except (ValueError, OSError, KeyError, json.JSONDecodeError, tarfile.TarError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
