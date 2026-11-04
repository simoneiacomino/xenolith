#!/usr/bin/env python3
"""Package a prebuilt Xenolith Linux binary as an engine component release."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

from distribution_metadata import validate_output
from engine_version import validate as validate_engine_version
from elf_requirements import inspect_tree, version_key


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def expected_build_snapshot(repo):
    paths = set()
    for pattern in ('*.c', '*.h', '*.cl', 'Makefile'):
        paths.update(path for path in repo.glob(pattern) if path.is_file())
    vendor = repo / 'vendor'
    if vendor.is_dir():
        paths.update(path for path in vendor.rglob('*') if path.is_file())
    paths.add(repo / 'LICENSE.md')
    return {path.relative_to(repo).as_posix(): digest(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-repo', type=Path, required=True)
    parser.add_argument('--engine-bin', type=Path, required=True)
    parser.add_argument('--engine-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-provenance', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--max-glibc', default='2.39')
    args = parser.parse_args()
    try:
        repo = args.engine_repo.resolve(strict=True)
        binary = args.engine_bin.resolve(strict=True)
        build_prov_path = args.build_provenance.resolve(strict=True)
        source_manifest_path = args.source_manifest.resolve(strict=True)
        if not repo.is_dir() or not binary.is_file() or binary.stat().st_size == 0:
            raise ValueError('engine repo and binary must be a directory and regular file')
        if not binary.stat().st_mode & 0o111:
            raise ValueError('engine binary must be executable')
        if binary.read_bytes()[:4] != b'\x7fELF':
            raise ValueError('engine binary must be an ELF executable')
        if not (repo / 'LICENSE.md').is_file():
            raise ValueError('engine repository is missing LICENSE.md')
        version = validate_engine_version(args.engine_version)
        if not re.fullmatch(r'[0-9]+\.[0-9]+', args.max_glibc):
            raise ValueError('--max-glibc must be major.minor')
        build_prov = json.loads(build_prov_path.read_text())
        source_manifest = json.loads(source_manifest_path.read_text())
        if build_prov.get('component') != 'engine' or build_prov.get('schema_version') != 1:
            raise ValueError('build provenance is not a supported engine build record')
        if build_prov.get('source_manifest_sha256') != digest(source_manifest_path):
            raise ValueError('build provenance does not match source manifest')
        if not isinstance(source_manifest, dict) or not source_manifest:
            raise ValueError('source manifest must be a nonempty object')
        if source_manifest != expected_build_snapshot(repo):
            raise ValueError('build source manifest does not match the engine build inputs')
        if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != build_prov.get('base_commit'):
            raise ValueError('build provenance source commit does not match engine checkout')
        output = validate_output(args.output, [repo], [binary, build_prov_path, source_manifest_path])
        if output.exists() and any(output.iterdir()):
            raise ValueError('output must be a fresh or empty directory')
        stage = Path(tempfile.mkdtemp(prefix='engine-package-'))
        try:
            root = stage / 'package'
            (root / 'bin').mkdir(parents=True)
            shutil.copy2(binary, root / 'bin/xenolith')
            shutil.copy2(repo / 'LICENSE.md', root / 'LICENSE.md')
            abi = inspect_tree(root)
            minimum = abi['glibc']
            if minimum and version_key(minimum) > version_key(args.max_glibc):
                raise ValueError(f'engine requires glibc {minimum}, exceeding {args.max_glibc}')
            archive = output / f'engine-{version}-linux-x86_64.tar.gz'
            output.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, 'w:gz') as tar:
                for entry in sorted(root.iterdir()):
                    tar.add(entry, arcname=entry.name, recursive=True)
            provenance = {
                'schema_version': 1,
                'component': 'engine',
                'version': version,
                'source_commit': build_prov['base_commit'],
                'binary_sha256': digest(binary),
                'package_files': {'bin/xenolith': digest(root / 'bin/xenolith'), 'LICENSE.md': digest(root / 'LICENSE.md')},
                'archive': {'file': archive.name, 'size_bytes': archive.stat().st_size, 'sha256': digest(archive)},
                'build_provenance_sha256': digest(build_prov_path),
                'source_manifest_sha256': digest(source_manifest_path),
                'build_provenance': build_prov,
                'abi': abi,
                'max_glibc': args.max_glibc,
            }
            (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            shutil.copyfile(build_prov_path, output / 'build-provenance.json')
            shutil.copyfile(source_manifest_path, output / 'source-manifest.json')
            files = [archive, output / 'provenance.json', output / 'build-provenance.json', output / 'source-manifest.json']
            (output / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(files)))
            print(f'Engine package: {archive}')
        finally:
            shutil.rmtree(stage)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
