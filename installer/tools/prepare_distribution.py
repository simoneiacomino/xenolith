#!/usr/bin/env python3
"""Assemble installer metadata from prebuilt, version-pinned component archives.

Each component repository owns its build and GitHub Release. This tool inspects
archives without executing their payloads and never rebuilds or republishes them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from elf_requirements import inspect_tree, version_key
from distribution_metadata import (component_url, render_installer, safe_source,
                                     safe_version, source_identity, validate_output)

MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
MAX_MEMBERS = 200000


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack_component(archive_path, root):
    """Extract bounded regular files into private staging, never following links."""
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError('Component archive must be a regular file, not a symlink')
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Component archive exceeds 2 GiB limit')
    seen = set()
    total = 0
    try:
        with tarfile.open(archive_path, 'r:gz') as archive:
            for member in archive:
                name = member.name.rstrip('/') if member.isdir() else member.name
                parts = name.split('/')
                if (not name or any(part in ('', '.', '..') for part in parts) or
                        not re.fullmatch(r'[A-Za-z0-9_@./+,-]+', name) or
                        ':' in name or name in seen):
                    raise ValueError(f'Unsafe or duplicate component archive path: {member.name!r}')
                seen.add(name)
                if len(seen) > MAX_MEMBERS:
                    raise ValueError('Component archive contains too many members')
                destination = root / name
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or member.mode & 0o7000:
                    raise ValueError(f'Component archive contains a link, special file or privileged mode: {name}')
                total += member.size
                if member.size < 0 or total > MAX_EXTRACTED_BYTES:
                    raise ValueError('Component archive exceeds 4 GiB extracted limit')
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f'Cannot read component archive member: {name}')
                with source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if destination.stat().st_size != member.size:
                    raise ValueError(f'Truncated component archive member: {name}')
                destination.chmod(member.mode & 0o777)
    except (tarfile.TarError, EOFError) as exc:
        raise ValueError(f'Invalid component archive: {exc}') from exc


def require_file(root, relative, executable=False):
    path = root / relative
    if not path.is_file() or (executable and not path.stat().st_mode & 0o111):
        raise ValueError(f'Component archive requires {relative}' + (' executable' if executable else ''))
    return path


def require_linux_elf(path):
    with path.open('rb') as stream:
        header = stream.read(20)
    if (len(header) != 20 or header[:7] != b'\x7fELF\x02\x01\x01' or header[7] not in (0, 3) or
            int.from_bytes(header[16:18], 'little') not in (2, 3) or header[18:20] != b'\x3e\x00'):
        raise ValueError(f'Component executable must be Linux x86-64 ELF: {path.name}')


def inspect_component(name, version, archive, root, max_glibc, development=False):
    unpack_component(archive, root)
    entry = {'engine': 'bin/xenolith', 'xenopi': 'bin/xenopi', 'codexe': 'bin/codex'}[name]
    executable = require_file(root, entry, executable=True)
    metadata = None
    if name == 'xenopi':
        executable = require_file(root, 'node/bin/node', executable=True)
        metadata_path = require_file(root, 'lib/xenopi/package.json')
        if metadata_path.stat().st_size > 1024 * 1024:
            raise ValueError('XenoPi metadata exceeds 1 MiB limit')
        metadata = json.loads(metadata_path.read_text())
        if (not isinstance(metadata, dict) or metadata.get('name') != 'xenopi' or
                metadata.get('version') != version):
            raise ValueError('XenoPi metadata must be an object with name xenopi and version matching --xenopi-version')
        require_file(root, 'lib/xenopi/dist/src/bin/xenopi.js')
    elif name == 'codexe':
        require_file(root, 'install/install.sh')
        require_file(root, 'codexe-package_SHA256SUMS')
        metadata_path = require_file(root, 'codex-package.json')
        if metadata_path.stat().st_size > 1024 * 1024:
            raise ValueError('CodeXe metadata exceeds 1 MiB limit')
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict) or metadata.get('version') != version:
            raise ValueError('CodeXe metadata version must match --codexe-version')
    if not development:
        require_linux_elf(executable)
    abi = inspect_tree(root)
    minimum = abi['glibc']
    if max_glibc and minimum and version_key(minimum) > version_key(max_glibc):
        raise ValueError(f'{name} requires glibc {minimum}, exceeding requested baseline {max_glibc}')
    return abi, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-repo', type=Path, required=True, help='Source of GPU/model metadata; does not prove archive source identity')
    parser.add_argument('--distribution-version', required=True)
    for name in ('engine', 'xenopi', 'codexe'):
        required = name == 'engine'
        parser.add_argument(f'--{name}-archive', type=Path, required=required)
        parser.add_argument(f'--{name}-version', required=required)
        parser.add_argument(f'--{name}-url', required=required)
    parser.add_argument('--cpu-flags', required=True, help='Explicit CPU ISA declaration for the prebuilt engine')
    parser.add_argument('--max-glibc', help='Reject any component exceeding this major.minor glibc baseline')
    parser.add_argument('--default-source', help='Embed the distribution metadata directory/URL and manifest checksum')
    parser.add_argument('--development-components', action='store_true', help='Local tests only: allow explicit loopback HTTP component URLs and simulated executables; cannot be promoted')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    template_path = Path(__file__).resolve().parents[1] / 'install.sh'
    try:
        safe_version(args.distribution_version)
        if not re.fullmatch(r'[a-z0-9_]+(?: [a-z0-9_]+)*', args.cpu_flags):
            raise ValueError('--cpu-flags must be a space-separated list of lowercase CPU flags')
        if args.max_glibc and not re.fullmatch(r'[0-9]+\.[0-9]+', args.max_glibc):
            raise ValueError('--max-glibc must be a major.minor version')
        if args.default_source is not None:
            safe_source(args.default_source)
        repo = args.engine_repo.resolve(strict=True)
        if not repo.is_dir():
            raise ValueError('--engine-repo must be a directory')
        components = {}
        for name in ('engine', 'xenopi', 'codexe'):
            archive, version, url = (getattr(args, f'{name}_{field}') for field in ('archive', 'version', 'url'))
            if not any(value is not None for value in (archive, version, url)):
                continue
            if any(value is None for value in (archive, version, url)):
                raise ValueError(f'--{name}-archive, --{name}-version and --{name}-url must be supplied together')
            safe_version(version)
            component_url(name, version, url, archive.name, development=args.development_components)
            components[name] = {'path': archive, 'version': version, 'url': url}
        output = validate_output(args.output, [repo], [item['path'] for item in components.values()])
        model_manifest = json.loads((repo / 'tests/vision/manifest.json').read_text())
        safe_version(model_manifest['model_revision'])
        records, abi, provenance_components = [], {}, {}
        # Staging and all validation precede any candidate output writes.
        with tempfile.TemporaryDirectory(prefix='xenolith-inspect-') as scratch:
            for name, item in components.items():
                root = Path(scratch) / name
                root.mkdir()
                abi[name], metadata = inspect_component(name, item['version'], item['path'], root,
                                                        args.max_glibc, args.development_components)
                size, digest = item['path'].stat().st_size, sha(item['path'])
                records.append((name, item['version'], item['url'], str(size), digest))
                provenance_components[name] = {
                    'version': item['version'], 'origin': 'prebuilt-component-release',
                    'archive': {'file': item['path'].name, 'url': item['url'], 'size_bytes': size, 'sha256': digest},
                    'archive_source_match': 'not-verified',
                }
                if metadata is not None:
                    metadata_path = root / ('lib/xenopi/package.json' if name == 'xenopi' else 'codex-package.json')
                    provenance_components[name]['package_metadata_sha256'] = sha(metadata_path)
        code = (repo / 'xenolith.c').read_text()
        classify = code.split('static xe_gpu_status xe_gpu_classify', 1)[1].split('void *xe_vision_alloc_shared', 1)[0]
        tested, rest = classify.split('return XE_GPU_TESTED;', 1)
        untested = rest.split('return XE_GPU_UNTESTED;', 1)[0]
        matrix = ''.join(f'{device}\t{state}\n' for state, text in (('tested', tested), ('untested', untested))
                         for device in re.findall(r'case 0x([0-9a-f]{4}):', text))
        if not matrix:
            raise ValueError('Engine metadata source has no supported GPU entries')
        model_records = []
        for name, item in zip(('model', 'projector'), model_manifest['artifacts']):
            if (not re.fullmatch(r'https://[^\s]+', item['url']) or
                    not isinstance(item['size_bytes'], int) or item['size_bytes'] <= 0 or
                    not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
                raise ValueError('Invalid model artifact metadata')
            model_records.append((name, model_manifest['model_revision'], item['url'], str(item['size_bytes']), item['sha256']))
        if not model_records:
            raise ValueError('Engine metadata source has no model artifact')
        template = template_path.read_text()
        # Validate rendering before output creation too.
        if args.default_source is not None:
            render_installer(template, args.default_source, '0' * 64)
        output.mkdir(parents=True, exist_ok=True)
        def add_metadata(name, filename, content):
            path = output / filename
            path.write_text(content)
            records.append((name, args.engine_version, filename, str(path.stat().st_size), sha(path)))
        add_metadata('gpu-matrix', 'gpu-matrix.tsv', matrix)
        content = f"cpu_flags\t{args.cpu_flags}\nglibc\t{abi['engine']['glibc'] or '0.0'}\n"
        for name in ('xenopi', 'codexe'):
            if name in abi:
                content += f"{name}_glibc\t{abi[name]['glibc'] or '0.0'}\n"
        add_metadata('requirements', 'requirements.tsv', content)
        (output / 'elf-requirements.json').write_text(json.dumps(abi, indent=2) + '\n')
        records += model_records
        (output / 'manifest.tsv').write_text('# xenolith-distribution-v1\n' + ''.join('\t'.join(r) + '\n' for r in records))
        manifest_hash = sha(output / 'manifest.tsv')
        (output / 'install.sh').write_text(render_installer(template, args.default_source, manifest_hash) if args.default_source else template)
        (output / 'install.sh').chmod(template_path.stat().st_mode & 0o777)
        provenance = {'schema_version': 1, 'distribution_version': args.distribution_version,
                      'development_components': args.development_components,
                      'components': provenance_components,
                      'engine_metadata_source': {'checkout': source_identity(repo), 'role': 'hardware-and-model-metadata',
                                                 'archive_source_match': 'not-verified'},
                      'abi_report_sha256': sha(output / 'elf-requirements.json'),
                      'installer': {'default_source': args.default_source,
                                    'pinned_manifest_sha256': manifest_hash if args.default_source else None}}
        (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        (output / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(output.iterdir()) if p.name != 'SHA256SUMS'))
        print(f'Distribution metadata: {output}')
        print(f'Manifest SHA-256: {manifest_hash}')
    except (ValueError, OSError, KeyError, IndexError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
