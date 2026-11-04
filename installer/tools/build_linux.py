#!/usr/bin/env python3
"""Build local Linux artifacts in an isolated Ubuntu 24.04 userspace.

This is maintainer tooling, never invoked by install.sh. It does not qualify
every driver/GPU and deliberately retains the engine's current AVX-VNNI ISA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from build_baseline import ENGINE_CFLAGS, ENGINE_CPU_FLAGS


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component', choices=('engine',), default='engine')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image', default='xenolith-build:ubuntu24-local')
    p.add_argument('--build-image', action='store_true')
    args = p.parse_args()
    source = args.source.resolve(strict=True)
    out = args.output.resolve()
    if not source.is_dir():
        p.error('--source must be a directory')
    if out == source or out.is_relative_to(source):
        p.error('--output must be outside --source to avoid recursively copying the build output')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error('--output must be empty; retain previous build evidence separately')
    if args.build_image:
        packaging = Path(__file__).resolve().parents[1] / 'packaging'
        run(['docker', 'build', '-t', args.image, '-f', packaging / 'Dockerfile.ubuntu24', packaging])
    # Container sees only a source snapshot and this output directory. Neither
    # the user's home nor Docker socket is mounted into the build container.
    work = out / 'source'
    work.mkdir()
    for pattern in ('*.c', '*.h', '*.cl', 'Makefile'):
        for path in source.glob(pattern):
            shutil.copy2(path, work / path.name)
    shutil.copytree(source / 'vendor', work / 'vendor')
    shutil.copy2(source / 'LICENSE.md', work / 'LICENSE.md')
    script = f'''set -eu
cd /work/source
# Ubuntu's ocloc23 spells the Xe-LP feature baseline tgllp, not xe-lp.
# Emit portable SPIR-V; native GPU code is still compiled at engine startup.
ocloc compile -file xenolith.cl -device tgllp -spv_only -output xenolith_gpu -output_no_suffix -out_dir . -options '-cl-std=CL3.0' -q
make -j2 "BASE_CFLAGS={ENGINE_CFLAGS}" "BUILD_COMMIT=$BUILD_COMMIT-local" xenolith
mkdir -p /work/artifact/bin
cp xenolith /work/artifact/bin/xenolith
cp LICENSE.md /work/artifact/
./xenolith config inspect --json > /work/inspect.json
'''
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    (out / 'build.sh').write_text(script)
    source_manifest = {str(path.relative_to(work)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(work.rglob('*')) if path.is_file() and not path.is_symlink()}
    (out / 'source-manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    image_info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image], text=True))[0]
    provenance = {'schema_version': 1, 'component': args.component, 'base_commit': commit,
                  'image_id': image_info['Id'], 'repo_digests': image_info.get('RepoDigests', []),
                  'baseline': 'Ubuntu 24.04 userspace', 'model_loaded': False,
                  'build_script_sha256': hashlib.sha256(script.encode()).hexdigest(),
                  'source_manifest_sha256': hashlib.sha256((out / 'source-manifest.json').read_bytes()).hexdigest(),
                  'cpu_flags': ENGINE_CPU_FLAGS}
    (out / 'build-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    run(['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
         '--cap-drop=ALL', '--security-opt=no-new-privileges', '--cpus=2',
         '--mount', f'type=bind,source={out},target=/work',
         '-e', 'HOME=/work/home',
         '-e', f'BUILD_COMMIT={commit[:9]}', args.image, 'sh', '/work/build.sh'])
    print(f'Built {args.component}: {out / "artifact"}. Inspect ABI and runtime requirements before distribution.')


if __name__ == '__main__':
    main()
