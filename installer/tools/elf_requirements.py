"""Read binary ABI requirements without executing packaged programs."""
import re
import subprocess


def inspect_elf(path):
    with path.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            return None
    versions = subprocess.check_output(['readelf', '--version-info', str(path)], text=True)
    dynamic = subprocess.check_output(['readelf', '--dynamic', str(path)], text=True)
    required = re.findall(r'GLIBC_(\d+\.\d+)', versions)
    return {'glibc': max(required, key=version_key) if required else None,
            'needed': sorted(set(re.findall(r'\(NEEDED\).*\[(.*?)\]', dynamic)))}


def version_key(value):
    return tuple(map(int, value.split('.')))


def inspect_tree(root):
    binaries = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        info = inspect_elf(path)
        if info is not None:
            binaries[str(path.relative_to(root))] = info
    versions = [info['glibc'] for info in binaries.values() if info['glibc']]
    return {'glibc': max(versions, key=version_key) if versions else None, 'binaries': binaries}
