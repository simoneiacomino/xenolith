"""Shared distribution identity, pinned component URLs and installer rendering."""
import hashlib
import re
import shlex
import subprocess
from urllib.parse import urlsplit


def safe_version(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', value):
        raise ValueError('Version must be 1-128 letters/digits/dots/underscores/hyphens, starting with a letter or digit')
    return value


def safe_source(value):
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError('Default source must not be empty or contain control characters')
    if value.startswith('/'):
        return value
    if any(char.isspace() for char in value):
        raise ValueError('Default source URL must percent-encode whitespace')
    url = urlsplit(value)
    # Accessing .port validates malformed and out-of-range port declarations.
    port = url.port
    if url.query or url.fragment or url.username or url.password:
        raise ValueError('Default source URL must be a directory URL without credentials, query or fragment')
    if url.scheme == 'https' and url.hostname:
        return value
    if url.scheme == 'http' and url.hostname in ('localhost', '127.0.0.1') and port:
        return value
    raise ValueError('Default source must be an absolute directory, HTTPS URL, or loopback HTTP URL with port')


def validate_output(output, directories, files=()):
    output = output.resolve()
    for source in directories:
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('--output must be outside and must not contain any input/source directory')
    for source in files:
        source = source.resolve()
        if output == source or source.is_relative_to(output):
            raise ValueError('--output must not contain an input file')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('--output must be absent or an empty directory; previous artifacts are preserved')
    return output


def source_identity(repo):
    """Checkout observation, never proof that a supplied binary came from it."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL)
    try:
        head = git('rev-parse', 'HEAD').decode().strip()
        status = git('status', '--porcelain=v1', '-z', '--untracked-files=all')
        diff = git('diff', '--binary', 'HEAD')
        return {'head': head, 'dirty': bool(status), 'tracked_diff_sha256': hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {'head': None, 'dirty': None, 'tracked_diff_sha256': None}


def render_installer(template, source, manifest_sha256):
    safe_source(source)
    if not re.fullmatch(r'[0-9a-f]{64}', manifest_sha256):
        raise ValueError('Manifest digest must be lowercase SHA-256')
    lines = template.splitlines(keepends=True)
    replacements = {'DEFAULT_SOURCE': source, 'DEFAULT_MANIFEST_SHA256': manifest_sha256}
    for key, value in replacements.items():
        matches = [i for i, line in enumerate(lines) if line.rstrip('\n') == key + "=''"]
        if len(matches) != 1:
            raise ValueError(f'Installer template must contain exactly one standalone {key}=\'\' line')
        i = matches[0]
        lines[i] = key + '=' + shlex.quote(value) + ('\n' if lines[i].endswith('\n') else '')
    return ''.join(lines)


COMPONENT_RELEASES = {
    'engine': ('simoneiacomino/xenolith', 'v'),
    'xenopi': ('simoneiacomino/xenopi', 'v'),
    'codexe': ('crrisci/codexe', 'codexe-v'),
}


def component_url(name, version, location, archive_name=None, development=False):
    """Validate an exact, version-pinned official Release asset URL.

    Explicit development mode also permits a loopback HTTP URL. Promotion never enables that mode. No local input path is inferred.
    """
    safe_version(version)
    if name not in COMPONENT_RELEASES:
        raise ValueError('Unknown component')
    if not isinstance(location, str) or not location or any(ord(c) < 32 or ord(c) == 127 for c in location):
        raise ValueError('Component URL must be nonempty and contain no control characters')
    parsed = urlsplit(location)
    port = parsed.port
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError('Component URL must not contain credentials, query or fragment')
    if development and location.startswith('/'):
        raise ValueError('Absolute component locations are not supported by the installer; serve the archive over loopback HTTP and pass its URL')
    if development and parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost') and port:
        if any(char.isspace() for char in location) or '\\' in location:
            raise ValueError('Development component URL must percent-encode whitespace and contain no backslashes')
        return location
    repo, prefix = COMPONENT_RELEASES[name]
    # Exact path spelling rejects encoded traversal, extra segments and /latest.
    expected = f'/{repo}/releases/download/{prefix}{version}/'
    asset = parsed.path[len(expected):] if parsed.path.startswith(expected) else ''
    if (parsed.scheme != 'https' or parsed.netloc != 'github.com' or
            not asset or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,180}\.tar\.gz', asset) or
            (archive_name is not None and asset != archive_name)):
        raise ValueError(f'{name} must use an exact official GitHub Release URL with tag {prefix}{version} and matching archive basename')
    return location
