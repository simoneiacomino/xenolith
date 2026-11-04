#!/usr/bin/env python3
"""Strict engine release SemVer; build metadata is omitted from public labels."""
import argparse
import re


_IDENTIFIER = r'(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)'
_BUILD_IDENTIFIER = r'[0-9A-Za-z-]+'
_SEMVER = re.compile(
    rf'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'
    rf'(?:-({_IDENTIFIER}(?:\.{_IDENTIFIER})*))?'
    rf'(?:\+({_BUILD_IDENTIFIER}(?:\.{_BUILD_IDENTIFIER})*))?'
)


def validate(value):
    if isinstance(value, str) and '+' in value:
        raise ValueError('engine release versions use SemVer without build metadata (+...); provenance records build identity')
    if not isinstance(value, str) or not _SEMVER.fullmatch(value):
        raise ValueError(f'invalid engine SemVer version: {value!r}')
    return value


def is_prerelease(value):
    validate(value)
    return '-' in value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version')
    parser.add_argument('--prerelease-value', action='store_true', help='print true or false after validation')
    args = parser.parse_args()
    try:
        validate(args.version)
    except ValueError as error:
        parser.error(str(error))
    if args.prerelease_value:
        print('true' if is_prerelease(args.version) else 'false')


if __name__ == '__main__':
    main()
