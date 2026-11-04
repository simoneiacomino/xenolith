#!/usr/bin/env python3
"""Fail closed unless both a component tag and release are absent."""
import argparse
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def require_absent(url, headers):
    request = Request(url, headers=headers, method='GET')
    try:
        with urlopen(request, timeout=20) as response:
            status = response.status
    except HTTPError as error:
        status = error.code
        error.close()
    except (OSError, URLError) as error:
        raise RuntimeError(f'GitHub API request failed: {error}') from error
    if status == 404:
        return
    if 200 <= status < 300:
        raise RuntimeError(f'tag or release already exists ({url})')
    raise RuntimeError(f'GitHub API returned HTTP {status} while checking {url}')


def check(repo, tag, *, token=None, api_url=None):
    token = token if token is not None else os.environ.get('GH_TOKEN')
    if not token:
        raise RuntimeError('GH_TOKEN is required to check release and tag existence')
    if '/' not in repo or not tag:
        raise ValueError('repository must be OWNER/REPO and tag must be nonempty')
    root = (api_url or os.environ.get('GITHUB_API_URL') or 'https://api.github.com').rstrip('/')
    encoded_tag = quote(tag, safe='')
    headers = {'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {token}',
               'X-GitHub-Api-Version': '2022-11-28'}
    require_absent(f'{root}/repos/{repo}/releases/tags/{encoded_tag}', headers)
    require_absent(f'{root}/repos/{repo}/git/ref/tags/{encoded_tag}', headers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    try:
        check(args.repository, args.tag)
        print(f'{args.tag}: no existing release or git tag found')
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
