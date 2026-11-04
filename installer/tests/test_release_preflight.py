"""Fail-closed release/tag existence guard and publication workflow contract."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import release_preflight


class Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ReleasePreflightTest(unittest.TestCase):
    def test_only_authenticated_404s_mean_absent(self):
        observed = []

        def open_request(request, timeout):
            observed.append((request.full_url, request.get_header('Authorization'), timeout))
            return Response(404)

        with patch.object(release_preflight, 'urlopen', open_request):
            release_preflight.check('owner/repo', 'v1.2.3', token='secret', api_url='https://api.example')
        self.assertEqual([entry[0] for entry in observed], [
            'https://api.example/repos/owner/repo/releases/tags/v1.2.3',
            'https://api.example/repos/owner/repo/git/ref/tags/v1.2.3'])
        self.assertTrue(all(entry[1] == 'Bearer secret' for entry in observed))

    def test_existing_release_or_tag_is_rejected(self):
        for status in (200, 201):
            with self.subTest(status=status), patch.object(release_preflight, 'urlopen', lambda *a, **k: Response(status)):
                with self.assertRaisesRegex(RuntimeError, 'already exists'):
                    release_preflight.check('owner/repo', 'v1', token='token', api_url='https://api.example')

    def test_auth_server_and_transport_errors_fail_closed(self):
        from urllib.error import HTTPError
        for status in (401, 403, 500):
            def fail(request, timeout, status=status):
                raise HTTPError(request.full_url, status, 'failure', {}, None)
            with self.subTest(status=status), patch.object(release_preflight, 'urlopen', fail):
                with self.assertRaisesRegex(RuntimeError, f'HTTP {status}'):
                    release_preflight.check('owner/repo', 'v1', token='token', api_url='https://api.example')
        with patch.object(release_preflight, 'urlopen', side_effect=OSError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                release_preflight.check('owner/repo', 'v1', token='token', api_url='https://api.example')
        with self.assertRaisesRegex(RuntimeError, 'GH_TOKEN is required'):
            release_preflight.check('owner/repo', 'v1', token='')

    def test_workflow_binds_target_commit_and_uses_notes_file(self):
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/release-engine.yml').read_text()
        self.assertIn('GH_TOKEN: ${{ github.token }}', workflow)
        self.assertIn('ARCHIVE_SHA256: ${{ inputs.archive_sha256 }}', workflow)
        self.assertIn('[[ "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]]', workflow)
        self.assertIn('release_preflight.py --repository "$GITHUB_REPOSITORY" --tag "$TAG"', workflow)
        self.assertIn('--target "$CANDIDATE_SHA"', workflow)
        self.assertIn('--notes-file "$RUNNER_TEMP/release-notes.md"', workflow)
        self.assertNotIn('`${{ inputs.archive_sha256 }}`', workflow)
        self.assertNotIn('"${{ inputs.archive_sha256 }}"', workflow)
        self.assertIn('engine_version.py "$VERSION" --prerelease-value', workflow)
        self.assertNotIn('alpha|beta|rc|pre', workflow)


if __name__ == '__main__':
    unittest.main()
