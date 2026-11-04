"""Exact candidate workflow and archive release gate tests."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import component_release


class ComponentReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='component-release-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
        self.commit = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        (self.repo / 'LICENSE.md').write_text('fixture license\n')
        self.binary = Path(shutil.which('true'))
        self.manifest = self.root / 'source-manifest.json'
        self.manifest.write_text(json.dumps({'LICENSE.md': self.sha(self.repo / 'LICENSE.md')}) + '\n')
        self.build = self.root / 'build-provenance.json'
        self.build.write_text(json.dumps({'schema_version': 1, 'component': 'engine', 'base_commit': self.commit,
            'source_manifest_sha256': self.sha(self.manifest)}) + '\n')
        self.out = self.root / 'artifact'
        subprocess.run(['python3', str(TOOLS / 'package_engine.py'), '--engine-repo', str(self.repo),
            '--engine-bin', str(self.binary), '--engine-version', '1.2.3', '--output', str(self.out),
            '--build-provenance', str(self.build), '--source-manifest', str(self.manifest)], check=True,
            stdout=subprocess.DEVNULL)
        self.run_file = self.root / 'run.json'
        self.write_run()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_run(self, **changes):
        record = {'id': 12345, 'status': 'completed', 'conclusion': 'success', 'event': 'workflow_dispatch',
                  'path': '.github/workflows/linux-candidate.yml', 'head_branch': 'main', 'head_sha': self.commit,
                  'repository': {'full_name': 'simoneiacomino/xenolith'},
                  'head_repository': {'full_name': 'simoneiacomino/xenolith'}}
        record.update(changes)
        self.run_file.write_text(json.dumps(record))

    def args(self, **changes):
        version = changes.get('version', '1.2.3')
        values = {'artifact_dir': self.out, 'version': version,
                  'sha256': self.sha(self.out / 'engine-1.2.3-linux-x86_64.tar.gz'),
                  'run_metadata': self.run_file, 'expected_run_id': '12345',
                  'expected_repository': 'simoneiacomino/xenolith', 'expected_source_sha': self.commit}
        values.update(changes)
        return type('VerifyArgs', (), values)

    def test_accepts_exact_candidate(self):
        component_release.verify(self.args())

    def test_rejects_wrong_approved_run_identity_and_version(self):
        for field, value, message in [
                ('expected_source_sha', '0' * 40, 'run metadata'),
                ('expected_repository', 'other/repo', 'run metadata'),
                ('expected_run_id', '9', 'run metadata'),
                ('version', '9.9.9', 'artifact file set'),
                ('version', '1.2.3-alpha.01', 'invalid engine SemVer'),
                ('sha256', '0' * 64, 'approved SHA-256')]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                component_release.verify(self.args(**{field: value}))

    def test_rejects_untrusted_run_source_workflow_event_or_conclusion(self):
        cases = [
            ('path', '.github/workflows/other.yml'), ('event', 'push'), ('status', 'in_progress'),
            ('conclusion', 'failure'), ('head_repository', {'full_name': 'other/repo'}),
            ('repository', {'full_name': 'other/repo'}), ('head_branch', 'feature/topic'),
        ]
        for field, value in cases:
            self.write_run(**{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'run metadata'):
                component_release.verify(self.args())
            self.write_run()

    def test_rejects_modified_archive_or_release_payload(self):
        archive = self.out / 'engine-1.2.3-linux-x86_64.tar.gz'
        archive.write_bytes(archive.read_bytes() + b'altered')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            component_release.verify(self.args())

    def test_rejects_modified_source_files_even_when_sums_are_rewritten(self):
        license_path = self.root / 'license'
        license_path.write_text('tampered')
        provenance = self.out / 'provenance.json'
        data = json.loads(provenance.read_text())
        data['package_files']['LICENSE.md'] = self.sha(license_path)
        provenance.write_text(json.dumps(data) + '\n')
        sums = self.out / 'SHA256SUMS'
        names = ['engine-1.2.3-linux-x86_64.tar.gz', 'provenance.json', 'build-provenance.json', 'source-manifest.json']
        sums.write_text(''.join(f'{self.sha(self.out / name)}  {name}\n' for name in names))
        with self.assertRaisesRegex(ValueError, 'payload does not match'):
            component_release.verify(self.args())


if __name__ == '__main__':
    unittest.main()
