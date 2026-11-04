"""Engine component packaging fixtures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import package_engine


class EnginePackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='engine-release-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=Fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
        self.commit = subprocess.check_output(['git', '-C', str(self.repo), 'rev-parse', 'HEAD'], text=True).strip()
        (self.repo / 'LICENSE.md').write_text('Fixture engine license\n')
        self.binary = Path(shutil.which('true'))
        self.manifest = self.root / 'source-manifest.json'
        self.manifest.write_text(json.dumps({'LICENSE.md': self.sha(self.repo / 'LICENSE.md')}) + '\n')
        self.build = self.root / 'build-provenance.json'
        self.build_data = {'schema_version': 1, 'component': 'engine', 'base_commit': self.commit,
                           'source_manifest_sha256': self.sha(self.manifest), 'cpu_flags': 'avx2'}
        self.build.write_text(json.dumps(self.build_data) + '\n')
        self.out = self.root / 'artifact'

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def package(self):
        # Exercise the CLI in a subprocess to preserve argparse and output guarantees.
        subprocess.run(['python3', str(TOOLS / 'package_engine.py'), '--engine-repo', str(self.repo),
            '--engine-bin', str(self.binary), '--engine-version', '1.2.3', '--output', str(self.out),
            '--build-provenance', str(self.build), '--source-manifest', str(self.manifest)], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_builds_flat_package_checksums_and_provenance(self):
        self.package()
        provenance = json.loads((self.out / 'provenance.json').read_text())
        self.assertEqual(provenance['source_commit'], self.commit)
        self.assertEqual(provenance['build_provenance_sha256'], self.sha(self.out / 'build-provenance.json'))
        sums = (self.out / 'SHA256SUMS').read_text()
        for name in ('engine-1.2.3-linux-x86_64.tar.gz', 'provenance.json', 'build-provenance.json', 'source-manifest.json'):
            self.assertIn(f'{self.sha(self.out / name)}  {name}', sums)
        with tarfile.open(self.out / 'engine-1.2.3-linux-x86_64.tar.gz') as tar:
            self.assertEqual(set(tar.getnames()), {'bin', 'bin/xenolith', 'LICENSE.md'})

    def test_refuses_reused_output_and_wrong_build_source(self):
        self.package()
        with self.assertRaises(subprocess.CalledProcessError):
            self.package()
        self.build_data['base_commit'] = '0' * 40
        self.build.write_text(json.dumps(self.build_data))
        with self.assertRaises(subprocess.CalledProcessError) as error:
            bad_output = self.root.parent / (self.root.name + '-other')
            subprocess.run(['python3', str(TOOLS / 'package_engine.py'), '--engine-repo', str(self.repo),
                '--engine-bin', str(self.binary), '--engine-version', '1.2.3', '--output', str(bad_output),
                '--build-provenance', str(self.build), '--source-manifest', str(self.manifest)],
                check=True, capture_output=True, text=True)
        self.assertIn('source commit does not match', error.exception.stderr)

    def test_rejects_build_metadata_version(self):
        output = self.root.parent / (self.root.name + '-build-metadata')
        with self.assertRaises(subprocess.CalledProcessError) as error:
            subprocess.run(['python3', str(TOOLS / 'package_engine.py'), '--engine-repo', str(self.repo),
                '--engine-bin', str(self.binary), '--engine-version', '1.2.3+build.1', '--output', str(output),
                '--build-provenance', str(self.build), '--source-manifest', str(self.manifest)],
                check=True, capture_output=True, text=True)
        self.assertIn('without build metadata', error.exception.stderr)


if __name__ == '__main__':
    unittest.main()
