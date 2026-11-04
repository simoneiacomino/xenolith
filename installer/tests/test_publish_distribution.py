#!/usr/bin/env python3
"""Local promotion tests with a deterministic fake AWS CLI."""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import urllib.error

INSTALLER = Path(__file__).resolve().parents[1]
TOOL = INSTALLER / "tools/publish_distribution.py"
WORKFLOW = INSTALLER.parent / ".github/workflows/publish-distribution.yml"
sys.path.insert(0, str(INSTALLER / "tools"))
from distribution_metadata import render_installer
import publish_distribution as publisher


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PublishDistribution(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="xe-publish-test-")
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.version = "candidate-1"
        self.source_base = "https://xenolith.it/distributions"
        self.source = f"{self.source_base}/{self.version}"
        self.bucket = "xenolith-test-bucket"
        self.state = self.root / "aws-state.json"
        self.calls = self.root / "aws-calls.jsonl"
        self.fake_aws = self.root / "aws"
        self.fake_aws.write_text("""#!/usr/bin/env python3
import base64, json, os, sys
args = sys.argv[1:]
with open(os.environ['FAKE_AWS_CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')
state_path = os.environ['FAKE_AWS_STATE']
try:
    with open(state_path) as f: state = json.load(f)
except FileNotFoundError: state = {}
def val(flag): return args[args.index(flag) + 1]
if args[0] != 's3api': sys.exit(90)
if args[1] == 'put-object':
    key = val('--key')
    if key == os.environ.get('FAKE_AWS_FAIL_KEY'):
        print('injected put failure', file=sys.stderr); sys.exit(42)
    if '--if-none-match' in args and key in state:
        print('PreconditionFailed', file=sys.stderr); sys.exit(1)
    state[key] = val('--checksum-sha256')
    with open(state_path, 'w') as f: json.dump(state, f)
    print('{}')
elif args[1] == 'head-object':
    key = val('--key')
    if key not in state:
        print('NotFound', file=sys.stderr); sys.exit(1)
    print(json.dumps({'ChecksumSHA256': state[key]}))
else: sys.exit(91)
""")
        self.fake_aws.chmod(0o755)
        self.make_bundle()

    def tearDown(self):
        self.temp.cleanup()

    def make_bundle(self, *, source=None, version=None, extra=None, component_url=None, duplicate_name=False):
        for item in self.bundle.iterdir():
            if item.is_file() or item.is_symlink(): item.unlink()
            else: raise AssertionError("test fixture should stay flat")
        artifact = b"engine archive bytes\n"
        self.remote_bytes = artifact
        artifact_hash = sha(artifact)
        gpu_matrix = b"device\tstatus\n"
        requirements = b"cpu_flags\tavx2\nglibc\t2.39\n"
        (self.bundle / "gpu-matrix.tsv").write_bytes(gpu_matrix)
        (self.bundle / "requirements.tsv").write_bytes(requirements)
        (self.bundle / "elf-requirements.json").write_text("{}\n")
        component_url = component_url or "https://github.com/simoneiacomino/xenolith/releases/download/v1/engine-1.tar.gz"
        model_url = "https://models.example.test/model.gguf"
        manifest = ("# xenolith-distribution-v1\n"
                    f"engine\t1\t{component_url}\t{len(artifact)}\t{artifact_hash}\n"
                    f"gpu-matrix\t1\tgpu-matrix.tsv\t{len(gpu_matrix)}\t{sha(gpu_matrix)}\n"
                    f"requirements\t1\trequirements.tsv\t{len(requirements)}\t{sha(requirements)}\n"
                    f"model\tmodel-r1\t{model_url}\t100\t{'a' * 64}\n").encode()
        if duplicate_name:
            manifest += f"engine\t1\t{component_url}?duplicate=1\t{len(artifact)}\t{artifact_hash}\n".encode()
        manifest_hash = sha(manifest)
        (self.bundle / "manifest.tsv").write_bytes(manifest)
        source = source or self.source
        template = (INSTALLER / "install.sh").read_text()
        installer = render_installer(template, source, manifest_hash).encode()
        (self.bundle / "install.sh").write_bytes(installer)
        installer_hash = sha(installer)
        prov = {"schema_version": 1, "distribution_version": version or self.version,
                "development_components": False,
                "components": {"engine": {"version": "1", "archive": {"file": "engine-1.tar.gz",
                    "url": component_url, "size_bytes": len(artifact), "sha256": artifact_hash}}},
                "installer": {"default_source": source, "pinned_manifest_sha256": manifest_hash}}
        (self.bundle / "provenance.json").write_text(json.dumps(prov) + "\n")
        sums = []
        for path in sorted(self.bundle.iterdir()):
            sums.append(f"{sha(path.read_bytes())}  {path.name}\n")
        (self.bundle / "SHA256SUMS").write_text("".join(sums))
        if extra:
            (self.bundle / extra).write_text("unlisted\n")
        self.manifest_hash = manifest_hash
        self.installer_hash = installer_hash

    def env(self, **updates):
        return {**os.environ, "FAKE_AWS_CALLS": str(self.calls), "FAKE_AWS_STATE": str(self.state), **updates}

    def command(self, *extra, env=None, success=True):
        argv = ["--distribution-dir", str(self.bundle),
            "--distribution-version", self.version, "--source-base", self.source_base,
            "--approved-manifest-sha256", self.manifest_hash,
            "--approved-installer-sha256", self.installer_hash,
            "--bucket", self.bucket, "--aws", str(self.fake_aws), *extra]
        class Response(io.BytesIO):
            status = 200
            headers = {}
            def geturl(self): return 'https://release-assets.githubusercontent.com/fixture'
        response = Response(self.remote_bytes)
        with patch.dict(os.environ, env or self.env()), \
             patch.object(publisher.urllib.request, 'build_opener') as opener, \
             contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            opener.return_value.open.return_value = response
            code = publisher.main(argv)
        result = subprocess.CompletedProcess(argv, code, out.getvalue(), err.getvalue())
        self.assertEqual(result.returncode == 0, success, result.stderr + result.stdout)
        return result

    def call_log(self):
        if not self.calls.exists(): return []
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def aws_state(self):
        return json.loads(self.state.read_text()) if self.state.exists() else {}

    def test_dry_run_validates_and_never_invokes_aws(self):
        result = self.command()
        self.assertIn("Dry run", result.stdout)
        self.assertEqual(self.call_log(), [])

    def test_execute_uploads_immutable_first_and_stable_last(self):
        self.command("--execute")
        calls = self.call_log()
        keys = [args[args.index("--key") + 1] for args in calls if args[1] == "put-object"]
        self.assertEqual(keys[-1], "install.sh")
        self.assertEqual(set(keys[:-1]), {f"distributions/{self.version}/{name}" for name in
                         ("manifest.tsv", "provenance.json", "SHA256SUMS", "install.sh", "gpu-matrix.tsv", "requirements.tsv", "elf-requirements.json")})
        immutable_puts = [args for args in calls if args[1] == "put-object" and args[args.index("--key") + 1] != "install.sh"]
        self.assertTrue(all(args[args.index("--if-none-match") + 1] == "*" for args in immutable_puts))
        self.assertTrue(all("--checksum-sha256" in args for args in calls if args[1] == "put-object"))
        self.assertTrue(all(args[args.index("--cache-control") + 1] == "public,max-age=31536000,immutable" for args in immutable_puts))
        stable = next(args for args in calls if args[1] == "put-object" and args[args.index("--key") + 1] == "install.sh")
        self.assertEqual(stable[stable.index("--cache-control") + 1], "no-cache,max-age=0,must-revalidate")

    def test_existing_identical_immutable_objects_are_accepted(self):
        self.command("--execute")
        self.calls.unlink()
        self.command("--execute")
        calls = self.call_log()
        self.assertTrue(any(args[1] == "head-object" for args in calls))
        self.assertEqual(sum(args[1] == "put-object" and args[args.index("--key") + 1] == "install.sh" for args in calls), 1)

    def test_existing_mismatching_immutable_stops_before_stable(self):
        key = f"distributions/{self.version}/elf-requirements.json"
        self.state.write_text(json.dumps({key: base64.b64encode(b"wrong checksum").decode()}))
        self.command("--execute", success=False)
        keys = [args[args.index("--key") + 1] for args in self.call_log()]
        self.assertNotIn("install.sh", keys)

    def test_immutable_upload_failure_preserves_stable(self):
        self.command("--execute", env=self.env(FAKE_AWS_FAIL_KEY=f"distributions/{self.version}/manifest.tsv"), success=False)
        calls = self.call_log()
        self.assertNotIn("install.sh", [args[args.index("--key") + 1] for args in calls])
        self.assertFalse(any(args[1] == "head-object" for args in calls))

    def test_bad_bundle_inputs_fail_before_any_aws_call(self):
        self.command("--execute", "--approved-installer-sha256", "0" * 64, success=False)
        self.assertEqual(self.call_log(), [])
        self.make_bundle(source="https://other.example.test/release")
        self.command("--execute", success=False)
        self.assertEqual(self.call_log(), [])
        self.make_bundle()
        self.command("--execute", "--source-base", "https://xenolith.it/other", success=False)
        self.assertEqual(self.call_log(), [])
        self.make_bundle(component_url="https://github.com/simoneiacomino/xenolith/releases/download/v1/wrong-name.tar.gz")
        self.command("--execute", success=False)
        self.assertEqual(self.call_log(), [])
        self.make_bundle(duplicate_name=True)
        self.command("--execute", success=False)
        self.assertEqual(self.call_log(), [])

    def test_component_download_mismatch_fails_before_first_aws_call(self):
        self.remote_bytes = b'wrong remote bytes'
        self.command('--execute', success=False)
        self.assertEqual(self.call_log(), [])

    def test_network_failure_fails_before_first_aws_call(self):
        argv = ['--distribution-dir', str(self.bundle), '--distribution-version', self.version,
                '--source-base', self.source_base, '--approved-manifest-sha256', self.manifest_hash,
                '--approved-installer-sha256', self.installer_hash, '--bucket', self.bucket, '--execute']
        with patch.object(publisher.urllib.request, 'build_opener') as opener, \
             contextlib.redirect_stderr(io.StringIO()):
            opener.return_value.open.side_effect = urllib.error.URLError('injected network failure')
            self.assertEqual(publisher.main(argv), 1)
        self.assertEqual(self.call_log(), [])

    def test_component_archives_are_rejected_even_when_listed_in_sums(self):
        archive = self.bundle / 'engine-1.tar.gz'
        archive.write_bytes(self.remote_bytes)
        with (self.bundle / 'SHA256SUMS').open('a') as sums:
            sums.write(f'{sha(self.remote_bytes)}  engine-1.tar.gz\n')
        self.command('--execute', success=False)
        self.assertEqual(self.call_log(), [])

    def test_offline_validation_cannot_execute(self):
        self.command('--local-validation')
        self.command('--local-validation', '--execute', success=False)
        self.assertEqual(self.call_log(), [])

    def test_checksum_file_rejects_unlisted_duplicate_and_corrupt_files(self):
        self.make_bundle(extra="surprise.bin")
        self.command(success=False)
        self.assertEqual(self.call_log(), [])
        (self.bundle / "SHA256SUMS").write_text("0" * 64 + "  ../outside\n")
        self.command(success=False)
        self.assertEqual(self.call_log(), [])
        (self.bundle / "surprise.bin").unlink()
        sums = (self.bundle / "SHA256SUMS").read_text()
        (self.bundle / "SHA256SUMS").write_text(sums + sums.splitlines()[0] + "\n")
        self.command(success=False)
        self.assertEqual(self.call_log(), [])

    def test_workflow_unpack_and_validation_steps_accept_local_fixture(self):
        workflow_text = WORKFLOW.read_text()
        unpack = self.workflow_run_block(workflow_text, "Safely unpack flat regular files without running bundle code")
        validate = self.workflow_run_block(workflow_text, "Validate distribution and retain exact verified bundle")
        archive = self.root / "candidate.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in self.bundle.iterdir():
                tar.add(path, arcname=path.name)
        runner_temp = self.root / "runner-temp"
        runner_temp.mkdir()
        archive_target = runner_temp / "distribution.tar.gz"
        archive_target.write_bytes(archive.read_bytes())
        env = self.env(RUNNER_TEMP=str(runner_temp), DISTRIBUTION_VERSION=self.version,
                       SOURCE_BASE=self.source_base, MANIFEST_SHA256=self.manifest_hash,
                       INSTALLER_SHA256=self.installer_hash, PUBLISH_BUCKET="validation-only")
        subprocess.run(["bash", "-euo", "pipefail", "-c", unpack], env=env, check=True, timeout=10)
        # This check exercises the shell workflow's argument wiring offline.
        # Remote bytes are independently exercised by the publisher tests.
        validate = validate.replace('python3 installer/tools/publish_distribution.py',
                                    'python3 installer/tools/publish_distribution.py --local-validation')
        subprocess.run(["bash", "-euo", "pipefail", "-c", validate], cwd=INSTALLER.parent,
                       env=env, check=True, capture_output=True, text=True, timeout=10)
        self.assertTrue((runner_temp / "distribution/install.sh").is_file())
        self.assertEqual(self.call_log(), [])

    def test_workflow_unpack_rejects_malicious_archive_members(self):
        unpack = self.workflow_run_block(WORKFLOW.read_text(), "Safely unpack flat regular files without running bundle code")
        outside_absolute = self.root / "outside-absolute.txt"
        cases = [
            [("../outside-relative.txt", tarfile.REGTYPE, b"escape")],
            [(str(outside_absolute), tarfile.REGTYPE, b"escape")],
            [("link", tarfile.SYMTYPE, str(outside_absolute).encode())],
            [("same.txt", tarfile.REGTYPE, b"one"), ("same.txt", tarfile.REGTYPE, b"two")],
            [("./same.txt", tarfile.REGTYPE, b"one"), ("same.txt", tarfile.REGTYPE, b"two")],
        ]
        for index, members in enumerate(cases):
            with self.subTest(member_names=[member[0] for member in members]):
                runner_temp = self.root / f"malicious-{index}"
                runner_temp.mkdir()
                outside_relative = runner_temp / "outside-relative.txt"
                archive = runner_temp / "distribution.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    for name, member_type, content in members:
                        info = tarfile.TarInfo(name)
                        info.type = member_type
                        if member_type == tarfile.SYMTYPE:
                            info.linkname = content.decode()
                        else:
                            info.size = len(content)
                        tar.addfile(info, io.BytesIO(content) if member_type == tarfile.REGTYPE else None)
                result = subprocess.run(["bash", "-euo", "pipefail", "-c", unpack],
                                        env=self.env(RUNNER_TEMP=str(runner_temp)), capture_output=True,
                                        text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(outside_relative.exists())
                self.assertFalse(outside_absolute.exists())
                self.assertFalse((runner_temp / "link").is_symlink())

    @staticmethod
    def workflow_run_block(text, step_name):
        step = text.index(f"      - name: {step_name}\n")
        run = text.index("        run: |\n", step)
        next_step = text.find("\n      - ", run)
        if next_step < 0:
            next_step = len(text)
        start = run + len("        run: |\n")
        lines = []
        for line in text[start:next_step].splitlines():
            if line.startswith("      - ") or (line and not line.startswith("          ")):
                break
            lines.append(line[10:] if line.startswith("          ") else "")
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
