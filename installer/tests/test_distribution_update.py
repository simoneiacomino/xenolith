"""End-to-end local fixture for managed distribution component updates."""
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import unittest


FIXTURE_PATH = Path(__file__).with_name("test_install.py")
SPEC = importlib.util.spec_from_file_location("installer_update_fixtures", FIXTURE_PATH)
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)


class DistributionUpdateTest(unittest.TestCase):
    def setUp(self):
        self.fixture = FIXTURES.InstallerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.prefix = self.fixture.home / ".local/share/xenolith/install"
        self.bindir = self.fixture.home / ".local/bin"
        self.config = self.fixture.config
        self.config.parent.mkdir(parents=True, exist_ok=True)
        self.config_bytes = b'{"schema_version":1,"model":"/fixture/model.gguf"}\n'
        self.config.write_bytes(self.config_bytes)
        self.session = self.fixture.home / ".local/share/codexe/sessions/saved-session.json"
        self.session.parent.mkdir(parents=True, exist_ok=True)
        self.session_bytes = b'{"session":"keep-this-conversation"}\n'
        self.session.write_bytes(self.session_bytes)

    @staticmethod
    def archive(files):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as package:
            for name, content in files.items():
                member = tarfile.TarInfo(name)
                member.mode = 0o755
                member.size = len(content)
                package.addfile(member, io.BytesIO(content))
        return output.getvalue()

    def package(self, component, version):
        if component == "engine":
            script = FIXTURES.FAKE_ENGINE.decode().replace(
                "a=sys.argv[1:]",
                f"a=sys.argv[1:]\nif a==['--version']: print('fixture-engine-{version}'); sys.exit(0)",
                1,
            )
            files = {"bin/xenolith": script.encode()}
        elif component == "xenopi":
            files = {"bin/xenopi": f"#!/bin/sh\necho fixture-xenopi-{version}\n".encode()}
        else:
            client = (
                "#!/bin/sh\n"
                f"if [ \"${{1:-}}\" = --version ]; then echo fixture-codexe-{version}; exit 0; fi\n"
                "exit 0\n"
            ).encode()
            files = {
                "bin/codex": client,
                "bin/codex-code-mode-host": b"#!/bin/sh\necho helper\n",
                "bin/codex-exec-server": b"#!/bin/sh\necho helper\n",
                "codex-path/rg": b"#!/bin/sh\necho rg 1\n",
                "codex-resources/bwrap": b"#!/bin/sh\necho bwrap 1\n",
                "codex-package.json": b'{"xenolith":{"nativeConfig":1}}',
                "VERSION": (version + "\n").encode(),
                "install/install.sh": b'''#!/bin/sh
set -eu
shift
while [ "$#" -gt 0 ]; do
 key=$1; value=$2; shift 2
 case "$key" in --package-dir) package=$value;; --prefix) prefix=$value;; --bin-dir) bindir=$value;; *) exit 2;; esac
done
root=$prefix/lib/codexe/standalone
version=$(cat "$package/VERSION")
mkdir -p "$root/releases/$version" "$bindir"
cp -R "$package/." "$root/releases/$version/"
printf '%s\\n' codexe-local-v1 > "$root/.codexe-local"
ln -sfn "releases/$version" "$root/current"
ln -sfn "$root/current/bin/codex" "$bindir/codexe"
''',
            }
        return self.archive(files)

    def set_distribution(self, version, corrupt_component=None):
        fixture = self.fixture
        fixture.records = [record for record in fixture.records if record[0] not in ("engine", "xenopi", "codexe")]
        self.package_bytes = {}
        for component in ("engine", "xenopi", "codexe"):
            artifact = self.package(component, version)
            self.package_bytes[component] = artifact
            fixture.add(component, artifact, component + ".tar.gz")
            fixture.records[-1][1] = version
        fixture.manifest()
        if corrupt_component:
            (fixture.source / (corrupt_component + ".tar.gz")).write_bytes(b"truncated package archive")

    def install_all(self):
        return self.fixture.install(
            "--skip-verify", "--interface", "all", "--prefix", str(self.prefix),
            "--bin-dir", str(self.bindir), "--no-modify-path",
        )

    def selected_version(self, component):
        command = self.bindir / component
        result = subprocess.run([str(command), "--version"], env=self.fixture.env,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def assert_preserved_user_data(self):
        self.assertEqual(self.config.read_bytes(), self.config_bytes)
        self.assertEqual(self.session.read_bytes(), self.session_bytes)

    def test_updates_versions_preserves_state_and_recovers_after_corrupt_component(self):
        self.set_distribution("A")
        initial = self.install_all()
        self.assertEqual(initial.returncode, 0, initial.stdout + initial.stderr)
        self.assertEqual(self.selected_version("xenolith"), "fixture-engine-A")
        self.assertEqual(self.selected_version("xenopi"), "fixture-xenopi-A")
        self.assertEqual(self.selected_version("codexe"), "fixture-codexe-A")
        self.assert_preserved_user_data()

        links_a = {name: (self.bindir / name).readlink() for name in ("xenolith", "xenopi", "codexe")}
        rerun = self.install_all()
        self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
        self.assertEqual({name: (self.bindir / name).readlink() for name in links_a}, links_a)
        self.assertEqual(self.selected_version("xenolith"), "fixture-engine-A")
        self.assertEqual(self.selected_version("xenopi"), "fixture-xenopi-A")
        self.assertEqual(self.selected_version("codexe"), "fixture-codexe-A")
        self.assert_preserved_user_data()

        self.set_distribution("B", corrupt_component="xenopi")
        failed = self.install_all()
        self.assertNotEqual(failed.returncode, 0, failed.stdout + failed.stderr)
        self.assertIn("Checksum/size mismatch", failed.stderr)
        # Installation proceeds component by component: the valid engine update
        # is selected, while XenoPi and CodeXe retain their last working versions.
        self.assertEqual(self.selected_version("xenolith"), "fixture-engine-B")
        self.assertEqual(self.selected_version("xenopi"), "fixture-xenopi-A")
        self.assertEqual(self.selected_version("codexe"), "fixture-codexe-A")
        self.assert_preserved_user_data()

        (self.fixture.source / "xenopi.tar.gz").write_bytes(self.package_bytes["xenopi"])
        recovered = self.install_all()
        self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
        self.assertEqual(self.selected_version("xenolith"), "fixture-engine-B")
        self.assertEqual(self.selected_version("xenopi"), "fixture-xenopi-B")
        self.assertEqual(self.selected_version("codexe"), "fixture-codexe-B")
        self.assert_preserved_user_data()

        self.assertIn("releases/engine-B-", str((self.bindir / "xenolith").readlink()))
        self.assertIn("releases/xenopi-B-", str((self.bindir / "xenopi").readlink()))
        codexe_root = self.prefix / "lib/codexe/standalone"
        self.assertEqual((codexe_root / "current").readlink(), Path("releases/B"))
        self.assertEqual((self.bindir / "codexe").readlink(), codexe_root / "current/bin/codex")
        self.assertTrue((codexe_root / "releases/A/bin/codex").is_file())
        self.assertTrue((codexe_root / "releases/B/bin/codex").is_file())


if __name__ == "__main__":
    unittest.main()
