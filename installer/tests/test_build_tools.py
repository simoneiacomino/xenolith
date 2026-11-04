"""Maintainer build-path/metadata regressions; no Docker build or GPU access."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load_tool(name):
    # Match direct-script imports while allowing isolated mocked entrypoints.
    with patch.object(sys, 'path', [str(TOOLS), *sys.path]):
        spec = importlib.util.spec_from_file_location(name, TOOLS / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class BuildTools(unittest.TestCase):
    def test_template_renderer_quotes_data_and_requires_exact_unique_lines(self):
        tool = load_tool('distribution_metadata')
        template = "#!/bin/sh\nDEFAULT_SOURCE=''\nDEFAULT_MANIFEST_SHA256=''\nprintf '%s\\n%s\\n' \"$DEFAULT_SOURCE\" \"$DEFAULT_MANIFEST_SHA256\"\n"
        with tempfile.TemporaryDirectory(prefix='candidate-shell-quote-') as temp:
            marker = Path(temp) / 'must-not-exist'
            source = f"{temp}/a quote's $(touch {marker})"
            script = tool.render_installer(template, source, 'a' * 64)
            result = subprocess.run(['sh'], input=script, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, source + '\n' + 'a' * 64 + '\n')
            self.assertFalse(marker.exists())
        for broken in (template.replace("DEFAULT_SOURCE=''", " DEFAULT_SOURCE=''"),
                       template + "DEFAULT_MANIFEST_SHA256=''\n"):
            with self.assertRaisesRegex(ValueError, 'exactly one standalone'):
                tool.render_installer(broken, '/tmp/local-candidate', 'b' * 64)
        for invalid in ('relative', 'http://example.invalid/files', 'https://example.invalid/files?q=1',
                        'https://example.invalid/#fragment', '/tmp/path\nnext'):
            with self.subTest(source=invalid), self.assertRaises(ValueError):
                tool.render_installer(template, invalid, 'b' * 64)

    def test_output_containment_rejected_before_directory_or_docker_side_effects(self):
        tool = load_tool('build_linux')
        with tempfile.TemporaryDirectory(prefix='build-path-test-') as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            alias = root / 'source-alias'
            alias.symlink_to(source, target_is_directory=True)
            for component in ('engine',):
                for out in (source, source / 'portable-build', alias / 'portable-build'):
                    with self.subTest(component=component, output=str(out)):
                        argv = ['build_linux.py', '--component', component, '--source', str(source),
                                '--output', str(out), '--build-image']
                        with patch.object(sys, 'argv', argv), patch.object(tool, 'run') as run, \
                             patch.object(tool.subprocess, 'check_output') as inspect, \
                             contextlib.redirect_stderr(io.StringIO()) as error:
                            with self.assertRaises(SystemExit) as raised:
                                tool.main()
                        self.assertEqual(raised.exception.code, 2)
                        self.assertIn('outside --source', error.getvalue())
                        run.assert_not_called()
                        inspect.assert_not_called()
                        self.assertEqual(list(source.iterdir()), [])

    def test_component_urls_are_exact_official_versioned_release_assets(self):
        tool = load_tool('distribution_metadata')
        for component, repo, prefix in [('engine', 'simoneiacomino/xenolith', 'v'),
                                        ('xenopi', 'simoneiacomino/xenopi', 'v'),
                                        ('codexe', 'crrisci/codexe', 'codexe-v')]:
            valid = f'https://github.com/{repo}/releases/download/{prefix}1.2.3/component.tar.gz'
            self.assertEqual(tool.component_url(component, '1.2.3', valid, 'component.tar.gz'), valid)
            for invalid in (valid.replace(repo, 'attacker/repo'), valid.replace(prefix + '1.2.3', 'latest'),
                            valid.replace(prefix + '1.2.3', prefix + '1.2.4'), valid + '?x=1',
                            valid + '#fragment', valid.replace('https:', 'http:'),
                            valid.replace('component.tar.gz', '../component.tar.gz'),
                            valid.replace('component.tar.gz', '%2e%2e%2fcomponent.tar.gz'),
                            valid.replace('github.com/', 'github.com:443/'),
                            valid.replace('github.com/', 'github.com/extra/')):
                with self.subTest(url=invalid), self.assertRaises(ValueError):
                    tool.component_url(component, '1.2.3', invalid)
            with self.assertRaises(ValueError):
                tool.component_url(component, '1.2.3', valid, 'wrong.tar.gz')
        with self.assertRaises(ValueError):
            tool.component_url('engine', '1', '/tmp/local.tar.gz')
        with self.assertRaisesRegex(ValueError, 'serve the archive over loopback HTTP'):
            tool.component_url('engine', '1', '/tmp/local.tar.gz', development=True)
        self.assertEqual(tool.component_url('engine', '1', 'http://127.0.0.1:8123/engine.tar.gz',
                                           development=True), 'http://127.0.0.1:8123/engine.tar.gz')


if __name__ == '__main__':
    unittest.main()
