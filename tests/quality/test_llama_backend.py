from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xe_quality import llama_backend as backend

NATIVE_WRAPPER = Path(__file__).resolve().parents[2] / "tools/quality/native/llama_driver.cpp"


class LlamaBackendTests(unittest.TestCase):
    def test_profiles_do_not_mutate_historical_options(self):
        self.assertIn('-DGGML_VULKAN=ON', backend.profile_options('vulkan'))
        self.assertIn('-DGGML_VULKAN=OFF', backend.profile_options('cpu-historical'))
        self.assertNotIn('-DGGML_VULKAN=ON', backend.OPTIONS)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            backend.profile_options('automatic')

    def test_invalid_device_fails_before_build(self):
        with self.assertRaisesRegex(ValueError, 'explicit Vulkan'):
            backend.prepare('source', None, 'driver', 'model', 'unused', device='CPU')

    def test_chunk_cooldown_rejects_cpu_before_build(self):
        with patch.object(backend.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Vulkan profile'):
                backend.prepare('source', None, 'driver', 'model', 'unused',
                                profile='cpu-historical', chunk_cooldown=True)
            run.assert_not_called()

    def test_chunk_cooldown_rejects_preserved_and_unknown_wrappers_before_work(self):
        # Reconstruct exact historical wrapper bytes without duplicating its source.
        # The asserted hash anchors this regression to the preserved Vulkan producer.
        old_lines, skipping = [], False
        for line in NATIVE_WRAPPER.read_text().splitlines(keepends=True):
            if line.strip() == '#ifdef XE_QUALITY_CHUNK_COOLDOWN':
                skipping = True
            elif skipping and line.strip() == '#endif':
                skipping = False
            elif not skipping:
                old_lines.append(line)
        preserved = ''.join(old_lines).replace('\n\n\nstatic std::vector<int>',
                                               '\n\nstatic std::vector<int>').encode()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root/'preserved-vulkan.cpp'
            old.write_bytes(preserved)
            self.assertEqual(backend.file_hash(old),
                             '5de74d8e67b864050fbd9fcfeae18adeb2239b287e0d687453b33a9d024ff8d6')
            modified = root/'modified-wrapper.cpp'
            modified.write_bytes(NATIVE_WRAPPER.read_bytes() + b'\n// changed source\n')
            unknown = root/'unknown.cpp'
            unknown.write_text('#define XE_QUALITY_CHUNK_COOLDOWN\n')
            for index, wrapper in enumerate((old, modified, unknown, root/'missing.cpp')):
                output = root/str(index)
                with self.subTest(wrapper=wrapper.name), \
                     patch.object(backend.subprocess, 'run') as run, \
                     patch.object(backend.subprocess, 'check_output') as check_output:
                    with self.assertRaisesRegex(ValueError, 'supported wrapper SHA256'):
                        backend.prepare('source', None, wrapper, 'model', output,
                                        chunk_cooldown=True)
                    self.assertFalse(output.exists())
                    run.assert_not_called()
                    check_output.assert_not_called()

    def test_cli_chunk_cooldown_is_explicit(self):
        argv = ['--source', 'source', '--driver-source', 'driver', '--model', 'model',
                '--output', 'output']
        with patch.object(backend, 'prepare') as prepare:
            self.assertEqual(backend.main(argv), 0)
            self.assertFalse(prepare.call_args.kwargs['chunk_cooldown'])
            self.assertEqual(backend.main([*argv, '--chunk-cooldown']), 0)
            self.assertTrue(prepare.call_args.kwargs['chunk_cooldown'])

    def test_runtime_probe_requires_loader_and_device(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.CompletedProcess([], 0, '', 'XE_QUALITY_VULKAN device=Vulkan0\n')
            with patch.object(backend.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    backend.runtime_probe('driver', Path(folder))

    def test_runtime_probe_hashes_dynamically_loaded_icd(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            loader, icd = root/'libvulkan.so.1', root/'libvulkan_test_icd.so'
            loader.write_bytes(b'loader')
            icd.write_bytes(b'icd')
            result = subprocess.CompletedProcess([], 0,
                f'runtime_file\t{loader}\nruntime_file\t{icd}\n',
                'XE_QUALITY_VULKAN device=Vulkan0 description=hardware\n')
            with patch.object(backend.subprocess, 'run', return_value=result):
                runtime, pins = backend.runtime_probe('driver', root)
            self.assertEqual(pins[str(icd)], backend.file_hash(icd))
            self.assertEqual(runtime['mapped_files'], sorted([str(loader), str(icd)]))
            self.assertTrue((root/'probe.stderr.txt').is_file())

    def test_prepare_default_vulkan_and_historical_compiler_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'source', root/'build-output'
            source.mkdir()
            (source/'CMakeLists.txt').write_text('pinned')
            driver, model = root/'driver.cpp', root/'model.gguf'
            driver.write_text('driver')
            model.write_bytes(b'model')
            for tool in ('gcc', 'g++', 'cmake', 'glslc'):
                (root/tool).write_bytes(tool.encode())
            def check_output(argv, **kwargs):
                if argv[:2] == ['git', 'rev-parse']:
                    return backend.COMMIT+'\n'
                if argv[:2] == ['git', 'status']:
                    return ''
                if argv[0] == 'ldd':
                    return ''
                return 'compiler 14.2.0\n'
            def run(argv, **kwargs):
                if '--build' in argv:
                    build = output/'build'
                    build.mkdir()
                    for name in ('llama', 'ggml', 'ggml-cpu', 'ggml-base', 'ggml-vulkan'):
                        (build/f'lib{name}.a').write_bytes(name.encode())
                    (root/'vulkan').mkdir(exist_ok=True)
                    (root/'vulkan/vulkan.h').write_bytes(b'vulkan header')
                    (root/'libvulkan.so').write_bytes(b'vulkan library')
                    (build/'CMakeCache.txt').write_text(
                        f'Vulkan_GLSLC_EXECUTABLE:FILEPATH={root}/glslc\nVulkan_INCLUDE_DIR:PATH={root}\nVulkan_LIBRARY:FILEPATH={root}/libvulkan.so\n')
                if '-o' in argv:
                    Path(argv[argv.index('-o')+1]).write_bytes(b'executable')
            with patch.object(backend.shutil, 'which', side_effect=lambda n: str(root/n)), \
                 patch.object(backend.subprocess, 'check_output', side_effect=check_output), \
                 patch.object(backend.subprocess, 'run', side_effect=run), \
                 patch.object(backend, 'runtime_probe', return_value=({'identity': ['physical GPU']}, {})):
                result = backend.prepare(source, None, driver, model, output)
                self.assertEqual(result['backend'], 'llama.cpp Vulkan')
                config = result['public_profile']['configuration']
                self.assertEqual(config['devices'], ['Vulkan0'])
                self.assertEqual(config['flash_attn_type'], 'OFF')
                self.assertEqual(config['n_gpu_layers'], 999)
                self.assertTrue(config['offload_kqv'])
                self.assertNotIn('-DXE_QUALITY_CHUNK_COOLDOWN', result['build_argv'][-1])
                self.assertNotIn('generation_controls', result['public_profile'])
                self.assertIn('-DGGML_VULKAN=ON', result['build_argv'][0])
                self.assertIn(str(root/'libvulkan.so'), result['build_argv'][-1])
                self.assertIn('-I'+str(root), result['build_argv'][-1])
                self.assertIn('LD_AUDIT', result['public_profile']['environment'])
                self.assertIn('HOME', result['public_profile']['environment'])
                self.assertIn('CMAKE_LIBRARY_PATH', result['public_profile']['build_environment'])
                self.assertIn(str(root/'glslc'), result['input_pins'])
                self.assertIn(str(root/'vulkan/vulkan.h'), result['input_pins'])
                output = root/'cooled-build-output'
                supported_driver = root/'supported-driver.cpp'
                supported_driver.write_bytes(NATIVE_WRAPPER.read_bytes())
                self.assertEqual(backend.file_hash(supported_driver), backend.CHUNK_COOLDOWN_WRAPPER_SHA256)
                cooled = backend.prepare(source, None, supported_driver, model, output, chunk_cooldown=True)
                self.assertIn('-DXE_QUALITY_CHUNK_COOLDOWN', cooled['build_argv'][-1])
                self.assertEqual(cooled['public_profile']['configuration'], config)
                self.assertEqual(cooled['public_profile']['generation_controls'], {
                    'synchronized_chunk_cooldown': True,
                    'ready_temperature_c': 60, 'stable_seconds': 5, 'poll_milliseconds': 200,
                    'boundary_timeout_seconds': 120, 'hard_stop_temperature_c': 95,
                    'compile_definition': 'XE_QUALITY_CHUNK_COOLDOWN',
                    'process_sigstop_pacing': False, 'performance_measurement': False})
                with self.assertRaisesRegex(ValueError, '15.2.0'):
                    backend.prepare(source, None, driver, model, root/'cpu', profile='cpu-historical')


if __name__ == '__main__':
    unittest.main()
