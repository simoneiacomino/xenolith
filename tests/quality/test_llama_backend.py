from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xe_quality import llama_backend as backend


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
                self.assertIn('-DGGML_VULKAN=ON', result['build_argv'][0])
                self.assertIn(str(root/'libvulkan.so'), result['build_argv'][-1])
                self.assertIn('-I'+str(root), result['build_argv'][-1])
                self.assertIn('LD_AUDIT', result['public_profile']['environment'])
                self.assertIn('HOME', result['public_profile']['environment'])
                self.assertIn('CMAKE_LIBRARY_PATH', result['public_profile']['build_environment'])
                self.assertIn(str(root/'glslc'), result['input_pins'])
                self.assertIn(str(root/'vulkan/vulkan.h'), result['input_pins'])
                with self.assertRaisesRegex(ValueError, '15.2.0'):
                    backend.prepare(source, None, driver, model, root/'cpu', profile='cpu-historical')


if __name__ == '__main__':
    unittest.main()
