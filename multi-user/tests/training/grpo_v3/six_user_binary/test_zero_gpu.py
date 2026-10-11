"""零GPU验证必须覆盖真实像素解码和可执行编译器。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_pipeline import require_module


class ZeroGPUTests(unittest.TestCase):
    def test_cli_scheduler_json_is_normalized_without_ignoring_invalid_values(self):
        zero_gpu = require_module(self, 'zero_gpu')
        self.assertEqual(zero_gpu.normalize_scheduler_kwargs('{"min_lr_rate": 0.1}'), {'min_lr_rate': 0.1})
        value = {'min_lr_rate': 0.1}
        self.assertIs(zero_gpu.normalize_scheduler_kwargs(value), value)
        self.assertEqual(zero_gpu.normalize_scheduler_kwargs(None), {})
        for bad in ('[]', 'true', 'not-json', [], True):
            with self.assertRaises(ValueError):
                zero_gpu.normalize_scheduler_kwargs(bad)

    def test_truncated_jpeg_with_valid_header_fails_full_decode(self):
        module=require_module(self,'zero_gpu')
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.jpg'
            Image.new('RGB',(64,64),'red').save(path)
            self.assertEqual(module.decode_images([path]),1)
            path.write_bytes(path.read_bytes()[:-20])
            with Image.open(path) as image:image.verify()
            with self.assertRaises(OSError):module.decode_images([path])

    def test_missing_host_compiler_is_a_failure(self):
        module=require_module(self,'zero_gpu')
        with tempfile.TemporaryDirectory() as d, patch.dict(module.os.environ,{'CC':'','CXX':''}), patch.object(module.shutil,'which',return_value=None):
            with self.assertRaises(FileNotFoundError):module.compiler_check(d)


if __name__=='__main__':unittest.main()
