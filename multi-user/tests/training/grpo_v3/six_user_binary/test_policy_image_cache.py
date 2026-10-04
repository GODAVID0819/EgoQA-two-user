"""验证 Policy 图像缓存不会改变像素、泄漏可变对象或忽略源文件更新。"""
import math
from io import BytesIO
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
from test_pipeline import require_module


class PolicyImageCacheTests(unittest.TestCase):
    def test_optional_budget_is_validated_before_runtime(self):
        module = require_module(self, 'policy_image_cache')
        self.assertTrue(hasattr(module, 'configured_cache_bytes'))
        self.assertEqual(module.configured_cache_bytes(None), 0)
        self.assertEqual(module.configured_cache_bytes(4), 4 * 1024**3)
        for value in (True, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                module.configured_cache_bytes(value)

    def test_workflow_preserves_explicit_cache_budget(self):
        module = require_module(self, 'workflow')
        c = {'project_root': '/scratch/p', 'train_python': '/train/python',
             'judge_python': '/judge/python', 'model': '/model', 'policy_image_cache_gb': 4}
        run = module.training_config(c, job_id='123', phase='formal', max_steps=60)
        self.assertEqual(run.get('policy_image_cache_gb'), 4)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'frame.png'
        Image.new('RGB', (8, 8), (12, 34, 56)).save(self.source)

    def install(self, *, max_bytes=4096):
        module = require_module(self, 'policy_image_cache')
        def resize(image, pixels):
            side = math.isqrt(pixels)
            return image.resize((side, side)) if image.width * image.height > pixels else image
        class Template:
            load_images = True
            mode = 'train'
            max_pixels = 16
            loads = 0
            @staticmethod
            def _load_image(value, _load):
                if isinstance(value, Image.Image):
                    return value
                if isinstance(value, dict):
                    value = value['bytes'] or value['path']
                Template.loads += 1
                if isinstance(value, bytes):
                    value = BytesIO(value)
                with Image.open(value) as image:
                    return image.convert('RGB')
            def _preprocess_inputs(self, inputs):
                inputs.images = [resize(self._load_image(value, True), self.max_pixels) for value in inputs.images]
                inputs.original_ran = True
        cache = module.install_policy_image_cache(max_bytes=max_bytes, template_cls=Template, resize_fn=resize)
        return Template, Template(), cache

    def encode(self, template, path=None, **kwargs):
        value = SimpleNamespace(images=[str(path or self.source)], objects={}, **kwargs)
        template._preprocess_inputs(value)
        self.assertTrue(value.original_ran)
        return value.images[0]

    def test_repeated_input_reuses_decode_and_preserves_pixels(self):
        cls, template, cache = self.install()
        first = self.encode(template)
        second = self.encode(template)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(cls.loads, 1)
        self.assertEqual(cache.snapshot()['hits'], 1)

    def test_framework_path_dictionaries_hit_the_cache(self):
        cls, template, cache = self.install()
        for _ in range(2):
            inputs = SimpleNamespace(images=[{'bytes': None, 'path': str(self.source)}], objects={})
            template._preprocess_inputs(inputs)
            self.assertEqual(inputs.images[0].getpixel((0, 0)), (12, 34, 56))
        self.assertEqual(cls.loads, 1)
        self.assertEqual(cache.snapshot()['hits'], 1)

    def test_embedded_bytes_override_path_and_are_not_file_cached(self):
        cls, template, cache = self.install()
        encoded = BytesIO()
        Image.new('RGB', (8, 8), (90, 80, 70)).save(encoded, format='PNG')
        for _ in range(2):
            inputs = SimpleNamespace(images=[{'bytes': encoded.getvalue(), 'path': str(self.source)}], objects={})
            template._preprocess_inputs(inputs)
            self.assertEqual(inputs.images[0].getpixel((0, 0)), (90, 80, 70))
        self.assertEqual(cls.loads, 2)
        self.assertEqual(cache.snapshot()['hits'], 0)

    def test_callers_cannot_modify_the_cached_image(self):
        _, template, _ = self.install()
        self.encode(template).putpixel((0, 0), (255, 0, 0))
        self.assertEqual(self.encode(template).getpixel((0, 0)), (12, 34, 56))

    def test_file_update_invalidates_prior_pixels(self):
        cls, template, _ = self.install()
        self.encode(template)
        prior = self.source.stat().st_mtime_ns
        Image.new('RGB', (8, 8), (7, 8, 9)).save(self.source)
        os.utime(self.source, ns=(prior + 2000000000, prior + 2000000000))
        self.assertEqual(self.encode(template).getpixel((0, 0)), (7, 8, 9))
        self.assertEqual(cls.loads, 2)

    def test_pixel_budget_is_part_of_the_cache_key(self):
        cls, template, _ = self.install()
        self.assertEqual(self.encode(template).size, (4, 4))
        template.max_pixels = 64
        self.assertEqual(self.encode(template).size, (8, 8))
        self.assertEqual(cls.loads, 2)

    def test_cache_evicts_instead_of_exceeding_memory_budget(self):
        cls, template, cache = self.install(max_bytes=48)
        other = self.root / 'other.png'
        Image.new('RGB', (8, 8), (1, 2, 3)).save(other)
        self.encode(template)
        self.encode(template, other)
        self.encode(template)
        self.assertEqual(cls.loads, 3)
        self.assertLessEqual(cache.snapshot()['bytes'], 48)

    def test_oversized_item_is_not_retained(self):
        cls, template, cache = self.install(max_bytes=16)
        self.encode(template)
        self.encode(template)
        self.assertEqual(cls.loads, 2)
        self.assertEqual(cache.snapshot()['bytes'], 0)

    def test_object_annotations_keep_original_preprocessing(self):
        cls, template, _ = self.install()
        for _ in range(2):
            inputs = SimpleNamespace(images=[str(self.source)], objects={'bbox': [0, 0, 4, 4]})
            template._preprocess_inputs(inputs)
            self.assertTrue(inputs.original_ran)
        self.assertEqual(cls.loads, 2)

    def test_path_only_template_keeps_its_original_contract(self):
        cls, template, _ = self.install()
        template.load_images = False
        self.encode(template)
        self.encode(template)
        self.assertEqual(cls.loads, 2)


if __name__ == '__main__':
    unittest.main()
