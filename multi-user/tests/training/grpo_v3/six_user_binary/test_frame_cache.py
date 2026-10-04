"""逐帧缓存必须保留像素、顺序和内存上限。"""
import unittest
from PIL import Image
from test_pipeline import require_module


class FrameCacheTests(unittest.TestCase):
    def test_reordered_views_reuse_identical_frames(self):
        module=require_module(self,'predictor')
        cache=module.FrameImageCache(max_bytes=300)
        calls=[]
        def load(values):
            calls.append(list(values))
            return [Image.new('RGB',(5,5),(v,0,0)) for v in values]
        first,audit=cache.get_many([(1,100),(2,100)],[1,2],load)
        second,audit=cache.get_many([(2,100),(1,100)],[2,1],load)
        self.assertEqual(calls,[[1,2]])
        self.assertIs(second[0],first[1])
        self.assertIs(second[1],first[0])
        self.assertEqual(audit['decoded_image_cache_misses'],0)
        self.assertEqual(second[0].getpixel((0,0)),(2,0,0))

    def test_capacity_and_processing_settings_are_respected(self):
        module=require_module(self,'predictor')
        cache=module.FrameImageCache(max_bytes=150)
        calls=[]
        def load(values):
            calls.extend(values)
            return [Image.new('RGB',(5,5)) for _ in values]
        cache.get_many([(1,100),(2,100)],[1,2],load)
        cache.get_many([(1,200)],[3],load)
        self.assertLessEqual(cache.current_bytes,150)
        cache.get_many([(1,100)],[1],load)
        self.assertEqual(calls,[1,2,3,1])


if __name__=='__main__':unittest.main()
