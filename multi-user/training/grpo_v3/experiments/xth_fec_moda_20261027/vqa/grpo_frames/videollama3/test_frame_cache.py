"""外部下载边界测试：JPEG原图→原缩放→无损PNG缓存。"""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import prepare_frames as prepare
import grpo_inputs as core


class FrameCacheTests(unittest.TestCase):
    def test_downloaded_jpeg_uses_exact_reference_pixels_and_cache_reuse(self):
        rng=np.random.default_rng(0)
        raw=Image.fromarray(rng.integers(0,256,(1408,1408,3),dtype=np.uint8))
        stream=io.BytesIO();raw.save(stream,format='JPEG')
        response=Mock(status_code=200,content=stream.getvalue())
        response.raise_for_status.return_value=None
        prepare.local.session=Mock();prepare.local.session.get.return_value=response
        relative='frames/packet/person/frame_0000_0.00s.jpg'
        config={'dataset_repo':'example/repo','dataset_revision':'revision'}
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(prepare.fetch_one(relative,Path(directory),config),'downloaded')
            cached=core.cache_path(directory,relative)
            with Image.open(io.BytesIO(stream.getvalue())) as source:
                expected=core.reference.rescale_image(source.convert('RGB'),24576)
                with Image.open(cached) as got:
                    np.testing.assert_array_equal(np.asarray(expected),np.asarray(got))
            prepare.local.session.get.reset_mock()
            self.assertEqual(prepare.fetch_one(relative,Path(directory),config),'reused')
            prepare.local.session.get.assert_not_called()


if __name__=='__main__':unittest.main()
