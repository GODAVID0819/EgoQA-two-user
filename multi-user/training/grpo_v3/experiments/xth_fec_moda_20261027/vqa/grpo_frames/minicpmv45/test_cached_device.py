"""复现缓存文字容器在设备搬运时错误递归整数时间ID的边界。"""
import unittest
from pathlib import Path
from transformers import AutoProcessor
from prepare_cpu import records
from native_inputs import text_inputs

class CachedDeviceTests(unittest.TestCase):
    def test_native_container_moves_to_cpu(self):
        root=Path(__file__).resolve().parent
        processor=AutoProcessor.from_pretrained(root/'upstream',trust_remote_code=True)
        requests,_=records(root)
        inputs=text_inputs(processor,[requests[0]])
        inputs.pop('image_sizes')
        moved=inputs.to('cpu')
        self.assertEqual(len(moved['image_bound'][0]),len(requests[0]['groups']))
        self.assertNotIn('temporal_ids',moved)

if __name__=='__main__':unittest.main()
