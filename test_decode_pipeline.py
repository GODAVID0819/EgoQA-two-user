import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
import unittest


def module():
    p=Path(__file__).with_name('decode_pipeline.py');s=importlib.util.spec_from_file_location('decode_pipeline',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class PipelineTests(unittest.TestCase):
    def test_parallel_decode_preserves_frame_order_and_count(self):
        m=module();pool=m.DecodePipeline(workers=3,chunk_size=2)
        self.addCleanup(pool.close)
        def decode(values):
            time.sleep(.001*(8-values[0]));return [v*10 for v in values]
        self.assertEqual(pool.decode(list(range(8)),decode),[v*10 for v in range(8)])

    def test_metrics_extract_real_counter_and_detect_recompute(self):
        m=module()
        values=[SimpleNamespace(name='vllm:num_preemptions',value=3),SimpleNamespace(name='vllm:gpu_cache_usage_perc',value=.7)]
        x=m.engine_metrics(values)
        self.assertEqual(x['preemptions'],3)
        self.assertTrue(m.should_reduce_batch({'preemptions':2},x,8))
        self.assertFalse(m.should_reduce_batch({'preemptions':3},x,8))
        self.assertFalse(m.should_reduce_batch({},x,8))

    def test_prefetch_is_cpu_background_work_and_returns_result(self):
        p=module().DecodePipeline(workers=2,chunk_size=2);self.addCleanup(p.close)
        future=p.prefetch(lambda x:sum(x),[1,2,3])
        self.assertEqual(future.result(timeout=5),6)


if __name__=='__main__':unittest.main()
