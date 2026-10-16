import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import sys
import unittest
from decode_pipeline import DecodePipeline

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'EgoQA-main-20260928/multi-user'))
s=importlib.util.spec_from_file_location('speed_predictor',HERE/'predictor.py')
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)


class Engine:
    def __init__(self):self.calls=[];self.preemptions=0
    def get_metrics(self):return [N(name='vllm:num_preemptions',value=self.preemptions)]
    def generate(self,payloads,*args,**kwargs):
        self.calls.append(len(payloads))
        if len(payloads)>4:self.preemptions+=1
        return [N(outputs=[N(logprobs=[{4:N(logprob=-1),5:N(logprob=-2)}])],num_cached_tokens=0,prompt_token_ids=[1]) for _ in payloads]


class PipelineTests(unittest.TestCase):
    def fixture(self):
        obj=p.VllmJudge.__new__(p.VllmJudge)
        obj.config={'max_num_seqs':8,'cpu_prefetch':True};obj.sleeping=False
        obj.llm=Engine();obj.adapters={};obj.sampling=None;obj.pass_id=4;obj.fail_id=5
        obj.pipeline=DecodePipeline(2);self.addCleanup(obj.pipeline.close)
        obj._effective_batch_size=8;obj._prepared={};obj._next_examples=[]
        obj.prepared_ids=[]
        def prepare(e):obj.prepared_ids.append(e.example_id);return {'prompt':e.example_id},{}
        obj.prepare=prepare
        return obj

    def examples(self,n):return [N(example_id=str(i),task=N(value='answerability')) for i in range(n)]

    def test_actual_preemptions_reduce_next_real_batch_without_dropping_requests(self):
        obj=self.fixture();examples=self.examples(8)
        self.assertEqual(len(obj.predict_many(examples)),8)
        self.assertEqual(len(obj.predict_many(examples)),8)
        self.assertEqual(obj.llm.calls,[8,4,4])

    def test_next_stage_cpu_preparation_is_reused_without_double_decode(self):
        obj=self.fixture();a,b=self.examples(2)
        obj.predict_many_with_prefetch([a],[b])
        obj.predict_many([b])
        self.assertEqual(obj.prepared_ids,['0','1'])


if __name__=='__main__':unittest.main()
