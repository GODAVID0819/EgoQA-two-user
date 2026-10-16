"""CPU模拟主训练并发分配，验证Guard不误用进程全局统计重建。"""
from contextlib import nullcontext
import importlib
import itertools
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

MIB=1024**2


class Stop:
    def __init__(self):self.count=0
    def is_set(self):return self.count>=2
    def wait(self,value):self.count+=1


class FakeCuda:
    def __init__(self,allocated=2*MIB,reserved=2*MIB):
        self.a=iter([0,allocated,0,5222*MIB]);self.r=iter([0,reserved,0,2*MIB])
    def set_device(self,*args):pass
    def default_stream(self,*args):return N()
    def Stream(self,**kwargs):return N(priority=0,wait_stream=lambda *x:None,synchronize=lambda:None)
    def stream(self,*args):return nullcontext()
    def CUDAGraph(self):return N(replay=lambda:None)
    def graph(self,*args,**kwargs):return nullcontext()
    def memory_allocated(self,*args):return next(self.a)
    def memory_reserved(self,*args):return next(self.r)
    def empty_cache(self):pass


class FakeTorch:
    float32='float32'
    def __init__(self,**kwargs):self.cuda=FakeCuda(**kwargs);self.allocations=0
    def ones(self,*args,**kwargs):self.allocations+=1;return N()
    def empty_like(self,*args):return N()
    def sin(self,*args,**kwargs):pass
    def inference_mode(self):return nullcontext()


class DeferredRebuildTests(unittest.TestCase):
    def execute(self,**allocation):
        runtime=importlib.import_module('training.grpo_v3.six_user_binary.utilization_runtime')
        torch=FakeTorch(**allocation);free=iter([24,4,24])
        nv=N(nvmlInit=lambda:None,nvmlDeviceGetMemoryInfo=lambda *x:N(free=next(free)*1024**3,used=0),
             nvmlDeviceGetUtilizationRates=lambda *x:N(gpu=0))
        guard=runtime.Guard({'target':85,'cancel':60,'reserve_gib':8,'max_prealloc_mib':16,'max_duty':.9},'unused')
        guard.stop=Stop();events=[];guard.record=lambda **value:events.append(value)
        ticks=itertools.count(100,2)
        with patch.dict('sys.modules',{'torch':torch,'pynvml':nv,
                'hpc.shared.cuda_device_identity':N(handle_for_cuda_device=lambda *x:0)}), \
             patch.dict('os.environ',{'EGOQA_UTILIZATION_METRICS':'','EGOQA_GUARD_CUDA_DEVICE':'0'}), \
             patch.object(runtime.time,'time',side_effect=lambda:next(ticks)):
            guard.run()
        return guard,torch,events

    def test_no_runtime_rebuild_during_concurrent_training_allocation(self):
        guard,torch,events=self.execute()
        self.assertIsNone(guard.error,'主训练5GiB并发分配不应被计入Guard预算')
        self.assertEqual(torch.allocations,1)
        self.assertIn('memory_yield',[e['event'] for e in events])
        self.assertIn('rebuild_deferred_to_safe_boundary',[e['event'] for e in events])

    def test_real_auxiliary_allocated_budget_still_rejects(self):
        guard,_,_=self.execute(allocated=25*MIB)
        self.assertIn('超过预算',guard.error)

    def test_real_auxiliary_reserved_budget_still_rejects(self):
        guard,_,_=self.execute(reserved=22*MIB)
        self.assertIn('超过预算',guard.error)


if __name__=='__main__':unittest.main()
