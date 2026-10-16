"""两个真实Gloo进程检验：先平均梯度、再裁剪/AdamW，覆盖零学习率遮蔽。"""
import argparse
import copy
import json
import os
import datetime
from types import SimpleNamespace as N
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6,dtype=torch.float32).reshape(2,3)/10)
        self.bias = torch.nn.Parameter(torch.zeros(2))
    def forward(self, x, branch=False):
        value = x @ self.weight.T
        return value if branch else value + self.bias


class Trainer:
    def __init__(self, model):
        self.model = model
        self.model_wrapped = model
        self.state = N(global_step=0)
        self.args = N(max_grad_norm=.2)
        self.accelerator = N(unwrap_model=lambda x:x.module if isinstance(x,DDP) else x,
                             distributed_type='MULTI_CPU',num_processes=2)
    def _clip_grad_norm(self, model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(),self.args.max_grad_norm)
    def _get_grad_norm(self,model,grad_norm=None):
        return grad_norm if grad_norm is not None else torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'))


def loss(model, rank, micro, missing=False):
    x=torch.tensor([[rank+1.,micro/4.,1.]])
    return (model(x,branch=missing and rank==0)**2).mean()/4


def run_case(repair,normal_ddp,missing):
    rank=dist.get_rank()
    model=Tiny();reference=copy.deepcopy(model)
    wrapped=DDP(model,find_unused_parameters=missing) if normal_ddp else model
    trainer=Trainer(wrapped)
    opt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=.01)
    refopt=torch.optim.AdamW(reference.parameters(),lr=0.,weight_decay=.01)
    for step,lr in enumerate([0.,.001,.0008]):
        trainer.state.global_step=step
        opt.param_groups[0]['lr']=refopt.param_groups[0]['lr']=lr
        for micro in range(4):
            loss(wrapped,rank,micro,missing).backward()
            for other_rank in [0,1]:
                (loss(reference,other_rank,micro,missing)/2).backward()
        norm=trainer._clip_grad_norm(wrapped)
        trainer._get_grad_norm(wrapped,norm)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),.2)
        opt.step();refopt.step()
        for value,target in zip(model.parameters(),reference.parameters()):
            gathered=[torch.empty_like(value) for _ in range(2)]
            dist.all_gather(gathered,value.detach())
            assert torch.equal(gathered[0],gathered[1]),'zero-LR hid rank-local optimizer state; first nonzero update diverged'
            assert torch.allclose(value,target,atol=2e-7,rtol=2e-6),'gradient averaging/clipping differs from global-batch reference'
        for p in model.parameters():
            for key in ['exp_avg','exp_avg_sq']:
                value=opt.state[p][key];g=[torch.empty_like(value) for _ in range(2)];dist.all_gather(g,value)
                assert torch.equal(g[0],g[1]),'AdamW moments differ across ranks'
        model.zero_grad();reference.zero_grad()
    print('CPU_CASE_PASSED',rank,normal_ddp,missing,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--rank',type=int);parser.add_argument('--init-file');a=parser.parse_args()
    torch.set_num_threads(1)
    if a.init_file:
        dist.init_process_group('gloo',init_method='file://'+a.init_file,rank=a.rank,world_size=2,
                                timeout=datetime.timedelta(seconds=25))
    else:
        dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=25))
    try:
        if not a.baseline:
            from gradient_sync import install
            records=[];install(Trainer,lambda trainer,stats:records.append(stats))
        run_case(not a.baseline,False,False)
        if not a.baseline:
            run_case(True,False,True)
            run_case(True,True,False)
            assert len(records)==9,'same optimizer update synchronized more than once'
            assert not torch.cuda.is_initialized()
            print('GRAD_SYNC_CPU_PASSED',json.dumps({'rank':dist.get_rank(),'cases':3,'updates':9,'gpu_operations':0,'audit':records[-1]}),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
