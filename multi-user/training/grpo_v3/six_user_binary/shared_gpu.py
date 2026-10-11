"""同步GRPO的单GPU分阶段复用；释放成功前绝不恢复另一模型。"""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import time


def audit(event, **values):
    path = os.environ.get('EGOQA_SHARED_GPU_TRACE')
    if path:
        with Path(path).open('a', encoding='utf-8') as out:
            out.write(json.dumps({'epoch': time.time(), 'event': event, **values}) + '\n')


@contextmanager
def judge_window(client, suspend_policy, resume_policy):
    suspend_policy()
    from .utilization_runtime import rearm_training_guard
    rearm_training_guard()
    audit('policy_offloaded')
    try:
        state = client.resource('wake')
        if state.get('sleeping') is not False:
            raise RuntimeError('Judge没有确认唤醒')
        audit('judge_awake', resource=state)
        yield
    finally:
        # 如果释放失败，保持Policy卸载并退出，避免恢复时显存重叠。
        state = client.resource('sleep')
        if state.get('sleeping') is not True:
            raise RuntimeError('Judge没有确认释放GPU')
        audit('judge_asleep', resource=state)
        resume_policy()
        audit('policy_restored')


def install_training_switch(client, trainer_cls=None):
    if trainer_cls is None:
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
        trainer_cls = GRPOTrainer
    original = trainer_cls._score_completions
    if getattr(original, '_egoqa_shared_gpu', False):
        raise RuntimeError('同一训练进程不能重复安装不同Judge的GPU切换')

    @wraps(original)
    def score(trainer, *args, **kwargs):
        import gc
        import torch
        if not trainer.args.offload_model or not trainer.args.offload_optimizer:
            raise RuntimeError('共享GPU要求真实卸载模型与优化器')
        if not trainer.engine.inner_model_executor.is_sleeping:
            raise RuntimeError('Policy vLLM必须先睡眠，才能启动Judge')
        models = [trainer.accelerator.unwrap_model(trainer.model)]
        if trainer.ref_model is not None:
            models.append(trainer.ref_model)
        def suspend():
            torch.cuda.synchronize()
            for model in models:
                trainer.offload_model(model)
            if getattr(trainer, 'optimizer', None):
                trainer.offload_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        def resume():
            for model in models:
                trainer.load_model(model)
            if getattr(trainer, 'optimizer', None):
                trainer.load_optimizer()
            torch.cuda.synchronize()
        with judge_window(client, suspend, resume):
            return original(trainer, *args, **kwargs)
    score._egoqa_shared_gpu = True
    trainer_cls._score_completions = score


@contextmanager
def evaluation_judge_window(client, engine):
    import torch
    def suspend():
        engine.engine.reset_prefix_cache()
        engine.engine.sleep(level=1)
        torch.cuda.empty_cache()
    with judge_window(client, suspend, engine.engine.wake_up):
        yield
