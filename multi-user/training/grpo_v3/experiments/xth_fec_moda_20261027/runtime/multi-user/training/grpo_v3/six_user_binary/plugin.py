"""三项质量奖励与同窗固定历史池距离，保持真实G4优势分组。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid

from training.grpo_v3.six_user_binary.data import validate_row
from training.grpo_v3.six_user_binary.reward import QUALITY_MODES, aggregate
from training.grpo_v3.six_user_binary.service import JudgeClient, validate_response
from training.grpo_v3.six_user_binary.diversity import score_group

if os.environ.get('EGOQA_QWEN_PARTIAL_PACKED_LORA') == '1':
    from training.grpo_v3.six_user_binary.packed_lora_compat import install
    install()

try:
    from swift.rewards import ORM, orms
except ModuleNotFoundError as exc:
    if exc.name != "swift":
        raise
    class ORM:
        def __init__(self, args=None, **kwargs):
            self.args = args
    orms = {}


def expand(value, count, name):
    values = value if isinstance(value, (list, tuple)) else [value]
    if len(values) == count:
        return list(values)
    if len(values) == 1:
        return list(values) * count
    raise ValueError(f"{name}: {len(values)} 条元数据不能对齐 {count} 条 completion")


def completion_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict) and value[0].get("role") == "assistant" and isinstance(value[0].get("content"), str):
        return value[0]["content"]
    raise TypeError("不支持的 ms-swift completion 格式")


class SixUserBinaryReward(ORM):
    def __init__(self, args=None, *, client=None, trace_path=None, reward_mode=None,
                 encoder=None, history_by_window=None, diversity_lambda=None, **kwargs):
        super().__init__(args, **kwargs)
        self.mode = reward_mode or os.environ.get("EGOQA_SIX_USER_REWARD_MODE")
        if self.mode not in QUALITY_MODES:
            raise ValueError(f"质量奖励只能是 {QUALITY_MODES}")
        self.history_metadata = None
        if diversity_lambda is None:
            config=json.loads(Path(os.environ['EGOQA_MODA_CONFIG']).read_text(encoding='utf-8'))
            diversity_lambda=config['diversity_lambda']
            if diversity_lambda:
                from training.grpo_v3.six_user_binary.history_pool import MiniLMEncoder, load_history
                encoder=MiniLMEncoder(config['encoder_directory'])
                history_by_window,self.history_metadata=load_history(config['history_pool_directory'])
        self.encoder=encoder
        self.history_by_window=history_by_window or {}
        self.diversity_lambda=diversity_lambda
        self.completed_groups=int(os.environ.get('EGOQA_GRPO_START_STEP','0'))
        self.client = client or JudgeClient(os.environ["EGOQA_SIX_USER_JUDGE_URL"],
                                           expected_instance=os.environ["EGOQA_SIX_USER_JUDGE_INSTANCE"])
        self.trace = Path(trace_path or os.environ["EGOQA_GRPO_V3_REWARD_TRACE"])
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.validated = set()
        from training.grpo_v3.six_user_binary.utilization_runtime import enable_training_guard
        self.utilization_guard = enable_training_guard()
        from training.grpo_v3.six_user_binary.policy_image_cache import enable_from_environment
        self.policy_image_cache = enable_from_environment()
        if os.environ.get('EGOQA_SHARED_GPU') == '1':
            from training.grpo_v3.six_user_binary.shared_gpu import install_training_switch
            install_training_switch(self.client)
        if client is None:
            self._record_backend(args)

    def _record_backend(self, args):
        from importlib.metadata import version, PackageNotFoundError
        audit = {"use_vllm": getattr(args, "use_vllm", None), "attn_impl": getattr(args, "attn_impl", None), "versions": {}}
        audit['policy_image_cache'] = self.policy_image_cache.snapshot() if self.policy_image_cache is not None else None
        audit.update(reward_revision='fec_moda_static_history_v1',diversity_lambda=self.diversity_lambda,
                     history_pool=self.history_metadata)
        for name in ("torch", "vllm", "transformers", "ms-swift", "flash-linear-attention", "fla-core", "flash-attn"):
            try:
                audit["versions"][name] = version(name)
            except PackageNotFoundError:
                audit["versions"][name] = None
        try:
            import torch
            from transformers.models.qwen3_5 import modeling_qwen3_5 as model
            audit.update(cuda_available=torch.cuda.is_available(),
                flash_sdpa_available=torch.backends.cuda.is_flash_attention_available(),
                flash_sdpa_enabled=torch.backends.cuda.flash_sdp_enabled(),
                linear_chunk_kernel=getattr(model.chunk_gated_delta_rule, "__module__", None),
                linear_recurrent_kernel=getattr(model.fused_recurrent_gated_delta_rule, "__module__", None))
        except Exception as exc:
            audit["inspection_error"] = f"{type(exc).__name__}: {exc}"
        (self.trace.parent / "acceleration_backend.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    def __call__(self, completions, **kwargs):
        count = len(completions)
        metadata = {}
        for key in ("dataset_root", "source_packet_id", "asker_index", "evidence_id", "required_users", "generator_image_paths"):
            value = kwargs[key]
            if key in {"required_users", "generator_image_paths"} and isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
                value = [value]
            metadata[key] = expand(value, count, key)
        rewards = []
        batch_id = uuid.uuid4().hex
        requests, bindings = [], []
        for i, raw in enumerate(completions):
            binding = {key: values[i] for key, values in metadata.items()}
            cache_key = json.dumps(binding, sort_keys=True)
            with self.lock:
                if cache_key not in self.validated:
                    validate_row(binding, require_prompt=False)
                    self.validated.add(cache_key)
            request = {**binding, "request_id": f"{batch_id}:{i}", "completion": completion_text(raw)}
            requests.append(request)
            bindings.append(binding)
        if not requests:
            return []
        start = time.perf_counter()
        # 兼容注入的单条客户端；真实服务走整组批量评分，失败不退化成零奖励。
        results = self.client.score_many(requests) if hasattr(self.client, "score_many") else [self.client.score(r) for r in requests]
        if len(results) != len(requests):
            raise RuntimeError("Judge 返回候选数量不一致")
        elapsed = time.perf_counter() - start
        checked=[validate_response(response,request) for request,response in zip(requests,results)]
        components=score_group(requests,checked,encoder=self.encoder,history_by_window=self.history_by_window,
                               diversity_lambda=self.diversity_lambda,mode=self.mode)
        records=[]
        step=self.completed_groups+1
        for slot,(binding,request,result,parts) in enumerate(zip(bindings,requests,checked,components)):
            record = {**parts, "reward_mode": self.mode, "global_step":step,"slot":slot,
                "reward_revision": "fec_moda_static_history_v1", "source_packet_id": binding["source_packet_id"],
                "asker_index": binding["asker_index"], "evidence_id": binding["evidence_id"],
                "completion": request["completion"], "judge_result": result,
                "score_batch_id": batch_id, "score_batch_size": len(requests), "score_batch_seconds": elapsed}
            if self.policy_image_cache is not None:
                record['policy_image_cache'] = self.policy_image_cache.snapshot()
            records.append(record)
            rewards.append(parts['reward'])
        average=sum(rewards)/len(rewards)
        metrics={'global_step':step,'rank':int(os.environ.get('RANK','0')),
                 'status':'reward_processed_pending_optimizer_confirmation','candidate_count':len(records),
                 'pass_count':sum(int(r['bonus_gate']) for r in records),'invalid_count':sum(r['judge_result']['status']=='invalid_completion' for r in records),
                 'training_mean_reward':average,'group_reward_variance':sum((r-average)**2 for r in rewards)/len(rewards),
                 'zero_advantage_group':max(rewards)==min(rewards)}
        for output_key,source_key in [('quality_reward_mean','quality_reward'),('diversity_distance_mean','diversity_distance'),
                                      ('diversity_bonus_mean','diversity_bonus')]:
            metrics[output_key]=sum(r[source_key] for r in records)/len(records)
        with self.lock:
            with self.trace.open('a',encoding='utf-8') as out:
                out.write(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
            with (self.trace.parent/'group_metrics.jsonl').open('a',encoding='utf-8') as out:
                out.write(json.dumps(metrics,ensure_ascii=False,allow_nan=False)+'\n')
            self.completed_groups=step
        return rewards


orms["egoqa_fec_moda_v1"] = SixUserBinaryReward

# 外部插件导入时注册正常结束筛选阶段的回调，不改变奖励或调度器。
from training.grpo_v3.six_user_binary import stage_stop as _stage_stop
