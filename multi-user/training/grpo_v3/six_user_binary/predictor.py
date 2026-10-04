"""复用 Judge 的独立图片与 verdict-token 语义，通过 vLLM 读取原始概率。"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import time
import uuid

from training.judge_sft.collator import (
    _apply_chat_template, _assert_thinking_disabled, adaptive_image_max_pixels,
    model_visible_prompt, qwen_vision_geometry,
)
from training.judge_sft.contracts import VERDICT_ASSISTANT_PREFIX

TASKS = ("formality", "groundedness", "answerability")


class FrameImageCache:
    """服务内只读图片缓存；提问者改变排列时仍复用相同像素。"""
    def __init__(self, max_bytes):
        if max_bytes <= 0:
            raise ValueError('图片缓存容量必须为正')
        self.max_bytes = int(max_bytes)
        self.current_bytes = 0
        self.images = OrderedDict()

    def get_many(self, keys, contents, loader):
        if len(keys) != len(contents):
            raise ValueError('图片缓存键与输入数量不一致')
        available, missing = {}, OrderedDict()
        for key, content in zip(keys, contents):
            if key in self.images:
                self.images.move_to_end(key)
                available[key] = self.images[key][0]
            elif key not in missing:
                missing[key] = content
        decoded = loader(list(missing.values())) if missing else []
        if len(decoded) != len(missing):
            raise RuntimeError('图片解码数量不一致')
        for key, image in zip(missing, decoded):
            available[key] = image
            size = image.width * image.height * len(image.getbands())
            if size > self.max_bytes:
                continue
            while self.images and self.current_bytes + size > self.max_bytes:
                _, (_, removed) = self.images.popitem(last=False)
                self.current_bytes -= removed
            self.images[key] = (image, size)
            self.current_bytes += size
        return [available[key] for key in keys], {
            'decoded_image_cache_misses': len(missing),
            'decoded_image_cache_hits': len(keys) - len(missing),
            'decoded_image_cache_bytes': self.current_bytes,
        }


def validate_config(config: dict) -> dict:
    model = Path(config["model_id"]).expanduser().resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    mode = config.get("judge_mode", "adapter")
    if mode not in {"baseline", "adapter"}:
        raise ValueError("judge_mode 必须为 baseline 或 adapter")
    tp = config.get("tensor_parallel_size", 1)
    if isinstance(tp, bool) or not isinstance(tp, int) or tp < 1:
        raise ValueError("tensor_parallel_size 必须是正整数")
    adapters = config.get("adapters", {})
    if mode == "baseline":
        if adapters:
            raise ValueError("baseline 模式不得配置 adapter")
        return {**config, "model_id": str(model), "judge_mode": mode, "adapters": {}}
    if set(adapters) != set(TASKS):
        raise ValueError("必须为 formality、groundedness、answerability 显式指定 adapter")
    normalized, ranks = {}, []
    for task, value in adapters.items():
        path = Path(value).expanduser().resolve()
        cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        weights = path / "adapter_model.safetensors"
        if not weights.is_file() or weights.stat().st_size == 0:
            raise FileNotFoundError(weights)
        if str(cfg.get("base_model_name_or_path", "")).rstrip("/\\").replace("\\", "/").split("/")[-1] != model.name:
            raise ValueError(f"adapter base model 不一致：{task}")
        rank = cfg.get("r")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("adapter rank 不合法")
        ranks.append(rank)
        normalized[task] = str(path)
    result = {**config, "model_id": str(model), "judge_mode": mode, "adapters": normalized, "max_lora_rank": max(ranks)}
    return result


def engine_options(config):
    options = dict(model=config["model_id"], dtype="bfloat16", trust_remote_code=True,
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        max_model_len=int(config.get("max_input_tokens", 262144)), max_num_seqs=int(config.get("max_num_seqs", 4)),
        max_num_batched_tokens=int(config.get("max_num_batched_tokens", 8192)),
        gpu_memory_utilization=float(config.get("gpu_memory_utilization", .85)),
        enable_chunked_prefill=True, enable_prefix_caching=True,
        mm_processor_cache_gb=float(config.get("mm_processor_cache_gb", 4)),
        additional_config={"gdn_prefill_backend": "triton"},
        limit_mm_per_prompt={"image": 1800}, logprobs_mode="raw_logprobs",
        enable_lora=config["judge_mode"] == "adapter")
    if config.get('shared_gpu'):
        options['enable_sleep_mode'] = True
    if options["enable_lora"]:
        options.update(max_lora_rank=config["max_lora_rank"], max_loras=1,
                       max_cpu_loras=len(set(config["adapters"].values())))
    return options


def media_uuids(frames, *, effective, min_pixels, patch_size, namespace):
    # 完成的数据包在当前服务生命周期内只读；命名空间避免跨服务误用旧缓存。
    return [json.dumps([namespace, str(path), effective, min_pixels, patch_size], separators=(",", ":"))
            for path in frames]


def binary_decision(output, *, pass_id: int, fail_id: int) -> dict:
    if pass_id == fail_id:
        raise ValueError("pass/fail token 必须不同")
    try:
        row = output.outputs[0].logprobs[0]
        def get(token):
            value = row[token] if token in row else row[str(token)]
            return float(value.logprob if hasattr(value, "logprob") else value["logprob"] if isinstance(value, dict) else value)
        positive, negative = get(pass_id), get(fail_id)
    except (IndexError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("vLLM 未返回所请求的 pass/fail 原始 logprob，禁止使用 top-token 代替") from exc
    if not math.isfinite(positive) or not math.isfinite(negative):
        raise RuntimeError("Judge 原始 logprob 非有限")
    margin = positive - negative
    probability = 1. / (1. + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1. + math.exp(margin))
    return {"pass_probability": probability, "pass_minus_fail_logit": margin,
            "pass_logprob": positive, "fail_logprob": negative,
            "verdict": "pass" if probability >= .5 else "fail"}


class VllmJudge:
    @classmethod
    def from_config(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __init__(self, config):
        self.config = validate_config(config)
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"], local_files_only=True, trust_remote_code=True)
        self.geometry = qwen_vision_geometry(self.processor)
        self.max_tokens = int(self.config.get("max_input_tokens", 262144))
        self.min_pixels = int(self.config.get("min_pixels", 3136))
        self.max_pixels = int(self.config.get("max_pixels", 262144))
        unique_paths = list(dict.fromkeys(self.config["adapters"].values()))
        self.llm = LLM(**engine_options(self.config))
        self.sleeping = False
        tokenizer = self.llm.get_tokenizer()
        ids = {}
        for word in ("pass", "fail"):
            tokens = tokenizer.encode(word, add_special_tokens=False)
            other = self.processor.tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) != 1 or tokens != other:
                raise RuntimeError("Judge processor 与 vLLM 的 verdict token 不一致或不是单 token")
            ids[word] = tokens[0]
        self.pass_id, self.fail_id = ids["pass"], ids["fail"]
        self.sampling = SamplingParams(temperature=0., max_tokens=1, detokenize=False,
                                      logprob_token_ids=[self.pass_id, self.fail_id])
        self.adapters = {}
        if self.config["judge_mode"] == "adapter":
            from vllm.lora.request import LoRARequest
            self.adapters = {task: LoRARequest(f"judge-{unique_paths.index(path)+1}", unique_paths.index(path)+1, path)
                             for task, path in self.config["adapters"].items()}
        self.image_cache = FrameImageCache(int(float(self.config.get('decoded_image_cache_gb', 8)) * 1024**3))
        self.cache_namespace = uuid.uuid4().hex
        self.identity = {"model_id": self.config["model_id"], "judge_mode": self.config["judge_mode"], "adapters": self.config["adapters"],
            "backend": "vllm_raw_verdict_logprobs", "max_input_tokens": self.max_tokens,
            "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
            "tensor_parallel_size": self.config.get("tensor_parallel_size", 1),
            "max_num_seqs": self.config.get("max_num_seqs", 4), "media_uuid_cache": True}

    def predict(self, example):
        return self.predict_many([example])[0]

    def resource(self, action):
        if not self.config.get('shared_gpu') or action not in {'sleep', 'wake'}:
            raise ValueError('资源切换仅用于显式共享GPU的Judge')
        target = action == 'sleep'
        start = time.perf_counter()
        if self.sleeping is not target:
            # 中途失败保留未知状态；下一次释放必须实际调用，不能跳过。
            self.sleeping = None
            if target:
                self.llm.reset_prefix_cache()
                self.llm.sleep(level=1)
            else:
                self.llm.wake_up()
            self.sleeping = target
        return {'sleeping': self.sleeping, 'elapsed_seconds': time.perf_counter() - start}

    def prepare(self, example):
        start = time.perf_counter()
        effective = adaptive_image_max_pixels(image_count=example.frame_count,
            configured_max_pixels=self.max_pixels, min_pixels=self.min_pixels,
            max_input_tokens=self.max_tokens, vision_token_pixel_area=self.geometry["merged_token_pixel_area"])
        content = [{"type": "image", "image": path, "min_pixels": self.min_pixels, "max_pixels": effective}
                   for path in example.frames]
        content.append({"type": "text", "text": model_visible_prompt(example)})
        messages = [{"role": "user", "content": content}]
        rendered = _apply_chat_template(self.processor, messages)
        _assert_thinking_disabled(rendered)
        payload = {"prompt": rendered + VERDICT_ASSISTANT_PREFIX}
        cache_audit = {}
        if example.frames:
            def decode(contents):
                images, videos = self.process_vision_info([{'role':'user','content':contents}], image_patch_size=self.geometry["patch_size"])
                if videos is not None and len(videos):
                    raise RuntimeError("Judge 独立图片输入意外产生视频")
                if images is None or len(images) != len(contents):
                    raise RuntimeError("Judge 解码帧数不匹配")
                return list(images)
            keys = [(str(path), effective, self.min_pixels, self.geometry['patch_size']) for path in example.frames]
            images, cache_audit = self.image_cache.get_many(keys, content[:-1], decode)
            payload["multi_modal_data"] = {"image": images}
            payload["multi_modal_uuids"] = {"image": media_uuids(example.frames, effective=effective,
                min_pixels=self.min_pixels, patch_size=self.geometry["patch_size"], namespace=self.cache_namespace)}
        audit = {"frame_count": example.frame_count, "effective_max_pixels": effective,
                "frame_order": [f.label for f in example.frame_sets], "prompt": example.prompt,
                "prepare_seconds": time.perf_counter() - start, **cache_audit}
        return payload, audit

    def predict_many(self, examples):
        if self.config.get('shared_gpu') and self.sleeping is not False:
            raise RuntimeError('共享Judge必须先完成显存唤醒才能评分')
        if not examples:
            return []
        tasks = {example.task.value for example in examples}
        if len(tasks) != 1:
            raise ValueError("Judge 批次必须属于同一种任务，防止 adapter 路由错误")
        results = []
        batch_size = int(self.config.get("max_num_seqs", 4))
        if batch_size < 1:
            raise ValueError("max_num_seqs 必须为正数")
        for offset in range(0, len(examples), batch_size):
            prepared = [self.prepare(e) for e in examples[offset:offset + batch_size]]
            start = time.perf_counter()
            outputs = self.llm.generate([p for p,_ in prepared], self.sampling,
                lora_request=self.adapters.get(next(iter(tasks))), use_tqdm=False)
            elapsed = time.perf_counter() - start
            if len(outputs) != len(prepared):
                raise RuntimeError("Judge 返回结果数量不匹配")
            for output, (_, audit) in zip(outputs, prepared):
                result = binary_decision(output, pass_id=self.pass_id, fail_id=self.fail_id)
                results.append({**result, **audit, "generate_batch_seconds": elapsed,
                    "generate_batch_size": len(prepared), "num_cached_tokens": getattr(output, "num_cached_tokens", None)})
        return results
