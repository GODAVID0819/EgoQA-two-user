"""双原生视频冻结答题器及 A-E teacher-forcing 序列评分。"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Mapping

from training.grpo_v3_answer_margin import LABELS


@dataclass(frozen=True)
class ScoreRequest:
    """一个 scorer 请求：严格包含两段视频、一个问题和五个选项。"""

    videos: tuple[str, str]
    question: str
    options: tuple[str, str, str, str, str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.videos, tuple)
            or len(self.videos) != 2
            or any(not isinstance(video, str) or not video.strip() for video in self.videos)
        ):
            raise ValueError("videos must be exactly two non-empty native-video paths")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if (
            not isinstance(self.options, tuple)
            or len(self.options) != 5
            or any(not isinstance(option, str) or not option.strip() for option in self.options)
        ):
            raise ValueError("options must be exactly five non-empty strings")

    def to_payload(self) -> dict[str, Any]:
        return {
            "videos": list(self.videos),
            "question": self.question,
            "options": list(self.options),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScoreRequest":
        if not isinstance(payload, Mapping) or set(payload) != {"videos", "question", "options"}:
            raise ValueError("score request must contain only videos, question and options")
        videos = payload["videos"]
        options = payload["options"]
        if not isinstance(videos, list) or not isinstance(options, list):
            raise ValueError("videos and options must be JSON arrays")
        return cls(tuple(videos), payload["question"], tuple(options))


@dataclass(frozen=True)
class LabelScore:
    label: str
    token_ids: list[int]
    token_logprobs: list[float]
    sequence_logprob: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "token_ids": list(self.token_ids),
            "token_logprobs": list(self.token_logprobs),
            "sequence_logprob": self.sequence_logprob,
        }


@dataclass(frozen=True)
class PromptAuditMaterial:
    """只供泄露审计、绝不拼入模型 prompt 的生成器侧材料。"""

    excluded_values: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.excluded_values, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.excluded_values.items()
        ):
            raise ValueError("excluded_values must map strings to strings")

    def to_payload(self) -> dict[str, str]:
        return dict(self.excluded_values)

    @classmethod
    def from_payload(cls, payload: Any) -> "PromptAuditMaterial":
        if payload is None:
            return cls({})
        if not isinstance(payload, Mapping):
            raise ValueError("audit_material must be an object")
        return cls(dict(payload))


@dataclass(frozen=True)
class PromptAudit:
    prompt_sha256: str
    passed: bool
    rules: list[str]
    hits: list[dict[str, str]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "prompt_sha256": self.prompt_sha256,
            "passed": self.passed,
            "rules": list(self.rules),
            "hits": [dict(hit) for hit in self.hits],
        }


@dataclass(frozen=True)
class ScoreResponse(Mapping[str, LabelScore]):
    scores: dict[str, LabelScore]
    prompt_audit: PromptAudit
    rendered_prompt: str

    def __getitem__(self, label: str) -> LabelScore:
        return self.scores[label]

    def __iter__(self) -> Iterator[str]:
        return iter(self.scores)

    def __len__(self) -> int:
        return len(self.scores)

    def to_payload(self) -> dict[str, Any]:
        return {
            "scores": {label: self.scores[label].to_payload() for label in LABELS},
            "prompt_audit": self.prompt_audit.to_payload(),
            "rendered_prompt": self.rendered_prompt,
        }

_GENERATOR_FIELD_PATTERNS = {
    "correct": re.compile(r'(?i)(?:["\']correct["\']\s*[:=：]|\bcorrect\s*[:=：])'),
    # 裸 `Answer:` 是本实验规定的模型作答前缀，因此只拦截引号键、等号或 generator marker。
    "answer": re.compile(
        r'(?i)(?:["\']answer["\']\s*[:=：]|\banswer\s*=|\bgenerator\s+answer\s*[:=：]?)'
    ),
    "rationale": re.compile(
        r'(?i)(?:["\']rationale["\']\s*[:=：]|\brationale\s*[:=：])'
    ),
    "self_check": re.compile(
        r'(?i)(?:["\']self[-_ ]check["\']\s*[:=：]|\bself[-_ ]check\s*[:=：])'
    ),
    "review": re.compile(r'(?i)(?:["\']review["\']\s*[:=：]|\breview\s*[:=：])'),
    "evidence_claims": re.compile(
        r'(?i)(?:["\']evidence[_ ]claims["\']\s*[:=：]|\bevidence[_ ]claims\s*[:=：])'
    ),
    "explanation": re.compile(
        r'(?i)(?:["\']explanation["\']\s*[:=：]|\bexplanation\s*[:=：])'
    ),
    "解释": re.compile(r'(?:["\']解释["\']\s*[:=：]|解释\s*[:=：])'),
}


def audit_rendered_prompt(
    prompt: str,
    request: ScoreRequest,
    material: PromptAuditMaterial,
) -> PromptAudit:
    """扫描实际模型文本；审计材料只参与比较，不参与 prompt 构造。"""

    hits: list[dict[str, str]] = []
    rules = ["generator_field_marker_scan", "excluded_value_scan"]
    for name, pattern in _GENERATOR_FIELD_PATTERNS.items():
        if pattern.search(prompt):
            hits.append({"rule": "generator_field_marker_scan", "field": name})

    allowed_sources = {request.question, *request.options}
    for field, raw_value in material.excluded_values.items():
        value = raw_value.strip()
        if not value:
            continue
        if value in allowed_sources:
            if "excluded_value_allowed_source" not in rules:
                rules.append("excluded_value_allowed_source")
            continue
        # 单字母 correct 标签无法与正常 A-E 指令可靠区分，只由字段标记规则审计。
        if field.casefold() == "correct" and value.upper() in LABELS:
            if "short_correct_label_skipped" not in rules:
                rules.append("short_correct_label_skipped")
            continue
        if value in prompt:
            hits.append({"rule": "excluded_value_scan", "field": field})
    return PromptAudit(
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        passed=not hits,
        rules=rules,
        hits=hits,
    )


def build_answer_prompt(
    question: str,
    options: tuple[str, str, str, str, str],
) -> str:
    """构造不接收生成器答案字段的最小五选一 prompt。"""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if (
        not isinstance(options, tuple)
        or len(options) != len(LABELS)
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError("options must be exactly five non-empty strings")
    option_lines = "\n".join(
        f"{label}. {option}" for label, option in zip(LABELS, options)
    )
    return (
        "Use the two ordered videos to solve the multiple-choice question.\n"
        f"Question: {question}\n"
        f"{option_lines}\n"
        "Choose exactly one label from A, B, C, D, or E. Do not explain.\n"
        "Answer:"
    )


def freeze_model(model: Any) -> int:
    """切换 eval 并冻结全部参数，返回可训练参数数（必须为零）。"""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_count != 0:
        raise RuntimeError("frozen answer scorer still has trainable parameters")
    return trainable_count


def _active_positions(attention_mask: Any, row: int) -> list[int]:
    values = attention_mask[row]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [index for index, value in enumerate(values) if int(value) != 0]


def _active_ids(input_ids: Any, row: int, positions: list[int]) -> list[int]:
    return [int(input_ids[row, position]) for position in positions]


def _scalar(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _move_to_model_device(inputs: Any, model: Any) -> Any:
    if hasattr(inputs, "to") and hasattr(model, "device"):
        return inputs.to(model.device)
    if not isinstance(inputs, dict) or not hasattr(model, "device"):
        return inputs
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


class FrozenAnswerScorer:
    """对五个完整标签 span 执行 teacher forcing 的 inference-only scorer。"""

    def __init__(self, model: Any, processor: Any, *, torch_module: Any | None = None):
        if torch_module is None:
            import torch as torch_module  # 延迟导入，保证纯逻辑单测无需 Torch。

        self.model = model
        self.processor = processor
        self.torch = torch_module
        self.trainable_parameter_count = freeze_model(model)

    def _encode(self, texts: list[str], videos: tuple[str, str]) -> Any:
        encoded = self.processor(
            text=texts,
            # Qwen3-VL processor 要求媒体按 batch 中占位符顺序展平：
            # 每条文本两个 video_pad，因此五标签批次恰好对应十个媒体项。
            videos=[video for _ in texts for video in videos],
            return_tensors="pt",
            padding=True,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise RuntimeError("processor response lacks input_ids or attention_mask")
        return encoded

    def _render_prompt(self, request: ScoreRequest) -> str:
        prompt = build_answer_prompt(request.question, request.options)
        if not hasattr(self.processor, "apply_chat_template"):
            return prompt
        user_text = prompt.removesuffix("Answer:").rstrip()
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video": request.videos[0]},
                {"type": "video", "video": request.videos[1]},
                {"type": "text", "text": user_text},
            ],
        }]
        rendered = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("processor returned an empty chat template")
        return rendered + "Answer:"

    def readiness(self) -> dict[str, Any]:
        parameters = list(self.model.parameters()) if self.model is not None else []
        trainable_count = sum(
            int(parameter.numel()) if hasattr(parameter, "numel") else 1
            for parameter in parameters
            if parameter.requires_grad
        )
        checks = {
            "model_exists": self.model is not None,
            "processor_exists": self.processor is not None,
            "eval_mode": self.model is not None and not bool(self.model.training),
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in parameters),
            "trainable_parameter_count_zero": trainable_count == 0,
        }
        healthy = all(checks.values()) and self.trainable_parameter_count == 0
        return {
            "status": "ok" if healthy else "unhealthy",
            "checks": checks,
            "trainable_parameter_count": trainable_count,
        }

    def score(
        self,
        request: ScoreRequest,
        *,
        audit_material: PromptAuditMaterial | None = None,
    ) -> ScoreResponse:
        if not isinstance(request, ScoreRequest):
            raise TypeError("request must be ScoreRequest")
        if self.model.training:
            raise RuntimeError("answer scorer model must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("answer scorer model must remain fully frozen")

        prompt = self._render_prompt(request)
        prompt_audit = audit_rendered_prompt(
            prompt,
            request,
            audit_material or PromptAuditMaterial({}),
        )
        if not prompt_audit.passed:
            raise RuntimeError(f"rendered scorer prompt failed leakage audit: {prompt_audit.hits}")
        prompt_batch = self._encode([prompt], request.videos)
        prompt_positions = _active_positions(prompt_batch["attention_mask"], 0)
        prompt_ids = _active_ids(prompt_batch["input_ids"], 0, prompt_positions)
        prompt_length = len(prompt_ids)
        if prompt_length < 1:
            raise RuntimeError("scorer prompt token span is empty")

        full_batch = self._encode([prompt + label for label in LABELS], request.videos)
        input_ids = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        label_spans: dict[str, list[int]] = {}
        label_positions: dict[str, list[int]] = {}
        for row, label in enumerate(LABELS):
            active_positions = _active_positions(attention_mask, row)
            full_ids = _active_ids(input_ids, row, active_positions)
            if full_ids[:prompt_length] != prompt_ids:
                raise RuntimeError(f"{label} encoding does not share the audited prompt prefix")
            label_ids = full_ids[prompt_length:]
            if not label_ids:
                raise RuntimeError(f"{label} label token span is empty")
            label_spans[label] = label_ids
            label_positions[label] = active_positions[prompt_length:]

        model_inputs = _move_to_model_device(full_batch, self.model)
        with self.torch.inference_mode():
            output = self.model(**model_inputs)
            logits = output.logits
            log_probs = self.torch.nn.functional.log_softmax(logits, dim=-1)

        results: dict[str, LabelScore] = {}
        for row, label in enumerate(LABELS):
            token_logprobs: list[float] = []
            for token_position, token_id in zip(label_positions[label], label_spans[label]):
                if token_position <= 0:
                    raise RuntimeError("cannot score a token without a preceding context token")
                value = _scalar(log_probs[row, token_position - 1, token_id])
                if not math.isfinite(value):
                    raise RuntimeError(f"{label} contains a non-finite token logprob")
                token_logprobs.append(value)
            sequence_logprob = float(sum(token_logprobs))
            if not math.isfinite(sequence_logprob):
                raise RuntimeError(f"{label} sequence logprob is non-finite")
            results[label] = LabelScore(
                label=label,
                token_ids=label_spans[label],
                token_logprobs=token_logprobs,
                sequence_logprob=sequence_logprob,
            )
        return ScoreResponse(results, prompt_audit, prompt)


def load_frozen_answer_scorer(
    model_name_or_path: str,
    *,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
) -> FrozenAnswerScorer:
    """延迟加载真实依赖；调用方应通过 CUDA_VISIBLE_DEVICES 隔离 GPU1。"""

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = getattr(torch, torch_dtype)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    ).to(device)
    return FrozenAnswerScorer(model, processor, torch_module=torch)
