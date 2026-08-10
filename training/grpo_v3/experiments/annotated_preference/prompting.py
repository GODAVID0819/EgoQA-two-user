"""紧凑 QA 偏好训练的生成提示词与完成序列化。"""

from __future__ import annotations

import hashlib
import json

from training.grpo_v3.experiments.human_preference_reviewer.v1.data import CandidateRecord

COMPACT_QA_CONTRACT = "compact_qa_v1"
PROMPT_REVISION = "annotated_pareto_compact_qa_v1"

_COMPACT_FIELDS = ("question", "options", "correct", "answer")

COMPACT_GENERATION_PROMPT = """<video>
<video>
The first video is the Speaker and the second video is the Provider. They are synchronized complete dual views of the same interaction.
Write a natural first-person or shared-memory question from the Speaker/asker perspective.
Generate one grounded multiple-choice QA based jointly on both videos.
Return only one JSON object, with no Markdown or explanation, in exactly this field order:
{"question":"...","options":["...","...","...","...","..."],"correct":"A","answer":"..."}
The options must contain exactly five non-empty, mutually exclusive choices of the same semantic type.
The combined visual evidence must make exactly one option semantically correct.
correct must be exactly one of A, B, C, D, or E.
answer must exactly equal the text of the option selected by correct.
The question must not contain names, timestamps, or meta-language such as dataset, video, or frame."""


def serialize_compact_completion(candidate: CandidateRecord) -> str:
    """将候选的模型可见四字段编码为紧凑、稳定的 UTF-8 JSON。"""
    features = candidate.model_features()
    payload = {field: features[field] for field in _COMPACT_FIELDS}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def build_compact_generation_prompt() -> str:
    """返回固定的 compact_qa_v1 生成提示词。"""
    return COMPACT_GENERATION_PROMPT


def prompt_sha256() -> str:
    """返回提示词 UTF-8 原文的稳定 SHA-256。"""
    return hashlib.sha256(COMPACT_GENERATION_PROMPT.encode("utf-8")).hexdigest()
