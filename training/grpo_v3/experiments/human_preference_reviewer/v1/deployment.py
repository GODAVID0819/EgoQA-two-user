"""Frozen score-only ordinal reviewer deployment backend."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint import (
    load_checkpoint_contract,
    load_lora_adapter,
    load_ordinal_heads,
    validate_checkpoint_runtime_contract,
)
from .config import ReviewerV1Config
from .data import CandidateRecord
from .lora import inject_reviewer_lora
from .modeling import ReviewerV1, resolve_hidden_size
from .prompting import build_messages, encode_candidate

SCORE_FORMAT_VERSION = "score_only_ordinal_reviewer_score_v1"
REWARD_POLICY_VERSION = "score_only_expected_score_040_040_020_v1"
SCORE_FIELDS = ("evidence_quality", "answerability", "qa_formality")
REWARD_WEIGHTS = {
    "evidence_quality": 0.4,
    "answerability": 0.4,
    "qa_formality": 0.2,
}


def reviewer_config_from_contract(contract: Mapping[str, Any]) -> ReviewerV1Config:
    """Reconstruct every serialized architecture/LoRA field without defaults."""

    field_names = [field.name for field in fields(ReviewerV1Config)]
    missing = [name for name in field_names if name not in contract]
    if missing:
        raise ValueError(f"checkpoint is missing ReviewerV1Config fields: {missing}")
    values = {name: contract[name] for name in field_names}
    targets = values["lora_target_modules"]
    if not isinstance(targets, (list, tuple)):
        raise ValueError("checkpoint lora_target_modules must be a list")
    values["lora_target_modules"] = tuple(str(name) for name in targets)
    config = ReviewerV1Config(**values)
    validate_checkpoint_runtime_contract(
        contract,
        stage=config.stage,
        active_heads=config.active_heads,
        lora_enabled=config.lora_enabled,
        model_name_or_path=config.model_name_or_path,
        reviewer_config=config.to_dict(),
    )
    return config


def scalarize_expected_scores(expected_scores: Mapping[str, float]) -> float:
    if set(expected_scores) != set(SCORE_FIELDS):
        raise ValueError("expected scores must contain exactly the three score-only fields")
    normalized = {}
    for field in SCORE_FIELDS:
        value = float(expected_scores[field])
        if not math.isfinite(value) or not 1.0 <= value <= 3.0:
            raise ValueError(f"expected score for {field} must be finite and in [1, 3]")
        normalized[field] = (value - 1.0) / 2.0
    return sum(REWARD_WEIGHTS[field] * normalized[field] for field in SCORE_FIELDS)


def _validated_cumulative(values: Sequence[float], field: str) -> list[float]:
    if len(values) != 2:
        raise ValueError(f"{field} must contain two cumulative probabilities")
    row = [float(value) for value in values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in row):
        raise ValueError(f"{field} cumulative probabilities must be finite and in [0, 1]")
    if row[0] + 1e-6 < row[1]:
        raise ValueError(f"{field} cumulative probabilities must be monotone")
    return row


def build_score_payload(
    *,
    review_key: str,
    evidence_id: str,
    cumulative_probabilities: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    if not str(review_key).strip() or not str(evidence_id).strip():
        raise ValueError("review_key and evidence_id are required")
    if set(cumulative_probabilities) != set(SCORE_FIELDS):
        raise ValueError("cumulative probabilities must contain exactly three score-only fields")
    cumulative = {
        field: _validated_cumulative(cumulative_probabilities[field], field)
        for field in SCORE_FIELDS
    }
    class_probabilities = {
        field: [1.0 - row[0], row[0] - row[1], row[1]]
        for field, row in cumulative.items()
    }
    expected_scores = {
        field: round(1.0 + row[0] + row[1], 12)
        for field, row in cumulative.items()
    }
    hard_scores = {
        field: 1 + sum(value >= 0.5 for value in row)
        for field, row in cumulative.items()
    }
    policy = {
        "version": REWARD_POLICY_VERSION,
        "transform": "weighted_normalized_expected_score",
        "normalization": "(expected_score-1)/2",
        "weights": dict(REWARD_WEIGHTS),
    }
    serialized = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    policy["sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return {
        "format_version": SCORE_FORMAT_VERSION,
        "review_key": str(review_key),
        "evidence_id": str(evidence_id),
        "cumulative_probabilities": cumulative,
        "class_probabilities": class_probabilities,
        "hard_scores": hard_scores,
        "expected_scores": expected_scores,
        "reward": scalarize_expected_scores(expected_scores),
        "reward_policy": policy,
    }


def _move_to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)
    return value


class FrozenScoreOnlyReviewer:
    def __init__(self, reviewer: Any, processor: Any, process_vision_info: Any, *, device: str, checkpoint: Path) -> None:
        self.reviewer = reviewer
        self.processor = processor
        self.process_vision_info = process_vision_info
        self.device = device
        self.checkpoint = checkpoint

    def readiness(self) -> dict[str, Any]:
        trainable = sum(
            parameter.numel() for parameter in self.reviewer.parameters()
            if parameter.requires_grad
        )
        checks = {
            "checkpoint_contract": (self.checkpoint / "reviewer_v1_config.json").is_file(),
            "lora_adapter": (self.checkpoint / "lora_adapter.pt").is_file(),
            "ordinal_heads": (self.checkpoint / "ordinal_heads.pt").is_file(),
            "processor": (self.checkpoint / "processor").is_dir(),
            "frozen": trainable == 0,
            "evaluation_mode": not self.reviewer.training,
        }
        return {
            "status": "ok" if all(checks.values()) else "unhealthy",
            "format_version": SCORE_FORMAT_VERSION,
            "checkpoint": str(self.checkpoint),
            "trainable_parameter_count": trainable,
            "checks": checks,
        }

    def score(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        record = CandidateRecord(
            candidate_id=str(candidate["review_key"]),
            evidence_id=str(candidate["evidence_id"]),
            display_order=0,
            question=str(candidate["question"]),
            options=tuple(str(value) for value in candidate["options"]),
            correct=str(candidate["correct"]),
            answer=str(candidate["answer"]),
            evidence_quality=None,
            answerability=None,
            qa_formality=None,
        )
        messages = build_messages(
            record,
            video_a_path=str(candidate["video_a_path"]),
            video_b_path=str(candidate["video_b_path"]),
            video_a_user=str(candidate["video_a_user"]),
            video_b_user=str(candidate["video_b_user"]),
        )
        encoded = encode_candidate(self.processor, self.process_vision_info, messages)
        self.reviewer.eval()
        with torch.no_grad():
            output = self.reviewer(**_move_to_device(encoded, self.device))
        logits = {
            "evidence_quality": output.evidence_threshold_logits,
            "answerability": output.answerability_threshold_logits,
            "qa_formality": output.formality_threshold_logits,
        }
        cumulative = {
            field: tensor[0].detach().float().sigmoid().cpu().tolist()
            for field, tensor in logits.items()
        }
        return build_score_payload(
            review_key=str(candidate["review_key"]),
            evidence_id=str(candidate["evidence_id"]),
            cumulative_probabilities=cumulative,
        )


def load_frozen_score_only_reviewer(
    checkpoint: str | Path,
    *,
    model: str | None = None,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
) -> FrozenScoreOnlyReviewer:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint = Path(checkpoint)
    contract = load_checkpoint_contract(checkpoint)
    config = reviewer_config_from_contract(contract)
    model_path = str(model or contract["model_name_or_path"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[torch_dtype]
    processor = AutoProcessor.from_pretrained(checkpoint / "processor", trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "attn_implementation": "sdpa"}
    try:
        backbone = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        backbone = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    if getattr(backbone.config, "model_type", None) != "qwen3_vl":
        raise ValueError("score-only reviewer requires a Qwen3-VL base model")
    backbone.to(device)
    backbone.config.use_cache = False
    backbone, actual_lora_targets = inject_reviewer_lora(backbone, config)
    validate_checkpoint_runtime_contract(
        contract,
        stage=config.stage,
        active_heads=config.active_heads,
        lora_enabled=config.lora_enabled,
        model_name_or_path=config.model_name_or_path,
        reviewer_config=config.to_dict(),
        actual_lora_targets=tuple(actual_lora_targets),
    )
    reviewer = ReviewerV1(
        backbone, resolve_hidden_size(backbone.config), active_heads=config.active_heads
    ).to(device)
    load_ordinal_heads(reviewer, checkpoint)
    load_lora_adapter(reviewer, checkpoint)
    for parameter in reviewer.parameters():
        parameter.requires_grad = False
    reviewer.eval()
    scorer = FrozenScoreOnlyReviewer(
        reviewer, processor, process_vision_info, device=device, checkpoint=checkpoint
    )
    if scorer.readiness()["status"] != "ok":
        raise RuntimeError(f"score-only reviewer readiness failed: {scorer.readiness()}")
    return scorer
