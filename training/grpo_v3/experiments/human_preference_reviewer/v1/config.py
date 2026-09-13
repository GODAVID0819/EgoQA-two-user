"""Configuration contract for Reviewer v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ReviewerV1Config:
    stage: str = "stage2"
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    num_score_levels: int = 3
    num_ordinal_thresholds: int = 2
    evidence_loss_weight: float = 0.4
    answerability_loss_weight: float = 0.4
    formality_loss_weight: float = 0.2
    last_n_shared_blocks: int = 2
    expected_shared_block_count: int = 36
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    include_mlp_lora: bool = False
    train_evidence_count: int = 60
    validation_evidence_count: int = 10
    locked_test_evidence_count: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.stage not in {"stage0", "stage1", "stage2"}:
            raise ValueError("stage must be one of stage0, stage1, stage2")
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.num_score_levels != 3:
            raise ValueError("Reviewer v1 requires exactly three ordered score levels")
        if self.num_ordinal_thresholds != self.num_score_levels - 1:
            raise ValueError("Reviewer v1 requires one cumulative threshold per score boundary")
        raw_weights = (
            self.evidence_loss_weight,
            self.answerability_loss_weight,
            self.formality_loss_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in raw_weights):
            raise ValueError("all score-loss weights must be finite and positive")
        if self.last_n_shared_blocks != 2:
            raise ValueError("Reviewer v1 requires the last two shared blocks")
        if self.expected_shared_block_count != 36:
            raise ValueError("Reviewer v1 pins Qwen3-VL-8B to 36 shared blocks")
        if not self.lora_target_modules:
            raise ValueError("at least one LoRA target module is required")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.lora_bias != "none":
            raise ValueError("Reviewer v1 keeps all base-model bias parameters frozen")
        counts = (
            self.train_evidence_count,
            self.validation_evidence_count,
            self.locked_test_evidence_count,
        )
        if not isinstance(counts[0], int) or counts[0] <= 0:
            raise ValueError("train evidence count must be a positive integer")
        if any(not isinstance(value, int) or value < 0 for value in counts[1:]):
            raise ValueError("validation and locked-test counts must be non-negative integers")

    @property
    def active_heads(self) -> tuple[str, ...]:
        if self.stage in {"stage0", "stage1"}:
            return ("evidence_quality",)
        return ("evidence_quality", "answerability", "qa_formality")

    @property
    def lora_enabled(self) -> bool:
        return self.stage in {"stage1", "stage2"}

    @property
    def loss_weights(self) -> dict[str, float]:
        raw = {
            "evidence_quality": float(self.evidence_loss_weight),
            "answerability": float(self.answerability_loss_weight),
            "qa_formality": float(self.formality_loss_weight),
        }
        total = sum(raw.values())
        return {name: value / total for name, value in raw.items()}

    @property
    def active_loss_weights(self) -> dict[str, float]:
        active = {name: self.loss_weights[name] for name in self.active_heads}
        total = sum(active.values())
        return {name: value / total for name, value in active.items()}

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["lora_target_modules"] = list(self.lora_target_modules)
        result["loss_weights"] = self.loss_weights
        return result
