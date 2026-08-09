"""Configuration contract for Reviewer v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReviewerV1Config:
    stage: str = "stage2"
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    num_labels: int = 3
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
        if self.num_labels != 3:
            raise ValueError("Reviewer v1 requires exactly three labels")
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
        if any(
            not isinstance(value, int) or value <= 0
            for value in (self.train_evidence_count, self.validation_evidence_count)
        ):
            raise ValueError("train and validation split counts must be positive integers")
        if not isinstance(self.locked_test_evidence_count, int) or self.locked_test_evidence_count < 0:
            raise ValueError("locked test count must be a non-negative integer")

    @property
    def active_heads(self) -> tuple[str, ...]:
        if self.stage in {"stage0", "stage1"}:
            return ("evidence_quality",)
        return ("evidence_quality", "answerability", "qa_formality")

    @property
    def lora_enabled(self) -> bool:
        return self.stage in {"stage1", "stage2"}

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["lora_target_modules"] = list(self.lora_target_modules)
        return result
