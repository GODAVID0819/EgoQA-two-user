"""Configuration contract for Reviewer v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReviewerV1Config:
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    num_labels: int = 3
    last_n_shared_blocks: int = 2
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    include_mlp_lora: bool = False
    train_evidence_count: int = 40
    validation_evidence_count: int = 10
    locked_test_evidence_count: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.num_labels != 3:
            raise ValueError("Reviewer v1 requires exactly three labels")
        if self.last_n_shared_blocks != 2:
            raise ValueError("Reviewer v1 requires the last two shared blocks")
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
        if any(not isinstance(value, int) or value <= 0 for value in counts):
            raise ValueError("split counts must be positive integers")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["lora_target_modules"] = list(self.lora_target_modules)
        return result
