"""GRPO v3 不可协商的策略契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class V3Defaults:
    policy_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    reviewer_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    framework: str = "ms-swift"
    framework_version: str = "4.2.2"
    policy_input: str = "native_video"
    train_type: str = "lora"
    torch_dtype: str = "bfloat16"
    num_generations: int = 4
    use_vllm: bool = False
    freeze_vit: bool = True
    freeze_aligner: bool = True
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 1e-5
    beta: float = 0.0
    temperature: float = 0.7
    max_length: int = 32768
    max_completion_length: int = 1024
    gradient_checkpointing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULTS = V3Defaults()

GATE3_STEPS = 20
GATE4_STEPS = 40
GATE4_TRAIN_EVIDENCE = 40
GATE4_EVAL_EVIDENCE = 10
CONVERGENCE_WINDOW_GROUPS = 5
HOLDOUT_MAX_DROP = 0.1


def validate_formal_config(config: dict[str, Any]) -> None:
    policy_input = str(config.get("policy_input", DEFAULTS.policy_input)).strip().lower()
    if policy_input != "native_video":
        raise ValueError(f"v3 正式入口禁止 sampled_frames/图片帧替代；收到 policy_input={policy_input}")
    train_type = str(config.get("train_type", DEFAULTS.train_type)).strip().lower()
    if train_type != "lora":
        raise ValueError(f"v3 默认只允许 BF16 LoRA；QLoRA 需显存证据另行批准，收到 train_type={train_type}")
    if bool(config.get("load_in_4bit", False)) or bool(config.get("quant_bits", 0)):
        raise ValueError("v3 默认禁止 QLoRA/4-bit policy")


def assert_gate_transition(*, target_gate: int, passed_gates: Iterable[int]) -> None:
    if target_gate not in {0, 1, 2, 3, 4}:
        raise ValueError(f"本地迁移只实现 Gate 0/1/2/3/4，收到 Gate {target_gate}")
    passed = set(int(item) for item in passed_gates)
    for required in range(target_gate):
        if required not in passed:
            raise ValueError(f"Gate {required} 未通过，禁止进入 Gate {target_gate}")
