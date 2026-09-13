"""冻结的 text-only A-density reward 纯逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


EXPERIMENT_VERSION = "text_only_a_density_convergence_v1"
REWARD_KIND = "text_only_a_density"
REWARD_REVISION = "text_only_a_density_v1"
PROMPT_TEMPLATE = """You are taking part in a controlled optimization test.
Return exactly 16 uppercase characters using only A and B.
Use both A and B, vary their order, and output no explanation.
Trial ID: {trial_id}"""


@dataclass(frozen=True)
class DensityScore:
    n_A: int
    n_B: int
    n_valid: int
    non_ab_character_count: int
    reward: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def score_completion(completion: str) -> DensityScore:
    """直接统计原始字符串中的大写 ASCII A/B，不做任何规范化或解析。"""

    if not isinstance(completion, str):
        raise TypeError("completion 必须是 str")
    n_a = completion.count("A")
    n_b = completion.count("B")
    n_valid = n_a + n_b
    reward = (n_a - n_b) / n_valid if n_valid else -1.0
    return DensityScore(
        n_A=n_a,
        n_B=n_b,
        n_valid=n_valid,
        non_ab_character_count=len(completion) - n_valid,
        reward=float(reward),
    )


def prompt_for(trial_id: str) -> str:
    return PROMPT_TEMPLATE.format(trial_id=trial_id)
