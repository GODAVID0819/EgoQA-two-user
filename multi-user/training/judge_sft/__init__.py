"""Verdict-token supervised fine-tuning for the three EgoLife judges."""

from .contracts import (
    DEFAULT_TASK_WEIGHTS,
    VERDICT_ASSISTANT_PREFIX,
    JudgeTask,
    Verdict,
    parse_complete_reason_fix_output,
    parse_verdict,
    render_locked_verdict_prefix,
)
from .inference import (
    GeneratedJudgeOutput,
    VerdictDecision,
    generate_complete_judge_output,
    select_verdict_from_next_token_logits,
)

__all__ = [
    "DEFAULT_TASK_WEIGHTS",
    "VERDICT_ASSISTANT_PREFIX",
    "JudgeTask",
    "Verdict",
    "GeneratedJudgeOutput",
    "VerdictDecision",
    "generate_complete_judge_output",
    "parse_complete_reason_fix_output",
    "parse_verdict",
    "render_locked_verdict_prefix",
    "select_verdict_from_next_token_logits",
]
