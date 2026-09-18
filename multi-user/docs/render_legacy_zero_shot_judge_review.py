"""Render the active legacy-zero-shot judge prompts for human contract review."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.prompts import (  # noqa: E402
    ARCHIVED_COMBINED_JUDGE_SCHEMA,
    ANSWERABILITY_SUFFICIENCY_SCHEMA,
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
    judge_schema_for_check,
)

OUTPUT_PATH = Path(__file__).with_name("LEGACY_ZERO_SHOT_JUDGE_PROMPTS_REVIEW.md")

USERS = [
    "SpeakerUser",
    "ProviderOne",
    "ProviderTwo",
    "ProviderThree",
    "ProviderFour",
    "ProviderFive",
]

QA_ITEM = {
    "qa_id": "QA_PLACEHOLDER_001",
    "question_type": "neutral",
    "question": "Where was the mug placed after I handed it over?",
    "options": [
        "On the wooden desk",
        "Beside the kitchen sink",
        "Near the front door",
        "On the living-room sofa",
        "Inside a dark backpack",
    ],
    "correct": "B",
    "answer": "Beside the kitchen sink",
    "required_users": USERS,
}

PACKET = {
    "evidence_id": "EVIDENCE_PLACEHOLDER_001",
    "required_users": USERS,
    "participant_names": USERS,
    "clips": [
        {
            "user": user,
            "agent_name": user,
            "media_role": (
                "speaker_full_video" if index == 0 else "provider_full_video"
            ),
            "is_pruned": False,
        }
        for index, user in enumerate(USERS)
    ],
}

SPEAKER_ONLY_CONDITION = {
    "condition_id": "speaker_only::SpeakerUser",
    "condition_type": "speaker_only",
    "users": ["SpeakerUser"],
}

ALL_SIX_CONDITION = {
    "condition_id": "combined_all_six_users::SpeakerUser+ProviderOne+ProviderTwo+ProviderThree+ProviderFour+ProviderFive",
    "condition_type": "combined_all_six_users",
    "users": USERS,
}


ARCHIVED_FORMALITY_CONTRACT: dict[str, Any] = {
    "review_passed": True,
    "checks": {
        "qa_formality": {
            "status": "PASS/FAIL",
            "reason": "one short explanation based only on this judge's assigned scope",
            "fix": "one specific repair instruction if FAIL; empty string if PASS",
            "semantic_subchecks": {
                "first_person_perspective": {
                    "status": "PASS/FAIL",
                    "reason": "whether the question uses a natural first-person or shared-memory perspective",
                },
                "naturalness_and_clarity": {
                    "status": "PASS/FAIL",
                    "reason": "whether the question and options are natural, concrete, clear, exclusive, and parallel",
                },
                "other_person_activity_query": {
                    "status": "PASS/FAIL",
                    "reason": "whether the answer target is a prohibited concurrent activity report",
                },
                "direct_name_leakage": {
                    "status": "PASS/FAIL",
                    "reason": "whether the user-facing QA directly names a participant",
                },
                "timestamp_citation": {
                    "status": "PASS/FAIL",
                    "reason": "whether the user-facing QA cites dataset-like temporal coordinates",
                },
            },
        }
    },
    "blocking_failures": ["names of failed checks that should block acceptance"],
    "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed",
}

ARCHIVED_GROUNDING_CONTRACT: dict[str, Any] = {
    "review_passed": True,
    "checks": {
        "evidence_groundedness": {
            "status": "PASS/FAIL",
            "reason": "one short explanation based only on this judge's assigned scope",
            "fix": "one specific repair instruction if FAIL; empty string if PASS",
        }
    },
    "blocking_failures": ["names of failed checks that should block acceptance"],
    "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed",
}

ARCHIVED_SCORED_CHECK: dict[str, Any] = {
    "status": "PASS/FAIL",
    "reason": "one short explanation based only on this judge's assigned scope",
    "fix": "one specific repair instruction if FAIL; empty string if PASS",
    "quality_score": "1/2/3 using the check-specific quality rubric",
    "quality_flag": "1_weak_or_reject, 2_acceptable, or 3_strong",
    "quality_reason": "required rationale for the quality score",
    "quota_rebuttal": "required only for an exceptional score of 3 after quota exhaustion; otherwise empty",
}

BINARY_REVIEWER_TRAINING_CONTRACT: dict[str, Any] = {
    "contract_version": "verdict_token_bce_sampled_frames_v3",
    "supervision_type": "next_token_pass_fail",
    "binary_label_names": {"0": "fail", "1": "pass"},
    "assistant_prefix": '{"verdict":"',
    "head_type": "none; use the language-model vocabulary logits",
    "binary_logit": "logit(pass) - logit(fail)",
    "loss_function": "BCEWithLogits",
    "class_weighting": "balanced_inverse_frequency_from_training_split_only",
    "class_weight_reduction": "mean-one normalization independently per judge",
    "task_loss_weights": {
        "qa_formality": 0.2,
        "evidence_groundedness": 0.4,
        "answerability": 0.4,
    },
    "trainable_parameters": (
        "language attention+MLP LoRA only: q/k/v/o/gate/up/down projections, "
        "rank 8, alpha 16"
    ),
    "inference_generation": (
        "constrain only the first generated token to the selected pass/fail token, "
        "then continue the same generation through the complete JSON contract"
    ),
}

CODE_COMPUTED_MERGED_RECORD: dict[str, Any] = {
    "review_passed": "boolean computed from deterministic checks, both judge verdicts, and the answerability gate",
    "checks": {
        "qa_formality": {
            "status": "PASS/FAIL derived from verdict",
            "reason": "copied from the model failure reason, or a code-generated pass summary",
            "fix": "copied from the model failure fix, or an empty string",
        },
        "evidence_groundedness": {
            "status": "PASS/FAIL derived from verdict",
            "reason": "copied from the model failure reason, or a code-generated pass summary",
            "fix": "copied from the model failure fix, or an empty string",
        },
        "answerability": {
            "status": "PASS/FAIL derived from the asker-only/all-six gate",
            "reason": "code-generated gate reason",
            "fix": "code-generated repair instruction, or an empty string",
        },
    },
    "blocking_failures": "array computed from failed checks",
    "feedback_to_generator": "string assembled by code from failed reasons and fixes",
}

def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def prompt_block(value: str) -> str:
    return "```text\n" + value.rstrip() + "\n```"


def render() -> str:
    decision_contract = judge_schema_for_check("qa_formality")
    sections = [
        "# Legacy-zero-shot judge prompts and output contracts",
        "",
        "> Review snapshot generated from the active prompt builders. All example and placeholder values are English. This document does not contain real videos, private annotations, or deleted samples.",
        "",
        "## Scope",
        "",
        "The direct six-user path makes four judgments: one qa_formality call, one evidence_groundedness call, one speaker-only answerability call, and one all-six answerability call. Every model decision starts with the same lowercase pass/fail verdict field.",
        "",
        "## New final-judge output contract",
        "",
        "Both qa_formality and evidence_groundedness use this exact ordered shape:",
        "",
        json_block(decision_contract),
        "",
        "Valid pass example:",
        "",
        json_block({"verdict": "pass", "reason": None, "fix": None}),
        "",
        "Valid fail example:",
        "",
        json_block(
            {
                "verdict": "fail",
                "reason": "The correct answer claims a placement that is never visible in the supplied videos.",
                "fix": "Replace the answer with a visibly supported placement or provide evidence that shows the claimed placement.",
            }
        ),
        "",
        "The model no longer emits review_passed, checks, nested status, semantic_subchecks, blocking_failures, why_generator_asked_this, feedback_to_generator, or 1-3 quality fields. The merger computes status, blocking failures, overall review_passed, and retry feedback from the authoritative verdict, reason, and fix.",
        "",
        "### Code-computed merged record (not a model output contract)",
        "",
        json_block(CODE_COMPUTED_MERGED_RECORD),
        "",
        "## Direct answerability contract",
        "",
        json_block(ANSWERABILITY_SUFFICIENCY_SCHEMA),
        "",
        "For each answerability condition, verdict is pass exactly when the supplied videos are sufficient. The code gate passes only for speaker-only=fail and all-six=pass. available_evidence and missing_evidence remain in the model output for structured audit.",
        "",
        "## Judge-model training contract",
        "",
        json_block(BINARY_REVIEWER_TRAINING_CONTRACT),
        "",
        "Training supplies the fixed assistant prefix as input and applies BCE only to the next-token margin logit(pass)-logit(fail). It adds no classifier head and does not supervise archived numerical scores or later JSON fields. At inference, the first generated token is locked from those two logits and the same generation continues through the full JSON contract. Class weights are estimated from the training split independently per judge, then normalized to preserve the 0.2/0.4/0.4 task-loss scale.",
        "",
        "## Prompt 1 — qa_formality",
        "",
        prompt_block(build_qa_formality_judge_prompt(QA_ITEM, PACKET, schema_errors=[])),
        "",
        "## Prompt 2 — evidence_groundedness",
        "",
        prompt_block(build_evidence_groundedness_judge_prompt(QA_ITEM, PACKET)),
        "",
        "## Prompt 3 — answerability / asker only",
        "",
        prompt_block(build_answerability_prompt(QA_ITEM, SPEAKER_ONLY_CONDITION)),
        "",
        "## Prompt 4 — answerability / all six",
        "",
        prompt_block(build_answerability_prompt(QA_ITEM, ALL_SIX_CONDITION)),
        "",
        "## Archived previous output contracts",
        "",
        "These contracts are preserved here for audit and possible rollback. They are not the active per-judge production contract.",
        "",
        "### Previous qa_formality contract",
        "",
        json_block(ARCHIVED_FORMALITY_CONTRACT),
        "",
        "### Previous evidence_groundedness contract",
        "",
        json_block(ARCHIVED_GROUNDING_CONTRACT),
        "",
        "### Previous scored check fields",
        "",
        json_block(ARCHIVED_SCORED_CHECK),
        "",
        "### Previous one-call combined judge contract",
        "",
        json_block(ARCHIVED_COMBINED_JUDGE_SCHEMA),
        "",
        "### Previous verbose response sample",
        "",
        json_block(
            {
                "review_passed": False,
                "checks": {
                    "evidence_groundedness": {
                        "status": "FAIL",
                        "reason": "The claimed object placement is not visible.",
                        "fix": "Revise the answer to match a visible placement.",
                    }
                },
                "blocking_failures": ["evidence_groundedness"],
                "feedback_to_generator": "Revise the answer to match a visible placement.",
            }
        ),
        "",
        "Historical prompt source snapshots also remain under docs/others/historical_prompts; no dataset examples or prior generated QA files are removed by this migration. In the source worktree, the previous ordinal training contract and exact old manifest snapshots are archived under score-only-reward-model/training/grpo_v3/experiments/human_preference_reviewer/ARCHIVED_ORDINAL_REVIEWER_V1.md and its archive/ordinal_v1_manifests directory. The synchronized OneDrive copy is isolated under score-only-reward-model-new so the previous score-only-reward-model directory remains untouched.",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    OUTPUT_PATH.write_text(render(), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
