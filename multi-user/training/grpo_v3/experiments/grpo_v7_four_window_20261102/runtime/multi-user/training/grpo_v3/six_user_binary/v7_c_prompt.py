"""从用户附件静态抽取；不导入或执行附件模块。"""

from __future__ import annotations

import json

from typing import Any

ANSWERABILITY_SCHEMA = {
    "choice": "exactly one of A/B/C/D/E",
    "answer_text": "selected option text",
    "evidence_used": "short explanation grounded only in the provided videos",
}

ANSWERABILITY_SUFFICIENCY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "reason",
        "available_evidence",
        "missing_evidence",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reason": {"type": "string", "minLength": 1},
        "available_evidence": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "missing_evidence": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}

STRICT_JSON_OUTPUT_CONTRACT = """Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.
"""

def build_answerability_prompt(qa_item: dict[str, Any], condition: dict[str, Any]) -> str:
    six_user_mode = len(qa_item.get("required_users") or []) == 6
    condition_type = str(condition.get("condition_type") or "")
    if six_user_mode:
        condition_brief = {
            key: condition.get(key)
            for key in ("condition_id", "condition_type", "users")
            if key in condition
        }
        if condition_type == "speaker_only":
            media_rules = (
                "- This condition contains only the full unpruned sampled speaker timeline. Evaluate "
                "only what that timeline visibly establishes; do not assume facts from omitted "
                "provider views."
            )
        elif condition_type == "combined_all_six_users":
            media_rules = (
                "- This condition contains the full unpruned sampled speaker timeline and all "
                "five full unpruned sampled provider timelines. Combine visible evidence across "
                "them when needed. "
                "Some provider views may be irrelevant."
            )
        else:
            media_rules = (
                "- Evaluate only the videos explicitly listed in this condition. Do not assume "
                "facts from omitted views."
            )
        return f"""You are an evidence-sufficiency judge for an EgoLife question.

{STRICT_JSON_OUTPUT_CONTRACT}

Determine whether the media supplied for this condition directly contains all visual facts needed to answer the question. Do not answer the question or state what the answer is.

Set verdict to pass only when every required subject, object, action, attribute, location, identity or continuity link, state, and temporal relation is visible and sufficiently clear. Set verdict to fail when any required fact is absent, occluded, ambiguous, contradictory, or requires guessing, outside knowledge, or omitted media.

Rules:
- The first JSON field must be `verdict`; decide it before generating the later explanation and evidence lists.
- Judge this condition independently. Do not assume speaker_only is insufficient or combined_all_six_users is sufficient.
- Use only the question to identify required facts; the question itself is not visual evidence.
- Do not state, infer, or reveal the answer.
- Describe evidence using short, answer-neutral fact descriptions.
- Identity and continuity require visible continuity or distinguishing evidence, not roles, timing, lookalikes, similar clothing, or similar objects.
- State changes require the same object or place and both visible states. A visible difference does not prove an unseen cause or intervention.
- Handoffs and follow-ups require visible evidence of the exchange, the same recipient, the same object, and the claimed later action, location, or state.
- Temporal relations require synchronized timing or supplied mappings, not equal playback positions or timestamp proximity.
- "Last" or "most recent" requires checking all qualifying covered events before the reference event.
- `verdict` must be exactly the lowercase string `pass` or `fail`. `reason` must identify the decisive visible support for pass or the first decisive missing, occluded, ambiguous, or contradictory fact for fail. `available_evidence` and `missing_evidence` must be JSON arrays of short strings.
{media_rules}

Condition:
{json.dumps(condition_brief, ensure_ascii=False, indent=2)}

Generated question:
{qa_item.get("question")}

Return exactly one JSON object that conforms to the JSON Schema below. Return a data instance, not the schema itself:
{json.dumps(ANSWERABILITY_SUFFICIENCY_SCHEMA, ensure_ascii=False, indent=2)}
"""
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(["A", "B", "C", "D", "E"], qa_item.get("options", []))
    )
    dependency_rules = (
        "- When both users' videos are provided, answer-bearing facts may be split across "
        "them and need not coexist in either single view.\n"
        "- It is acceptable for the evidence-provider-only condition to answer an "
        "ordinary missing-detail question when that video independently establishes "
        "the requested detail."
    )
    return f"""Answer this EgoLife multiple-choice question using only the videos provided for this condition.

{STRICT_JSON_OUTPUT_CONTRACT}

Condition:
{json.dumps(condition, ensure_ascii=False, indent=2)}

Question:
{qa_item.get("question")}

Options:
{options}

Rules:
- You must choose exactly one answer: A, B, C, D, or E.
- Never return "insufficient", "unknown", a refusal, multiple choices, or an empty choice. If the evidence is incomplete or ambiguous, make the best forced choice from the provided condition.
- Base the forced choice on the provided condition, not common-sense priors, omitted videos, or clues from how the answer options are written.
- Use only visible evidence and supplied metadata from this condition.
{dependency_rules}

Return exactly one valid JSON object with this exact shape:
{json.dumps(ANSWERABILITY_SCHEMA, ensure_ascii=False, indent=2)}
"""

def build_c_prompt(question, users):
    if not isinstance(question,str) or not question.strip() or len(users)!=6 or len(set(users))!=6:
        raise ValueError('C 必须有题干和六个有序真实用户')
    return build_answerability_prompt(
        {'question':question,'required_users':list(users)},
        {'condition_id':'combined_all_six_users::'+'+'.join(users),
         'condition_type':'combined_all_six_users','users':list(users)})
