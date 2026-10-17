"""Answerability Judge(ckpt-340)的prompt。

PROMPT_TEMPLATE 逐字取自训练manifest(联合条件：speaker单独不足 且 六人合并足够)。
ALL_SIX_TEMPLATE 是奖励实际使用的版本：只删去“speaker单独不足”那一条及相关行，其余逐字保留，
只判断六人合并是否足以回答。340没有用这个精确文本训练过。
来源：/scratch/hm2991/egolife_judge_sft/answerability_842_question_split/manifests_video_split/train/answerability_joint.jsonl
(674 行全部是同一模板，只有问题不同)
"""
from __future__ import annotations

PROMPT_TEMPLATE = 'You are the cross-user answerability judge for an EgoLife question.\n\nOutput contract:\n- Return exactly one valid JSON object and nothing else.\n- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.\n- The first JSON field must be `verdict`.\n\nYou are given six synchronized egocentric timelines.\n\nThe first image group is the speaker\'s own timeline.\nThe remaining five image groups are provider timelines.\n\nJudge the following joint condition:\n\nPASS if and only if BOTH requirements hold:\n1. The speaker\'s own timeline alone is insufficient to answer the question.\n2. The combined six-user timelines contain all visual evidence needed to answer the question.\n\nFAIL if either requirement does not hold.\n\nImportant:\n- Evaluate the speaker-only condition using only the first image group.\n- Evaluate the all-six condition using all six image groups together.\n- Do not assume that speaker-only must fail.\n- Do not assume that all-six must pass.\n- Do not answer the question.\n- Do not use outside knowledge.\n- Missing, ambiguous, occluded, or unsupported required evidence makes the relevant condition insufficient.\n- Identity, continuity, state changes, and temporal relations must be visually supported.\n\nGenerated question:\n{question}\n\nReturn exactly one JSON object with this shape:\n{\n  "verdict": "pass/fail",\n  "reason": "brief reason",\n  "available_evidence": [],\n  "missing_evidence": []\n}\n'


_ALL_SIX_EDITS = (
    ("Judge the following joint condition:\n\nPASS if and only if BOTH requirements hold:\n"
     "1. The speaker's own timeline alone is insufficient to answer the question.\n"
     "2. The combined six-user timelines contain all visual evidence needed to answer the question.\n\n"
     "FAIL if either requirement does not hold.",
     "Judge the following condition:\n\nPASS if and only if the combined six-user timelines contain "
     "all visual evidence needed to answer the question.\n\nFAIL otherwise."),
    ("- Evaluate the speaker-only condition using only the first image group.\n", ""),
    ("- Do not assume that speaker-only must fail.\n", ""),
)


def _derive_all_six(template: str) -> str:
    for old, new in _ALL_SIX_EDITS:
        if template.count(old) != 1:
            raise RuntimeError("SFT模板与预期不符，无法派生all-six prompt")
        template = template.replace(old, new)
    return template


ALL_SIX_TEMPLATE = _derive_all_six(PROMPT_TEMPLATE)


def _fill(template: str, question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("answerability prompt 需要非空问题")
    # 用 replace 而非 format：模板里的 JSON 大括号不是占位符。 (Use replace instead of format: the JSON braces in the template are not placeholders.)
    return template.replace("{question}", question)


def build_joint_answerability_prompt(question: str) -> str:
    return _fill(PROMPT_TEMPLATE, question)


def build_all_six_answerability_prompt(question: str) -> str:
    return _fill(ALL_SIX_TEMPLATE, question)
