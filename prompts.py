"""Prompts for EgoLife two-user video question-answer generation and review."""

from __future__ import annotations

import json
from typing import Any


VIDEO_GENERATION_SCHEMA = {
    "qa_id": "string",
    "question_type": "commonality, difference, or neutral",
    "question": "natural first-person or shared-memory question",
    "options": ["option A", "option B", "option C", "option D", "option E"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "required_users": ["asker user first", "evidence-provider user second"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": "specific visible fact from this user's own video",
            "timeframe": "specific start-end time range, or an approximate moment, in this user's video",
            "frames_used": ["video-level evidence or approximate moment labels"],
        }
    ],
    "referred_timestamps": [
        {
            "user": "name",
            "timestamp_seconds": 0.0,
            "moment": "brief visual moment used as evidence",
        }
    ],
    "single_user_answerability": {
        "Jake": "insufficient because the asker alone only provides ...",
        "Alice": "sufficient/insufficient because the evidence provider alone ...",
    },
    "combined_answerability": "sufficient because the required users' videos together support exactly one option",
    "generator_rationale": "why this is a natural speaker-side question with a missing visual detail",
    "why_two_users_needed": "why the asker view and evidence-provider view are both needed",
    "per_user_evidence_claims": [
        {"user": "name", "claim": "claim grounded in that user's own video"}
    ],
    "review": {
        "generator_self_check": "why the asker alone cannot answer this, and why this is not a shallow activity or shared-view question",
        "status": "draft",
    },
}


DISCOVERED_RELATION_SCHEMA = {
    "information_needs": [
        {
            "need": "natural first-person question someone in the situation might ask",
            "speaker_user": "user whose own experience anchors the question",
            "other_required_users": ["users whose views provide missing information"],
            "what_speaker_knows_sees": "visual fact available to the speaker user",
            "what_others_know_see": {
                "Alice": "visual fact available only from this user's view"
            },
            "only_clear_when_combining": "answer-relevant relation or detail that becomes clear only when the users' views are combined",
            "why_natural_to_ask": "why this would arise naturally in the situation",
            "likely_answerable_by_one_video_alone": "yes/no/uncertain, with a brief reason",
        }
    ],
    "selected_relation": {
        "need": "chosen question or information need",
        "speaker_user": "chosen speaker user",
        "other_required_users": ["chosen supporting users"],
        "what_speaker_knows_sees": "speaker-side visual anchor",
        "what_others_know_see": {
            "Alice": "missing visual detail"
        },
        "only_clear_when_combining": "combined relation to turn into a multiple-choice question",
        "why_natural_to_ask": "situated reason",
        "likely_answerable_by_one_video_alone": "no, because ...",
    },
    "selection_reason": "why this relation is more natural, and less answerable from one user's video alone, than the alternatives",
}


ANSWERABILITY_SCHEMA = {
    "choice": "A/B/C/D/E or insufficient",
    "answer_text": "selected option text, or empty string if insufficient",
    "evidence_used": "short explanation grounded only in the provided videos",
    "insufficient_reason": "explain what is missing if choice is insufficient",
}


GENERATION_MODES = ("baseline", "clip_guided", "discovery", "discovery_control")


STRICT_JSON_OUTPUT_CONTRACT = """Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.
"""


QUESTION_TYPE_GENERATION_INSTRUCTIONS = {
    "commonality": (
        "Create a commonality question only when the shared state, consequence, or follow-up "
        "becomes clear by combining a speaker-side anchor from one required user's video with "
        "a related missing detail visible only in another required user's video. Do not ask "
        "about an object, action, or room state that each single video reveals independently."
    ),
    "difference": (
        "Create a difference question whose answer identifies a meaningful contrast, "
        "asymmetry, or complementary detail between the required users' egocentric videos."
    ),
}


QUESTION_TYPE_DISCOVERY_HINTS = {
    "commonality": (
        "Prefer a relation where one user's anchor and another user's missing detail together establish "
        "a shared state, consequence, or follow-up."
    ),
    "difference": (
        "Prefer a relation where the users' views reveal a meaningful asymmetry or complementary detail."
    ),
}


QUESTION_TYPE_MULTIPLE_CHOICE_INSTRUCTIONS = {
    "commonality": (
        "Turn the relation into a question whose answer is clear only after combining the required users' views."
    ),
    "difference": (
        "Turn the relation into a question about a meaningful contrast, asymmetry, or complementary detail."
    ),
}


ANTI_ACTIVITY_QUERY_GUIDANCE = """Avoid shallow other-person activity questions:
- Do not make the main question "what was the other person doing while I was doing X?"
- Bad pattern: "When I was washing dishes, what was the other person doing?" The answer is only another person's concurrent activity.
- Bad pattern: "While I was eating, what was the other person doing on the laptop?" This still asks for a person-level activity, not the speaker's missing detail.
- Better target: "Was the stove still on after I walked away?" The speaker has a concrete uncertainty and another view resolves the object state.
- Better target: "Which mug was still on the counter after I left the table?" The answer is a specific object/location detail, not a general activity.
- Use these examples only to understand the distinction. Do not copy their wording, objects, or structure unless the videos genuinely support that exact situation.
"""


POSITIVE_EXAMPLES_GUIDANCE = """Positive examples of strong two-user information gaps:
These are taste examples, not templates. Do not copy their wording, objects, or sentence structure. Generate a new question only when the current videos genuinely support it.

- I could see the blackboard in front of me, but the writing was too small for me to read. What did it say?
  Strong because the speaker's view establishes the board and the uncertainty, while another view supplies the readable text.

- I remember us entering the market, but I missed what ended up in the cart first. Which item was added?
  Strong because the speaker's view anchors the situation, while another view reveals the specific item.

- I was focused on taping the paper craft, so where did the extra cards get placed?
  Strong because the speaker's view anchors the activity and the missing materials, while another view shows the placement.

- I can remember this side of the room, but what was hanging on the wall I did not face?
  Strong because the speaker's view establishes the room context, while another view supplies the missing wall detail.

- I left the kitchen before checking the stove again. Was anything still cooking?
  Strong because the speaker's view creates a concrete uncertainty about an object state, while another view resolves it directly.
"""


JUDGE_CHECK_SCHEMA = {
    "status": "PASS/FAIL",
    "reason": "short evidence-grounded explanation",
    "fix": "specific repair instruction if the status is FAIL; empty string if PASS",
    "quality_score": "1/2/3 using the check-specific quality rubric",
    "quality_flag": "1_weak_or_reject, 2_acceptable, or 3_strong",
    "quality_reason": "brief reason for the quality score; this does not determine pass/fail status",
}


QA_FORMALITY_QUALITY_RUBRIC = """qa_formality quality_score rubric:
- 3 / 3_strong: The JSON and multiple-choice structure are clean, the wording is natural and specific, the question preserves first-person or shared-memory perspective without direct names, and it asks for a concrete missing object, state, location, or outcome rather than a shallow activity report.
- 2 / 2_acceptable: The question-answer item is acceptable but less elegant: wording is mildly generic, stiff, or template-like; options are merely adequate; or the perspective is understandable but not especially natural. It still has no blocking structure, name, or activity-query problems.
- 1 / 1_weak_or_reject: The question-answer item has a blocking formality or structure issue, directly names a person in the question, uses dataset-observer wording, is invalid as a five-option multiple-choice question, or is mainly a "what was the other person doing" query.

Scoring instructions:
- Decide PASS/FAIL first using the qa_formality rules. Then assign quality_score using this rubric.
- The quality_score is for analysis and training signal; it must not override the pass/fail decision.
- Also return quality_flag and quality_reason.
"""


EVIDENCE_GROUNDEDNESS_QUALITY_RUBRIC = """evidence_groundedness quality_score rubric:
- 3 / 3_strong: The videos clearly demonstrate the speaker-side anchor and the evidence-provider missing detail; the answer-relevant object, action, or state is plainly visible, temporally aligned with the claims, and central enough that the relation is easy to verify.
- 2 / 2_acceptable: The answer is still supported, but the evidence is weaker: the object, action, or state is blurry, brief, partially occluded, peripheral, not the focal point, only visible in a small part of the scene, or the timestamps/claims are somewhat coarse. This can still PASS if the support is sufficient.
- 1 / 1_weak_or_reject: The visual support is missing, invented, ambiguous, answerable from the speaker alone, based on unrelated timestamp stitching, or too unclear to verify. This should normally be FAIL.

Scoring instructions:
- Decide PASS/FAIL first using the evidence_groundedness rules. Then assign quality_score using this rubric.
- The quality_score is for analysis and training signal; it must not override the pass/fail decision.
- Also return quality_flag and quality_reason.
"""


QA_FORMALITY_CHECK_SCHEMA = {
    **JUDGE_CHECK_SCHEMA,
    "semantic_subchecks": {
        "other_person_activity_query": {
            "status": "PASS/FAIL",
            "reason": (
                "whether the question merely asks what another person was doing while the speaker "
                "was doing something, instead of asking for a concrete missing detail tied to the "
                "speaker's own information need"
            ),
        },
        "direct_name_leakage": {
            "status": "PASS/FAIL",
            "reason": (
                "whether the question directly names a required user or otherwise exposes a "
                "participant name that could give away who supplies the answer"
            ),
        }
    },
}


JUDGE_SCHEMA = {
    "review_passed": True,
    "checks": {
        "qa_formality": QA_FORMALITY_CHECK_SCHEMA,
        "evidence_groundedness": JUDGE_CHECK_SCHEMA,
    },
    "blocking_failures": ["names of failed checks that should block acceptance"],
    "why_generator_asked_this": "brief explanation of why the generator may have asked this",
    "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed",
}


def judge_schema_for_check(check_name: str) -> dict[str, Any]:
    check_schema = QA_FORMALITY_CHECK_SCHEMA if check_name == "qa_formality" else JUDGE_CHECK_SCHEMA
    return {
        "review_passed": True,
        "checks": {
            check_name: check_schema,
        },
        "blocking_failures": ["names of failed checks that should block acceptance"],
        "why_generator_asked_this": "brief explanation of why the generator may have asked this",
        "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed",
    }


def temporal_pruning_brief(temporal_pruning: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only pruning facts useful to the VLM prompt."""

    if not isinstance(temporal_pruning, dict):
        return None
    return {
        "applied": True,
        "kept_duration_seconds": temporal_pruning.get("kept_duration_seconds"),
        "removed_duration_seconds": temporal_pruning.get("removed_duration_seconds"),
        "protection_target_kept_seconds": temporal_pruning.get("protection_target_kept_seconds"),
    }


def video_packet_brief(packet: dict[str, Any]) -> str:
    required_users = list(packet.get("required_users") or [])
    speaker_user = required_users[0] if required_users else None
    evidence_provider_user = required_users[1] if len(required_users) > 1 else None
    clips = []
    for clip in packet.get("clips", []):
        gaze_summary = clip.get("gaze_summary") if isinstance(clip.get("gaze_summary"), dict) else {}
        clip_brief = {
            "user": clip.get("agent_name"),
            "day": clip.get("day"),
            "clip_clock": clip.get("clip_clock"),
            "local_video": clip.get("local_video"),
            "generator_media_mode": clip.get("generator_media_mode"),
            "pruning_summary": temporal_pruning_brief(clip.get("temporal_pruning")),
            "projection_status": gaze_summary.get("projection_status"),
        }
        clips.append({key: value for key, value in clip_brief.items() if value is not None})
    return json.dumps(
        {
            "evidence_id": packet.get("evidence_id"),
            "required_users": required_users,
            "role_contract": {
                "speaker_user": speaker_user,
                "evidence_provider_user": evidence_provider_user,
                "required_users_order": (
                    "required_users[0] is the asker whose view anchors the question; "
                    "aim for a question that is not answerable from that view alone. "
                    "required_users[1] is the evidence provider whose view supplies the missing "
                    "detail. It is acceptable if the evidence provider alone can answer, as long "
                    "as that is logged."
                ),
            },
            "prompt_requirement": (
                "Use the visual media directly. Try to generate a speaker-side question that "
                "makes sense from required_users[0]'s own experience, needs both users' videos "
                "to answer cleanly, and is resolved by a concrete visible detail from "
                "required_users[1]."
            ),
            "clips": clips,
        },
        ensure_ascii=False,
        indent=2,
    )


def _frame_summary(frame: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(frame, dict):
        return {}
    return {
        "timestamp_seconds": frame.get("timestamp_seconds"),
        "path": frame.get("path"),
    }


def clip_guidance_brief(packet: dict[str, Any], *, max_rows_per_user: int = 3) -> dict[str, Any]:
    """Return compact CLIP retrieval hints for prompt injection."""

    clip_meta = packet.get("clip_exclusiveness")
    if not isinstance(clip_meta, dict):
        return {
            "available": False,
            "note": "No CLIP exclusiveness metadata is attached to this evidence packet.",
        }

    exclusive_by_user = {}
    for user, rows in (clip_meta.get("exclusive_frames_by_user") or {}).items():
        compact_rows = []
        for row in list(rows or [])[:max_rows_per_user]:
            if not isinstance(row, dict):
                continue
            if "left_frame" in row:
                own_frame = row.get("left_frame")
                closest_other = row.get("closest_right_frame")
            else:
                own_frame = row.get("right_frame")
                closest_other = row.get("closest_left_frame")
            compact_rows.append(
                {
                    "own_frame": _frame_summary(own_frame),
                    "closest_other_frame": _frame_summary(closest_other),
                    "novelty": row.get("novelty"),
                    "closest_similarity": row.get("closest_similarity"),
                }
            )
        exclusive_by_user[user] = compact_rows

    anchors = []
    for row in list(clip_meta.get("anchors") or [])[:max_rows_per_user]:
        if not isinstance(row, dict):
            continue
        anchors.append(
            {
                "similarity": row.get("similarity"),
                str(clip_meta.get("left_user") or "left_user"): _frame_summary(row.get("left_frame")),
                str(clip_meta.get("right_user") or "right_user"): _frame_summary(row.get("right_frame")),
            }
        )

    return {
        "available": True,
        "model_id": clip_meta.get("model_id"),
        "rank": clip_meta.get("rank"),
        "score": clip_meta.get("score"),
        "window": clip_meta.get("window"),
        "metrics": clip_meta.get("metrics"),
        "candidate_user_specific_moments": exclusive_by_user,
        "candidate_shared_anchors": anchors,
        "warning": (
            clip_meta.get("interpretation_warning")
            or "CLIP hints are retrieval cues only. They do not prove semantic difference or answerability."
        ),
    }


def clip_guidance_block(packet: dict[str, Any]) -> str:
    brief = clip_guidance_brief(packet)
    return f"""CLIP retrieval hints, for attention guidance only:
{json.dumps(brief, ensure_ascii=False, indent=2)}

Rules for using these hints:
- Treat CLIP as a pointer to candidate moments where the users may see different things.
- Verify every semantic claim from the raw videos before using it in the generated question-answer item.
- Do not mention CLIP, embeddings, novelty, similarity, frame paths, or retrieval scores in the question or answer options.
- It is okay to ignore a CLIP hint if the raw videos do not support a natural cross-user information need.
"""


def object_guidance_brief(packet: dict[str, Any], *, max_objects: int = 6) -> dict[str, Any]:
    """Return compact object-detection hints for prompt injection."""

    hints = packet.get("object_hints")
    if not isinstance(hints, dict) or not hints.get("available"):
        return {
            "available": False,
            "note": "No object-detection hints are attached to this evidence packet.",
        }

    rows = []
    for obj in list(hints.get("key_objects") or [])[:max_objects]:
        if not isinstance(obj, dict):
            continue
        rows.append(
            {
                "user": obj.get("user"),
                "object_name": obj.get("object_name") or obj.get("name"),
                "timestamp_seconds": obj.get("timestamp_seconds"),
                "bbox_normalized_ymin_xmin_ymax_xmax": obj.get("gemini_bbox"),
                "bbox_pixel_xyxy": obj.get("bbox"),
                "selection_score": obj.get("selection_score"),
            }
        )
    return {
        "available": bool(rows),
        "detector_model": hints.get("detector_model"),
        "detector_philosophy": hints.get("detector_philosophy"),
        "candidate_key_objects": rows,
        "warning": (
            "Object hints are VLM detections from sampled frames. They are attention anchors only; "
            "verify object identity, location, state, and cross-user relation from the raw videos."
        ),
    }


def object_guidance_block(packet: dict[str, Any]) -> str:
    brief = object_guidance_brief(packet)
    if not brief.get("available"):
        return ""
    return f"""Object-detection hints, for attention guidance only:
{json.dumps(brief, ensure_ascii=False, indent=2)}

Rules for using these hints:
- Prefer forming a natural speaker-side question related to one of these candidate key objects when the raw videos support it.
- Treat the detected object name and bounding box as a pointer, not a fact.
- Verify the object and answer from the raw videos before using it in the question, answer, options, or evidence fields.
- Do not mention object detection, bounding boxes, coordinates, detector models, sampled frames, or hint scores in the question or answer options.
- Ignore these hints if they do not support a natural two-user information need.
"""


def _feedback_block(feedback: str | None) -> str:
    return (
        "\nPrevious reviewer/evaluator feedback to address:\n"
        f"{feedback}\n"
        "Revise the new question, options, answer, and evidence to address this feedback. "
        "Do not repeat the rejected issue.\n"
        if feedback
        else ""
    )


def _numbered_lines(lines: list[str]) -> str:
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


def build_video_generation_prompt(
    packet: dict[str, Any],
    question_type: str,
    feedback: str | None = None,
    generation_mode: str = "baseline",
) -> str:
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    if generation_mode == "discovery":
        return build_relation_mcq_prompt(
            packet,
            question_type,
            discovered_relation={},
            feedback=feedback,
        )
    type_instruction = QUESTION_TYPE_GENERATION_INSTRUCTIONS.get(question_type)
    type_requirement = (
        f'The question_type must be "{question_type}": {type_instruction}'
        if type_instruction
        else ""
    )
    feedback_block = _feedback_block(feedback)
    retrieval_block = clip_guidance_block(packet) if generation_mode == "clip_guided" else ""
    object_block = object_guidance_block(packet)
    if question_type == "neutral":
        task_lines = [
            "Generate exactly one five-option multiple-choice question.",
            "Make the question a speaker-side information need: required_users[0]'s view should explain why the question naturally comes up, but should not already make the answer obvious.",
            "Try to generate a question that needs both users' videos: required_users[0] supplies the speaker-side context, and required_users[1] supplies the decisive missing visual detail.",
            "Shared timestamps, proximity, or unrelated simultaneous actions are not enough.",
            "The available evidence should make exactly one answer option correct.",
            "Fill the evidence field with each needed user's visible fact and a specific timeframe.",
            "Return every field in the JSON shape exactly. Do not omit single_user_answerability, combined_answerability, generator_rationale, why_two_users_needed, per_user_evidence_claims, referred_timestamps, or review.",
            "The answer field must exactly equal the option text indicated by correct, and correct must be one letter: A, B, C, D, or E.",
        ]
        design_block = """Design rules:
- Treat required_users[0] as the asker. The question should sound like that person trying to remember or verify something from their own experience or querying external information that they do not possess.
- Anchor the question in something visible to required_users[0]: an object they handled, a place they entered, a surface they looked toward, a task they were doing, or a state they could not fully verify.
- Ask for a related missing detail supplied by required_users[1]: object identity, location, text, count, state, placement, outcome, follow-up, or another concrete visual fact.
- Do not stitch together unrelated scenes just because they happen during the same interval.
- Do not ask what required_users[1] was doing while the speaker was doing something unless the answer is a concrete missing object, state, location, outcome, or explanation.
- Avoid fixed openings such as "After I ...", "When I ...", or "Once I ...". Use them only when that is genuinely the most natural wording for the specific situation.
- Prefer compact everyday wording over formal language.
"""
        guidelines_block = """Guidelines:
1) Ask in a natural, informal, everyday way, like someone looking back on their own experience.
2) Use first-person or shared-memory wording such as "I", "me", "my", "we", or "our".
3) Do not name the required users in the question or options. Refer to people naturally only if the visible situation requires it.
4) Make all five options multi-word, plausible, mutually exclusive, and parallel in length and style.
5) Keep distractors grounded in the same scene type. Do not make the correct option obvious by specificity, grammar, or option length.
6) When 2D gaze coordinates are available, they appear as <gaze_coordinate> values indicating the user's attended image area. You may use nearby visible objects, regions, or actions as evidence, but do not invent exact gaze-object claims if the projection is unclear.
7) single_user_answerability must have one entry for each required user. The required_users[0] entry should say whether that view is sufficient or insufficient and why; the required_users[1] entry may say "sufficient because ..." if the evidence provider alone can answer.
8) combined_answerability must explicitly say "sufficient because ..." and explain why the required users' videos together support exactly one option.
9) Before returning, run the asker-alone test. Prefer a question where required_users[0] alone is insufficient, but do not reject merely because required_users[1] alone can answer.
"""
    else:
        task_lines = [
            "Generate exactly one five-option multiple-choice question.",
            *([type_requirement] if type_requirement else []),
            "The question must use visual evidence from one required user's contextual clue and another required user's complementary detail, but it should not follow a fixed wording pattern.",
            "Try to make the question need visual evidence from at least two required users. Timestamp overlap is not enough.",
            "required_users[0] is the asker, and that user's video alone must be insufficient. required_users[1] is the evidence provider who provides supporting evidence.",
            "Fill the evidence field with each needed user's visual fact and a specific timeframe.",
            "Return every field in the JSON shape exactly. Do not omit single_user_answerability, combined_answerability, generator_rationale, why_two_users_needed, per_user_evidence_claims, referred_timestamps, or review.",
            "The answer field must exactly equal the text of options[correct], and correct must be one letter: A, B, C, D, or E.",
        ]
        design_block = """Design rules:
- Treat required_users[0] as the asker. The question must be asked in a natural way from that user's first-person perspective. Do not stitch together two unrelated scenes and form a question.
- required_users[0]'s own video alone must not reveal the correct answer.
- Treat required_users[1] as the evidence provider that adds the missing visual detail. It is acceptable if required_users[1]'s video alone can answer the question, because that user supplies the missing visual evidence.
- The question should combine a contextual clue from required_users[0]'s own view with complementary information from required_users[1]'s view.
- Do not use a fixed question template or repeat a stock opening. In particular, avoid opening with a temporal setup clause such as "After I ...", "When I ...", or "Once I ...".
- Prefer varied everyday forms: checking where something ended up, identifying which object/action mattered, clarifying what changed, asking what was still true, or resolving a small uncertainty from the speaker's perspective.
- Do not ask what another person was doing while the speaker was doing something; that is a shallow activity query, not a speaker-side information need.
- Do not ask a generic comparison of two views, rooms, or camera angles.
"""
        guidelines_block = """Guidelines:
1) Ask in a natural, informal, everyday way, like someone looking back on their memories. Use varied question forms such as where, which, what changed, what remained, what ended up happening, or which detail explains the situation.
2) Use first-person or shared-memory wording from an AR-glasses user's perspective, such as "I", "me", "my", "we", or "our".
3) Do not name users in the question or answers.
4) Keep the question specific, concrete, conversational, and visually grounded.
5) Options must be multi-word, plausible, parallel in length/style, and have exactly one correct answer.
6) When 2D gaze coordinates are available, they are provided as <gaze_coordinate> values indicating the user's attended image area. You may ask about visible objects, regions, or actions near that area.
7) single_user_answerability must be an object with one entry for each required user. The required_users[0] entry must explicitly say "insufficient because ..."; the required_users[1] entry may say "sufficient because ..." if the evidence provider alone can answer.
8) combined_answerability must explicitly say "sufficient because ..." and explain why the combined videos support the correct option.
9) Before returning, mentally run the asker-alone test. If required_users[0] alone can answer the question, rewrite it. Do not reject merely because required_users[1] alone can answer.
"""
    return f"""You are generating one natural, evidence-grounded multiple-choice question from raw egocentric videos.

{STRICT_JSON_OUTPUT_CONTRACT}

Input: raw videos from multiple people during the same time interval. They may be near each other, or in different places. Look directly at the videos and use only visual evidence, video metadata, and the provided 2D gaze coordinates when available. Do not use captions, subtitles, transcripts, or pre-written observations.

Your task:
{_numbered_lines(task_lines)}

{guidelines_block}

{ANTI_ACTIVITY_QUERY_GUIDANCE}

{design_block}

{POSITIVE_EXAMPLES_GUIDANCE}

{retrieval_block}

{object_block}

{feedback_block}
Evidence packet metadata:
{video_packet_brief(packet)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_relation_discovery_prompt(
    packet: dict[str, Any],
    question_type: str,
    feedback: str | None = None,
) -> str:
    type_hint = QUESTION_TYPE_DISCOVERY_HINTS.get(question_type)
    target_block = f'\nTarget question_type: "{question_type}". {type_hint}\n' if type_hint else ""
    return f"""You are planning one template-free EgoLife two-user multiple-choice question from raw egocentric videos.

Do not write the multiple-choice question yet. First discover possible cross-user information needs.
Use only the raw videos, metadata, and available gaze summary. Do not use captions, transcripts, or outside knowledge.
{target_block}

List 3-5 possible cross-user information needs.
For each, identify:
- what required_users[0], the asker, knows or sees
- what required_users[1], the evidence provider, knows or sees
- what is only clear when combining them
- why someone in the situation would naturally ask this
- whether required_users[0] alone could answer it

Then select exactly one relation that is natural, visually grounded, and not answerable from required_users[0]'s video alone. It is acceptable if required_users[1]'s video alone can answer.
Avoid examples, stock phrasing, and fixed templates. Think in terms of the situation, not in terms of question patterns.
Do not select a relation whose main question is just what required_users[1] was doing while required_users[0] was doing something.

{ANTI_ACTIVITY_QUERY_GUIDANCE}

{object_guidance_block(packet)}

{_feedback_block(feedback)}
Evidence packet metadata:
{video_packet_brief(packet)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(DISCOVERED_RELATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_relation_mcq_prompt(
    packet: dict[str, Any],
    question_type: str,
    discovered_relation: dict[str, Any],
    feedback: str | None = None,
) -> str:
    type_instruction = QUESTION_TYPE_MULTIPLE_CHOICE_INSTRUCTIONS.get(question_type)
    requirement_lines = [
        "Generate exactly one five-option multiple-choice question.",
        *(
            [f'The question_type must be "{question_type}". {type_instruction}']
            if type_instruction
            else []
        ),
        "required_users[0] is the asker; write the question from that user's perspective.",
        "required_users[0]'s video alone must be insufficient.",
        "required_users[1] is the evidence provider; it is acceptable if that user's video alone can answer because they supply the missing visual detail. The combined required users' videos must make exactly one option correct.",
        "Do not use words such as video, footage, recording, frame, dataset, camera, clip, caption, subtitle, timestamp, CLIP, embedding, similarity, or novelty in the question or options.",
        "Options must be multi-word, plausible, parallel in length/style, and have exactly one correct answer.",
        "Fill the evidence fields with each needed user's visual fact and a specific timeframe.",
        "Return every field in the JSON shape exactly.",
        "The answer field must exactly equal the text of options[correct], and correct must be one letter: A, B, C, D, or E.",
    ]
    return f"""You are writing one natural first-person EgoLife multiple-choice question from a discovered cross-user relation.

Use the discovered relation to write one natural first-person multiple-choice question.
You may choose the wording freely.
Do not reuse examples or phrasing.
Do not follow a fixed template.

Input: raw videos from multiple people during the same time interval. Look directly at the videos and use only visual evidence, video metadata, and the provided gaze summary when available.

Requirements:
{_numbered_lines(requirement_lines)}

{ANTI_ACTIVITY_QUERY_GUIDANCE}

{POSITIVE_EXAMPLES_GUIDANCE}

{object_guidance_block(packet)}

Discovered relation:
{json.dumps(discovered_relation, ensure_ascii=False, indent=2)}

{_feedback_block(feedback)}
Evidence packet metadata:
{video_packet_brief(packet)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_qa_formality_judge_prompt(
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    *,
    schema_errors: list[str] | None = None,
) -> str:
    schema_errors = list(schema_errors or [])
    schema_status = "PASS" if not schema_errors else "FAIL"
    return f"""You are the qa_formality judge for a two-user multiple-choice question generated using egocentric videos.

{STRICT_JSON_OUTPUT_CONTRACT}

You will see a generated question-answer item plus deterministic schema/formality results. Judge only qa_formality.

qa_formality asks whether the generated question-answer item is natural and well-formed:
- The question should sound like a natural first-person or shared-memory question from someone in the situation, not like a dataset observer.
- The question-answer item must be a valid five-option multiple-choice question: exactly five non-empty, mutually exclusive options, labeled A-E, with exactly one correct answer option; answer exactly matches options[correct].
- The question_type field, if used, must be commonality, difference, or neutral. If it is commonality or difference, it must match the wording of the question.
- Run semantic_subchecks.other_person_activity_query and semantic_subchecks.direct_name_leakage explicitly.
- FAIL this subcheck when the question is essentially "while/when I was doing something, what was the other/named person doing?" and the answer is only that person's concurrent activity.
- FAIL the other_person_activity_query subcheck for variants such as "what was the other person doing nearby?", "what were they doing on the laptop?", or "what activity were they engaged in?" unless the question asks for a concrete missing object, state, location, outcome, consequence, explanation, or follow-up.
- FAIL direct_name_leakage when the question directly names a required user or participant, because naming the person can reveal which user's view supplies the answer.
- Bad direct-name example: "When I left the counter, where did Alice put the mug?" must FAIL because the question names the evidence provider and is simply a shallow concurrent-activity question.
- Better name-free version: "After I left the counter, where did the mug end up?" may PASS if supported, because it asks for the missing object location without naming a participant.
- Bad activity example: "When I was washing dishes, what was the other person doing?" with answer "reading a book" must FAIL because it only asks for another person's activity.
- Better qa_formality examples: "Was the stove left on after I walked away?" or "Which mug was still on the counter after I left?" may PASS if supported, because each asks for a specific missing state/location/object.
- The examples show the distinction only. Do not require or reward copying their templates.
- Do not fail merely because the wording contains "while" or mentions another person. Fail only when the semantic relation is a shallow concurrent-activity query rather than a speaker-side information need.
- If semantic_subchecks.other_person_activity_query is FAIL, set checks.qa_formality.status to FAIL, include "qa_formality" in blocking_failures, and give feedback telling the generator to ask for a concrete missing object/state/location/outcome instead of another person's activity.
- If semantic_subchecks.direct_name_leakage is FAIL, set checks.qa_formality.status to FAIL, include "qa_formality" in blocking_failures, and give feedback telling the generator to remove participant names from the question.
- PASS qa_formality only if the deterministic schema branch passes, the question wording and multiple-choice structure are acceptable, and neither semantic subcheck is FAIL.

{QA_FORMALITY_QUALITY_RUBRIC}

Deterministic schema branch:
{json.dumps({"status": schema_status, "errors": schema_errors}, ensure_ascii=False, indent=2)}

Video set metadata:
{video_packet_brief(packet)}

Generated question-answer item:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(judge_schema_for_check("qa_formality"), ensure_ascii=False, indent=2)}
"""


def build_evidence_groundedness_judge_prompt(qa_item: dict[str, Any], packet: dict[str, Any]) -> str:
    return f"""You are the evidence_groundedness judge for a two-user multiple-choice question generated using egocentric videos.

{STRICT_JSON_OUTPUT_CONTRACT}

You will see the same raw egocentric videos used by the generator. Judge only evidence_groundedness.

evidence_groundedness asks whether the question-answer item is supported only by the provided videos and metadata:
- The correct answer, evidence claims, referred timestamps, and per-user claims must be grounded in concrete visible moments or supplied metadata.
- Do not use outside knowledge, captions, transcripts, filenames alone, or assumptions not visible in the videos or metadata.
- Treat required_users[0] as the asker and required_users[1] as the evidence provider.
- The asker's view should visibly support the contextual anchor described in the item, and the evidence provider's view should visibly support the claimed answer-bearing detail.
- Do not decide whether either user's video alone is sufficient to select the correct option. Single-user and combined-video sufficiency are evaluated separately by answerability.
- PASS only when the correct answer and all material evidence, timestamp, and per-user claims are clearly supported by the provided videos or metadata, the asker’s contextual anchor and the evidence provider’s answer-bearing detail form a coherent situated relation, and no claim relies on outside knowledge, invented gaze evidence, or unrelated clip stitching.
- FAIL if the question merely stitches unrelated clips by timestamp, or makes a generic comparison of views rather than a situated speaker-side memory gap plus supported missing detail.
- If 2D gaze projection is unavailable, FAIL invented exact gaze-to-object claims; visible object/action claims are still allowed when grounded in the video itself.

{EVIDENCE_GROUNDEDNESS_QUALITY_RUBRIC}

Video set metadata:
{video_packet_brief(packet)}

Generated question-answer item:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(judge_schema_for_check("evidence_groundedness"), ensure_ascii=False, indent=2)}
"""


def build_answerability_prompt(qa_item: dict[str, Any], condition: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(["A", "B", "C", "D", "E"], qa_item.get("options", []))
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
- Choose A, B, C, D, or E only if the provided videos are sufficient.
- If the condition does not contain enough evidence, set choice to "insufficient".
- Do not guess from common sense or from the answer options.
- Do not use information from users or videos outside this condition.
- Base the answer on visible evidence, supplied metadata, and available gaze information only.
- It is acceptable for the evidence-provider-only condition to answer correctly when the evidence provider's video alone contains the missing visual detail; report that choice normally.

Return exactly one valid JSON object with this exact shape:
{json.dumps(ANSWERABILITY_SCHEMA, ensure_ascii=False, indent=2)}
"""
