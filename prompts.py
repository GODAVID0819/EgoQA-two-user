"""Prompts for two-user EgoLife QA generation and review."""

from __future__ import annotations

import json
from typing import Any


VIDEO_GENERATION_SCHEMA = {
    "qa_id": "string",
    "question_type": "commonality or difference",
    "question": "natural first-person question from an AR-glasses user's perspective; do not mention timestamps, video, footage, recordings, frames, or cameras",
    "options": ["A option", "B option", "C option", "D option", "E option"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "category": "one of: social_interaction, task_coordination, theory_of_mind, temporal_reasoning, environmental_interaction",
    "required_users": ["at least two user names"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": "visual fact from this user's own video",
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
        "Jake": "insufficient because Jake alone only provides ...",
        "Alice": "insufficient because Alice alone only provides ...",
    },
    "combined_answerability": "sufficient because the required users' videos together support exactly one option",
    "generator_rationale": "why this is a natural question and why the missing detail depends on multiple users",
    "why_two_users_needed": "why each required user provides necessary, non-redundant visual evidence",
    "per_user_evidence_claims": [
        {"user": "name", "claim": "claim grounded in that user's own video"}
    ],
    "review": {
        "generator_self_check": "why one user alone cannot answer this, and why the question is not just asking what both users saw",
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
        "only_clear_when_combining": "combined relation to turn into an MCQ",
        "why_natural_to_ask": "situated reason",
        "likely_answerable_by_one_video_alone": "no, because ...",
    },
    "selection_reason": "why this relation is more natural, and less answerable from one user's video alone, than the alternatives",
}


ANSWERABILITY_SCHEMA = {
    "choice": "A/B/C/D/E or insufficient",
    "answer_text": "selected option text, or empty string if insufficient",
    "confidence": 0.0,
    "evidence_used": "short explanation grounded only in the provided videos",
    "insufficient_reason": "explain what is missing if choice is insufficient",
}


GENERATION_MODES = ("baseline", "clip_guided", "discovery")


JUDGE_CHECK_SCHEMA = {
    "status": "PASS/FAIL/UNCERTAIN",
    "reason": "short evidence-grounded explanation",
    "fix": "specific repair instruction if the status is FAIL or UNCERTAIN; empty string if PASS",
}


JUDGE_SCHEMA = {
    "review_passed": True,
    "checks": {
        "first_person_naturalness": JUDGE_CHECK_SCHEMA,
        "agent_perspective": JUDGE_CHECK_SCHEMA,
        "source_scope": JUDGE_CHECK_SCHEMA,
        "question_type_semantics": JUDGE_CHECK_SCHEMA,
        "multi_video_necessity": JUDGE_CHECK_SCHEMA,
        "visual_grounding": JUDGE_CHECK_SCHEMA,
        "mcq_option_quality": JUDGE_CHECK_SCHEMA,
        "gaze_safety": JUDGE_CHECK_SCHEMA,
        "human_auditability": JUDGE_CHECK_SCHEMA,
    },
    "blocking_failures": ["names of failed checks that should block acceptance"],
    "why_generator_asked_this": "brief explanation of why the generator may have asked this",
    "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed",
}


def video_packet_brief(packet: dict[str, Any]) -> str:
    clips = []
    for clip in packet.get("clips", []):
        clips.append(
            {
                "user": clip.get("agent_name"),
                "day": clip.get("day"),
                "clip_clock": clip.get("clip_clock"),
                "video_url": clip.get("video_url"),
                "local_video": clip.get("local_video"),
                "gaze_summary": clip.get("gaze_summary"),
                "projection_status": clip.get("gaze_summary", {}).get("projection_status"),
            }
        )
    return json.dumps(
        {
            "evidence_id": packet.get("evidence_id"),
            "required_users": packet.get("required_users"),
            "requirement": packet.get("requirement"),
            "clips": clips,
            "source_urls": packet.get("source_urls"),
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
- Verify every semantic claim from the raw videos before using it in the QA.
- Do not mention CLIP, embeddings, novelty, similarity, frame paths, or retrieval scores in the question or answer options.
- It is okay to ignore a CLIP hint if the raw videos do not support a natural cross-user information need.
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
    type_instruction = {
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
    }[question_type]
    feedback_block = _feedback_block(feedback)
    retrieval_block = clip_guidance_block(packet) if generation_mode == "clip_guided" else ""
    example_block = """One good example of natural multi-user QA design.
This example shows the desired reasoning pattern only. Do not copy its objects, activities, answers, names, or options into the new QA item.
Do not treat it as evidence for the current videos. For the actual QA, use only the current raw videos and packet metadata.

Good example: setup check followed by missing room state
Video situation:
- One person checks a device/timer/setup near a practice or presentation room, then walks toward the stairwell.
- Another person's view still shows the front of that room, where an exercise or dance tutorial continues on the big screen.
Good question:
- "After I checked the setup and walked toward the stairwell, what was still going on at the front of the room I had just left?"
Why good:
- It starts from what the speaker experienced: checking the setup and leaving.
- The other video answers the missing follow-up state after the speaker left.
- The answer requires combining the speaker's anchor event with another user's visual evidence.

Compact design rules:
- A good question starts from one user's own anchor event and asks for a related missing detail supplied by another user's video.
- The speaker's video must not already reveal the correct answer; another user's video must add the missing visual detail.
- If either single user's video can select the correct option, discard the question and create a different one.
- Do not make a question just because clips share a timestamp.
- Do not ask what both users saw, noticed, or looked at.
- Do not ask what both users did, handled, had, shared, or were doing together.
- Do not ask "what was the other person doing nearby" if the speaker's own view can already show it.
- Do not ask a generic comparison of two views, rooms, or camera angles.
"""
    return f"""You are an assistant generating one meaningful, contextually grounded MCQ from raw egocentric videos.

Input: raw videos from multiple people during the same time interval. They may be near each other, or in different places. Look directly at the videos and use only visual evidence, video metadata, and the provided 2D gaze coordinates when available. Do not use captions, subtitles, transcripts, or pre-written observations.

Your job:
1. Generate exactly one five-option multiple-choice question.
2. The question_type must be "{question_type}": {type_instruction}
3. The question must start from one user's speaker-side anchor event, then ask about a related missing detail that is visible only in another required user's video.
4. The question must require visual evidence from at least two required users. Timestamp overlap is not enough.
5. Any single required user's video alone must be insufficient; the combined required users' videos must make exactly one option correct.
6. Fill the evidence field with each needed user's visual fact and a specific timeframe.
7. Return every field in the JSON shape exactly. Do not omit category, single_user_answerability, combined_answerability, generator_rationale, why_two_users_needed, per_user_evidence_claims, referred_timestamps, or review.
8. The answer field must exactly equal the text of options[correct], and correct must be one letter: A, B, C, D, or E.

Guidelines:
1) Ask in a natural, informal, everyday way, like someone looking back on their memories.
For example, "Where did I put my glasses when I was having lunch with Tasha and Alice?"
2) Use first-person or shared-memory wording from an AR-glasses user's perspective, such as "I", "me", "my", "we", or "our".
3) Do not name a required user in the question or the answer when the question is asked from that person's perspective.
For example, if the question is asked from Jake's perspective, Jake's name should not appear in the question or the answer.
4) Do not use words such as video, footage, recording, frame, dataset, camera, clip, caption, subtitle, or timestamp in the question or options.
5) Keep the question specific, concrete, conversational, and visually grounded.
6) Options must be multi-word, plausible, parallel in length/style, and have exactly one correct answer.
7) False options may use names such as Jake, Alice, Tasha, Lucia, Katrina, or Shure when helpful, but follow guideline 3.
8) When 2D gaze coordinates are available, they are provided as <gaze_coordinate> values indicating the user's attended image area. You may ask about visible objects, regions, or actions near that area.
9) single_user_answerability must be an object with one entry for each required user, and each entry must explicitly say "insufficient because ...".
10) combined_answerability must explicitly say "sufficient because ..." and explain why the combined videos support the correct option.
11) Before returning, mentally run the single-user test. If Jake alone, Alice alone, or any other single required user can answer the question, rewrite it.
12) Avoid these rejected patterns: "What did we both...", "What did we all...", "What did everyone...", "What did I and Alice both...", "What were we doing together...", and "What was the other person doing nearby?".
13) For commonality questions, the commonality must be the relationship between the speaker's anchor and another user's missing detail, not merely a shared object or shared action visible in both views.

{example_block}

{retrieval_block}

{feedback_block}
Evidence packet metadata:
{video_packet_brief(packet)}

Return one valid JSON object only, with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_relation_discovery_prompt(
    packet: dict[str, Any],
    question_type: str,
    feedback: str | None = None,
) -> str:
    type_hint = {
        "commonality": (
            "Prefer a relation where one user's anchor and another user's missing detail together establish "
            "a shared state, consequence, or follow-up."
        ),
        "difference": (
            "Prefer a relation where the users' views reveal a meaningful asymmetry or complementary detail."
        ),
    }[question_type]
    return f"""You are planning one template-free EgoLife two-user MCQ from raw egocentric videos.

Do not write the MCQ yet. First discover possible cross-user information needs.
Use only the raw videos, metadata, and available gaze summary. Do not use captions, transcripts, or outside knowledge.

Target question_type: "{question_type}". {type_hint}

List 3-5 possible cross-user information needs.
For each, identify:
- what the speaker user knows or sees
- what each other required user knows or sees
- what is only clear when combining them
- why someone in the situation would naturally ask this
- whether it is likely answerable by one video alone

Then select exactly one relation that is natural, visually grounded, and least likely to be answerable from one user's video alone.
Avoid examples, stock phrasing, and fixed templates. Think in terms of the situation, not in terms of question patterns.

{_feedback_block(feedback)}
Evidence packet metadata:
{video_packet_brief(packet)}

Return one valid JSON object only, with this exact shape:
{json.dumps(DISCOVERED_RELATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_relation_mcq_prompt(
    packet: dict[str, Any],
    question_type: str,
    discovered_relation: dict[str, Any],
    feedback: str | None = None,
) -> str:
    type_instruction = {
        "commonality": (
            "Turn the relation into a question whose answer is clear only after combining the required users' views."
        ),
        "difference": (
            "Turn the relation into a question about a meaningful contrast, asymmetry, or complementary detail."
        ),
    }[question_type]
    return f"""You are writing one natural first-person EgoLife MCQ from a discovered cross-user relation.

Use the discovered relation to write one natural first-person MCQ.
You may choose the wording freely.
Do not reuse examples or phrasing.
Do not follow a fixed template.

Input: raw videos from multiple people during the same time interval. Look directly at the videos and use only visual evidence, video metadata, and the provided gaze summary when available.

Requirements:
1. Generate exactly one five-option multiple-choice question.
2. The question_type must be "{question_type}". {type_instruction}
3. The question must feel like something an AR-glasses user or someone in the situation would naturally ask.
4. The answer must require visual evidence from at least two required users.
5. Any single required user's video alone must be insufficient; the combined required users' videos must make exactly one option correct.
6. Do not use words such as video, footage, recording, frame, dataset, camera, clip, caption, subtitle, timestamp, CLIP, embedding, similarity, or novelty in the question or options.
7. Options must be multi-word, plausible, parallel in length/style, and have exactly one correct answer.
8. Fill the evidence fields with each needed user's visual fact and a specific timeframe.
9. Return every field in the JSON shape exactly.
10. The answer field must exactly equal the text of options[correct], and correct must be one letter: A, B, C, D, or E.

Discovered relation:
{json.dumps(discovered_relation, ensure_ascii=False, indent=2)}

{_feedback_block(feedback)}
Evidence packet metadata:
{video_packet_brief(packet)}

Return one valid JSON object only, with this exact shape:
{json.dumps(VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_judger_prompt(qa_item: dict[str, Any], packet: dict[str, Any]) -> str:
    return f"""You are a strict reviewer for EgoLife video-first two-user MCQ generation.

You will see the same raw egocentric videos used by the generator. Judge whether the generated question is acceptable.

Return every check in the JSON schema. Judge all checks, but focus most carefully on multi_video_necessity.

Brief checks:
1. first_person_naturalness: natural first-person memory/AR-assistant wording.
2. agent_perspective: no dataset-observer wording, and no video/footage/recording/frame/camera/clip/timestamp language in the question or options.
3. source_scope: answerable from provided raw videos and metadata only.
4. question_type_semantics: commonality means shared or jointly verified; difference means meaningful contrast, asymmetry, or complementary detail.
6. visual_grounding: correct option and evidence claims are visually grounded in concrete moments.
7. mcq_option_quality: exactly five plausible options and exactly one correct answer.
8. gaze_safety: do not invent exact gaze-to-object claims when 2D gaze is unavailable.
9. human_auditability: enough user/video/time evidence exists for a human to inspect later.

Main check, 5. multi_video_necessity:
- Judge whether the QA has a situated cross-video dependency, not just two synchronized clips.
- PASS only if one required user's video provides a speaker-side anchor event and another required user's video provides a missing visual detail that is simultaneous, follow-up, or otherwise naturally related.
- PASS only if both videos are necessary: removing either user's video would make the question unanswerable or leave more than one plausible option.
- PASS only if the connection would be a plausible memory or AR-assistant question from someone involved in the situation.
- FAIL if the question merely stitches together two clips because they happen during the same time interval.
- FAIL if the activities are unrelated, such as one person discussing/checking a device while another person is washing dishes, unless the question identifies a concrete shared task or natural dependency.
- FAIL if the question asks what both users saw, both noticed, or both looked at; do not ask what both users saw or noticed because one user may not know the other user's perception.
- FAIL if the question is a generic comparison of two views, rooms, or camera angles rather than a speaker anchor plus missing visual detail.
- FAIL if the question generically asks what "the other person", "everyone else", or "others" were doing nearby, in the room, or at the same time, unless it names a concrete missing visual detail naturally tied to the speaker-side anchor.
- FAIL if a single user's video already reveals the correct answer.
- UNCERTAIN if the videos do not clearly show the anchor, the missing visual detail, or the relation between them.
- In the reason, explicitly name the speaker-side anchor, the missing visual detail, and why the second video is or is not needed.

Contrastive example for multi_video_necessity:
- PASS: One video shows the speaker checking a setup and leaving toward a stairwell; another video still shows the front of that room, where a tutorial continues. A good question asks what was still happening after the speaker left. The first video gives the anchor; the second supplies the missing follow-up detail.
- FAIL: One video shows someone discussing/checking a device setup while another shows dishwashing. If no shared task or natural dependency is visible, this is only timestamp alignment and should fail.

Use FAIL for a clear violation, UNCERTAIN when the videos do not provide enough evidence to verify the check, and PASS only when the check is satisfied.

Video set metadata:
{video_packet_brief(packet)}

Generated QA:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return one valid JSON object only, with this exact shape:
{json.dumps(JUDGE_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_answerability_prompt(qa_item: dict[str, Any], condition: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(["A", "B", "C", "D", "E"], qa_item.get("options", []))
    )
    return f"""Answer this EgoLife multiple-choice question using only the videos provided for this condition.

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

Return one valid JSON object only with this exact shape:
{json.dumps(ANSWERABILITY_SCHEMA, ensure_ascii=False, indent=2)}
"""
