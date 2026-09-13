"""Prompts for EgoLife two-user video question-answer generation and review."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .schema import extract_json_object


VIDEO_GENERATION_SCHEMA = {
    "qa_id": "string",
    "question_type": "neutral",
    "question": "natural first-person or shared-memory question",
    "options": ["option A", "option B", "option C", "option D", "option E"],
    "correct": "A/B/C/D/E",
    "answer": "exact text of the correct option",
    "required_users": ["asker user first", "evidence-provider user second"],
    "evidence": [
        {
            "user": "name",
            "needed_fact": (
                "specific directly visible fact from this user's supplied visual evidence"
            ),
            "timeframe": (
                "specific supported time range or approximate moment in this user's "
                "supplied visual evidence"
            ),
            "frames_used": [
                "supplied-visual-evidence references or approximate moment labels"
            ],
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
    "combined_answerability": (
        "sufficient because the required users' supplied visual evidence together "
        "supports exactly one option"
    ),
    "generator_rationale": (
        "why this is a natural first-person information need and how the supplied visual "
        "evidence supports the question"
    ),
    "why_two_users_needed": (
        "how the available views contribute the facts or temporal relation needed to answer, without "
        "overstating either view's individual necessity"
    ),
    "per_user_evidence_claims": [
        {
            "user": "name",
            "claim": "claim grounded in that user's supplied visual evidence",
        }
    ],
    "review": {
        "generator_self_check": "why the asker alone cannot answer this, why every first-person or shared-memory claim is supported by the asker's own visual evidence rather than only the evidence provider's video, and why the wording is natural and timestamp-free",
        "status": "draft",
    },
}


# Archived discovery mode still requests category fields. Keep a separate schema so
# restoring that mode does not conflict with the category-free production schema.
ARCHIVED_VIDEO_GENERATION_SCHEMA = {
    **VIDEO_GENERATION_SCHEMA,
    "category": "one exact category label from the archived taxonomy",
    "category_rationale": "why the category fits and what each required view contributes",
}


# Archived discovery-mode schema. Kept for reading old prompt artifacts and
# reproducing the retired ablation offline; it is not reachable from production.
DISCOVERED_RELATION_SCHEMA = {
    "information_needs": [
        {
            "category": "one exact category label from the two-user category taxonomy",
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
        "category": "one exact category label from the two-user category taxonomy",
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


ANSWERABILITY_SCHEMA_VERSION = "answerability_evidence_sufficiency_v1"


ANSWERABILITY_SCHEMA = {
    "answerable": True,
    "reason": "short evidence-sufficiency explanation grounded only in the provided condition",
    "missing_information": "empty string if answerable; otherwise the exact missing or ambiguous fact",
}


# Production exposes baseline only.
GENERATION_MODES = ("baseline",)
# Archived generation modes:
# ARCHIVED_GENERATION_MODES = ("clip_guided", "discovery", "discovery_control")


# Offline analysis taxonomy only. Production generators and judges must never
# render this catalog or request category fields from a generated QA item.
QUESTION_CATEGORY_DEFINITIONS = {
    # Equal-status reasoning families. They describe the dominant grounded relation;
    # the separate answerability fields decide what either user can answer alone.
    "object_identification": (
        "Identify a concrete object or resolve its type, contents, shape, text, color, material, "
        "or another visible attribute when the asker-side context establishes what is being asked "
        "about and the evidence-provider view supplies the clearest identifying detail."
    ),
    "object_tracking_and_location": (
        "Relate the asker-side context for an object or person to visible evidence about where it "
        "came from, who carried or handed it off, how it moved, where it was placed, or where it "
        "ended up."
    ),
    "quantity_and_comparison": (
        "Resolve a clearly visible count, set, inventory change, addition or removal, or compare "
        "corresponding objects, states, outcomes, or roles. Count only unambiguous items and do not "
        "infer hidden quantities."
    ),
    "state_change_and_verification": (
        "Determine or verify the visible state of an object, device, room, container, or task, such "
        "as on or off, open or closed, empty or full, clean or dirty, intact or damaged, completed "
        "or incomplete, or changed or unchanged."
    ),
    "task_execution_and_completion": (
        "Resolve the concrete item, method, step, result, or completion of a purposeful task such "
        "as cooking, cleaning, organizing, making, shopping, or operating equipment. Ask about the "
        "task-relevant detail, not merely what another person was doing."
    ),
    "interaction_and_response": (
        "Resolve a concrete exchange, gesture, request, handoff, greeting, response, or follow-up "
        "between people or entities. The question must concern the linked interaction or response, "
        "not an unrelated concurrent activity."
    ),
    "temporal_sequence_and_continuation": (
        "Resolve what visibly happened before, after, next, or at the end of a supported event "
        "sequence. Do not infer continuation from timestamp overlap, proximity, or similar-looking "
        "objects alone."
    ),
    "cross_view_concurrent_activity": (
        "Match a concrete event interval from either user's synchronized view to a concrete event "
        "interval in the other view, or identify which complete cross-view activity pair overlaps. "
        "The temporal relation must be answer-bearing rather than decorative, both views must be "
        "needed to establish the match, and the user-facing question must not cite a timestamp."
    ),
    "other": (
        "Choose this when none of the provided categories fits and you have a better coherent, "
        "grounded reasoning category in mind. Use imagination to create and name that category in "
        "category_rationale, but invent only the category concept, never video facts or connections."
    ),
}


QUESTION_CATEGORY_EXAMPLES = {
    "object_identification": (
        "In a pair of videos, one view establishes the shared craft-room context and the other "
        "clearly shows blue shark-shaped paper cutouts being taped to a white sheet, so the "
        "question asks what shape the blue cutouts were."
    ),
    "object_tracking_and_location": (
        "In a pair of videos, one view shows paper crafts being handled inside and the other shows "
        "the greeting cards spread across the outdoor patio table under an umbrella, so the "
        "question asks which surface the cards ended up on."
    ),
    "quantity_and_comparison": (
        "In a pair of videos, one view establishes the table activity while the other gives a clear "
        "top-down view of a metal bowl containing four eggs, so the question asks how many eggs "
        "were in the bowl."
    ),
    "state_change_and_verification": (
        "In a pair of videos, one view shows a red refrigerator from across the room and the other "
        "clearly shows its door wide open while a person looks inside, so the question asks whether "
        "the refrigerator door was open or closed."
    ),
    "task_execution_and_completion": (
        "In a pair of videos, one view shows a shopper focused on the store shelves and the other "
        "shows checkout being completed with a phone used to scan a code, so the question asks how "
        "the shopping was paid for."
    ),
    "interaction_and_response": (
        "In a pair of videos, one angle shows the group beginning a toast and the other shows a "
        "participant joining by raising a glass of orange juice, so the question asks how that "
        "person joined the toast."
    ),
    "temporal_sequence_and_continuation": (
        "In a pair of videos, one view shows the group entering the fruit section and taking a "
        "yellow basket while the other later shows several large spiky durians inside that basket, "
        "so the question asks what was added after the basket was selected."
    ),
    "cross_view_concurrent_activity": (
        "In a pair of synchronized videos, one view contains several bounded activities and the "
        "other contains several different events. A valid question either fixes one concrete event "
        "from either view and asks which event in the other view overlapped, or asks which complete "
        "cross-view pair happened at about the same time."
    ),
    "other": (
        "Anything you think does not fit the provided categories perfectly."
    ),
}


# Audit trail for the real benchmark pairs used to write the examples above. These IDs are
# intentionally not rendered into the model prompt; the model sees only the scene descriptions.
QUESTION_CATEGORY_EXAMPLE_EVIDENCE_IDS = {
    "object_identification": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY6_12530000_A2_A4_0-1",
    "object_tracking_and_location": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY6_11133000_A4_A5_0-1",
    "quantity_and_comparison": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY4_11360000_A2_A4_0-1",
    "state_change_and_verification": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY5_11460000_A2_A4_0-1",
    "task_execution_and_completion": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY5_16253000_A4_A6_0-1",
    "interaction_and_response": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY2_18360000_A2_A4_0-1",
    "temporal_sequence_and_continuation": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY1_17193000_A1_A3_0-1",
    "cross_view_concurrent_activity": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY4_18220000_A1_A3_0-1",
    "other": "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED_DAY6_20330000_A2_A6_0-1",
}


LEGACY_QUESTION_CATEGORY_MERGES = {
    "cross_view_transfer_chain": "object_tracking_and_location",
    "route_or_destination_continuation": "object_tracking_and_location",
    "object_location": "object_tracking_and_location",
    "object_movement": "object_tracking_and_location",
    "cross_view_state_transition": "state_change_and_verification",
    "device_or_object_state_verification": "state_change_and_verification",
    "cross_view_action_outcome": "task_execution_and_completion",
    "collaborative_task_dependency": "task_execution_and_completion",
    "instruction_or_request_to_execution": "task_execution_and_completion",
    "temporal_cross_view_continuation": "temporal_sequence_and_continuation",
    "distributed_event_ordering": "temporal_sequence_and_continuation",
    "cross_view_reference_resolution": "object_identification",
    "complementary_viewpoint_resolution": "object_identification",
    "cross_view_quantity_reconciliation": "quantity_and_comparison",
    "cross_user_comparison": "quantity_and_comparison",
    "entity_action_or_social_interaction": "interaction_and_response",
    "action_outcome_and_task": "task_execution_and_completion",
    "reference_and_viewpoint_resolution": "object_identification",
    "social_and_entity_interaction": "interaction_and_response",
}


def question_category_guidance() -> str:
    lines = [
        "Broad two-user reasoning categories:",
        "- Choose the one category from these broad families that best fits the given "
        "video samples.",
        "- Choose exactly one category label and copy it verbatim into the JSON category field.",
        "- In the JSON category_rationale field, explain why it fits and state the grounded "
        "contribution made by each required user's view. Do not claim that a view is individually "
        "insufficient unless the separate answerability test supports that claim.",
        "- These categories are intentionally broad. Choose the dominant reasoning family rather "
        "than inventing or searching for a narrow subtype.",
        "- All category families have equal status. Do not prefer a label because of its list "
        "position, perceived difficulty, breadth, or specificity.",
        "- Category selection does not determine answerability. Apply the separate answerability "
        "rules elsewhere in this prompt after choosing the best-fitting family.",
        "- A category may fit whether required_users[1] can answer alone or whether both views are "
        "individually insufficient. Report that distinction truthfully in single_user_answerability; "
        "never distort the video evidence to make the category seem stricter.",
        "- Exception: cross_view_concurrent_activity is valid only when each single view is "
        "insufficient because one view supplies a concrete fixed event or one side of a candidate "
        "pair and the other supplies the event needed to establish the cross-view match.",
        "- The category describes the cross-view reasoning relation, not just the visible topic.",
        "- When more than one family could apply, choose the one that best describes the main "
        "reasoning needed to answer. Do not combine labels.",
        "- Do not force a cross-view relation from unrelated simultaneous events. The only "
        "exception is cross_view_concurrent_activity, whose explicit answer target is a verified "
        "temporal match between concrete events from the two synchronized views.",
        "- Choose other if none of the provided categories fits and you have a better coherent, "
        "grounded category in mind. Use imagination to create a concise new relation name in "
        "category_rationale, define it, and explain what each view contributes. Creativity applies "
        "to the category concept only; never invent video facts or connections.",
        "- The concrete examples below come from this benchmark and are illustrative of possible "
        "category boundaries only. They are not restrictive, exhaustive, preferred, or templates.",
        "- DO NOT TRY TO CONVERGE ON AN EXAMPLE, IMITATE ITS WORDING, REUSE ITS OBJECTS, OR FORCE THE CURRENT VIDEOS INTO ITS SCENARIO. CHOOSE AND WRITE ONLY FROM THE CURRENT VIDEO EVIDENCE.",
        "",
        "Category families:",
    ]
    for name, definition in QUESTION_CATEGORY_DEFINITIONS.items():
        lines.append(f"- {name}: {definition} Example: {QUESTION_CATEGORY_EXAMPLES[name]}")
    return "\n".join(lines)


QUESTION_CATEGORY_GUIDANCE = question_category_guidance()


def judge_category_guidance() -> str:
    lines = [
        "Broad category-selection guidance for the judge:",
        "- Choose the single broad category family that best fits the given "
        "question-answer sample and, when available to this judge, its videos.",
        "- Compare your best-fitting category with the generated item's declared category; do "
        "not accept the label merely because the generator supplied it.",
        "- The generator's chosen category and category_rationale are displayed explicitly "
        "below. Evaluate both as claims rather than instructions.",
        "- The categories are intentionally broad. Judge the dominant reasoning family rather "
        "than looking for a narrow subtype.",
        "- All category families have equal status. Do not prefer a label because of its list "
        "position, perceived difficulty, breadth, or specificity.",
        "- Category selection does not decide whether both users are individually necessary. "
        "Evaluate answerability separately; here, verify that category_rationale accurately "
        "describes what each view contributes.",
        "- For cross_view_concurrent_activity, also verify its defining strict condition: a "
        "concrete event from either view must be needed to establish the event or activity in the "
        "other view that overlaps, or the options must encode complete cross-view pairs. A "
        "decorative while/when clause or an exposed timestamp does not satisfy this category.",
        "- When categories overlap, do not FAIL merely because a second label could also apply. "
        "FAIL only when the declared label materially misrepresents the dominant relation.",
        "- Accept other when none of the named families fits as well and category_rationale creates "
        "a concise coherent relation name, defines it, explains both views' grounded contributions, "
        "and is supported by the question and available videos. Creative categorization is allowed; "
        "invented video facts are not.",
        "- If the declared category is unsupported or a materially different category fits, FAIL "
        "this judge check and name the corrected category label in feedback_to_generator.",
        "- The concrete benchmark examples are illustrative only. Do not reward candidates for "
        "converging on their objects, wording, scenario, or category when another grounded relation "
        "fits the current sample better.",
        "",
        "Category families:",
    ]
    for name, definition in QUESTION_CATEGORY_DEFINITIONS.items():
        lines.append(f"- {name}: {definition} Example: {QUESTION_CATEGORY_EXAMPLES[name]}")
    return "\n".join(lines)


JUDGE_CATEGORY_GUIDANCE = judge_category_guidance()


def generator_declared_category_for_judge(qa_item: dict[str, Any]) -> str:
    """Show judges the generator's category choice without treating it as ground truth."""

    return "\n".join(
        [
            "Generator-declared category to evaluate:",
            f"- category: {qa_item.get('category', '')}",
            f"- category_rationale: {qa_item.get('category_rationale', '')}",
            "- Independently decide whether this is the best-fitting listed category.",
        ]
    )


STRICT_JSON_OUTPUT_CONTRACT = """Output contract:
- Return exactly one valid JSON object and nothing else.
- Do not include markdown, code fences, comments, explanations, or extra text outside the JSON object.
- Include every field shown in the requested JSON shape, even when a value is brief.
"""


QUESTION_TYPE_GENERATION_INSTRUCTIONS = {
    "commonality": (
        "Create a commonality question only when the shared state, consequence, or follow-up "
        "becomes clear by combining a speaker-side anchor from one required user's visual "
        "evidence with a related missing detail visible only in another required user's "
        "visual evidence. Do not ask about an object, action, or room state that each "
        "single-user evidence set reveals independently."
    ),
    "difference": (
        "Create a difference question whose answer identifies a meaningful contrast, "
        "asymmetry, or complementary detail between the required users' visual evidence."
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


# Archived concurrent-activity experiment. Production prompts must not render
# this block; it is retained only so the retired experiment can be reproduced.
ARCHIVED_CONCURRENT_ACTIVITY_GUIDANCE = """Concurrent-activity guidance:
- A concurrent question may use a concrete event from either user's synchronized view as the relative temporal key and ask which concrete event or activity in the other view occurred at the same time.
- A second valid form asks which complete pair of activities, one associated with each view, overlapped.
- The strict dependency must be real: each single-user condition lacks a required side of the temporal match, while the combined synchronized views establish exactly one answer.
- The temporal clause is invalid when it is merely decorative, the fixed event is vague, the question exposes a clock time or timecode, or the answer can be selected without cross-view temporal alignment.
- The evidence and generator_rationale must record the concrete events and their original-video intervals. Timestamp proximity or equal positions in independently pruned videos are not proof of concurrency.
- A shallow prompt such as "What was the other person doing?" still fails because it expresses no concrete temporal relation.
- Use examples only to understand the structural distinction. Never copy their activities, objects, people, or setting.
"""


RESTORED_GENERATOR_COVERAGE_GUIDANCE = """Restored design safeguards:
- Do not reveal or strongly suggest the correct answer in the question stem; place candidate answers in the options.
- Prefer casual everyday wording over formal language. Be creative in tone and wording, just like how somebody would naturally ask everyday.
"""


# Retired example retained for offline reproduction only. It is intentionally
# excluded from all production prompt builders.
ARCHIVED_CONCURRENT_ACTIVITY_EXAMPLES = """- Cross-view concurrent activity. Fix a concrete event from either user's view and ask which event in the other view happened at the same time. In each video, there may be multiple actions that the user takes, for example "walking down the stairs", "watching a video", "reaching for a mug", etc. Pick one from either user's video, and ask which event in the other user's video occured around the same time.
  Example: "What was I doing when the person with pink hair chopped the vegetables for dinner?" Here, the answer option should be the event the asker's video supported, and "the person in pink hair chopping vegetables" is the event that the evidence provider's video shows.
"""


def question_wording_direction(packet: dict[str, Any]) -> str:
    """Assign a reproducible opening style so independent calls do not collapse."""

    identity = {
        "evidence_id": packet.get("evidence_id"),
        "required_users": packet.get("required_users"),
        "clips": [
            {
                "user": clip.get("user"),
                "day": clip.get("day"),
                "clip_clock": clip.get("clip_clock"),
                "local_video": clip.get("local_video"),
                "video_url": clip.get("video_url"),
            }
            for clip in packet.get("clips") or []
            if isinstance(clip, dict)
        ],
    }
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    bucket = hashlib.sha256(serialized.encode("utf-8")).digest()[0] % 3
    if bucket:
        return """Per-item wording direction: question_first
- For this item, lead with the missing-information request and place any first-person or temporal anchor later in the sentence.
- Do not begin this item with a scene-setting clause such as "I was ...", "We were ...", "When I ...", "While I ...", or "After I ...".
- This assignment controls sentence structure only; choose the strongest grounded relation independently."""
    return """Per-item wording direction: context_first_allowed
- For this item, a concise first-person or relative-time setup may come first when it makes the question natural and clear.
- A form such as "I was ..., but ..." is allowed here; use it only when the setup identifies a necessary event, object, or uncertainty rather than serving as filler.
- Question-first wording is also acceptable if a context-first opening would be awkward. Choose the strongest grounded relation independently."""


# Archived point-scoring schema retained for offline analysis only. The production
# prompt builders below deliberately do not route to this schema, even when an old
# caller still passes pass_fail_only=False.
JUDGE_CHECK_SCHEMA = {
    "status": "PASS/FAIL",
    "reason": "one short explanation based only on this judge's assigned scope",
    "fix": "one specific repair instruction if FAIL; empty string if PASS",
    "quality_score": "1/2/3 using the check-specific quality rubric",
    "quality_flag": "1_weak_or_reject, 2_acceptable, or 3_strong",
    "quality_reason": "required rationale for this attempt's quality score; this does not determine pass/fail status",
    "quota_rebuttal": "required explicit rebuttal only when assigning 3 after the 48-assignment quota is exhausted; otherwise empty string",
}

DEFAULT_QUALITY_QUOTA = 48

PASS_FAIL_ONLY_CHECK_SCHEMA = {
    "status": "PASS/FAIL",
    "reason": "one short explanation based only on this judge's assigned scope",
    "fix": "one specific repair instruction if FAIL; empty string if PASS",
}

PASS_FAIL_ONLY_INSTRUCTION = """Binary decision contract:
- Return only the fields in the requested JSON schema.
- Do not include reasoning, markdown, or code fences outside the JSON object.
- Keep every reason and fix to one sentence and no more than 40 words.
- Do not assign a numerical score, quality label, rank, quota, or comparison against other candidates.
"""

JUDGE_OUTPUT_SCHEMA_MARKER = "Return exactly one valid JSON object with this exact shape:"

JUDGE_FIRST_VERDICT_INSTRUCTION = """Authoritative first-verdict contract:
- Apply exactly the same judge criteria and return the same detailed checks and feedback requested below.
- The first JSON field must be verdict, with exactly one lowercase value: pass or fail.
- verdict is the authoritative overall decision for this model judge. Decide it before generating checks, subchecks, reasons, fixes, blocking_failures, or feedback.
- Every later status and blocking_failures entry must be consistent with verdict, but those later fields do not override it.
- Do not emit review_passed. The lowercase verdict field replaces that boolean.
- Return exactly one valid JSON object and no markdown, analysis, or text outside it.
"""

JUDGE_MINIMAL_VERDICT_PROBE_INSTRUCTION = """Independent entropy-probe contract:
- Apply the judge criteria above silently to the provided candidate and media.
- This is a separate diagnostic judgment. You are not given the production judge's answer.
- Return exactly one valid JSON object and no markdown, analysis, or text outside it.
- The object must contain exactly one field: verdict.
- verdict must be exactly one lowercase value: pass or fail.
- Do not return review_passed, checks, subchecks, reasons, fixes, blocking_failures, feedback, scores, or any other field.
"""


QA_FORMALITY_QUALITY_RUBRIC = """qa_formality quality_score rubric:
- 3 / 3_strong: The JSON and five-option structure are clean, the question is natural and clearly first-person or shared-memory, references are unambiguous, and no participant names or timestamp citations appear.
- 2 / 2_acceptable: The item is acceptable but mildly stiff, generic, or uneven in option style. It still has no blocking schema, perspective, name, timestamp, or ambiguity problem.
- 1 / 1_weak_or_reject: The item has a blocking schema or semantic-form issue, lacks first-person perspective, directly names a participant, cites a timestamp, or is unnatural or ambiguous.

Scoring instructions:
- Decide PASS/FAIL first using the qa_formality rules. Then assign quality_score using this rubric.
- The quality_score is for analysis and training signal; it must not override the pass/fail decision.
- For every attempt, return quality_flag and a concrete quality_reason explaining why this candidate earned that score.
"""


EVIDENCE_GROUNDEDNESS_QUALITY_RUBRIC = """evidence_groundedness quality_score rubric:
- 3 / 3_strong: The videos clearly demonstrate the speaker-side anchor and the evidence-provider missing detail; the answer-relevant object, action, or state is plainly visible, temporally aligned with the claims, and central enough that the relation is easy to verify.
- 2 / 2_acceptable: The answer is still supported, but the evidence is weaker: the object, action, or state is blurry, brief, partially occluded, peripheral, not the focal point, only visible in a small part of the scene, or the timestamps/claims are somewhat coarse. This can still PASS if the support is sufficient.
- 1 / 1_weak_or_reject: The visual support is missing, invented, ambiguous, answerable from the speaker alone, based on unrelated timestamp stitching, or too unclear to verify. This should normally be FAIL.

Scoring instructions:
- Decide PASS/FAIL first using the evidence_groundedness rules. Then assign quality_score using this rubric.
- The quality_score is for analysis and training signal; it must not override the pass/fail decision.
- For every attempt, return quality_flag and a concrete quality_reason explaining why this candidate earned that score.
"""


def quality_quota_prompt(
    *,
    previous_three_point_assignments: int,
    quota: int = DEFAULT_QUALITY_QUOTA,
) -> str:
    """Render the run-global, per-judge-category quota without an item summary."""

    previous = max(0, int(previous_three_point_assignments))
    limit = max(1, int(quota))
    remaining = max(0, limit - previous)
    return f"""Global 3-point quota for this judge category:
-The prompt budget for this category is at most {limit} 3-point assignments.
-Previous 3-point assignments already observed: {previous}.
-Remaining 3-point capacity before this candidate: {remaining}.
-MAKE SURE THE TOTAL 3-POINT ASSIGNMENT COUNT IS STRICTLY BELOW THE GIVEN QUOTA; DO NOT ASSIGN 3-POINT AFTER THE QUOTA HAS BEEN REACHED.

- Apply the 1/2/3 rubric honestly to this candidate, while following the quota instruction above.
- Always explain this attempt's score in quality_reason.
- If you assign quality_score 3 when the remaining capacity is 0, quota_rebuttal is mandatory. Explicitly rebut the quota instruction and explain why this candidate still warrants 3 points despite being told not to exceed the quota.
- For every other score, return quota_rebuttal as an empty string.
- The quota and score must not alter the independent PASS/FAIL decision.
"""


QA_FORMALITY_CHECK_SCHEMA = {
    **JUDGE_CHECK_SCHEMA,
    "semantic_subchecks": {
        "first_person_perspective": {
            "status": "PASS/FAIL",
            "reason": (
                "whether the question is written as a natural first-person or shared-memory "
                "question using I, me, my, we, us, or our"
            ),
        },
        "naturalness_and_clarity": {
            "status": "PASS/FAIL",
            "reason": (
                "whether the question is conversational, concrete, unambiguous, and paired with "
                "clear, mutually exclusive, parallel options"
            ),
        },
        "direct_name_leakage": {
            "status": "PASS/FAIL",
            "reason": (
                "whether the user-facing question or options directly name a required user or "
                "another participant"
            ),
        },
        "timestamp_citation": {
            "status": "PASS/FAIL",
            "reason": (
                "whether the user-facing question and options avoid clock times, timecodes, "
                "timestamps, frame numbers, seconds-from-start, and minute-mark citations"
            ),
        },
    },
}

QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES = tuple(
    QA_FORMALITY_CHECK_SCHEMA["semantic_subchecks"]
)


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


def formality_participant_names(
    packet: dict[str, Any],
    qa_item: dict[str, Any] | None = None,
) -> list[str]:
    """Collect known participant names without exposing other packet metadata."""

    candidates = []
    for value in (
        packet.get("required_users"),
        (qa_item or {}).get("required_users"),
        packet.get("participant_names"),
    ):
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, (list, tuple, set)):
            candidates.extend(value)
    for clip in packet.get("clips") or []:
        if isinstance(clip, dict):
            candidates.append(clip.get("agent_name") or clip.get("user"))

    names = []
    seen = set()
    for candidate in candidates:
        name = str(candidate or "").strip()
        key = name.casefold()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    return names


def user_facing_participant_name_errors(
    qa_item: dict[str, Any],
    participant_names: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Return deterministic errors for known participant names in question/options."""

    fields: list[tuple[str, str]] = [("question", str(qa_item.get("question") or ""))]
    for index, option in enumerate(qa_item.get("options") or []):
        fields.append((f"options[{index}]", str(option or "")))

    errors = []
    for field_name, value in fields:
        for participant_name in participant_names or []:
            name = str(participant_name or "").strip()
            if not name:
                continue
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", value, re.IGNORECASE):
                errors.append(
                    f"{field_name} contains a prohibited participant name: {name!r}"
                )
                break
    return errors


def qa_formality_errors(
    qa_item: dict[str, Any],
    schema_errors: list[str] | None = None,
    *,
    participant_names: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Combine deterministic QA-schema and known participant-name errors."""

    errors = list(schema_errors or [])
    errors.extend(user_facing_participant_name_errors(qa_item, participant_names))
    return list(dict.fromkeys(errors))


def formality_context_brief(
    packet: dict[str, Any],
    qa_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose only participant names needed for text-only name-leakage detection."""

    return {"participant_names": formality_participant_names(packet, qa_item)}


def formality_qa_item_brief(qa_item: dict[str, Any]) -> dict[str, Any]:
    """Expose only user-facing QA fields and deterministic answer metadata."""

    return {
        "question_type": qa_item.get("question_type"),
        "question": qa_item.get("question"),
        "options": qa_item.get("options"),
        "correct": qa_item.get("correct"),
        "answer": qa_item.get("answer"),
    }


def judge_schema_for_check(
    check_name: str,
    *,
    pass_fail_only: bool = True,
) -> dict[str, Any]:
    # Production schema is unconditionally binary. Archived scored-schema routing:
    # use_scored_schema = not pass_fail_only
    # check_schema = QA_FORMALITY_CHECK_SCHEMA or JUDGE_CHECK_SCHEMA
    if check_name == "qa_formality":
        check_schema = {
            **PASS_FAIL_ONLY_CHECK_SCHEMA,
            "semantic_subchecks": QA_FORMALITY_CHECK_SCHEMA["semantic_subchecks"],
        }
    else:
        check_schema = PASS_FAIL_ONLY_CHECK_SCHEMA
    schema = {
        "review_passed": True,
        "checks": {
            check_name: check_schema,
        },
        "blocking_failures": ["names of failed checks that should block acceptance"],
        "feedback_to_generator": "specific revision instructions if review_passed is false; use an empty string if it passed",
    }
    # Archived scored-schema field:
    # schema["why_generator_asked_this"] = "brief explanation ..."
    return schema


def build_judge_first_verdict_prompt(
    review_prompt: str,
    check_name: str,
) -> str:
    """Put the authoritative lowercase production verdict before judge details."""

    if check_name not in {"qa_formality", "evidence_groundedness"}:
        raise ValueError(f"unsupported first-verdict judge: {check_name}")
    if JUDGE_OUTPUT_SCHEMA_MARKER not in review_prompt:
        raise ValueError("judge prompt does not contain the expected output-schema marker")
    rubric_prompt, schema_text = review_prompt.rsplit(JUDGE_OUTPUT_SCHEMA_MARKER, 1)
    try:
        detailed_schema = json.loads(schema_text.strip())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"judge prompt output schema is not valid JSON: {exc}") from exc
    if not isinstance(detailed_schema, dict):
        raise ValueError("judge prompt output schema must be a JSON object")
    detailed_schema.pop("review_passed", None)
    detailed_schema.pop("verdict", None)
    feedback_contract = detailed_schema.get("feedback_to_generator")
    if isinstance(feedback_contract, str):
        detailed_schema["feedback_to_generator"] = feedback_contract.replace(
            "review_passed is false",
            "verdict is fail",
        )
    first_verdict_schema = {
        "verdict": "pass/fail",
        **detailed_schema,
    }
    return f"""{rubric_prompt}

{JUDGE_FIRST_VERDICT_INSTRUCTION}

Judge decision field: {check_name}

{JUDGE_OUTPUT_SCHEMA_MARKER}
{json.dumps(first_verdict_schema, ensure_ascii=False, indent=2)}
"""


def build_judge_first_verdict_sidecar_prompt(
    review_prompt: str,
    check_name: str,
) -> str:
    """Compatibility alias for the offline sidecar experiment."""

    return build_judge_first_verdict_prompt(review_prompt, check_name)


def build_judge_minimal_verdict_probe_prompt(
    review_prompt: str,
    check_name: str,
) -> str:
    """Keep a judge's rubric and candidate, but request only a pass/fail verdict."""

    if check_name not in {"qa_formality", "evidence_groundedness"}:
        raise ValueError(f"unsupported minimal-verdict judge: {check_name}")
    if JUDGE_OUTPUT_SCHEMA_MARKER not in review_prompt:
        raise ValueError("judge prompt does not contain the expected output-schema marker")
    rubric_prompt, _ = review_prompt.rsplit(JUDGE_OUTPUT_SCHEMA_MARKER, 1)
    minimal_schema = {"verdict": "pass/fail"}
    return f"""{rubric_prompt.rstrip()}

{JUDGE_MINIMAL_VERDICT_PROBE_INSTRUCTION}

Judge decision field: {check_name}

{JUDGE_OUTPUT_SCHEMA_MARKER}
{json.dumps(minimal_schema, ensure_ascii=False, indent=2)}
"""


def _pruned_to_original_time_map(
    keep_intervals: list[list[float]] | list[tuple[float, float]] | None,
) -> list[dict[str, float]]:
    """Map concatenated pruned-video positions to the original synchronized timeline."""

    segments = []
    pruned_cursor = 0.0
    for interval in keep_intervals or []:
        if not isinstance(interval, (list, tuple)) or len(interval) < 2:
            continue
        original_start = float(interval[0])
        original_end = float(interval[1])
        if original_end <= original_start:
            continue
        pruned_start = pruned_cursor
        pruned_end = pruned_start + original_end - original_start
        segments.append(
            {
                "pruned_start_seconds": round(pruned_start, 3),
                "pruned_end_seconds": round(pruned_end, 3),
                "original_start_seconds": round(original_start, 3),
                "original_end_seconds": round(original_end, 3),
            }
        )
        pruned_cursor = pruned_end
    return segments


def temporal_pruning_brief(temporal_pruning: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only pruning facts useful to the VLM prompt."""

    if not isinstance(temporal_pruning, dict):
        return None
    brief = {
        "applied": True,
        "kept_duration_seconds": temporal_pruning.get("kept_duration_seconds"),
        "removed_duration_seconds": temporal_pruning.get("removed_duration_seconds"),
        "protection_target_kept_seconds": temporal_pruning.get("protection_target_kept_seconds"),
    }
    for key in (
        "temporal_pairing_policy",
        "max_pair_time_difference_seconds",
        "split_noncontiguous_clusters",
    ):
        if temporal_pruning.get(key) is not None:
            brief[key] = temporal_pruning[key]
    keep_intervals = temporal_pruning.get("keep_intervals")
    if isinstance(keep_intervals, list):
        time_map = _pruned_to_original_time_map(keep_intervals)
        if time_map:
            brief["pruned_to_original_time_map"] = time_map
            brief["temporal_alignment_contract"] = (
                "Map activity intervals from pruned playback time to original time before comparing "
                "the two users. The pruned videos concatenate retained intervals independently, so "
                "equal pruned playback positions do not establish a temporal relationship."
            )
    return brief


SAMPLED_FRAME_GENERATOR_MEDIA_MODES = {
    "centroid_frames_only",
    "retained_cluster_frames_only",
}


def generator_uses_sampled_frames(packet: dict[str, Any]) -> bool:
    if packet.get("generator_media_mode") in SAMPLED_FRAME_GENERATOR_MEDIA_MODES:
        return True
    return any(
        isinstance(clip, dict)
        and clip.get("generator_media_mode") in SAMPLED_FRAME_GENERATOR_MEDIA_MODES
        for clip in packet.get("clips") or []
    )


def video_packet_brief(packet: dict[str, Any]) -> str:
    required_users = list(packet.get("required_users") or [])
    speaker_user = required_users[0] if required_users else None
    evidence_provider_user = required_users[1] if len(required_users) > 1 else None
    clips = []
    sampled_media_modes: set[str] = set()
    sampled_frame_modes = SAMPLED_FRAME_GENERATOR_MEDIA_MODES

    for clip in packet.get("clips", []):
        generator_media_mode = clip.get("generator_media_mode")
        pruning_summary = (
            None
            if generator_media_mode == "retained_cluster_frames_only"
            else temporal_pruning_brief(clip.get("temporal_pruning"))
        )
        clip_brief = {
            "user": clip.get("agent_name") or clip.get("user"),
            "day": clip.get("day"),
            "clip_clock": clip.get("clip_clock"),
            "duration_seconds": clip.get("duration_seconds"),
            "segment_count": clip.get("segment_count"),
            "local_video": clip.get("local_video"),
            "generator_media_mode": generator_media_mode,
            "pruning_summary": pruning_summary,
        }

        if generator_media_mode in sampled_frame_modes:
            frame_count = sum(
                isinstance(frame, dict) for frame in clip.get("frames", [])
            )
            if frame_count:
                sampled_media_modes.add(str(generator_media_mode))
                clip_brief["generator_frame_input"] = {
                    "frame_count": frame_count,
                    "ordering": "chronological within this user",
                }

        clips.append({key: value for key, value in clip_brief.items() if value is not None})

    generator_media_contract = None
    if sampled_media_modes:
        if sampled_media_modes == {"centroid_frames_only"}:
            mode = "retained_clip_cluster_centroid_images_only"
        elif sampled_media_modes == {"retained_cluster_frames_only"}:
            mode = "retained_clip_cluster_member_images_only"
        else:
            mode = "mixed_sampled_frame_modes"

        generator_media_contract = {
            "mode": mode,
            "image_group_order": (
                "Images are grouped in required_users order and are chronological within "
                "each user's group. No per-frame indices or timestamps are included."
            ),
        }

    brief = {
        "evidence_id": packet.get("evidence_id"),
        "required_users": required_users,
        "role_contract": {
            "speaker_user": speaker_user,
            "evidence_provider_user": evidence_provider_user,
            "required_users_order": (
                "required_users[0] is the asker and the question must use that user's natural "
                "first-person or shared-memory perspective. That user's view alone should be "
                "insufficient. required_users[1] supplies additional evidence; report each "
                "user's individual answerability truthfully. The combined evidence must support "
                "exactly one answer. Every "
                "first-person factual claim must be supported specifically by required_users[0]'s "
                "visual evidence; never attribute something visible only to required_users[1] "
                "to the asker's experience."
            ),
        },
        "prompt_requirement": (
            "Use the visual media directly and write the strongest natural, grounded question "
            "supported by the current evidence. Do not cite timestamps in the user-facing "
            "question or options."
        ),
        "clips": clips,
    }
    if generator_media_contract is not None:
        brief["generator_media_contract"] = generator_media_contract
    return json.dumps(brief, ensure_ascii=False, indent=2)


def _feedback_block(
    feedback: str | None,
    *,
    previous_generation: str | None = None,
) -> str:
    if not feedback and not previous_generation:
        return ""

    rejected_item: dict[str, Any] = {}
    if previous_generation:
        try:
            parsed = extract_json_object(previous_generation)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = {}
        for field in ("question", "options", "correct", "answer"):
            value = parsed.get(field)
            if value not in (None, "", []):
                rejected_item[field] = value

    blocks = ["\nRetry context:"]
    if rejected_item:
        blocks.extend(
            [
                "<previous_rejected_item>",
                json.dumps(rejected_item, ensure_ascii=False, indent=2),
                "</previous_rejected_item>",
            ]
        )
    if feedback:
        blocks.extend(
            [
                "<exact_judge_feedback>",
                feedback,
                "</exact_judge_feedback>",
            ]
        )
    blocks.append(
        "Generate a new question, options, answer, and evidence that avoid this failure. "
        "Use the failed sample only as context; do not assume any of its claims are correct, "
        "and do not repeat the failed pattern."
    )
    return "\n".join(blocks) + "\n"


def _numbered_lines(lines: list[str]) -> str:
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


# FALLBACK SNAPSHOT: the complete previous-morning generator/judge prompt supplied
# as prompts(9).py is preserved as prompts-previous-morning-2026-07-21.py. It is
# intentionally not imported or rendered anywhere. If the active prompt underperforms,
# compare or restore its prompt builders and guidance constants from that snapshot.
def build_video_generation_prompt(
    packet: dict[str, Any],
    question_type: str,
    feedback: str | None = None,
    generation_mode: str = "baseline",
    previous_generation: str | None = None,
) -> str:
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")

    type_instruction = QUESTION_TYPE_GENERATION_INSTRUCTIONS.get(question_type)
    type_requirement = (
        f'The question_type must be "{question_type}": {type_instruction}'
        if type_instruction
        else ""
    )
    feedback_block = _feedback_block(
        feedback,
        previous_generation=previous_generation,
    )
    sampled_frame_input = generator_uses_sampled_frames(packet)
    if sampled_frame_input:
        generator_opening = """You are generating one natural, evidence-grounded multiple-choice question from egocentric visual evidence.

Input: chronologically ordered sampled frames retained from multiple people's egocentric recordings during the same synchronized interval.

These images are sparse samples rather than continuous video. Use only objects, states, actions, and relationships that are directly visible in the supplied frames and supported by the provided packet metadata.

Do not infer an action, transition, handoff, state change, temporal sequence, or event occurring between sampled frames unless both relevant states or moments are visibly supported by the supplied evidence.

Before returning the question, verify that every claim in the question, answer, options, evidence fields, and rationale is grounded in the supplied sampled frames.

Do not use captions, subtitles, transcripts, or outside knowledge."""
        relation_source = "the supplied sampled frames"
        timestamp_instruction = (
            "Original timestamps may be supplied as internal metadata for ordering and "
            "evidence bookkeeping. Do not treat timestamp proximity as proof of an unseen "
            "event or transition. Do not include participant names, clock times, timestamps, "
            "timecodes, frame numbers, seconds from the start, minute marks, filenames, or "
            "clip positions in the question or options."
        )
        evidence_timeframe_instruction = (
            "Fill the evidence field with each needed user's directly visible fact and the "
            "specific supplied-frame moment or supported timeframe."
        )
        single_user_visibility = "the supplied sampled frames"
        interval_wording = "the frame sets come from the same synchronized interval"
    else:
        generator_opening = """You are generating one natural, evidence-grounded multiple-choice question from raw egocentric videos.

Input: raw videos from multiple people during the same time interval. They may be near each other or in different places. Look directly at the videos and use only visual evidence and video metadata. Do not use captions, subtitles, transcripts, or outside knowledge."""
        relation_source = "the supplied videos"
        timestamp_instruction = (
            "The timestamp is supplied on the top-left corner of each frame to infer exact "
            "timing. However, do not include participant names, clock times, timestamps, "
            "timecodes, frame numbers, seconds from the start, minute marks, filenames, or "
            "clip positions in the question or options. Precise times belong only in internal "
            "evidence and referred_timestamps fields."
        )
        evidence_timeframe_instruction = (
            "Fill the evidence field with each needed user's visible fact and a specific "
            "original-video timeframe."
        )
        single_user_visibility = "the supplied video"
        interval_wording = "the videos share a time interval"

    task_lines = [
        "Generate exactly one five-option multiple-choice question.",
        *([type_requirement] if type_requirement else []),
        "Treat required_users[0] as the asker and write a natural first-person or shared-memory question from that user's perspective.",
        "Make the question a speaker-side information need: required_users[0]'s view should explain why the question naturally comes up, but should not already make the answer obvious.",
        "Every first-person factual claim about what the speaker saw, noticed, did, handled, or experienced must be directly supported by required_users[0]'s supplied visual evidence. Never present an action, observation, object, person, or event visible only in required_users[1]'s evidence as something the speaker saw or did. Provider-only evidence may supply the unknown or answer-bearing detail, but it must not be attributed to the asker.",
        "Hard attribution audit before returning: inspect every clause using I, me, my, we, us, or our. If the clause claims or implies that the asker saw, noticed, did, handled, possessed, visited, or experienced something supported only by required_users[1]'s evidence, discard or rewrite the question. Combined-video support does not make a provider-only fact part of the asker's experience.",
        f"Choose the strongest natural, grounded question supported by {relation_source} without following a prescribed question family or template.",
        "required_users[0]'s view alone must be insufficient. The question should not be answered by the asker on their own; it must require additional evidence from required_users[1].",
        "The available evidence must make exactly one answer option correct.",
        timestamp_instruction,
        evidence_timeframe_instruction,
        "The answer field must exactly equal the correct option's text, and the correct option must be one of A, B, C, D, or E.",
    ]

    guidelines_block = f"""Guidelines:
- Use natural, informal, everyday first-person or shared-memory wording with I, me, my, we, us, or our.
- Use we, us, or our only for an experience that is genuinely shared and supported as part of required_users[0]'s experience. Do not use shared-memory wording to reattribute provider-only evidence to the asker.
- A provider-only object, action, person, place, or event may be the unknown being asked about, but the stem must not claim or imply that the asker previously saw, handled, visited, participated in, or remembers it. The asker-side setup itself must come from required_users[0]'s evidence.
- Do not ask required_users[1] a second-person question and do not name any participant in the question or options. 
- Do not refer to locations ambiguously, for example merely saying "the other room". Always specify with more detail; Perhaps identify the room as the living room or bedroom, or find some details that distinguishes the room and refer to that, something like "the room with a blue painting on the wall".
- Use a concise appearance-and-location description when needed referring to people. For example, "the person in the white shirt standing next to the television", or "the person with pink hair and wearing a pink blouse whose next to the bed".
- Make all five options multi-word, plausible, mutually exclusive, and parallel in grammar, length, and specificity. Keep distractor options grounded in the same scene type. Do not make the correct option obvious by specificity, grammar, or option length.
- Lead with the information request and place first-person context later only when it helps identify the event or object. Vary the wording and do not default to an opening first-person setup clause.
- single_user_answerability must contain one truthful entry for each required user. Do not manufacture insufficiency to fit an intended relation. For example, do not say an item was occluded or blurry when it was clearly visible in {single_user_visibility}.
- combined_answerability must explicitly say "sufficient because ..." and explain why the available views together support exactly one option.
- Do not stitch unrelated scenes together, invent person or object continuity, or exaggerate a cross-view dependency merely because {interval_wording}.
"""

    return f"""{generator_opening}

{STRICT_JSON_OUTPUT_CONTRACT}

Your task:
{_numbered_lines(task_lines)}

{guidelines_block}

{RESTORED_GENERATOR_COVERAGE_GUIDANCE}

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
    """Archived discovery-planning prompt retained for offline reproduction."""

    target_block = f'\nTarget question_type: "{question_type}".\n' if question_type else ""
    return f"""You are planning one template-free EgoLife two-user multiple-choice question from raw egocentric videos.

Do not write the multiple-choice question yet. First discover possible cross-user information needs.
Use only the raw videos and metadata. Do not use captions, transcripts, or outside knowledge.
{target_block}

List 3-5 possible cross-user information needs.
For each, identify:
- the single best-fit category from the taxonomy below
- what required_users[0], the asker, knows or sees
- what required_users[1], the evidence provider, knows or sees
- what is only clear when combining them
- why someone in the situation would naturally ask this
- whether required_users[0] alone could answer it

Then select exactly one relation that is natural and visually grounded. required_users[0] must be unable to answer alone; required_users[1] may be sufficient or insufficient. Do not invent a dependency between the views.
Avoid examples, stock phrasing, and fixed templates. Think in terms of the situation, not in terms of question patterns.
Select a concurrent-activity relation only when a concrete event from either view is needed to establish which concrete event in the other view overlaps, or when the options encode complete cross-view pairs; otherwise it is a shallow activity query.

{QUESTION_CATEGORY_GUIDANCE}

{ARCHIVED_CONCURRENT_ACTIVITY_GUIDANCE}

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
    """Archived discovery-to-MCQ prompt retained for offline reproduction."""

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
        "required_users[1] is the evidence provider and may be able to answer alone. Report that truthfully; do not invent a dependency on required_users[0]. The combined required users' videos must make exactly one option correct.",
        "Copy selected_relation.category exactly into category and explain both users' distinct contributions in category_rationale.",
        "Do not use participant names, clock times, timecodes, timestamps, frame numbers, seconds-from-start, minute marks, or words such as video, footage, recording, frame, dataset, camera, clip, caption, subtitle, CLIP, embedding, similarity, or novelty in the question or options.",
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

Input: raw videos from multiple people during the same time interval. Look directly at the videos and use only visual evidence and video metadata.

Requirements:
{_numbered_lines(requirement_lines)}

{ARCHIVED_CONCURRENT_ACTIVITY_GUIDANCE}

{QUESTION_CATEGORY_GUIDANCE}

{ARCHIVED_CONCURRENT_ACTIVITY_EXAMPLES}

Discovered relation:
{json.dumps(discovered_relation, ensure_ascii=False, indent=2)}

{_feedback_block(feedback)}
Evidence packet metadata:
{video_packet_brief(packet)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(ARCHIVED_VIDEO_GENERATION_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_qa_formality_judge_prompt(
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    *,
    schema_errors: list[str] | None = None,
    pass_fail_only: bool = True,
    previous_three_point_assignments: int = 0,
    quality_quota: int = DEFAULT_QUALITY_QUOTA,
) -> str:
    participant_names = formality_participant_names(packet, qa_item)
    schema_errors = qa_formality_errors(
        qa_item,
        schema_errors,
        participant_names=participant_names,
    )
    schema_status = "PASS" if not schema_errors else "FAIL"
    binary_block = PASS_FAIL_ONLY_INSTRUCTION

    return f"""You are the qa_formality judge for a two-user multiple-choice question. You are a pure text-only semantic judge and do not see the videos.

{STRICT_JSON_OUTPUT_CONTRACT}

Judge only the deterministic schema result and the user-facing question and options. Do not use hidden generator intent to rescue unclear wording.

Run every semantic subcheck explicitly:

1. first_person_perspective
- PASS only when the question sounds like a natural first-person or shared-memory question from someone in the situation and uses I, me, my, we, us, or our.
- The options do not need first-person pronouns.
- FAIL third-person wording or questions with no asker perspective.

2. naturalness_and_clarity
- PASS when the question is conversational, concrete, grammatical, and unambiguous, and the five options answer the same question in mutually exclusive, reasonably parallel forms.
- FAIL vague references, incompatible option types, dataset language such as video/clip/frame/camera/evidence provider, or wording that would be unnatural for someone recalling their experience.
- Judge semantic form only, not whether the described facts are true.

3. direct_name_leakage
- FAIL when the question or any option directly names a required user or another participant. PASS otherwise.
- Natural descriptive references such as "the person in the dark jacket beside the television" are allowed.
- Required-user names below are provided only for this text comparison.

4. timestamp_citation
- FAIL when the question or any option cites a clock time, timestamp, timecode, frame number, seconds-from-start, minute mark, or similar dataset-like temporal coordinate.
- Examples that FAIL include "around 12:53", "at 00:42", "at timestamp 35.2", "during the first 15 seconds", and "near frame 200".
- Natural relative wording such as while, when, before, after, later, at the same time, or a few minutes later is allowed.
- Internal evidence timeframes are outside this judge's scope and are not shown.

5. ambiguous_reference
- FAIL when the question contains ambiguous references to people, places or other visual details.
- FAIL when the question is asked in a second-person perspective, for example "what were you doing".
- Examples that FAIL include "the other room", "the other person", "the cup", etc.

Deterministic structure rules:
- The deterministic schema branch must PASS.
- The item must contain exactly five non-empty options in A-E order, one correct letter, and an answer that exactly matches the selected option. The option strings themselves do not need A./B./C./D./E. prefixes.

Decision rules:
- If any semantic subcheck is FAIL, set checks.qa_formality.status to FAIL, include "qa_formality" in blocking_failures, and provide one specific semantic repair.
- PASS qa_formality only when the deterministic schema branch passes and every semantic subcheck passes.
- Keep each reason and fix concise.

{binary_block}

Deterministic schema/formality branch:
{json.dumps({"status": schema_status, "errors": schema_errors}, ensure_ascii=False, indent=2)}

Known participant names for leakage detection only:
{json.dumps(formality_context_brief(packet, qa_item), ensure_ascii=False, indent=2)}

User-facing question-answer item:
{json.dumps(formality_qa_item_brief(qa_item), ensure_ascii=False, indent=2)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(judge_schema_for_check("qa_formality", pass_fail_only=True), ensure_ascii=False, indent=2)}
"""


def build_evidence_groundedness_judge_prompt(
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    *,
    pass_fail_only: bool = True,
    previous_three_point_assignments: int = 0,
    quality_quota: int = DEFAULT_QUALITY_QUOTA,
) -> str:
    rationale_rule = (
        "- Use generator_rationale only to understand the intended relation; treat every claim in it as unverified until confirmed against the full original videos."
        if "generator_rationale" in qa_item
        else "- Infer no hidden generator interpretation; judge the question, declared answer, material option claims, and videos shown."
    )
    binary_block = PASS_FAIL_ONLY_INSTRUCTION

    return f"""You are the evidence_groundedness judge for a two-user multiple-choice question generated from egocentric videos.

{STRICT_JSON_OUTPUT_CONTRACT}

You will see the full original videos for this evidence packet, which may be fuller than the sampled visual media shown to the generator. Judge only visual and temporal grounding. Do not fail for names, missing first-person wording, awkward phrasing, timestamp citations, or schema style. Do not decide whether a single-user condition is sufficient.

evidence_groundedness asks whether the material claims and declared answer are supported by the videos and metadata:
{rationale_rule}
- Verify every material factual claim in the question stem and declared correct answer against concrete visible moments or supplied metadata.
- Be very strict and verify every claim made in the question.
- Incorrect distractors do not need to occur in the videos for an ordinary object, state, action, or location MCQ; they must simply not make the declared answer ambiguous.
- For a comparison whose options make concrete claims about both operands, verify the declared complete relation and ensure no alternative option is also supported.
- Treat every object, action, person, state, identity, and continuity description as unverified. The generator may hallucinate or misidentify them.
- Verify every first-person factual claim specifically against required_users[0]'s video. If a claimed speaker observation, action, handled object, or experience is visible only in required_users[1]'s video, FAIL the item even when the claim is grounded somewhere in the combined video set. Treat we, us, and our the same way unless the claimed experience is genuinely shared and supported as part of required_users[0]'s experience.
- Do not rescue an attribution error because the provider-only fact is visible and the declared answer is otherwise correct. Combined-video grounding is insufficient: the asker-side claim itself must be visible in required_users[0]'s video, or the item is FAIL.
- When the generator received sampled still frames, do not accept a claimed transition, continuous action, or intermediate event merely because it seems plausible between adjacent images; verify it directly in the full original videos.
- Do not use outside knowledge, captions, transcripts, filenames alone, or assumptions not visible in the videos or metadata.
- Treat required_users[0] as the asker and required_users[1] as the evidence provider. Verify every asker-side and provider-side claim against the corresponding user's visual evidence.
- For identity or role linkage, verify enough visible continuity or distinguishing evidence to establish same-person versus different-person rather than inferring identity from roles, timing, or option wording.
- For a post-handoff follow-up, verify the initial exchange, same recipient, same object, and claimed later action/location/state. FAIL links based only on lookalikes, similar objects, or temporal proximity.
- For state verification, verify the exact object and observed state. Accept a claimed change only when both earlier and later states are visible.
- For any temporal claim, verify the claimed events and their relation on the original synchronized timeline. Do not compare equal playback positions in independently pruned videos; use original-video time or supplied pruned-to-original maps.
- FAIL a temporal relation when it is false, vague, or inferred only from timestamp proximity instead of verified synchronized intervals.
- PASS only when the question stem and declared correct answer are clearly supported and exactly one option remains correct.

{binary_block}

Video set metadata:
{video_packet_brief(packet)}

Generated question-answer item:
{json.dumps(qa_item, ensure_ascii=False, indent=2)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(judge_schema_for_check("evidence_groundedness", pass_fail_only=True), ensure_ascii=False, indent=2)}
"""


def build_answerability_prompt(qa_item: dict[str, Any], condition: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(["A", "B", "C", "D", "E"], qa_item.get("options", []))
    )
    return f"""You are the answerability judge for an EgoLife multiple-choice question.

Decide, using only the videos provided for this condition, whether the evidence is sufficient for a competent viewer to determine one uniquely supported answer. Evaluate evidence sufficiency; do not answer the question and do not report an option letter or option text.

{STRICT_JSON_OUTPUT_CONTRACT}

Condition:
{json.dumps(condition, ensure_ascii=False, indent=2)}

Question:
{qa_item.get("question")}

Options:
{options}

Rules:
- Set answerable to true only when the provided evidence contains the question-relevant facts needed to distinguish one option from the alternatives without guessing.
- Set answerable to false when a required fact or cross-view link is absent, visually unclear, only assumed, or when multiple options remain plausible. State the exact missing or ambiguous information.
- Judge what the evidence can establish, not whether a forced guess might happen to match a hidden answer key. You are not graded on selecting the dataset's declared answer, and you must not return a selection.
- Base the verdict only on the provided condition, not common-sense priors, omitted videos, filenames, or clues from option wording.
- Use only visible evidence and supplied metadata from this condition.
- Treat each condition independently. Do not use or remember evidence from another condition.
- When both users' videos are provided (or a condition contains more than two users), relevant facts may be combined across them and need not coexist in one view.
- The options define the possible answers but are not evidence. Do not infer that a more detailed, natural, or likely-sounding option is supported.
- The reason must identify the concrete visible support when answerable, or the concrete evidence gap when unanswerable.
- missing_information must be an empty string when answerable is true, and a concise description of the unresolved fact when answerable is false.

Return exactly one valid JSON object with this exact shape:
{json.dumps(ANSWERABILITY_SCHEMA, ensure_ascii=False, indent=2)}
"""


def build_judge_json_repair_prompt(raw_response: str, expected_schema: dict[str, Any]) -> str:
    """Build a one-shot formatting repair prompt without asking the judge to reconsider."""

    return f"""Your previous judge response was not valid JSON. Preserve the same decision and content, but return only one valid JSON object matching the schema below. Do not add markdown, code fences, analysis, or new reasoning. Keep every reason and fix to one sentence and no more than 40 words.

Previous response:
{raw_response}

Required schema:
{json.dumps(expected_schema, ensure_ascii=False, indent=2)}
"""
