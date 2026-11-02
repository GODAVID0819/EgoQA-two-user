"""从附件静态抽取的生成器函数和已审查依赖，仅用于生成输入。"""

from __future__ import annotations

import json

from typing import Any

def extract_json_object(text):
    from training.judge_sft.prepare_real_data import _repo_module
    return _repo_module("schema").extract_json_object(text)

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
        "generator_self_check": "why the asker alone cannot answer this, why every first-person or shared-memory claim is supported by the asker's own visual evidence rather than only the evidence provider's video, why the wording is natural and timestamp-free, and why the answer target is not another person's concurrent activity",
        "status": "draft",
    },
}

GENERATION_MODES = ("baseline",)

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

ANTI_ACTIVITY_QUERY_GUIDANCE = """Concurrent-activity restriction:
- Do not generate a question whose answer is what one person was doing while or when another person was doing something else.
- This restriction applies even when the anchor event is concrete, the supplied evidence comes from synchronized recordings, or the temporal overlap can be verified.
- Do not generate reverse-direction variants that fix an evidence-provider event and ask what the speaker was doing.
- Do not generate options that encode pairs of concurrent activities.
- Instead, ask for a concrete missing object, identity, state, location, placement, outcome, consequence, explanation, interaction result, or follow-up that reflects a natural speaker-side information need.
"""

RESTORED_GENERATOR_COVERAGE_GUIDANCE = """Restored design safeguards:
- Do not reveal or strongly suggest the correct answer in the question stem; place candidate answers in the options.
- Prefer casual everyday wording over formal language. Be creative in tone and wording, just like how somebody would naturally ask everyday.
- Shared clock time or proximity is not a relation by itself. Do not generate a question whose answer is another person's concurrent activity. Never expose time values in the question or options.
"""

OPTIONAL_LONG_HORIZON_GUIDANCE = """Optional long-horizon selections:
The supplied evidence may cover up to 10 minutes per user. Inspect the full available span and look for useful relations whose necessary evidence is separated by tens of seconds or several minutes, rather than defaulting to one short local event.

The five patterns below are optional, equal-status selections, not required categories or output labels. Use one as the dominant question structure only when every necessary observation is visibly supported; use none when the evidence does not support one. A natural question may overlap more than one pattern, but do not pile patterns together merely to make it seem difficult.

For any selected long-horizon pattern:
- The speaker's supplied evidence alone must remain insufficient, and at least one provider view or provider combination must supply necessary answer-bearing evidence. Not every provider needs to contribute.
- Prefer multiple distinct evidence intervals and wider temporal separation, but never trade away visual clarity, natural wording, or a unique answer merely to increase the time span.
- Do not invent an intermediate event, object identity, causal explanation, state transition, or temporal link to fit a pattern. Sparse sampled frames support only the moments and states they directly show, not unseen activity between them.
- Do not cite clock times, timestamps, timecodes, frame numbers, seconds from the start, or minute marks in the question or options. The model answering the question should locate the relevant moments from the visual evidence.

1. Object trajectory
- Track one persistent physical object across separated observations and multiple users, such as origin -> interaction or handoff -> relocation -> later observation.
- Ask where it came from, where it ended up, or how it moved between people or locations. Require separated observations from the speaker and at least one provider, or from multiple provider views when the speaker supplies the natural information need.
- Use this only when distinguishing visual features, continuous handling, or another visible link supports that it is the same object. Similar-looking objects are not enough.
- Example structure: "Where did the mug I later picked up originally come from?"

2. Cross-user before/after state
- Compare visibly different earlier and later states of the same object, location, or environment, with different users contributing necessary observations or a visible intervening change.
- Ask what changed, appeared, disappeared, or was relocated only when the before/after identity and both states are clear.
- Do not infer how a change happened merely because two visible states differ. If the question asks who or what caused the change, that intervening action must also be visible.
- Example structure: "Compared with how I left the kitchen counter, what was missing when it was used later?"

3. Same-user revisit with a cross-user intervention
- Use the structure: the speaker observes state 1 -> leaves -> a provider visibly interacts with or changes the scene -> the speaker returns to state 2.
- The revisit alone is insufficient. Make the provider's intervening evidence necessary to answer, especially when the speaker's before and after views already reveal the surface difference.
- Skip this selection when the visits are effectively identical or the speaker's evidence alone answers the question.
- Example structure: "Where did the person in the blue shirt put the item I had left beside the stove before I returned?"

4. Last-seen or most-recent interaction
- Search backward across separated candidate encounters to identify who last handled an object, where it was last seen, what was most recently done with it, or which observation most recently established its location before a later reference event.
- Use words such as "last" or "most recent" only when the supplied evidence covers the relevant interval well enough to rule out a later qualifying event before the reference event.
- Prefer cases in which multiple provider views supply competing candidate interactions, so the result requires a real cross-user temporal comparison rather than the immediately preceding local moment.
- Example structure: "Where was the notebook last seen before I found it?"

5. Cross-user temporal ordering
- Compare two or more clearly identifiable, semantically related events from different users and determine which happened first, whether one was before or after another, or what visible event fell between them.
- Infer order from synchronized evidence and any supplied original timing metadata. Equal frame positions across different users' groups do not establish simultaneity or event order.
- Do not choose arbitrary unrelated actions merely because they have an order, and do not turn this into a prohibited question asking what another person was doing at the same time.
- Example structure: "What was the order of the window closing, my entering the room, and the lamp turning on?"

When several valid long-horizon questions are available, prefer wider separation between required observations, multiple distinct evidence intervals, stronger speaker-to-provider dependency, clearer support for every referenced event, and a natural reason the temporal relation matters. A long input does not by itself justify a long-horizon question, and a question answerable from one short local segment does not qualify.
"""

SAMPLED_FRAME_GENERATOR_MEDIA_MODES = {
    "asker_full_provider_pruned",
    "all_clustering_frames_only",
    "centroid_frames_only",
    "full_unpruned_sampled_frames_only",
    "retained_cluster_frames_only",
    "speaker_all_clustering_frames_five_provider_retained_cluster_frames",
    "six_full_unpruned_sampled_frame_sets",
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
    """Render prompt-safe aggregate metadata.

    The visual inputs are supplied to the runner separately.  This text intentionally
    omits file paths, frame indices, frame timestamps, cluster membership, and exact
    retained-interval maps so the prompt cannot expose an exact frame mapping.
    """

    required_users = list(packet.get("required_users") or [])
    speaker_user = required_users[0] if required_users else None
    evidence_provider_users = required_users[1:]
    evidence_provider_user = evidence_provider_users[0] if evidence_provider_users else None
    six_user_mode = len(required_users) == 6
    provider_users = (
        list(packet.get("provider_users") or required_users[1:])
        if six_user_mode
        else []
    )
    clips = []
    sampled_media_modes: set[str] = set()
    sampled_frame_modes = SAMPLED_FRAME_GENERATOR_MEDIA_MODES

    for clip in packet.get("clips", []):
        generator_media_mode = clip.get("generator_media_mode")
        clip_brief = {
            "user": clip.get("agent_name") or clip.get("user"),
            "day": clip.get("day"),
            "duration_seconds": clip.get("duration_seconds"),
            "segment_count": clip.get("segment_count"),
            "generator_media_mode": generator_media_mode,
            "media_role": clip.get("media_role"),
            "is_pruned": clip.get("is_pruned"),
        }

        if generator_media_mode in sampled_frame_modes:
            frame_count = sum(
                1 for frame in clip.get("frames", []) if isinstance(frame, dict)
            )
            if frame_count:
                sampled_media_modes.add(str(generator_media_mode))
                context_sampling = clip.get("context_sampling")
                aggregate_sampling = None
                if isinstance(context_sampling, dict):
                    aggregate_sampling = {
                        key: context_sampling.get(key)
                        for key in (
                            "policy",
                            "analysis_sample_fps",
                            "source_frame_count",
                            "model_input_frame_count",
                            "effective_model_input_fps",
                            "aggregate_frame_budget",
                        )
                        if context_sampling.get(key) is not None
                    }
                clip_brief["generator_frame_input"] = {
                    "frame_count": frame_count,
                    "ordering": "chronological within this user",
                    "aggregate_sampling": aggregate_sampling,
                }

        clips.append({key: value for key, value in clip_brief.items() if value is not None})

    generator_media_contract = None
    if sampled_media_modes:
        if sampled_media_modes == {"centroid_frames_only"}:
            mode = "retained_clip_cluster_centroid_images_only"
            frame_description = "isolated representative still images"
        elif sampled_media_modes == {"full_unpruned_sampled_frames_only"}:
            mode = "six_full_unpruned_sampled_frame_sets"
            frame_description = "one-per-second samples from all six complete videos"
        elif sampled_media_modes == {"retained_cluster_frames_only"}:
            mode = "retained_clip_cluster_member_images_only"
            frame_description = "sampled still images from retained cluster content"
        elif sampled_media_modes == {
            "all_clustering_frames_only",
            "retained_cluster_frames_only",
        }:
            mode = "asker_all_clustering_frames_provider_retained_cluster_frames"
            frame_description = (
                "all CLIP-sampled asker images and sampled provider images from "
                "clusters that survived pruning"
            )
        else:
            mode = "mixed_sampled_frame_modes"
            frame_description = "sampled still images"

        generator_media_contract = {
            "mode": mode,
            "image_group_order": (
                "Images are grouped in required_users order and are chronological within "
                "each user's group. No per-frame indices or timestamps are included."
            ),
            "limitations": (
                f"These are {frame_description}, not a continuous video. Adjacent supplied "
                "images may be separated by omitted footage. Do not infer unseen motion, "
                "transitions, duration, or events between images. Original timestamps are "
                "internal evidence metadata and must not appear in the question or options."
            ),
        }

    context_budget = packet.get("generator_context_budget")
    if six_user_mode and isinstance(context_budget, dict):
        required_users_order = (
            "required_users[0] is the speaker. required_users[1] through required_users[5] "
            "are providers. The generator receives every sampled speaker frame and every "
            "sampled member of each provider cluster that survived pruning, in chronological "
            "order; it receives no pruned MP4. The packet records the analysis sample rate and "
            "aggregate frame ceiling. The speaker input must naturally "
            "motivate the question but remain insufficient to answer it, while one or more "
            "provider inputs supply necessary answer-bearing evidence. An unused provider does "
            "not invalidate the item."
        )
    elif six_user_mode:
        required_users_order = (
            "required_users[0] is the speaker. required_users[1] through required_users[5] "
            "are providers. For generation, the speaker input contains every frame sampled for "
            "CLIP clustering (normally 30 one-per-second images), while each provider input "
            "contains only sampled members of clusters that survived pruning. The speaker input "
            "must naturally motivate the question but remain insufficient to answer it, while "
            "the combined six-user image input must support one unique answer. One or more "
            "provider views may support the answer; an unused provider does not invalidate the item."
        )
    else:
        required_users_order = (
            "required_users[0] is the asker and the question must use that user's natural "
            "first-person or shared-memory perspective. That user's view alone should be "
            "insufficient. required_users[1] supplies additional evidence; report each user's "
            "individual answerability truthfully. The combined evidence must support exactly "
            "one answer."
        )

    brief = {
        "evidence_id": packet.get("evidence_id"),
        "required_users": required_users,
        "role_contract": {
            "speaker_user": speaker_user,
            "evidence_provider_user": evidence_provider_user,
            "evidence_provider_users": evidence_provider_users,
            "provider_users": provider_users,
            "required_users_order": required_users_order,
        },
        "prompt_requirement": (
            "Use the visual media directly and write the strongest natural, grounded question "
            "supported by the current evidence. Do not cite timestamps in the user-facing "
            "question or options."
        ),
        "clips": clips,
    }
    if isinstance(context_budget, dict):
        brief["generator_context_budget"] = {
            key: context_budget.get(key)
            for key in (
                "policy",
                "analysis_sample_fps",
                "aggregate_frame_budget",
                "source_frame_count",
                "model_input_frame_count",
                "per_user_source_frame_counts",
                "per_user_model_input_frame_counts",
                "estimated_visual_tokens",
                "estimated_input_tokens_with_text_reserve",
            )
            if context_budget.get(key) is not None
        }
    if generator_media_contract is not None:
        brief["generator_media_contract"] = generator_media_contract
    return json.dumps(brief, ensure_ascii=False, indent=2)

def _generator_packet_brief(packet: dict[str, Any]) -> str:
    """Describe supplied frames without exposing sampling or selection machinery."""

    brief_text = video_packet_brief(packet)
    if not generator_uses_sampled_frames(packet):
        return brief_text

    brief = json.loads(brief_text)
    required_users = list(packet.get("required_users") or [])
    speaker_user = required_users[0] if required_users else None
    brief.pop("generator_context_budget", None)
    for clip in brief["clips"]:
        clip.pop("generator_media_mode", None)
        clip.pop("is_pruned", None)
        clip["media_role"] = "speaker" if clip.get("user") == speaker_user else "provider"
        frame_input = clip.get("generator_frame_input")
        if isinstance(frame_input, dict):
            frame_input.pop("aggregate_sampling", None)

    if len(required_users) == 6:
        brief["role_contract"]["required_users_order"] = (
            "required_users[0] is the speaker. required_users[1] through required_users[5] "
            "are providers. Each user's input consists of frames ordered chronologically "
            "within that user's group. The speaker input must naturally motivate the "
            "question but remain insufficient to answer it, while one or more provider "
            "inputs supply necessary answer-bearing evidence. An unused provider does "
            "not invalidate the item."
        )

    brief["generator_media_contract"] = {
        "mode": "frame_sets",
        "image_group_order": (
            "Images are grouped in required_users order and are chronological within "
            "each user's group. No per-frame indices or timestamps are included."
        ),
        "limitations": (
            "These are individual frames, not continuous video. Do not assume a fixed "
            "time interval between frames. Do not infer unseen motion, transitions, "
            "duration, or events between images. Original timestamps are internal "
            "evidence metadata and must not appear in the question or options."
        ),
    }
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
    required_users = list(packet.get("required_users") or [])
    six_user_mode = len(required_users) == 6
    if sampled_frame_input:
        generator_opening = """You are generating one natural, evidence-grounded multiple-choice question from egocentric visual evidence.

Input: chronologically ordered sampled frames selected from multiple people's egocentric recordings during the same synchronized interval.

These images are sparse samples rather than continuous video. Use only objects, states, actions, and relationships that are directly visible in the supplied frames and supported by the provided packet metadata.

Do not infer an action, transition, handoff, state change, temporal sequence, or event occurring between sampled frames unless both relevant states or moments are visibly supported by the supplied evidence.

Before returning the question, verify that every claim in the question, answer, options, evidence fields, and rationale is grounded in the supplied sampled frames.

Do not use captions, subtitles, transcripts, pre-written observations, or outside knowledge."""
        relation_source = "the supplied sampled frames"
        timestamp_instruction = (
            "The text prompt does not include exact per-frame paths, indices, or timestamps. "
            "Use the supplied images in "
            "their visual order, and do not treat proximity as proof of an unseen event or "
            "transition. Do not include participant names, clock times, timestamps, timecodes, "
            "frame numbers, seconds from the start, minute marks, filenames, or clip positions "
            "in the question or options."
        )
        evidence_timeframe_instruction = (
            "Fill the evidence field with each needed user's directly visible fact and the "
            "specific supplied-frame moment or supported timeframe."
        )
        single_user_visibility = "the supplied sampled frames"
        interval_wording = "the frame sets come from the same synchronized interval"
    else:
        generator_opening = """You are generating one natural, evidence-grounded multiple-choice question from raw egocentric videos.

Input: raw videos from multiple people during the same time interval. They may be near each other or in different places. Look directly at the videos and use only visual evidence and video metadata. Do not use captions, subtitles, transcripts, pre-written observations, or outside knowledge."""
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

    six_user_media_line = (
        "The generator receives frames from six people's egocentric recordings during "
        "the same synchronized interval. Frames are grouped by user and ordered "
        "chronologically within each user's group."
    )
    dependency_lines = (
        [
            "required_users[0] is the speaker, and required_users[1] through required_users[5] are providers.",
            "Ask a question that the speaker would naturally have and genuinely want to ask after what they experienced; do not manufacture a benchmark-style query merely because a provider view contains an extra detail.",
            six_user_media_line,
            "The speaker's sampled frames alone must remain insufficient to answer the question; one or more provider frame sets must supply the missing answer-bearing evidence.",
            "The combined six-user image input must directly support exactly one correct option.",
            "One or more provider views may supply the answer or the cross-view relation needed to identify it.",
            "Do not require every provider to contribute, and do not reject a natural question merely because some provider views are irrelevant.",
        ]
        if six_user_mode
        else [
            "required_users[0]'s view alone must be insufficient. The question should not be answered by the asker on their own; it must require additional evidence from required_users[1]."
        ]
    )
    provider_reference = (
        "required_users[1] through required_users[5]"
        if six_user_mode
        else "required_users[1]"
    )
    output_schema = VIDEO_GENERATION_SCHEMA
    if six_user_mode:
        output_schema = {
            **VIDEO_GENERATION_SCHEMA,
            "required_users": [
                "speaker user first",
                "first provider second",
                "second provider third",
                "third provider fourth",
                "fourth provider fifth",
                "fifth provider sixth",
            ],
            "single_user_answerability": {
                "Speaker": "insufficient because the speaker alone only provides ...",
                "ProviderOne": "sufficient/insufficient based only on this provider view ...",
                "ProviderTwo": "sufficient/insufficient based only on this provider view ...",
                "ProviderThree": "sufficient/insufficient based only on this provider view ...",
                "ProviderFour": "sufficient/insufficient based only on this provider view ...",
                "ProviderFive": "sufficient/insufficient based only on this provider view ...",
            },
            "combined_answerability": (
                "sufficient because the combined six-user image input supports exactly one option"
            ),
            "why_two_users_needed": (
                "legacy field name: explain why the speaker view alone is insufficient and "
                "which provider view or provider combination supplies the missing evidence; "
                "do not claim that every provider is necessary unless the videos establish it"
            ),
        }
    task_lines = [
        "Generate exactly one five-option multiple-choice question.",
        *([type_requirement] if type_requirement else []),
        "Treat required_users[0] as the asker and write a natural first-person or shared-memory question from that user's perspective.",
        "Make the question a speaker-side information need: required_users[0]'s view should explain why the question naturally comes up, but should not already make the answer obvious.",
        f"Choose the strongest natural, grounded question supported by {relation_source} without following a prescribed question family or template.",
        *dependency_lines,
        "The available evidence must make exactly one answer option correct.",
        timestamp_instruction,
        evidence_timeframe_instruction,
        "The answer field must exactly equal the correct option's text, and the correct option must be one of A, B, C, D, or E.",
    ]

    guidelines_block = f"""Guidelines:
- Use natural, informal, everyday first-person or shared-memory wording with I, me, my, we, us, or our.
- Do not ask {provider_reference} a second-person question and do not name any participant in the question or options.
- Do not refer to locations ambiguously, for example merely saying "the other room". Always specify with more detail; Perhaps identify the room as the living room or bedroom, or find some details that distinguishes the room and refer to that, something like "the room with a blue painting on the wall".
- Use a concise appearance-and-location description when needed referring to people. For example, "the person in the white shirt standing next to the television", or "the person with pink hair and wearing a pink blouse whose next to the bed".
- First decide the shortest natural answer that fully resolves the question. A short noun phrase or a single word is fine when it is clear; do not pad an option with extra visual details merely to make it more sophisticated.
- Keep all five options at the same answer type and level of detail: place versus place, object versus object, person versus person, outcome versus outcome, etc. If the question asks for a location, name the place at the level needed to answer it; omit a container, route, appearance, material, or extra action unless that detail is necessary to distinguish the correct choice.
- Put supporting visual details in evidence and rationale rather than the answer or options when they are not needed to identify the choice. Preserve details that are needed for a clear, unique answer; do not make the correct option so broad that another option could also be true.
- Write four distinct distractors that could logically answer this exact question in the described situation but are not supported as the correct answer. They need not be observed, but must not contradict the question's premise, mix answer types, or assign an impossible action or role to an object or person. If four plausible alternatives are unavailable, choose a different grounded question.
- Make the five options mutually exclusive at the same granularity and roughly parallel in grammar, length, and specificity. Avoid near-duplicates, an option nested inside another, and a correct option that is noticeably longer or more detailed than the distractors. Do not signal the answer through wording or specificity.
- Lead with the information request, then place first-person context when it helps identify the event or object. Prefer a direct interrogative opening such as what, which, where, who, how, did, was, or were. You can also start with "I was ...", "We were ...", "When I ...", "While I ...", "After I ...", "Once I ...", etc, then proceed to asking the question.
- single_user_answerability must contain one truthful entry for each required user. Do not manufacture insufficiency to fit an intended relation. For example, do not say an item was occluded or blurry when it was clearly visible in {single_user_visibility}.
- combined_answerability must explicitly say "sufficient because ..." and explain why the available views together support exactly one option.
- Do not stitch unrelated scenes together, invent person or object continuity, or exaggerate a cross-view dependency merely because {interval_wording}.
"""

    return f"""{generator_opening}

{STRICT_JSON_OUTPUT_CONTRACT}

Your task:
{_numbered_lines(task_lines)}

{guidelines_block}

{ANTI_ACTIVITY_QUERY_GUIDANCE}

{RESTORED_GENERATOR_COVERAGE_GUIDANCE}

{OPTIONAL_LONG_HORIZON_GUIDANCE}

{feedback_block}
Evidence packet metadata:
{_generator_packet_brief(packet)}

Return exactly one valid JSON object with this exact shape:
{json.dumps(output_schema, ensure_ascii=False, indent=2)}
"""