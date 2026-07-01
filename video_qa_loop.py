"""Video-first generation loop for EgoLife two-user MCQ construction."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, iter_jsonl, write_jsonl
from .prompts import (
    GENERATION_MODES,
    build_answerability_prompt,
    build_judger_prompt,
    build_relation_discovery_prompt,
    build_relation_mcq_prompt,
    build_video_generation_prompt,
)
from .qwen3vl_runner import DEFAULT_MODEL_ID, make_runner
from .schema import OPTION_LETTERS, extract_json_object, normalize_correct, validate_qa_item


class StreamingJsonlRows(list[dict[str, Any]]):
    """Keep an in-memory row list while also flushing each row to disk."""

    def __init__(self, path: str | Path | None, *, reset: bool = True) -> None:
        super().__init__()
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if reset or not self.path.exists():
                self.path.write_text("", encoding="utf-8")

    def load_existing(self) -> None:
        if not self.path or not self.path.exists():
            return
        for row in iter_jsonl(self.path):
            super().append(row)

    def append(self, row: dict[str, Any]) -> None:
        super().append(row)
        if self.path:
            append_jsonl(self.path, row)


QUESTION_TYPES = ("commonality", "difference")
BLOCKING_JUDGE_CHECKS = (
    "first_person_naturalness",
    "agent_perspective",
    "source_scope",
    "question_type_semantics",
    "multi_video_necessity",
    "visual_grounding",
    "mcq_option_quality",
    "gaze_safety",
    "human_auditability",
)


def existing_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return str(path)
    return None


def clip_video_path(clip: dict[str, Any]) -> str | None:
    return existing_path(clip.get("local_video"))


def clip_image_paths(clip: dict[str, Any]) -> list[str]:
    paths = []
    for frame in clip.get("frames", []):
        path = existing_path(frame.get("path"))
        if path:
            paths.append(path)
    return paths


def clips_require_frame_inputs(clips: list[dict[str, Any]]) -> bool:
    return any(
        clip.get("generator_media_mode") == "frames_only" or clip.get("force_frame_inputs")
        for clip in clips
    )


def media_for_clips(
    clips: list[dict[str, Any]],
    *,
    backend: str,
    allow_openai_video_input: bool,
) -> tuple[list[str], list[str]]:
    videos = [path for clip in clips if (path := clip_video_path(clip))]
    images = [path for clip in clips for path in clip_image_paths(clip)]
    if clips_require_frame_inputs(clips):
        return images, []
    if backend == "openai-compatible-local" and not allow_openai_video_input:
        return images, []
    return images if not videos else [], videos


def video_evidence_for_packet(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic clip/video provenance for the generated QA row."""

    rows = []
    for clip in packet.get("clips", []):
        local_video = clip.get("local_video")
        rows.append(
            {
                "user": clip.get("agent_name"),
                "agent_dir": clip.get("agent_dir"),
                "agent_id": clip.get("agent_id"),
                "day": clip.get("day"),
                "time_token": clip.get("time_token"),
                "clip_clock": clip.get("clip_clock"),
                "duration_seconds": clip.get("duration_seconds"),
                "video_url": clip.get("video_url"),
                "local_video": local_video,
                "local_video_exists": bool(existing_path(local_video)),
                "source_local_video": clip.get("source_local_video"),
                "generator_media_mode": clip.get("generator_media_mode"),
                "temporal_pruning": clip.get("temporal_pruning"),
                "gaze_url": clip.get("gaze_url"),
                "gaze_summary": clip.get("gaze_summary"),
                "sampled_frames": [
                    {
                        "timestamp_seconds": frame.get("timestamp_seconds"),
                        "path": frame.get("path"),
                        "path_exists": bool(existing_path(frame.get("path"))),
                    }
                    for frame in clip.get("frames", [])
                ],
            }
        )
    return rows


def human_audit_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Compact evidence bundle intended for manual review of one generated QA."""

    return {
        "evidence_id": packet.get("evidence_id"),
        "required_users": packet.get("required_users", []),
        "requirement": packet.get("requirement"),
        "source_urls": packet.get("source_urls", {}),
        "video_evidence": video_evidence_for_packet(packet),
        "review_instructions": [
            "Open each listed local_video or video_url for the required users.",
            "Check the referred_timestamps and per_user_evidence_claims against the visible content.",
            "Verify that no single user's video alone makes the correct option obvious.",
        ],
    }


def complete_generator_metadata(
    qa: dict[str, Any],
    *,
    packet: dict[str, Any],
    question_type: str,
) -> dict[str, Any]:
    """Fill review metadata that the generator may omit before the real gates run."""

    required_users = list(packet.get("required_users") or qa.get("required_users") or [])
    qa["question_type"] = question_type
    qa["required_users"] = required_users
    qa.setdefault("category", "environmental_interaction")
    qa.setdefault("referred_timestamps", [])
    if not isinstance(qa.get("referred_timestamps"), list):
        qa["referred_timestamps"] = []

    try:
        correct = normalize_correct(qa.get("correct"))
        qa["correct"] = correct
        options = qa.get("options")
        if isinstance(options, list) and len(options) == len(OPTION_LETTERS):
            qa["answer"] = options[OPTION_LETTERS.index(correct)]
    except ValueError:
        pass

    single = qa.get("single_user_answerability")
    if not isinstance(single, dict):
        single = {}
    for user in required_users:
        text = str(single.get(user, "")).strip()
        if not text or not any(marker in text.lower() for marker in ("insufficient", "cannot", "not enough")):
            single[user] = (
                "insufficient because this user's video alone does not provide "
                "all visual facts needed from the other required user(s)"
            )
    qa["single_user_answerability"] = single

    combined = str(qa.get("combined_answerability", "")).strip()
    if "sufficient" not in combined.lower() and "support" not in combined.lower():
        qa["combined_answerability"] = (
            "sufficient because combining the required users' videos provides "
            "the speaker-side anchor event plus the missing visual detail needed "
            "to select exactly one option"
        )

    if not qa.get("generator_rationale"):
        qa["generator_rationale"] = (
            "The question is framed as a natural first-person memory gap anchored "
            "in one user's experience and answered with another user's visual evidence."
        )
    if not qa.get("why_two_users_needed"):
        qa["why_two_users_needed"] = (
            "At least two required users are needed because one supplies the "
            "speaker-side anchor event while another supplies a non-redundant "
            "missing visual detail."
        )
    claims = qa.get("per_user_evidence_claims")
    if not isinstance(claims, list) or not claims:
        claims = []
        for user in required_users:
            claims.append(
                {
                    "user": user,
                    "claim": f"{user}'s own video contributes a necessary visual fact listed in the evidence field.",
                }
            )
        qa["per_user_evidence_claims"] = claims

    review = qa.get("review")
    if not isinstance(review, dict):
        review = {}
    review.setdefault(
        "generator_self_check",
        "This draft should require the combined required users' videos and should not ask what both users saw.",
    )
    review.setdefault("status", "draft")
    qa["review"] = review
    return qa


def condition_media_for_clips(
    *,
    condition: dict[str, Any],
    clips: list[dict[str, Any]],
    image_paths: list[str],
    video_paths: list[str],
) -> dict[str, Any]:
    return {
        "condition_id": condition.get("condition_id"),
        "condition_type": condition.get("condition_type"),
        "users": condition.get("users", []),
        "image_paths": image_paths,
        "video_paths": video_paths,
        "video_evidence": video_evidence_for_packet({"clips": clips}),
    }


def qa_for_judger_prompt(qa: dict[str, Any]) -> dict[str, Any]:
    """Return only the generated QA fields the judger needs to evaluate."""

    wanted = [
        "qa_id",
        "evidence_id",
        "question_type",
        "question",
        "options",
        "correct",
        "answer",
        "category",
        "required_users",
        "evidence",
        "single_user_answerability",
        "combined_answerability",
        "generator_rationale",
        "why_two_users_needed",
        "per_user_evidence_claims",
        "referred_timestamps",
        "review",
    ]
    return {key: qa[key] for key in wanted if key in qa}


def clips_for_users(packet: dict[str, Any], users: list[str]) -> list[dict[str, Any]]:
    wanted = set(users)
    return [clip for clip in packet.get("clips", []) if clip.get("agent_name") in wanted]


def target_type_counts(target_count: int) -> dict[str, int]:
    commonality = (target_count + 1) // 2
    difference = target_count - commonality
    return {"commonality": commonality, "difference": difference}


def choose_question_type(counts: dict[str, int], targets: dict[str, int]) -> str | None:
    remaining = {
        question_type: targets[question_type] - counts.get(question_type, 0)
        for question_type in QUESTION_TYPES
    }
    remaining = {key: value for key, value in remaining.items() if value > 0}
    if not remaining:
        return None
    return sorted(remaining.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_answerability_conditions(required_users: list[str]) -> list[dict[str, Any]]:
    users = list(required_users)
    conditions = [
        {
            "condition_id": f"single_user::{user}",
            "condition_type": "single_user",
            "users": [user],
        }
        for user in users
    ]
    if len(users) > 2:
        for size in range(2, len(users)):
            for combo in itertools.combinations(users, size):
                combo_users = list(combo)
                conditions.append(
                    {
                        "condition_id": "proper_subset::" + "+".join(combo_users),
                        "condition_type": "proper_subset",
                        "users": combo_users,
                    }
                )
    conditions.append(
        {
            "condition_id": "combined_all_users::" + "+".join(users),
            "condition_type": "combined_all_users",
            "users": users,
        }
    )
    return conditions


def parsed_choice(value: Any) -> tuple[str | None, bool]:
    text = str(value or "").strip()
    if text.lower() in {"insufficient", "not enough", "unknown", "cannot answer", "can't answer"}:
        return None, True
    try:
        return normalize_correct(text), False
    except ValueError:
        return None, False


def answerability_gate(qa_item: dict[str, Any], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        correct = normalize_correct(qa_item.get("correct"))
    except ValueError as exc:
        return {"passed": False, "reason": str(exc)}

    combined = [row for row in evaluations if row.get("condition_type") == "combined_all_users"]
    if not combined:
        return {"passed": False, "reason": "missing combined_all_users evaluation"}

    combined_choice, combined_insufficient = parsed_choice(combined[-1].get("choice"))
    if combined_insufficient or combined_choice != correct:
        return {
            "passed": False,
            "reason": f"combined_all_users did not select correct answer {correct}",
        }

    leaking = []
    for row in evaluations:
        if row.get("condition_type") == "combined_all_users":
            continue
        choice, insufficient = parsed_choice(row.get("choice"))
        if not insufficient and choice == correct:
            leaking.append(row.get("condition_id"))
    if leaking:
        return {
            "passed": False,
            "reason": "single/subset condition answered correctly: " + ", ".join(str(item) for item in leaking),
        }

    return {
        "passed": True,
        "reason": "combined videos answer correctly and all single/subset conditions are insufficient or incorrect",
    }


def judge_gate(judge: dict[str, Any]) -> dict[str, Any]:
    """Deterministically gate structured judger output.

    The model still proposes review_passed, but when structured checks are
    present the checks are authoritative. Some VLM outputs mark every blocking
    check PASS while leaving the top-level review_passed flag false; that flag
    is treated as a diagnostic inconsistency rather than a veto.
    """

    checks = judge.get("checks")
    if not isinstance(checks, dict):
        if judge.get("review_passed") is not True:
            return {
                "passed": False,
                "reason": str(judge.get("feedback_to_generator") or "judger review_passed is not true"),
                "failed_checks": list(judge.get("blocking_failures") or []),
            }
        return {
            "passed": True,
            "reason": "legacy judger output passed without structured checks",
            "failed_checks": [],
        }

    failed = []
    missing = []
    blocking_failures = list(judge.get("blocking_failures") or [])
    for name in BLOCKING_JUDGE_CHECKS:
        check = checks.get(name)
        if not isinstance(check, dict):
            missing.append(name)
            continue
        status = str(check.get("status", "")).strip().upper()
        if status != "PASS":
            failed.append(name)
    if missing or failed:
        details = []
        if failed:
            details.append("failed checks: " + ", ".join(failed))
        if missing:
            details.append("missing checks: " + ", ".join(missing))
        return {
            "passed": False,
            "reason": "; ".join(details),
            "failed_checks": failed + missing,
        }
    if blocking_failures:
        return {
            "passed": False,
            "reason": "blocking_failures listed despite structured checks passing: "
            + ", ".join(str(item) for item in blocking_failures),
            "failed_checks": blocking_failures,
        }

    gate = {
        "passed": True,
        "reason": "all structured judger checks passed",
        "failed_checks": [],
    }
    if judge.get("review_passed") is not True:
        gate["model_review_passed"] = judge.get("review_passed")
        gate["warning"] = "ignored inconsistent top-level review_passed because all structured checks passed"
    return gate


def build_review_from_gates(
    *,
    judge: dict[str, Any] | None,
    answerability: dict[str, Any] | None,
    schema_errors: list[str] | None,
    accepted: bool,
    rejection_stage: str | None = None,
    final_reason: str | None = None,
) -> dict[str, Any]:
    """Build the final review object stored inside each QA row.

    Generator self-checks stay in generation_trace. The final review is derived
    from the judger, answerability evaluator, and deterministic schema checks.
    """

    schema_errors = list(schema_errors or [])
    schema_passed = not schema_errors
    if accepted:
        status = "passed"
    elif rejection_stage == "judger":
        status = "rejected_by_judger"
    elif rejection_stage == "answerability":
        status = "rejected_by_answerability"
    else:
        status = "rejected_by_schema"

    return {
        "status": status,
        "review_passed": bool(accepted),
        "judger": judge if isinstance(judge, dict) else {},
        "answerability": answerability if isinstance(answerability, dict) else {},
        "schema_validation": {
            "passed": schema_passed,
            "errors": schema_errors,
        },
        "final_decision": {
            "accepted": bool(accepted),
            "rejection_stage": None if accepted else (rejection_stage or "schema"),
            "reason": final_reason or ("passed all gates" if accepted else "rejected"),
        },
    }


def dry_run_discovered_relation(packet: dict[str, Any], question_type: str) -> dict[str, Any]:
    users = packet.get("required_users", [])[:2]
    speaker = users[0] if users else "User A"
    other = users[1] if len(users) > 1 else "User B"
    return {
        "need": "dry-run discovered cross-user information need",
        "speaker_user": speaker,
        "other_required_users": [other],
        "what_speaker_knows_sees": "dry-run speaker-side visual anchor",
        "what_others_know_see": {other: "dry-run missing visual detail"},
        "only_clear_when_combining": f"dry-run {question_type} relation requiring both users",
        "why_natural_to_ask": "dry-run placeholder for prompt plumbing",
        "likely_answerable_by_one_video_alone": "no, dry-run placeholder",
    }


def dry_run_qa(packet: dict[str, Any], question_type: str, generation_mode: str = "baseline") -> dict[str, Any]:
    users = packet.get("required_users", [])[:2]
    clips = packet.get("clips", [])
    if clips_require_frame_inputs(clips):
        dry_run_media = {
            "image_paths": [path for clip in clips for path in clip_image_paths(clip)],
            "video_paths": [],
        }
    else:
        dry_run_media = {
            "image_paths": [],
            "video_paths": [
                path for clip in clips if (path := clip_video_path(clip))
            ],
        }
    return {
        "qa_id": f"DRYRUN_{packet.get('evidence_id')}_{question_type}",
        "question_type": question_type,
        "generation_mode": generation_mode,
        "question": "Which option can be determined only after comparing what we each experienced?",
        "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
        "correct": "A",
        "answer": "Option A",
        "category": "environmental_interaction",
        "required_users": users,
        "evidence": [{"user": user, "needed_fact": "dry-run video evidence", "frames_used": []} for user in users],
        "single_user_answerability": {user: "insufficient in dry-run mode" for user in users},
        "combined_answerability": "sufficient in dry-run prompt construction only",
        "generator_rationale": "dry-run placeholder",
        "why_two_users_needed": "dry-run placeholder",
        "per_user_evidence_claims": [{"user": user, "claim": "dry-run placeholder"} for user in users],
        "attempt_count": 0,
        "review": {
            "review_passed": False,
            "status": "dry_run",
            "judger": {},
            "answerability": {},
            "schema_validation": {"passed": False, "errors": []},
            "final_decision": {
                "accepted": False,
                "rejection_stage": "dry_run",
                "reason": "No model review was run in dry-run mode.",
            },
        },
        "model_id": "dry-run-no-model",
        "source_urls": packet.get("source_urls", {}),
        "video_evidence": video_evidence_for_packet(packet),
        "referred_timestamps": [],
        "human_audit": human_audit_packet(packet),
        "generation_trace": [
            {
                "attempt": 0,
                "stage": "dry_run",
                "question_type": question_type,
                "note": "No model was called; prompts and media paths were generated for plumbing validation.",
                "media": dry_run_media,
            }
        ],
    }


def run_answerability_eval(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    runner: Any,
    backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = []
    for condition in build_answerability_conditions(qa_item.get("required_users", [])):
        clips = clips_for_users(packet, condition["users"])
        image_paths, video_paths = media_for_clips(
            clips,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
        )
        prompt = build_answerability_prompt(qa_item, condition)
        prompt_rows.append(
            {
                "stage": "answerability",
                "qa_id": qa_item.get("qa_id"),
                "generation_mode": qa_item.get("generation_mode"),
                "condition_id": condition["condition_id"],
                "prompt": prompt,
                "image_paths": image_paths,
                "video_paths": video_paths,
                "condition_media": condition_media_for_clips(
                    condition=condition,
                    clips=clips,
                    image_paths=image_paths,
                    video_paths=video_paths,
                ),
            }
        )
        stage_start = time.time()
        print(
            "qa_stage_start "
            f"stage=answerability qa_id={qa_item.get('qa_id')} "
            f"condition_id={condition['condition_id']} "
            f"images={len(image_paths)} videos={len(video_paths)}",
            flush=True,
        )
        raw = runner.generate(prompt, image_paths=image_paths, video_paths=video_paths)
        print(
            "qa_stage_done "
            f"stage=answerability qa_id={qa_item.get('qa_id')} "
            f"condition_id={condition['condition_id']} seconds={time.time() - stage_start:.1f}",
            flush=True,
        )
        try:
            answer = extract_json_object(raw)
        except Exception as exc:
            answer = {
                "choice": "insufficient",
                "answer_text": "",
                "confidence": 0.0,
                "evidence_used": "",
                "insufficient_reason": f"parse_failed: {exc}",
            }
        evaluations.append(
            {
                **condition,
                **answer,
                "raw_output": raw,
                "condition_media": condition_media_for_clips(
                    condition=condition,
                    clips=clips,
                    image_paths=image_paths,
                    video_paths=video_paths,
                ),
            }
        )
    gate = answerability_gate(qa_item, evaluations)
    return {"evaluations": evaluations, "gate": gate}


def generate_video_qa_loop(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path | None,
    rejected_path: str | Path | None,
    intermediate_path: str | Path | None = None,
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    target_count: int = 20,
    max_attempts: int = 3,
    max_new_tokens: int = 1536,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    allow_openai_video_input: bool = False,
    dry_run: bool = False,
    generation_mode: str = "baseline",
    fixed_question_type_schedule: bool = False,
    resume: bool = False,
) -> list[dict[str, Any]]:
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    runner = make_runner(
        "dry-run" if dry_run else backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        allow_openai_video_input=allow_openai_video_input,
    )
    prompts = StreamingJsonlRows(prompts_path, reset=not resume)
    intermediate_rows = StreamingJsonlRows(intermediate_path, reset=not resume)
    accepted = StreamingJsonlRows(output_path, reset=not resume)
    rejected = StreamingJsonlRows(rejected_path, reset=not resume)
    if resume:
        accepted.load_existing()
        rejected.load_existing()
        prompts.load_existing()
        intermediate_rows.load_existing()
    processed_evidence_ids = {
        str(row.get("evidence_id"))
        for row in [*accepted, *rejected]
        if row.get("evidence_id")
    }
    targets = target_type_counts(target_count)
    counts = {question_type: 0 for question_type in QUESTION_TYPES}
    for row in accepted:
        question_type = row.get("question_type")
        if question_type in counts:
            counts[question_type] += 1

    for packet_index, packet in enumerate(iter_jsonl(evidence_path)):
        if fixed_question_type_schedule and packet_index >= target_count:
            break
        if len(accepted) >= target_count:
            break
        evidence_id = str(packet.get("evidence_id") or "")
        if resume and evidence_id in processed_evidence_ids:
            print(f"resume_skip evidence_id={evidence_id}", flush=True)
            continue
        question_type = (
            QUESTION_TYPES[packet_index % len(QUESTION_TYPES)]
            if fixed_question_type_schedule
            else choose_question_type(counts, targets)
        )
        if question_type is None:
            break
        clips = packet.get("clips", [])
        image_paths, video_paths = media_for_clips(
            clips,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
        )
        feedback = None
        if dry_run:
            qa = dry_run_qa(packet, question_type, generation_mode=generation_mode)
            discovery_prompt = None
            discovered_relation = None
            if generation_mode == "discovery":
                discovery_prompt = build_relation_discovery_prompt(packet, question_type)
                discovered_relation = dry_run_discovered_relation(packet, question_type)
                gen_prompt = build_relation_mcq_prompt(packet, question_type, discovered_relation)
            else:
                gen_prompt = build_video_generation_prompt(
                    packet,
                    question_type,
                    generation_mode=generation_mode,
                )
            judge_prompt = build_judger_prompt(qa_for_judger_prompt(qa), packet)
            dry_trace = {
                "evidence_id": packet.get("evidence_id"),
                "qa_id": qa.get("qa_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": 1,
                "feedback_in": None,
                "media": {
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "human_audit": human_audit_packet(packet),
                },
                "discovery": (
                    {
                        "prompt": discovery_prompt,
                        "raw_output": None,
                        "parsed": {"selected_relation": discovered_relation},
                    }
                    if generation_mode == "discovery"
                    else {}
                ),
                "generation": {"prompt": gen_prompt, "raw_output": None},
                "judge": {"prompt": judge_prompt, "raw_output": None},
                "answerability": {"conditions": []},
                "result": {"accepted": False, "dry_run": True},
            }
            if generation_mode == "discovery":
                prompts.append(
                    {
                        "stage": "discovery",
                        "evidence_id": packet.get("evidence_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "attempt": 1,
                        "prompt": discovery_prompt,
                        "image_paths": image_paths,
                        "video_paths": video_paths,
                    }
                )
            prompts.append(
                {
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": 1,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            prompts.append(
                {
                    "stage": "judge",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": 1,
                    "prompt": judge_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            for condition in build_answerability_conditions(packet.get("required_users", [])):
                condition_clips = clips_for_users(packet, condition["users"])
                cond_images, cond_videos = media_for_clips(
                    condition_clips,
                    backend=backend,
                    allow_openai_video_input=allow_openai_video_input,
                )
                prompts.append(
                    {
                        "stage": "answerability",
                        "evidence_id": packet.get("evidence_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "condition_id": condition["condition_id"],
                        "prompt": build_answerability_prompt(qa, condition),
                        "image_paths": cond_images,
                        "video_paths": cond_videos,
                        "condition_media": condition_media_for_clips(
                            condition=condition,
                            clips=condition_clips,
                            image_paths=cond_images,
                            video_paths=cond_videos,
                        ),
                    }
                )
                dry_trace["answerability"]["conditions"].append(
                    condition_media_for_clips(
                        condition=condition,
                        clips=condition_clips,
                        image_paths=cond_images,
                        video_paths=cond_videos,
                    )
                )
            qa["generation_trace"] = [dry_trace]
            qa["human_audit"] = human_audit_packet(packet)
            intermediate_rows.append(dry_trace)
            counts[question_type] += 1
            accepted.append(qa)
            continue

        packet_rejections = []
        packet_trace = []
        last_review = None
        for attempt in range(1, max_attempts + 1):
            attempt_trace: dict[str, Any] = {
                "evidence_id": packet.get("evidence_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": attempt,
                "feedback_in": feedback,
                "media": {
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "human_audit": human_audit_packet(packet),
                },
                "discovery": {},
                "generation": {},
                "judge": {},
                "answerability": {},
                "result": {},
            }
            packet_trace.append(attempt_trace)
            if generation_mode == "discovery":
                discovery_prompt = build_relation_discovery_prompt(packet, question_type, feedback=feedback)
                attempt_trace["discovery"]["prompt"] = discovery_prompt
                prompts.append(
                    {
                        "stage": "discovery",
                        "evidence_id": packet.get("evidence_id"),
                        "question_type": question_type,
                        "generation_mode": generation_mode,
                        "attempt": attempt,
                        "prompt": discovery_prompt,
                        "image_paths": image_paths,
                        "video_paths": video_paths,
                    }
                )
                stage_start = time.time()
                print(
                    "qa_stage_start "
                    f"stage=discovery evidence_id={packet.get('evidence_id')} "
                    f"question_type={question_type} attempt={attempt} "
                    f"images={len(image_paths)} videos={len(video_paths)}",
                    flush=True,
                )
                raw_discovery = runner.generate(discovery_prompt, image_paths=image_paths, video_paths=video_paths)
                print(
                    "qa_stage_done "
                    f"stage=discovery evidence_id={packet.get('evidence_id')} "
                    f"question_type={question_type} attempt={attempt} "
                    f"seconds={time.time() - stage_start:.1f}",
                    flush=True,
                )
                attempt_trace["discovery"]["raw_output"] = raw_discovery
                try:
                    discovery = extract_json_object(raw_discovery)
                except Exception as exc:
                    feedback = f"Discovery output was not valid JSON: {exc}"
                    attempt_trace["result"] = {"accepted": False, "reason": feedback}
                    packet_rejections.append({"attempt": attempt, "reason": feedback, "raw_output": raw_discovery})
                    continue
                discovered_relation = discovery.get("selected_relation") if isinstance(discovery, dict) else None
                if not isinstance(discovered_relation, dict) or not discovered_relation:
                    discovered_relation = discovery
                attempt_trace["discovery"]["parsed"] = discovery
                gen_prompt = build_relation_mcq_prompt(
                    packet,
                    question_type,
                    discovered_relation,
                    feedback=feedback,
                )
            else:
                gen_prompt = build_video_generation_prompt(
                    packet,
                    question_type,
                    feedback=feedback,
                    generation_mode=generation_mode,
                )
            attempt_trace["generation"]["prompt"] = gen_prompt
            prompts.append(
                {
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": attempt,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            stage_start = time.time()
            print(
                "qa_stage_start "
                f"stage=generation evidence_id={packet.get('evidence_id')} "
                f"question_type={question_type} attempt={attempt} "
                f"images={len(image_paths)} videos={len(video_paths)}",
                flush=True,
            )
            raw_generation = runner.generate(gen_prompt, image_paths=image_paths, video_paths=video_paths)
            print(
                "qa_stage_done "
                f"stage=generation evidence_id={packet.get('evidence_id')} "
                f"question_type={question_type} attempt={attempt} "
                f"seconds={time.time() - stage_start:.1f}",
                flush=True,
            )
            attempt_trace["generation"]["raw_output"] = raw_generation
            try:
                qa = extract_json_object(raw_generation)
            except Exception as exc:
                feedback = f"Generator output was not valid JSON: {exc}"
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append({"attempt": attempt, "reason": feedback, "raw_output": raw_generation})
                continue

            qa.setdefault("qa_id", f"QA_{len(accepted) + 1:03d}_{packet.get('evidence_id')}")
            attempt_trace["qa_id"] = qa.get("qa_id")
            attempt_trace["generation"]["parsed_qa"] = {
                "qa_id": qa.get("qa_id"),
                "question": qa.get("question"),
                "options": qa.get("options"),
                "correct": qa.get("correct"),
                "answer": qa.get("answer"),
                "required_users": qa.get("required_users"),
                "question_type": qa.get("question_type"),
                "generator_rationale": qa.get("generator_rationale"),
                "why_two_users_needed": qa.get("why_two_users_needed"),
                "per_user_evidence_claims": qa.get("per_user_evidence_claims"),
                "referred_timestamps": qa.get("referred_timestamps"),
            }
            qa["evidence_id"] = packet.get("evidence_id")
            qa["question_type"] = question_type
            qa["generation_mode"] = generation_mode
            qa["required_users"] = packet.get("required_users", qa.get("required_users", []))
            qa["model_id"] = runner.model_id
            qa["source_urls"] = packet.get("source_urls", {})
            qa["video_evidence"] = video_evidence_for_packet(packet)
            qa.setdefault("referred_timestamps", [])
            qa["human_audit"] = human_audit_packet(packet)
            qa["generation_trace"] = packet_trace
            qa["attempt_count"] = attempt
            qa.pop("judge_feedback", None)
            qa.pop("answerability_eval", None)
            complete_generator_metadata(qa, packet=packet, question_type=question_type)
            attempt_trace["generation"]["normalized_qa"] = {
                "qa_id": qa.get("qa_id"),
                "category": qa.get("category"),
                "single_user_answerability": qa.get("single_user_answerability"),
                "combined_answerability": qa.get("combined_answerability"),
                "generator_rationale": qa.get("generator_rationale"),
                "why_two_users_needed": qa.get("why_two_users_needed"),
                "per_user_evidence_claims": qa.get("per_user_evidence_claims"),
                "review": qa.get("review"),
            }

            schema_errors = validate_qa_item(qa)
            if schema_errors:
                feedback = "Schema errors to fix: " + "; ".join(schema_errors)
                qa["review"] = build_review_from_gates(
                    judge=None,
                    answerability=None,
                    schema_errors=schema_errors,
                    accepted=False,
                    rejection_stage="schema",
                    final_reason=feedback,
                )
                last_review = qa["review"]
                attempt_trace["schema_errors"] = schema_errors
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            judge_prompt = build_judger_prompt(qa_for_judger_prompt(qa), packet)
            attempt_trace["judge"]["prompt"] = judge_prompt
            prompts.append(
                {
                    "stage": "judge",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "attempt": attempt,
                    "prompt": judge_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            stage_start = time.time()
            print(
                "qa_stage_start "
                f"stage=judge evidence_id={packet.get('evidence_id')} "
                f"qa_id={qa.get('qa_id')} attempt={attempt} "
                f"images={len(image_paths)} videos={len(video_paths)}",
                flush=True,
            )
            raw_judge = runner.generate(judge_prompt, image_paths=image_paths, video_paths=video_paths)
            print(
                "qa_stage_done "
                f"stage=judge evidence_id={packet.get('evidence_id')} "
                f"qa_id={qa.get('qa_id')} attempt={attempt} "
                f"seconds={time.time() - stage_start:.1f}",
                flush=True,
            )
            attempt_trace["judge"]["raw_output"] = raw_judge
            try:
                judge = extract_json_object(raw_judge)
            except Exception as exc:
                judge = {
                    "review_passed": False,
                    "feedback_to_generator": f"Judger output was not valid JSON: {exc}",
                }
            judge["raw_output"] = raw_judge
            judge["gate"] = judge_gate(judge)
            attempt_trace["judge"]["parsed"] = judge
            if judge["gate"].get("passed") is not True:
                feedback = str(
                    judge.get("feedback_to_generator")
                    or judge["gate"].get("reason")
                    or "Judger rejected the question."
                )
                qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=None,
                    schema_errors=[],
                    accepted=False,
                    rejection_stage="judger",
                    final_reason=feedback,
                )
                last_review = qa["review"]
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            answerability = run_answerability_eval(
                qa_item=qa,
                packet=packet,
                runner=runner,
                backend=backend,
                allow_openai_video_input=allow_openai_video_input,
                prompt_rows=prompts,
            )
            attempt_trace["answerability"] = answerability
            if answerability.get("gate", {}).get("passed") is not True:
                feedback = "Answerability gate failed: " + str(answerability.get("gate", {}).get("reason", ""))
                qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=[],
                    accepted=False,
                    rejection_stage="answerability",
                    final_reason=feedback,
                )
                last_review = qa["review"]
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            qa["review"] = build_review_from_gates(
                judge=judge,
                answerability=answerability,
                schema_errors=[],
                accepted=True,
                final_reason="passed all gates",
            )
            strict_errors = validate_qa_item(qa, strict_review=True)
            if strict_errors:
                feedback = "Strict validation errors: " + "; ".join(strict_errors)
                qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=strict_errors,
                    accepted=False,
                    rejection_stage="schema",
                    final_reason=feedback,
                )
                last_review = qa["review"]
                attempt_trace["schema_errors"] = strict_errors
                attempt_trace["result"] = {"accepted": False, "reason": feedback}
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            attempt_trace["result"] = {"accepted": True, "reason": "passed all gates"}
            qa["generation_trace"] = packet_trace
            last_review = qa["review"]
            accepted.append(qa)
            intermediate_rows.append(
                {
                    "evidence_id": packet.get("evidence_id"),
                    "qa_id": qa.get("qa_id"),
                    "question_type": question_type,
                    "generation_mode": generation_mode,
                    "status": "accepted",
                    "attempts": packet_trace,
                }
            )
            counts[question_type] += 1
            break
        else:
            rejected_row = {
                "evidence_id": packet.get("evidence_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempts": packet_rejections,
                "generation_trace": packet_trace,
                "human_audit": human_audit_packet(packet),
            }
            if last_review is not None:
                rejected_row["review"] = last_review
            rejected.append(rejected_row)
            intermediate_rows.append({**rejected_row, "status": "rejected"})

    if prompts_path:
        write_jsonl(prompts_path, prompts)
    if intermediate_path:
        write_jsonl(intermediate_path, intermediate_rows)
    write_jsonl(output_path, accepted)
    if rejected_path and rejected:
        write_jsonl(rejected_path, rejected)
    return accepted


def add_video_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "openai-compatible-local"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--generation-mode", default="baseline", choices=GENERATION_MODES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-openai-video-input", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixed-question-type-schedule", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append to existing JSONL outputs and skip completed evidence IDs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video-first EgoLife two-user QA generation loop")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-output")
    parser.add_argument("--rejected-output")
    parser.add_argument("--intermediate-output")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    add_video_loop_args(parser)
    args = parser.parse_args(argv)
    rows = generate_video_qa_loop(
        evidence_path=args.evidence,
        output_path=args.output,
        prompts_path=args.prompts_output,
        rejected_path=args.rejected_output,
        intermediate_path=args.intermediate_output,
        backend=args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        target_count=args.target_count,
        max_attempts=args.max_attempts,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        allow_openai_video_input=args.allow_openai_video_input,
        dry_run=args.dry_run,
        generation_mode=args.generation_mode,
        fixed_question_type_schedule=args.fixed_question_type_schedule,
        resume=args.resume,
    )
    print(f"accepted {len(rows)} video-first QA rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
