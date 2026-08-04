"""Two-pass local generation and judging for models that cannot coexist in VRAM.

The generate phase writes every requested attempt before any model judge runs.
The judge phase later evaluates every parseable attempt and selects the earliest
attempt that passes all production gates.  This intentionally differs from the
integrated loop: deferred attempts cannot use judge feedback from earlier ones.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl
from .prompts import (
    GENERATION_MODES,
    build_video_generation_prompt,
    formality_participant_names,
    qa_formality_errors,
)
from .qwen3vl_runner import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MODEL_ID,
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_P,
    DEFAULT_VIDEO_FPS,
    GENERATOR_DECODING_MODES,
    OpenAICompatibleLocalRunner,
    make_runner,
)
from .schema import extract_json_object, validate_qa_item
from .video_qa_loop import (
    DEFAULT_JUDGE_MODEL_ID,
    DEFAULT_QUESTION_TYPES,
    JUDGE_VIDEO_SOURCES,
    QUESTION_TYPES,
    StreamingJsonlRows,
    TEMPORAL_REASONING_MODE,
    build_review_from_gates,
    complete_generator_metadata,
    generator_decode_config,
    human_audit_packet,
    media_for_clips,
    packet_with_temporal_reasoning_media,
    parse_question_types,
    run_parallel_review_judges,
    video_evidence_for_packet,
)


STAGED_SCHEMA_VERSION = "deferred_judging_v1"
DEFERRED_RETRY_FEEDBACK = (
    "Judging is deferred until the generator model is unloaded. Produce a distinct "
    "alternative candidate for the same evidence packet. Independently check JSON, "
    "answer choices, cross-user necessity, grounding, and answerability."
)
BACKEND_CHOICES = (
    "transformers-local",
    "transformers-local-memory-safe",
    "openai-compatible-local",
)


class StagedOpenAICompatibleLocalRunner(OpenAICompatibleLocalRunner):
    """Sidecar-only vLLM request settings; shared runner behavior stays unchanged."""

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        max_new_tokens: int,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        allow_video_input: bool,
        disable_thinking: bool,
        video_fps: float,
    ) -> None:
        super().__init__(
            model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            timeout=int(os.getenv("STAGED_VLM_TIMEOUT_SECONDS", "7200")),
            allow_video_input=allow_video_input,
        )
        if max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")
        self.max_image_pixels = int(max_image_pixels)
        self.disable_thinking = disable_thinking
        self.video_fps = float(video_fps)

    def _extra_request_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mm_processor_kwargs": {
                "fps": self.video_fps,
                "do_sample_frames": True,
                "max_pixels": self.max_image_pixels,
            }
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload


def _make_staged_runner(
    backend: str,
    *,
    model_id: str,
    base_url: str,
    max_new_tokens: int,
    max_image_pixels: int,
    dtype: str,
    allow_cpu: bool,
    allow_openai_video_input: bool,
    disable_thinking: bool,
    video_fps: float,
) -> Any:
    if backend == "openai-compatible-local":
        return StagedOpenAICompatibleLocalRunner(
            model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            allow_video_input=allow_openai_video_input,
            disable_thinking=disable_thinking,
            video_fps=video_fps,
        )
    return make_runner(
        backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        allow_openai_video_input=allow_openai_video_input,
        disable_thinking=disable_thinking,
        video_fps=video_fps,
    )


def _validate_common_settings(
    *,
    generation_mode: str,
    question_types: tuple[str, ...],
) -> None:
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    if not question_types:
        raise ValueError("question_types must include at least one question type")
    unknown = [value for value in question_types if value not in QUESTION_TYPES]
    if unknown:
        raise ValueError(f"unknown question_types: {unknown}")


def _scheduled_question_type(
    packet_index: int,
    question_types: tuple[str, ...],
) -> str:
    return question_types[packet_index % len(question_types)]


def _pending_review() -> dict[str, Any]:
    return {
        "status": "pending_deferred_judging",
        "review_passed": False,
        "judger": {},
        "answerability": {},
        "schema_validation": {"passed": False, "errors": []},
        "final_decision": {
            "accepted": False,
            "rejection_stage": "pending_deferred_judging",
            "reason": "The generator phase completed; the judge model has not run yet.",
        },
    }


def _normalized_candidate_qa(
    qa: dict[str, Any],
    *,
    packet: dict[str, Any],
    packet_index: int,
    question_type: str,
    generation_mode: str,
    attempt: int,
    model_id: str,
    decode_config: dict[str, Any],
    attempt_trace: dict[str, Any],
) -> dict[str, Any]:
    evidence_id = packet.get("evidence_id")
    generated_qa_id = qa.get("qa_id")
    qa["qa_id"] = f"QA_{packet_index + 1:03d}_{evidence_id}_A{attempt}"
    if generated_qa_id and generated_qa_id != qa["qa_id"]:
        qa["generator_qa_id"] = generated_qa_id
    qa["evidence_id"] = evidence_id
    qa["question_type"] = question_type
    qa["generation_mode"] = generation_mode
    qa["generator_decode"] = decode_config
    qa["required_users"] = packet.get("required_users", qa.get("required_users", []))
    qa["model_id"] = model_id
    qa["source_urls"] = packet.get("source_urls", {})
    qa["video_evidence"] = video_evidence_for_packet(packet)
    qa.setdefault("referred_timestamps", [])
    qa["human_audit"] = human_audit_packet(packet)
    qa["attempt_count"] = attempt
    qa["generation_trace"] = [attempt_trace]
    qa["review"] = _pending_review()
    qa.pop("judge_feedback", None)
    qa.pop("answerability_eval", None)
    complete_generator_metadata(qa, packet=packet, question_type=question_type)
    return qa


def generate_video_qa_candidates(
    *,
    evidence_path: str | Path,
    candidates_path: str | Path,
    prompts_path: str | Path | None,
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    target_count: int = 20,
    max_attempts: int = 3,
    max_new_tokens: int = 4096,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    allow_openai_video_input: bool = False,
    disable_thinking: bool = True,
    video_fps: float = DEFAULT_VIDEO_FPS,
    generation_mode: str = "baseline",
    question_types: tuple[str, ...] | None = None,
    resume: bool = False,
    generator_decode_mode: str = "sampling",
    generator_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    generator_top_p: float = DEFAULT_SAMPLING_TOP_P,
    generator_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Generate and persist all attempts without instantiating a judge runner."""

    active_question_types = tuple(question_types or DEFAULT_QUESTION_TYPES)
    _validate_common_settings(
        generation_mode=generation_mode,
        question_types=active_question_types,
    )
    if generator_decode_mode not in GENERATOR_DECODING_MODES:
        raise ValueError(f"unknown generator_decode_mode: {generator_decode_mode}")
    if target_count < 1:
        raise ValueError("target_count must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    runner = _make_staged_runner(
        backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        allow_openai_video_input=allow_openai_video_input,
        disable_thinking=disable_thinking,
        video_fps=video_fps,
    )
    decode_config = generator_decode_config(
        generator_decode_mode=generator_decode_mode,
        generator_temperature=generator_temperature,
        generator_top_p=generator_top_p,
        generator_top_k=generator_top_k,
    )
    candidates = StreamingJsonlRows(candidates_path, reset=not resume)
    prompts = StreamingJsonlRows(prompts_path, reset=not resume)
    if resume:
        candidates.load_existing()
        prompts.load_existing()
    completed = {
        (str(row.get("evidence_id") or ""), int(row.get("attempt") or 0))
        for row in candidates
    }
    existing_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        existing_by_evidence[str(row.get("evidence_id") or "")].append(row)

    print(
        "staged_generator_config "
        f"backend={backend} model={runner.model_id} target_packets={target_count} "
        f"attempts_per_packet={max_attempts} deferred_judge_feedback=true",
        flush=True,
    )
    for packet_index, source_packet in enumerate(iter_jsonl(evidence_path)):
        if packet_index >= target_count:
            break
        packet = source_packet
        if generation_mode == TEMPORAL_REASONING_MODE:
            packet = packet_with_temporal_reasoning_media(packet)
        evidence_id = str(packet.get("evidence_id") or "")
        question_type = _scheduled_question_type(packet_index, active_question_types)
        image_paths, video_paths = media_for_clips(
            packet.get("clips", []),
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role="generator",
        )
        prior_rows = sorted(
            existing_by_evidence.get(evidence_id, []),
            key=lambda row: int(row.get("attempt") or 0),
        )
        previous_generation = next(
            (
                str(row.get("raw_output"))
                for row in reversed(prior_rows)
                if row.get("raw_output") is not None
            ),
            None,
        )
        for attempt in range(1, max_attempts + 1):
            if (evidence_id, attempt) in completed:
                print(
                    f"staged_generation_resume_skip evidence_id={evidence_id} attempt={attempt}",
                    flush=True,
                )
                continue
            feedback = DEFERRED_RETRY_FEEDBACK if attempt > 1 else None
            prompt = build_video_generation_prompt(
                packet,
                question_type,
                feedback=feedback,
                generation_mode=generation_mode,
                previous_generation=previous_generation if attempt > 1 else None,
            )
            prompt_row = {
                "stage": "staged_generation",
                "staged_schema_version": STAGED_SCHEMA_VERSION,
                "evidence_id": packet.get("evidence_id"),
                "packet_index": packet_index,
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": attempt,
                "prompt": prompt,
                "feedback_in": feedback,
                "image_paths": image_paths,
                "video_paths": video_paths,
                "generator_decode": decode_config,
            }
            prompts.append(prompt_row)
            print(
                "qa_stage_start "
                f"stage=staged_generation evidence_id={evidence_id} attempt={attempt} "
                f"images={len(image_paths)} videos={len(video_paths)}",
                flush=True,
            )
            if generator_decode_mode == "sampling":
                raw_generation = runner.generate(
                    prompt,
                    image_paths=image_paths,
                    video_paths=video_paths,
                    decoding_mode=generator_decode_mode,
                    temperature=generator_temperature,
                    top_p=generator_top_p,
                    top_k=generator_top_k,
                )
            else:
                raw_generation = runner.generate(
                    prompt,
                    image_paths=image_paths,
                    video_paths=video_paths,
                )
            print(
                f"qa_stage_done stage=staged_generation evidence_id={evidence_id} attempt={attempt}",
                flush=True,
            )
            previous_generation = str(raw_generation)
            attempt_trace: dict[str, Any] = {
                "evidence_id": packet.get("evidence_id"),
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": attempt,
                "feedback_in": feedback,
                "previous_generation_in": (
                    "present" if attempt > 1 and previous_generation else None
                ),
                "media": {
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                    "media_role": "generator",
                    "human_audit": human_audit_packet(packet),
                },
                "generation": {"prompt": prompt, "raw_output": raw_generation},
                "generator_decode": decode_config,
                "judge": {},
                "answerability": {},
                "result": {"status": "pending_deferred_judging"},
            }
            candidate_row: dict[str, Any] = {
                "staged_schema_version": STAGED_SCHEMA_VERSION,
                "evidence_id": packet.get("evidence_id"),
                "packet_index": packet_index,
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": attempt,
                "generator_model_id": runner.model_id,
                "generator_decode": decode_config,
                "raw_output": raw_generation,
                "generator_parse_passed": False,
                "qa": None,
                "generation_trace": attempt_trace,
            }
            try:
                qa = extract_json_object(raw_generation)
            except Exception as exc:
                candidate_row["generator_parse_error"] = str(exc)
                attempt_trace["result"] = {
                    "accepted": False,
                    "stage": "generator_parse",
                    "reason": str(exc),
                }
            else:
                qa = _normalized_candidate_qa(
                    qa,
                    packet=packet,
                    packet_index=packet_index,
                    question_type=question_type,
                    generation_mode=generation_mode,
                    attempt=attempt,
                    model_id=runner.model_id,
                    decode_config=decode_config,
                    attempt_trace=attempt_trace,
                )
                schema_errors = qa_formality_errors(
                    qa,
                    validate_qa_item(qa),
                    participant_names=formality_participant_names(packet, qa),
                )
                candidate_row["generator_parse_passed"] = True
                candidate_row["pre_judge_schema_errors"] = schema_errors
                candidate_row["qa"] = qa
                attempt_trace["qa_id"] = qa.get("qa_id")
                attempt_trace["schema_errors"] = schema_errors
            candidates.append(candidate_row)
            completed.add((evidence_id, attempt))
            existing_by_evidence[evidence_id].append(candidate_row)
    return candidates


def _group_candidates(
    candidates_path: str | Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(candidates_path):
        evidence_id = str(row.get("evidence_id") or "")
        grouped.setdefault(evidence_id, []).append(row)
    return [
        (
            evidence_id,
            sorted(rows, key=lambda row: int(row.get("attempt") or 0)),
        )
        for evidence_id, rows in grouped.items()
    ]


def judge_video_qa_candidates(
    *,
    evidence_path: str | Path,
    candidates_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path,
    judged_candidates_path: str | Path,
    prompts_path: str | Path | None,
    intermediate_path: str | Path | None,
    backend: str,
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    target_count: int | None = None,
    max_new_tokens: int = 4096,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    allow_openai_video_input: bool = False,
    disable_thinking: bool = True,
    video_fps: float = DEFAULT_VIDEO_FPS,
    judge_video_source: str = "full",
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Judge every candidate after the generator server has been unloaded."""

    if judge_video_source not in JUDGE_VIDEO_SOURCES:
        raise ValueError(
            f"unknown judge_video_source {judge_video_source!r}; "
            f"expected one of {JUDGE_VIDEO_SOURCES}"
        )
    runner = _make_staged_runner(
        backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        allow_openai_video_input=allow_openai_video_input,
        disable_thinking=disable_thinking,
        video_fps=video_fps,
    )
    evidence_by_id = {
        str(packet.get("evidence_id") or ""): packet
        for packet in iter_jsonl(evidence_path)
    }
    candidate_groups = _group_candidates(candidates_path)
    if target_count is not None:
        candidate_groups = candidate_groups[:target_count]

    accepted = StreamingJsonlRows(output_path, reset=not resume)
    rejected = StreamingJsonlRows(rejected_path, reset=not resume)
    judged_candidates = StreamingJsonlRows(judged_candidates_path, reset=not resume)
    prompts = StreamingJsonlRows(prompts_path, reset=not resume)
    intermediate = StreamingJsonlRows(intermediate_path, reset=not resume)
    if resume:
        accepted.load_existing()
        rejected.load_existing()
        judged_candidates.load_existing()
        prompts.load_existing()
        intermediate.load_existing()
    processed_evidence_ids = {
        str(row.get("evidence_id") or "") for row in [*accepted, *rejected]
    }
    existing_judgements = {
        (
            str(row.get("evidence_id") or ""),
            int(row.get("attempt") or 0),
        ): row
        for row in judged_candidates
    }
    print(
        "staged_judge_config "
        f"backend={backend} model={runner.model_id} candidate_packets={len(candidate_groups)} "
        "selection=earliest_passing_attempt judge_every_parseable_attempt=true",
        flush=True,
    )

    for evidence_id, rows in candidate_groups:
        if resume and evidence_id in processed_evidence_ids:
            missing_attempts = [
                int(row.get("attempt") or 0)
                for row in rows
                if (evidence_id, int(row.get("attempt") or 0))
                not in existing_judgements
            ]
            if missing_attempts:
                raise ValueError(
                    f"inconsistent staged judge resume for {evidence_id}: final packet "
                    f"exists but judged attempts are missing: {missing_attempts}"
                )
            print(f"staged_judge_resume_skip evidence_id={evidence_id}", flush=True)
            continue
        if evidence_id not in evidence_by_id:
            raise ValueError(f"candidate evidence_id is missing from evidence file: {evidence_id}")
        packet = evidence_by_id[evidence_id]
        generation_mode = str(rows[0].get("generation_mode") or "baseline")
        if generation_mode == TEMPORAL_REASONING_MODE:
            packet = packet_with_temporal_reasoning_media(packet)
        clips = packet.get("clips", [])
        full_image_paths, full_video_paths = media_for_clips(
            clips,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role=judge_video_source,
        )
        packet_traces: list[dict[str, Any]] = []
        packet_results: list[dict[str, Any]] = []
        passing: list[dict[str, Any]] = []

        for candidate in rows:
            attempt = int(candidate.get("attempt") or 0)
            existing = existing_judgements.get((evidence_id, attempt))
            if resume and existing is not None:
                if existing.get("judge_model_id") != runner.model_id:
                    raise ValueError(
                        f"staged judge resume model mismatch for {evidence_id} attempt {attempt}: "
                        f"existing={existing.get('judge_model_id')} active={runner.model_id}"
                    )
                reused = deepcopy(existing)
                reused_trace = deepcopy(reused.get("generation_trace") or {})
                packet_results.append(reused)
                packet_traces.append(reused_trace)
                if reused.get("accepted") is True:
                    passing.append(reused)
                print(
                    f"staged_judge_attempt_resume_skip evidence_id={evidence_id} attempt={attempt}",
                    flush=True,
                )
                continue
            trace = deepcopy(candidate.get("generation_trace") or {})
            if not isinstance(trace, dict):
                trace = {"attempt": attempt, "generation": {}, "result": {}}
            trace["attempt"] = attempt
            qa_source = candidate.get("qa")
            if not candidate.get("generator_parse_passed") or not isinstance(qa_source, dict):
                reason = str(candidate.get("generator_parse_error") or "generator output was not parseable")
                trace["result"] = {
                    "accepted": False,
                    "stage": "generator_parse",
                    "reason": reason,
                }
                judged_row = {
                    **candidate,
                    "judge_model_id": runner.model_id,
                    "judged": False,
                    "accepted": False,
                    "reason": reason,
                    "generation_trace": trace,
                }
                judged_candidates.append(judged_row)
                existing_judgements[(evidence_id, attempt)] = judged_row
                packet_results.append(judged_row)
                packet_traces.append(trace)
                continue

            qa = deepcopy(qa_source)
            qa["review_model_id"] = runner.model_id
            qa["review_model_ids"] = {
                "qa_formality": runner.model_id,
                "evidence_groundedness": runner.model_id,
                "answerability": runner.model_id,
            }
            qa["judge_video_source"] = judge_video_source
            qa["generation_trace"] = [trace]
            schema_errors = qa_formality_errors(
                qa,
                validate_qa_item(qa),
                participant_names=formality_participant_names(packet, qa),
            )
            judge, answerability, judge_trace = run_parallel_review_judges(
                qa_item=qa,
                packet=packet,
                schema_errors=schema_errors,
                runner=runner,
                qa_formality_runner=runner,
                media_backend=backend,
                allow_openai_video_input=allow_openai_video_input,
                prompt_rows=prompts,
                full_image_paths=full_image_paths,
                full_video_paths=full_video_paths,
                attempt=attempt,
                judge_media_role=judge_video_source,
                include_generator_rationale=False,
                pass_fail_only=True,
                quality_quota_counts=None,
                record_decision_entropy=False,
            )
            trace["judge"] = judge_trace
            trace["answerability"] = answerability
            judge_passed = judge.get("gate", {}).get("passed") is True
            reason = str(
                judge.get("feedback_to_generator")
                or judge.get("gate", {}).get("reason")
                or ("passed all gates" if judge_passed else "Judger rejected the question.")
            )
            if judge_passed:
                qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=[],
                    accepted=True,
                    final_reason="passed all gates",
                )
                strict_errors = validate_qa_item(qa, strict_review=True)
                if strict_errors:
                    judge_passed = False
                    reason = "Strict validation errors: " + "; ".join(strict_errors)
                    qa["review"] = build_review_from_gates(
                        judge=judge,
                        answerability=answerability,
                        schema_errors=strict_errors,
                        accepted=False,
                        rejection_stage="schema",
                        final_reason=reason,
                    )
            else:
                qa["review"] = build_review_from_gates(
                    judge=judge,
                    answerability=answerability,
                    schema_errors=schema_errors,
                    accepted=False,
                    rejection_stage="judger",
                    final_reason=reason,
                )
            trace["result"] = {"accepted": judge_passed, "reason": reason}
            qa["generation_trace"] = [trace]
            judged_row = {
                **candidate,
                "judge_model_id": runner.model_id,
                "judged": True,
                "accepted": judge_passed,
                "reason": reason,
                "qa": qa,
                "generation_trace": trace,
            }
            judged_candidates.append(judged_row)
            existing_judgements[(evidence_id, attempt)] = judged_row
            packet_results.append(judged_row)
            packet_traces.append(trace)
            if judge_passed:
                passing.append(judged_row)

        if passing:
            winner = min(passing, key=lambda row: int(row.get("attempt") or 0))
            selected_qa = deepcopy(winner["qa"])
            selected_qa["generation_trace"] = packet_traces
            selected_qa["attempt_count"] = int(winner.get("attempt") or 0)
            selected_qa["staged_selection"] = {
                "schema_version": STAGED_SCHEMA_VERSION,
                "policy": "earliest_passing_attempt",
                "selected_attempt": selected_qa["attempt_count"],
                "attempts_generated": len(rows),
                "attempts_judged": sum(result.get("judged") is True for result in packet_results),
                "deferred_judge_feedback": True,
            }
            final_errors = validate_qa_item(selected_qa, strict_review=True)
            if final_errors:
                raise ValueError(
                    f"selected staged QA failed final validation for {evidence_id}: {final_errors}"
                )
            accepted.append(selected_qa)
            intermediate.append(
                {
                    "evidence_id": evidence_id,
                    "status": "accepted",
                    "selected_attempt": selected_qa["attempt_count"],
                    "attempts": packet_results,
                }
            )
        else:
            rejected_row = {
                "evidence_id": evidence_id,
                "question_type": rows[0].get("question_type"),
                "generation_mode": generation_mode,
                "judge_video_source": judge_video_source,
                "judge_model_id": runner.model_id,
                "attempts": packet_results,
                "generation_trace": packet_traces,
                "human_audit": human_audit_packet(packet),
                "staged_selection": {
                    "schema_version": STAGED_SCHEMA_VERSION,
                    "policy": "earliest_passing_attempt",
                    "selected_attempt": None,
                    "attempts_generated": len(rows),
                    "attempts_judged": sum(result.get("judged") is True for result in packet_results),
                    "deferred_judge_feedback": True,
                },
            }
            rejected.append(rejected_row)
            intermediate.append({**rejected_row, "status": "rejected"})
    return accepted


def _add_runner_args(parser: argparse.ArgumentParser, *, default_model: str) -> None:
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="openai-compatible-local")
    parser.add_argument("--model-id", default=default_model)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-image-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-openai-video-input", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-pass video QA generation and judging")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate all attempts without loading a judge")
    generate.add_argument("--evidence", required=True)
    generate.add_argument("--candidates-output", required=True)
    generate.add_argument("--prompts-output")
    generate.add_argument("--target-count", type=int, default=20)
    generate.add_argument("--max-attempts", type=int, default=3)
    generate.add_argument("--generation-mode", choices=GENERATION_MODES, default="baseline")
    generate.add_argument("--question-types", default="commonality,difference")
    generate.add_argument("--generator-decode-mode", choices=GENERATOR_DECODING_MODES, default="sampling")
    generate.add_argument("--generator-temperature", type=float, default=DEFAULT_SAMPLING_TEMPERATURE)
    generate.add_argument("--generator-top-p", type=float, default=DEFAULT_SAMPLING_TOP_P)
    generate.add_argument("--generator-top-k", type=int)
    _add_runner_args(generate, default_model=DEFAULT_MODEL_ID)

    judge = sub.add_parser("judge", help="Judge all saved attempts and select one per packet")
    judge.add_argument("--evidence", required=True)
    judge.add_argument("--candidates", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--rejected-output", required=True)
    judge.add_argument("--judged-candidates-output", required=True)
    judge.add_argument("--prompts-output")
    judge.add_argument("--intermediate-output")
    judge.add_argument("--target-count", type=int)
    judge.add_argument("--judge-video-source", choices=JUDGE_VIDEO_SOURCES, default="full")
    _add_runner_args(judge, default_model=DEFAULT_JUDGE_MODEL_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        rows = generate_video_qa_candidates(
            evidence_path=args.evidence,
            candidates_path=args.candidates_output,
            prompts_path=args.prompts_output,
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
            disable_thinking=args.disable_thinking,
            video_fps=args.video_fps,
            generation_mode=args.generation_mode,
            question_types=parse_question_types(args.question_types),
            resume=args.resume,
            generator_decode_mode=args.generator_decode_mode,
            generator_temperature=args.generator_temperature,
            generator_top_p=args.generator_top_p,
            generator_top_k=args.generator_top_k,
        )
        print(f"generated {len(rows)} deferred-judge candidate rows")
        return 0
    rows = judge_video_qa_candidates(
        evidence_path=args.evidence,
        candidates_path=args.candidates,
        output_path=args.output,
        rejected_path=args.rejected_output,
        judged_candidates_path=args.judged_candidates_output,
        prompts_path=args.prompts_output,
        intermediate_path=args.intermediate_output,
        backend=args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        target_count=args.target_count,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        allow_openai_video_input=args.allow_openai_video_input,
        disable_thinking=args.disable_thinking,
        video_fps=args.video_fps,
        judge_video_source=args.judge_video_source,
        resume=args.resume,
    )
    print(f"accepted {len(rows)} deferred-judge video QA rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
