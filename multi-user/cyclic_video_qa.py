"""Feedback-driven video-QA rounds for mutually exclusive local model residency.

Each generation round produces one attempt for every non-terminal evidence packet.
After the generator is removed from VRAM, a judge round evaluates those attempts.
Rejected feedback and the previous raw generation become inputs to the next round.
The Slurm launcher owns model loading/unloading; this module owns durable state.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import threading
from pathlib import Path
from typing import Any

from .cluster_member_frame_sidecar import GENERATOR_MEDIA_MODE
from .io_utils import iter_jsonl, write_json, write_jsonl
from .prompts import (
    GENERATION_MODES,
    build_video_generation_prompt,
    formality_participant_names,
    qa_formality_errors,
)
from .qwen3vl_runner import (
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_SAMPLING_TEMPERATURE,
    DEFAULT_SAMPLING_TOP_P,
    DEFAULT_VIDEO_FPS,
    GENERATOR_DECODING_MODES,
)
from .schema import extract_json_object, validate_qa_item
from .staged_video_qa import (
    BACKEND_CHOICES,
    StagedOpenAICompatibleLocalRunner,
    _make_staged_runner,
    _normalized_candidate_qa,
)
from .video_qa_loop import (
    DEFAULT_JUDGE_MODEL_ID,
    DEFAULT_QUESTION_TYPES,
    JUDGE_VIDEO_SOURCES,
    QUESTION_TYPES,
    StreamingJsonlRows,
    TEMPORAL_REASONING_MODE,
    build_review_from_gates,
    generator_decode_config,
    human_audit_packet,
    media_for_clips,
    packet_with_temporal_reasoning_media,
    parse_question_types,
    run_parallel_review_judges,
)


CYCLIC_SCHEMA_VERSION = "feedback_rounds_v1"


def _read_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    active = Path(path)
    return list(iter_jsonl(active)) if active.exists() else []


def _terminal_evidence_ids(
    accepted_path: str | Path,
    rejected_path: str | Path,
) -> set[str]:
    return {
        str(row.get("evidence_id") or "")
        for row in [*_read_rows(accepted_path), *_read_rows(rejected_path)]
    }


def _row_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row.get("evidence_id") or ""), int(row.get("attempt") or 0)): row
        for row in rows
    }


def verify_cyclic_media_routing(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    generator_prompts_path: str | Path | None = None,
    judge_prompts_path: str | Path | None = None,
    expected_packet_count: int | None = None,
    expected_judge_media_role: str = "full",
    max_images_per_prompt: int | None = None,
) -> dict[str, Any]:
    """Prove retained-frame generation and full-video visual judging."""

    evidence = list(iter_jsonl(evidence_path))
    if expected_packet_count is not None and len(evidence) != expected_packet_count:
        raise ValueError(
            f"expected {expected_packet_count} evidence packets, found {len(evidence)}"
        )
    if expected_judge_media_role != "full":
        raise ValueError(
            "the cyclic cluster-member sidecar requires full-video visual judges"
        )

    expected_generator_images: dict[str, list[str]] = {}
    expected_full_videos: dict[str, list[str]] = {}
    all_full_videos: set[str] = set()
    maximum_generator_frame_count = 0
    for packet in evidence:
        evidence_id = str(packet.get("evidence_id") or "")
        clips = packet.get("clips")
        if (
            not evidence_id
            or packet.get("generator_media_mode") != GENERATOR_MEDIA_MODE
            or not isinstance(clips, list)
            or len(clips) != 2
        ):
            raise ValueError(
                f"{evidence_id or '<missing>'}: invalid retained-cluster-frame packet"
            )
        image_paths: list[str] = []
        for clip in clips:
            if (
                clip.get("generator_media_mode") != GENERATOR_MEDIA_MODE
                or clip.get("force_frame_inputs") is not True
                or clip.get("local_video")
                or clip.get("generator_local_video")
            ):
                raise ValueError(
                    f"{evidence_id}: generator clip is not strictly frame-only"
                )
            frames = clip.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"{evidence_id}: generator clip has no retained frames")
            for frame in frames:
                path = str(frame.get("path") or "") if isinstance(frame, dict) else ""
                if not path or not Path(path).is_file():
                    raise FileNotFoundError(
                        f"{evidence_id}: retained generator frame is missing: {path!r}"
                    )
                image_paths.append(path)
        maximum_generator_frame_count = max(
            maximum_generator_frame_count,
            len(image_paths),
        )
        if max_images_per_prompt is not None and len(image_paths) > max_images_per_prompt:
            raise ValueError(
                f"{evidence_id}: {len(image_paths)} retained frames exceed the vLLM "
                f"per-prompt limit of {max_images_per_prompt}"
            )

        full_images, full_videos = media_for_clips(
            clips,
            backend="openai-compatible-local",
            allow_openai_video_input=True,
            media_role="full",
        )
        if full_images or len(full_videos) != len(clips):
            raise ValueError(
                f"{evidence_id}: full-video judge media is incomplete or image-routed"
            )
        expected_generator_images[evidence_id] = image_paths
        expected_full_videos[evidence_id] = full_videos
        all_full_videos.update(full_videos)

    generation_rows = _read_rows(generator_prompts_path)
    cyclic_generation_rows = [
        row for row in generation_rows if row.get("stage") == "cyclic_generation"
    ]
    observed_generation_ids: set[str] = set()
    for row in cyclic_generation_rows:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id not in expected_generator_images:
            raise ValueError(
                f"generator prompt references unknown evidence packet {evidence_id!r}"
            )
        observed_generation_ids.add(evidence_id)
        if list(row.get("image_paths") or []) != expected_generator_images[evidence_id]:
            raise ValueError(
                f"{evidence_id}: generator prompt did not use the complete ordered "
                "retained-frame list"
            )
        if row.get("video_paths"):
            raise ValueError(f"{evidence_id}: generator unexpectedly received a video")
    if generator_prompts_path:
        missing = sorted(set(expected_generator_images) - observed_generation_ids)
        if missing:
            raise ValueError(
                "packets missing cyclic generator prompts: " + ", ".join(missing[:5])
            )

    judge_rows = _read_rows(judge_prompts_path)
    evidence_judge_rows = [
        row for row in judge_rows if row.get("stage") == "evidence_groundedness_judge"
    ]
    answerability_rows = [
        row for row in judge_rows if row.get("stage") == "answerability"
    ]
    for row in evidence_judge_rows:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id not in expected_full_videos:
            raise ValueError(
                f"visual judge prompt references unknown evidence packet {evidence_id!r}"
            )
        if row.get("media_role") != expected_judge_media_role:
            raise ValueError(
                f"{evidence_id}: visual judge media role is {row.get('media_role')!r}"
            )
        if row.get("image_paths"):
            raise ValueError(
                f"{evidence_id}: visual judge unexpectedly received generator frames"
            )
        if list(row.get("video_paths") or []) != expected_full_videos[evidence_id]:
            raise ValueError(
                f"{evidence_id}: evidence judge did not receive both full original videos"
            )
    for row in answerability_rows:
        videos = list(row.get("video_paths") or [])
        if row.get("media_role") != expected_judge_media_role:
            raise ValueError(
                f"answerability media role is {row.get('media_role')!r}, expected "
                f"{expected_judge_media_role!r}"
            )
        if row.get("image_paths"):
            raise ValueError(
                "answerability judge unexpectedly received generator frame images"
            )
        if not videos or any(path not in all_full_videos for path in videos):
            raise ValueError(
                "answerability judge did not receive only full original videos"
            )

    report = {
        "verified": True,
        "cyclic_schema_version": CYCLIC_SCHEMA_VERSION,
        "evidence_packet_count": len(evidence),
        "generator_media_mode": GENERATOR_MEDIA_MODE,
        "generator_frame_only": True,
        "generator_prompt_count": len(cyclic_generation_rows),
        "generator_video_count": 0,
        "maximum_generator_frame_count": maximum_generator_frame_count,
        "max_images_per_prompt": max_images_per_prompt,
        "judge_media_role": expected_judge_media_role,
        "judge_full_video_only": True,
        "evidence_judge_prompt_count": len(evidence_judge_rows),
        "answerability_prompt_count": len(answerability_rows),
    }
    write_json(output_path, report)
    return report


def _validate_generation_settings(
    *,
    attempt: int,
    max_attempts: int,
    generation_mode: str,
    question_types: tuple[str, ...],
    generator_decode_mode: str,
) -> None:
    if not 1 <= attempt <= max_attempts:
        raise ValueError(f"attempt must be in [1, {max_attempts}], got {attempt}")
    if generation_mode not in GENERATION_MODES:
        raise ValueError(f"unknown generation_mode: {generation_mode}")
    if generator_decode_mode not in GENERATOR_DECODING_MODES:
        raise ValueError(f"unknown generator_decode_mode: {generator_decode_mode}")
    if not question_types:
        raise ValueError("question_types must include at least one question type")
    unknown = [value for value in question_types if value not in QUESTION_TYPES]
    if unknown:
        raise ValueError(f"unknown question_types: {unknown}")


class RoundRobinJudgeRunner:
    """Thread-safe request distributor across identical local judge replicas."""

    def __init__(self, runners: list[Any]) -> None:
        if not runners:
            raise ValueError("at least one judge runner is required")
        model_ids = {str(runner.model_id) for runner in runners}
        if len(model_ids) != 1:
            raise ValueError(f"judge replicas must serve the same model: {sorted(model_ids)}")
        self.runners = runners
        self.model_id = runners[0].model_id
        self.supports_choice_logits = all(
            getattr(runner, "supports_choice_logits", False) for runner in runners
        )
        self._next_index = 0
        self._lock = threading.Lock()

    def _next(self) -> Any:
        with self._lock:
            runner = self.runners[self._next_index]
            self._next_index = (self._next_index + 1) % len(self.runners)
            return runner

    def generate(self, *args: Any, **kwargs: Any) -> str:
        return self._next().generate(*args, **kwargs)

    def generate_with_choice_logits(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._next().generate_with_choice_logits(*args, **kwargs)


class SerializedJudgeRunner:
    """Serialize calls into one in-process Transformers model replica.

    ``run_parallel_review_judges`` launches the three logical review branches in
    threads.  A local Transformers model should not execute three independent
    ``generate`` calls concurrently on one 80-GB device.  Four Slurm worker
    processes still judge four packets concurrently; this lock only serializes
    the branches assigned to the same GPU/model instance.
    """

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.model_id = runner.model_id
        self.supports_choice_logits = getattr(
            runner,
            "supports_choice_logits",
            False,
        )
        self._lock = threading.Lock()

    def generate(self, *args: Any, **kwargs: Any) -> str:
        with self._lock:
            return self.runner.generate(*args, **kwargs)

    def generate_with_choice_logits(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            return self.runner.generate_with_choice_logits(*args, **kwargs)


def _make_replicated_judge_runner(
    *,
    backend: str,
    model_id: str,
    base_urls: tuple[str, ...],
    max_new_tokens: int,
    max_image_pixels: int,
    dtype: str,
    allow_cpu: bool,
    allow_openai_video_input: bool,
    disable_thinking: bool,
    video_fps: float,
) -> Any:
    if backend != "openai-compatible-local":
        if len(base_urls) != 1:
            raise ValueError("multiple base URLs are supported only for openai-compatible-local")
        runner = _make_staged_runner(
            backend,
            model_id=model_id,
            base_url=base_urls[0],
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            dtype=dtype,
            allow_cpu=allow_cpu,
            allow_openai_video_input=allow_openai_video_input,
            disable_thinking=disable_thinking,
            video_fps=video_fps,
        )
        if backend.startswith("transformers-local"):
            return SerializedJudgeRunner(runner)
        return runner
    replicas = [
        StagedOpenAICompatibleLocalRunner(
            model_id,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
            max_image_pixels=max_image_pixels,
            allow_video_input=allow_openai_video_input,
            disable_thinking=disable_thinking,
            video_fps=video_fps,
        )
        for base_url in base_urls
    ]
    return RoundRobinJudgeRunner(replicas)


def generate_video_qa_round(
    *,
    evidence_path: str | Path,
    candidates_path: str | Path,
    judged_candidates_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    prompts_path: str | Path | None,
    attempt: int,
    max_attempts: int,
    backend: str,
    model_id: str,
    base_url: str,
    target_count: int,
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
    """Generate one feedback-conditioned attempt for every pending packet."""

    active_question_types = tuple(question_types or DEFAULT_QUESTION_TYPES)
    _validate_generation_settings(
        attempt=attempt,
        max_attempts=max_attempts,
        generation_mode=generation_mode,
        question_types=active_question_types,
        generator_decode_mode=generator_decode_mode,
    )
    if attempt > 1 and not resume:
        raise ValueError("generation rounds after attempt 1 require resume=True")
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
    candidates = StreamingJsonlRows(candidates_path, reset=not resume)
    prompts = StreamingJsonlRows(prompts_path, reset=not resume)
    if resume:
        candidates.load_existing()
        prompts.load_existing()
    candidate_by_key = _row_index(list(candidates))
    judgement_by_key = (
        _row_index(_read_rows(judged_candidates_path)) if resume else {}
    )
    terminal_ids = (
        _terminal_evidence_ids(accepted_path, rejected_path) if resume else set()
    )
    decode_config = generator_decode_config(
        generator_decode_mode=generator_decode_mode,
        generator_temperature=generator_temperature,
        generator_top_p=generator_top_p,
        generator_top_k=generator_top_k,
    )
    generated_this_round: list[dict[str, Any]] = []
    print(
        "cyclic_generator_config "
        f"attempt={attempt}/{max_attempts} model={runner.model_id} "
        f"terminal_packets={len(terminal_ids)}",
        flush=True,
    )

    for packet_index, source_packet in enumerate(iter_jsonl(evidence_path)):
        if packet_index >= target_count:
            break
        packet = source_packet
        evidence_id = str(packet.get("evidence_id") or "")
        if evidence_id in terminal_ids:
            print(
                f"cyclic_generation_terminal_skip evidence_id={evidence_id} attempt={attempt}",
                flush=True,
            )
            continue
        key = (evidence_id, attempt)
        if key in candidate_by_key:
            if not resume:
                raise ValueError(f"candidate already exists without resume: {key}")
            print(
                f"cyclic_generation_resume_skip evidence_id={evidence_id} attempt={attempt}",
                flush=True,
            )
            continue
        previous_candidate = candidate_by_key.get((evidence_id, attempt - 1))
        previous_judgement = judgement_by_key.get((evidence_id, attempt - 1))
        if attempt > 1 and (previous_candidate is None or previous_judgement is None):
            raise ValueError(
                f"cannot generate attempt {attempt} for {evidence_id}: "
                "the previous candidate or judgement is missing"
            )
        if attempt > 1 and previous_judgement.get("accepted") is True:
            raise ValueError(
                f"cannot retry accepted packet {evidence_id} from attempt {attempt - 1}"
            )
        feedback = (
            str(previous_judgement.get("reason") or "The previous attempt was rejected.")
            if previous_judgement
            else None
        )
        previous_generation = (
            str(previous_candidate.get("raw_output")) if previous_candidate else None
        )
        if generation_mode == TEMPORAL_REASONING_MODE:
            packet = packet_with_temporal_reasoning_media(packet)
        question_type = active_question_types[packet_index % len(active_question_types)]
        image_paths, video_paths = media_for_clips(
            packet.get("clips", []),
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
            media_role="generator",
        )
        prompt = build_video_generation_prompt(
            packet,
            question_type,
            feedback=feedback,
            generation_mode=generation_mode,
            previous_generation=previous_generation,
        )
        prompts.append(
            {
                "stage": "cyclic_generation",
                "cyclic_schema_version": CYCLIC_SCHEMA_VERSION,
                "evidence_id": packet.get("evidence_id"),
                "packet_index": packet_index,
                "question_type": question_type,
                "generation_mode": generation_mode,
                "attempt": attempt,
                "feedback_in": feedback,
                "previous_generation_included": previous_generation is not None,
                "prompt": prompt,
                "image_paths": image_paths,
                "video_paths": video_paths,
                "generator_decode": decode_config,
            }
        )
        print(
            "qa_stage_start "
            f"stage=cyclic_generation evidence_id={evidence_id} attempt={attempt} "
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
            f"qa_stage_done stage=cyclic_generation evidence_id={evidence_id} attempt={attempt}",
            flush=True,
        )
        trace: dict[str, Any] = {
            "evidence_id": packet.get("evidence_id"),
            "question_type": question_type,
            "generation_mode": generation_mode,
            "attempt": attempt,
            "feedback_in": feedback,
            "previous_generation_in": previous_generation,
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
            "result": {"status": "pending_round_judgement"},
        }
        candidate: dict[str, Any] = {
            "cyclic_schema_version": CYCLIC_SCHEMA_VERSION,
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
            "generation_trace": trace,
        }
        try:
            qa = extract_json_object(raw_generation)
        except Exception as exc:
            candidate["generator_parse_error"] = str(exc)
            trace["result"] = {
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
                attempt_trace=trace,
            )
            schema_errors = qa_formality_errors(
                qa,
                validate_qa_item(qa),
                participant_names=formality_participant_names(packet, qa),
            )
            candidate["generator_parse_passed"] = True
            candidate["pre_judge_schema_errors"] = schema_errors
            candidate["qa"] = qa
            trace["qa_id"] = qa.get("qa_id")
            trace["schema_errors"] = schema_errors
        candidates.append(candidate)
        candidate_by_key[key] = candidate
        generated_this_round.append(candidate)
    return generated_this_round


def _packet_history(
    *,
    evidence_id: str,
    through_attempt: int,
    candidate_by_key: dict[tuple[str, int], dict[str, Any]],
    judgement_by_key: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    traces = []
    attempts = []
    for prior_attempt in range(1, through_attempt + 1):
        key = (evidence_id, prior_attempt)
        candidate = candidate_by_key.get(key)
        judgement = judgement_by_key.get(key)
        if candidate is None:
            continue
        attempts.append(judgement or candidate)
        trace_source = judgement or candidate
        trace = deepcopy(trace_source.get("generation_trace") or {})
        if isinstance(trace, dict):
            traces.append(trace)
    return traces, attempts


def _finalize_round_judgement(
    *,
    evidence_id: str,
    candidate: dict[str, Any],
    judged_row: dict[str, Any],
    active_packet: dict[str, Any],
    attempt: int,
    max_attempts: int,
    judge_video_source: str,
    judge_model_id: str,
    judge_replica_count: int,
    candidate_by_key: dict[tuple[str, int], dict[str, Any]],
    judgement_by_key: dict[tuple[str, int], dict[str, Any]],
    accepted: StreamingJsonlRows,
    rejected: StreamingJsonlRows,
    intermediate: StreamingJsonlRows,
    terminal_ids: set[str],
) -> dict[str, Any] | None:
    traces, attempt_rows = _packet_history(
        evidence_id=evidence_id,
        through_attempt=attempt,
        candidate_by_key=candidate_by_key,
        judgement_by_key=judgement_by_key,
    )
    if judged_row.get("accepted") is True:
        selected_qa = deepcopy(judged_row["qa"])
        selected_qa["generation_trace"] = traces
        selected_qa["attempt_count"] = attempt
        selected_qa["cyclic_selection"] = {
            "schema_version": CYCLIC_SCHEMA_VERSION,
            "policy": "first_passing_feedback_round",
            "selected_attempt": attempt,
            "max_attempts": max_attempts,
            "judge_replica_count": judge_replica_count,
        }
        final_errors = validate_qa_item(selected_qa, strict_review=True)
        if final_errors:
            raise ValueError(
                f"selected cyclic QA failed final validation for {evidence_id}: {final_errors}"
            )
        accepted.append(selected_qa)
        terminal_ids.add(evidence_id)
        intermediate.append(
            {
                "evidence_id": evidence_id,
                "attempt": attempt,
                "status": "accepted",
                "attempts": attempt_rows,
            }
        )
        return selected_qa
    if attempt == max_attempts:
        rejected_row = {
            "evidence_id": evidence_id,
            "question_type": candidate.get("question_type"),
            "generation_mode": candidate.get("generation_mode"),
            "judge_video_source": judge_video_source,
            "judge_model_id": judge_model_id,
            "attempts": attempt_rows,
            "generation_trace": traces,
            "human_audit": human_audit_packet(active_packet),
            "cyclic_selection": {
                "schema_version": CYCLIC_SCHEMA_VERSION,
                "policy": "first_passing_feedback_round",
                "selected_attempt": None,
                "max_attempts": max_attempts,
                "judge_replica_count": judge_replica_count,
            },
        }
        rejected.append(rejected_row)
        terminal_ids.add(evidence_id)
        intermediate.append({**rejected_row, "attempt": attempt, "status": "rejected"})
        return rejected_row
    intermediate.append(
        {
            "evidence_id": evidence_id,
            "attempt": attempt,
            "status": "retry_pending",
            "reason": judged_row.get("reason"),
            "judgement": judged_row,
        }
    )
    return None


def judge_video_qa_round(
    *,
    evidence_path: str | Path,
    candidates_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    judged_candidates_path: str | Path,
    prompts_path: str | Path | None,
    intermediate_path: str | Path | None,
    attempt: int,
    max_attempts: int,
    backend: str,
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    base_urls: tuple[str, ...] = ("http://127.0.0.1:8100/v1",),
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
    shard_index: int = 0,
    shard_count: int = 1,
    judge_replica_count: int | None = None,
    state_accepted_path: str | Path | None = None,
    state_rejected_path: str | Path | None = None,
    state_judged_candidates_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Judge one attempt round and finalize passes or last-attempt failures."""

    if not 1 <= attempt <= max_attempts:
        raise ValueError(f"attempt must be in [1, {max_attempts}], got {attempt}")
    if attempt > 1 and not resume:
        raise ValueError("judge rounds after attempt 1 require resume=True")
    if judge_video_source not in JUDGE_VIDEO_SOURCES:
        raise ValueError(f"unknown judge_video_source: {judge_video_source}")
    if not base_urls:
        raise ValueError("at least one judge base URL is required")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    active_replica_count = (
        judge_replica_count if judge_replica_count is not None else len(base_urls)
    )
    if active_replica_count <= 0:
        raise ValueError("judge_replica_count must be positive")
    state_paths = (
        state_accepted_path,
        state_rejected_path,
        state_judged_candidates_path,
    )
    isolated_worker_state = any(path is not None for path in state_paths)
    if isolated_worker_state and not all(path is not None for path in state_paths):
        raise ValueError(
            "isolated judge workers require all three state input paths"
        )
    runner = _make_replicated_judge_runner(
        backend=backend,
        model_id=model_id,
        base_urls=base_urls,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
        allow_openai_video_input=allow_openai_video_input,
        disable_thinking=disable_thinking,
        video_fps=video_fps,
    )
    evidence_rows = list(iter_jsonl(evidence_path))
    if target_count is not None:
        evidence_rows = evidence_rows[:target_count]
    evidence_rows = [
        packet
        for packet_index, packet in enumerate(evidence_rows)
        if packet_index % shard_count == shard_index
    ]
    evidence_by_id = {
        str(packet.get("evidence_id") or ""): packet for packet in evidence_rows
    }
    candidate_rows = _read_rows(candidates_path)
    candidate_by_key = _row_index(candidate_rows)

    accepted = StreamingJsonlRows(
        accepted_path,
        reset=True if isolated_worker_state else not resume,
    )
    rejected = StreamingJsonlRows(
        rejected_path,
        reset=True if isolated_worker_state else not resume,
    )
    judged = StreamingJsonlRows(
        judged_candidates_path,
        reset=True if isolated_worker_state else not resume,
    )
    prompts = StreamingJsonlRows(
        prompts_path,
        reset=True if isolated_worker_state else not resume,
    )
    intermediate = StreamingJsonlRows(
        intermediate_path,
        reset=True if isolated_worker_state else not resume,
    )
    if resume and not isolated_worker_state:
        accepted.load_existing()
        rejected.load_existing()
        judged.load_existing()
        prompts.load_existing()
        intermediate.load_existing()
    prior_accepted = _read_rows(state_accepted_path) if isolated_worker_state else []
    prior_rejected = _read_rows(state_rejected_path) if isolated_worker_state else []
    prior_judged = (
        _read_rows(state_judged_candidates_path) if isolated_worker_state else []
    )
    terminal_ids = {
        str(row.get("evidence_id") or "")
        for row in [*prior_accepted, *prior_rejected, *accepted, *rejected]
    }
    judgement_by_key = _row_index([*prior_judged, *judged])
    finalized_this_round: list[dict[str, Any]] = []
    print(
        "cyclic_judge_config "
        f"attempt={attempt}/{max_attempts} model={runner.model_id} "
        f"replicas={active_replica_count} shard={shard_index}/{shard_count} "
        f"terminal_packets={len(terminal_ids)}",
        flush=True,
    )

    for evidence_id, packet in evidence_by_id.items():
        if evidence_id in terminal_ids:
            continue
        key = (evidence_id, attempt)
        candidate = candidate_by_key.get(key)
        if candidate is None:
            raise ValueError(
                f"pending packet {evidence_id} has no candidate for attempt {attempt}"
            )
        generation_mode = str(candidate.get("generation_mode") or "baseline")
        active_packet = packet_with_temporal_reasoning_media(packet) if (
            generation_mode == TEMPORAL_REASONING_MODE
        ) else packet
        if key in judgement_by_key:
            if not resume:
                raise ValueError(f"judgement already exists without resume: {key}")
            existing_judge_model = str(
                judgement_by_key[key].get("judge_model_id") or ""
            )
            if existing_judge_model != runner.model_id:
                raise ValueError(
                    "resume judge model mismatch for "
                    f"{key}: existing={existing_judge_model!r} "
                    f"requested={runner.model_id!r}"
                )
            print(
                f"cyclic_judge_reuse evidence_id={evidence_id} attempt={attempt}",
                flush=True,
            )
            finalized = _finalize_round_judgement(
                evidence_id=evidence_id,
                candidate=candidate,
                judged_row=judgement_by_key[key],
                active_packet=active_packet,
                attempt=attempt,
                max_attempts=max_attempts,
                judge_video_source=judge_video_source,
                judge_model_id=runner.model_id,
                judge_replica_count=active_replica_count,
                candidate_by_key=candidate_by_key,
                judgement_by_key=judgement_by_key,
                accepted=accepted,
                rejected=rejected,
                intermediate=intermediate,
                terminal_ids=terminal_ids,
            )
            if finalized is not None:
                finalized_this_round.append(finalized)
            continue
        trace = deepcopy(candidate.get("generation_trace") or {})
        if not isinstance(trace, dict):
            trace = {"attempt": attempt, "generation": {}, "result": {}}
        qa_source = candidate.get("qa")
        if not candidate.get("generator_parse_passed") or not isinstance(qa_source, dict):
            reason = str(
                candidate.get("generator_parse_error")
                or "Generator output was not valid JSON."
            )
            trace["result"] = {
                "accepted": False,
                "stage": "generator_parse",
                "reason": reason,
            }
            judged_row = {
                **candidate,
                "judge_model_id": runner.model_id,
                "judge_replica_count": active_replica_count,
                "judged": False,
                "accepted": False,
                "reason": reason,
                "generation_trace": trace,
            }
        else:
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
                participant_names=formality_participant_names(active_packet, qa),
            )
            full_image_paths, full_video_paths = media_for_clips(
                active_packet.get("clips", []),
                backend=backend,
                allow_openai_video_input=allow_openai_video_input,
                media_role=judge_video_source,
            )
            judge_result, answerability, judge_trace = run_parallel_review_judges(
                qa_item=qa,
                packet=active_packet,
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
            passed = judge_result.get("gate", {}).get("passed") is True
            reason = str(
                judge_result.get("feedback_to_generator")
                or judge_result.get("gate", {}).get("reason")
                or ("passed all gates" if passed else "Judger rejected the question.")
            )
            if passed:
                qa["review"] = build_review_from_gates(
                    judge=judge_result,
                    answerability=answerability,
                    schema_errors=[],
                    accepted=True,
                    final_reason="passed all gates",
                )
                strict_errors = validate_qa_item(qa, strict_review=True)
                if strict_errors:
                    passed = False
                    reason = "Strict validation errors: " + "; ".join(strict_errors)
                    qa["review"] = build_review_from_gates(
                        judge=judge_result,
                        answerability=answerability,
                        schema_errors=strict_errors,
                        accepted=False,
                        rejection_stage="schema",
                        final_reason=reason,
                    )
            else:
                qa["review"] = build_review_from_gates(
                    judge=judge_result,
                    answerability=answerability,
                    schema_errors=schema_errors,
                    accepted=False,
                    rejection_stage="judger",
                    final_reason=reason,
                )
            trace["result"] = {"accepted": passed, "reason": reason}
            qa["generation_trace"] = [trace]
            judged_row = {
                **candidate,
                "judge_model_id": runner.model_id,
                "judge_replica_count": active_replica_count,
                "judged": True,
                "accepted": passed,
                "reason": reason,
                "qa": qa,
                "generation_trace": trace,
            }

        judged.append(judged_row)
        judgement_by_key[key] = judged_row
        finalized = _finalize_round_judgement(
            evidence_id=evidence_id,
            candidate=candidate,
            judged_row=judged_row,
            active_packet=active_packet,
            attempt=attempt,
            max_attempts=max_attempts,
            judge_video_source=judge_video_source,
            judge_model_id=runner.model_id,
            judge_replica_count=active_replica_count,
            candidate_by_key=candidate_by_key,
            judgement_by_key=judgement_by_key,
            accepted=accepted,
            rejected=rejected,
            intermediate=intermediate,
            terminal_ids=terminal_ids,
        )
        if finalized is not None:
            finalized_this_round.append(finalized)
    return finalized_this_round


def _merge_unique_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    label: str,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if any(value is None for value in key):
            raise ValueError(f"{label} row is missing merge key {key_fields}: {row}")
        previous = by_key.get(key)
        if previous is not None:
            if previous != row:
                raise ValueError(f"conflicting {label} rows for key {key}")
            continue
        by_key[key] = row
        merged.append(row)
    return merged


def _deduplicate_exact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        marker = repr(row)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(row)
    return merged


def merge_local_judge_shards(
    *,
    candidates_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    judged_candidates_path: str | Path,
    prompts_path: str | Path,
    intermediate_path: str | Path,
    worker_output_root: str | Path,
    attempt: int,
    replica_count: int,
    model_id: str,
) -> dict[str, Any]:
    """Validate and atomically append successful local-judge shard outputs."""

    if replica_count <= 0:
        raise ValueError("replica_count must be positive")
    worker_root = Path(worker_output_root)
    worker_files = {
        "accepted": "accepted.jsonl",
        "rejected": "rejected.jsonl",
        "judged": "judged.jsonl",
        "prompts": "prompts.jsonl",
        "intermediate": "intermediate.jsonl",
    }
    deltas: dict[str, list[dict[str, Any]]] = {
        name: [] for name in worker_files
    }
    for worker_index in range(replica_count):
        worker_dir = worker_root / f"worker_{worker_index}"
        for name, filename in worker_files.items():
            path = worker_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"judge worker {worker_index} did not produce {path}"
                )
            deltas[name].extend(_read_rows(path))

    existing_accepted = _read_rows(accepted_path)
    existing_rejected = _read_rows(rejected_path)
    existing_judged = _read_rows(judged_candidates_path)
    existing_prompts = _read_rows(prompts_path)
    existing_intermediate = _read_rows(intermediate_path)

    accepted = _merge_unique_rows(
        [*existing_accepted, *deltas["accepted"]],
        key_fields=("evidence_id",),
        label="accepted",
    )
    rejected = _merge_unique_rows(
        [*existing_rejected, *deltas["rejected"]],
        key_fields=("evidence_id",),
        label="rejected",
    )
    accepted_ids = {str(row["evidence_id"]) for row in accepted}
    rejected_ids = {str(row["evidence_id"]) for row in rejected}
    overlap = sorted(accepted_ids & rejected_ids)
    if overlap:
        raise ValueError(
            "evidence packets cannot be both accepted and rejected: "
            + ", ".join(overlap[:5])
        )

    judged = _merge_unique_rows(
        [*existing_judged, *deltas["judged"]],
        key_fields=("evidence_id", "attempt"),
        label="judged",
    )
    for row in deltas["judged"]:
        if int(row.get("attempt") or 0) != attempt:
            raise ValueError("judge shard emitted a row for the wrong attempt")
        if str(row.get("judge_model_id") or "") != model_id:
            raise ValueError("judge shard emitted a row for the wrong model")
        if int(row.get("judge_replica_count") or 0) != replica_count:
            raise ValueError("judge shard emitted the wrong replica count")

    terminal_before = {
        str(row.get("evidence_id") or "")
        for row in [*existing_accepted, *existing_rejected]
    }
    expected_keys = {
        (str(row.get("evidence_id") or ""), attempt)
        for row in _read_rows(candidates_path)
        if int(row.get("attempt") or 0) == attempt
        and str(row.get("evidence_id") or "") not in terminal_before
    }
    observed_keys = {
        (str(row.get("evidence_id") or ""), int(row.get("attempt") or 0))
        for row in judged
        if int(row.get("attempt") or 0) == attempt
    }
    delta_keys = {
        (str(row.get("evidence_id") or ""), int(row.get("attempt") or 0))
        for row in deltas["judged"]
    }
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(delta_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "local judge shard coverage mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    prompts = _deduplicate_exact_rows([*existing_prompts, *deltas["prompts"]])
    intermediate = _deduplicate_exact_rows(
        [*existing_intermediate, *deltas["intermediate"]]
    )
    merged_outputs = {
        Path(accepted_path): accepted,
        Path(rejected_path): rejected,
        Path(judged_candidates_path): judged,
        Path(prompts_path): prompts,
        Path(intermediate_path): intermediate,
    }
    temporary_paths: dict[Path, Path] = {}
    for output_path, rows in merged_outputs.items():
        temporary = output_path.with_name(
            f".{output_path.name}.attempt{attempt}.merge.tmp"
        )
        write_jsonl(temporary, rows)
        temporary_paths[output_path] = temporary
    for output_path, temporary in temporary_paths.items():
        temporary.replace(output_path)

    report = {
        "merged": True,
        "attempt": attempt,
        "replica_count": replica_count,
        "model_id": model_id,
        "expected_judgements": len(expected_keys),
        "merged_judgements": len(deltas["judged"]),
        "accepted_delta": len(deltas["accepted"]),
        "rejected_delta": len(deltas["rejected"]),
        "prompt_delta": len(deltas["prompts"]),
        "intermediate_delta": len(deltas["intermediate"]),
    }
    write_json(worker_root / "merge_report.json", report)
    return report


def _base_urls(value: str) -> tuple[str, ...]:
    rows = tuple(part.strip().rstrip("/") for part in value.split(",") if part.strip())
    if not rows:
        raise argparse.ArgumentTypeError("base URLs must contain at least one URL")
    return rows


def _add_common_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="openai-compatible-local")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-image-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-openai-video-input", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cyclic local video-QA generation and judging")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-round")
    generate.add_argument("--evidence", required=True)
    generate.add_argument("--candidates-output", required=True)
    generate.add_argument("--judged-candidates", required=True)
    generate.add_argument("--accepted", required=True)
    generate.add_argument("--rejected", required=True)
    generate.add_argument("--prompts-output")
    generate.add_argument("--attempt", required=True, type=int)
    generate.add_argument("--max-attempts", type=int, default=3)
    generate.add_argument("--target-count", type=int, default=20)
    generate.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    generate.add_argument("--generation-mode", choices=GENERATION_MODES, default="baseline")
    generate.add_argument("--question-types", default="commonality,difference")
    generate.add_argument("--generator-decode-mode", choices=GENERATOR_DECODING_MODES, default="sampling")
    generate.add_argument("--generator-temperature", type=float, default=DEFAULT_SAMPLING_TEMPERATURE)
    generate.add_argument("--generator-top-p", type=float, default=DEFAULT_SAMPLING_TOP_P)
    generate.add_argument("--generator-top-k", type=int)
    _add_common_runner_args(generate)

    judge = sub.add_parser("judge-round")
    judge.add_argument("--evidence", required=True)
    judge.add_argument("--candidates", required=True)
    judge.add_argument("--accepted", required=True)
    judge.add_argument("--rejected", required=True)
    judge.add_argument("--judged-candidates-output", required=True)
    judge.add_argument("--prompts-output")
    judge.add_argument("--intermediate-output")
    judge.add_argument("--attempt", required=True, type=int)
    judge.add_argument("--max-attempts", type=int, default=3)
    judge.add_argument("--target-count", type=int)
    judge.add_argument("--base-urls", type=_base_urls, required=True)
    judge.add_argument("--judge-video-source", choices=JUDGE_VIDEO_SOURCES, default="full")
    judge.add_argument("--shard-index", type=int, default=0)
    judge.add_argument("--shard-count", type=int, default=1)
    judge.add_argument("--judge-replica-count", type=int)
    judge.add_argument("--state-accepted")
    judge.add_argument("--state-rejected")
    judge.add_argument("--state-judged-candidates")
    _add_common_runner_args(judge)

    merge = sub.add_parser("merge-judge-shards")
    merge.add_argument("--candidates", required=True)
    merge.add_argument("--accepted", required=True)
    merge.add_argument("--rejected", required=True)
    merge.add_argument("--judged-candidates", required=True)
    merge.add_argument("--prompts", required=True)
    merge.add_argument("--intermediate", required=True)
    merge.add_argument("--worker-output-root", required=True)
    merge.add_argument("--attempt", required=True, type=int)
    merge.add_argument("--replica-count", required=True, type=int)
    merge.add_argument("--model-id", required=True)

    verify = sub.add_parser("verify-routing")
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--generator-prompts")
    verify.add_argument("--judge-prompts")
    verify.add_argument("--output", required=True)
    verify.add_argument("--expected-packet-count", type=int)
    verify.add_argument("--max-images-per-prompt", type=int)
    verify.add_argument(
        "--expected-judge-media-role",
        choices=JUDGE_VIDEO_SOURCES,
        default="full",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-routing":
        report = verify_cyclic_media_routing(
            evidence_path=args.evidence,
            generator_prompts_path=args.generator_prompts,
            judge_prompts_path=args.judge_prompts,
            output_path=args.output,
            expected_packet_count=args.expected_packet_count,
            expected_judge_media_role=args.expected_judge_media_role,
            max_images_per_prompt=args.max_images_per_prompt,
        )
        print(
            "verified_cyclic_media_routing=true "
            f"packets={report['evidence_packet_count']} "
            f"generator_prompts={report['generator_prompt_count']} "
            f"visual_judge_prompts="
            f"{report['evidence_judge_prompt_count'] + report['answerability_prompt_count']}"
        )
        return 0
    if args.command == "generate-round":
        rows = generate_video_qa_round(
            evidence_path=args.evidence,
            candidates_path=args.candidates_output,
            judged_candidates_path=args.judged_candidates,
            accepted_path=args.accepted,
            rejected_path=args.rejected,
            prompts_path=args.prompts_output,
            attempt=args.attempt,
            max_attempts=args.max_attempts,
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
            generation_mode=args.generation_mode,
            question_types=parse_question_types(args.question_types),
            resume=args.resume,
            generator_decode_mode=args.generator_decode_mode,
            generator_temperature=args.generator_temperature,
            generator_top_p=args.generator_top_p,
            generator_top_k=args.generator_top_k,
        )
        print(f"generated_round_candidates={len(rows)} attempt={args.attempt}")
        return 0
    if args.command == "merge-judge-shards":
        report = merge_local_judge_shards(
            candidates_path=args.candidates,
            accepted_path=args.accepted,
            rejected_path=args.rejected,
            judged_candidates_path=args.judged_candidates,
            prompts_path=args.prompts,
            intermediate_path=args.intermediate,
            worker_output_root=args.worker_output_root,
            attempt=args.attempt,
            replica_count=args.replica_count,
            model_id=args.model_id,
        )
        print(
            "merged_local_judge_shards=true "
            f"attempt={report['attempt']} "
            f"judgements={report['expected_judgements']} "
            f"replicas={report['replica_count']}"
        )
        return 0
    rows = judge_video_qa_round(
        evidence_path=args.evidence,
        candidates_path=args.candidates,
        accepted_path=args.accepted,
        rejected_path=args.rejected,
        judged_candidates_path=args.judged_candidates_output,
        prompts_path=args.prompts_output,
        intermediate_path=args.intermediate_output,
        attempt=args.attempt,
        max_attempts=args.max_attempts,
        backend=args.backend,
        model_id=args.model_id,
        base_urls=args.base_urls,
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
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        judge_replica_count=args.judge_replica_count,
        state_accepted_path=args.state_accepted,
        state_rejected_path=args.state_rejected,
        state_judged_candidates_path=args.state_judged_candidates,
    )
    print(f"finalized_round_packets={len(rows)} attempt={args.attempt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
