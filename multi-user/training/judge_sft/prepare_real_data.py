"""Join completed human labels to production generation and frame artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Iterable

from .collator import render_frame_order_blocks
from .contracts import VERDICT_ASSISTANT_PREFIX, JudgeTask, Verdict, normalize_verdict


LABEL_SCHEMA = "egolife_six_user_binary_labeling_v2"
OUTPUT_SCHEMA = "egolife_judge_sft_real_data_v2"
GENERATION_FILES = (
    "qa_mcq.jsonl",
    "qa_mcq.intermediate.jsonl",
    "loop_outcomes.jsonl",
    "video_first_prompts.jsonl",
    "run_summary.json",
)
LOCAL_REPO_PACKAGE = "_egolife_multi_user_judge_training"
FULL_JUDGE_MEDIA_MODE = "full_unpruned_sampled_frames_only"


@dataclass(frozen=True)
class AttemptRecord:
    candidate_id: str
    packet_id: str
    evidence_id: str
    asker_index: int
    asker_user: str
    attempt: int
    qa: dict[str, Any]
    schema_errors: tuple[str, ...]


def _repo_module(name: str) -> Any:
    """Import a sibling module from this exact ``multi-user`` checkout.

    The parent repository is also an ``egolife_two_user_qa`` Python package.
    Importing through that name can silently select the parent copy instead of
    this package's current prompts.  A private package alias makes resolution
    deterministic even though the directory name ``multi-user`` is not a valid
    Python package name.
    """

    package_root = Path(__file__).resolve().parents[2]
    package = sys.modules.get(LOCAL_REPO_PACKAGE)
    if package is None:
        package = types.ModuleType(LOCAL_REPO_PACKAGE)
        package.__path__ = [str(package_root)]
        sys.modules[LOCAL_REPO_PACKAGE] = package
    return importlib.import_module(f"{LOCAL_REPO_PACKAGE}.{name}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _load_full_judge_view(
    dataset_root: str | Path,
    packet_id: str,
    asker_index: int,
) -> dict[str, Any]:
    """Build the judge view directly from packet.json using all sampled frames."""

    dataset_root = Path(dataset_root).resolve()
    packet_dir = dataset_root / "packets" / packet_id
    packet = _read_json(packet_dir / "packet.json")
    users = list(packet.get("users") or [])
    if len(users) != 6:
        raise ValueError(f"{packet_id}: packet must contain exactly six users")
    if not 0 <= int(asker_index) < len(users):
        raise ValueError(f"{packet_id}: invalid asker index {asker_index}")

    asker_index = int(asker_index)
    order = [asker_index] + [index for index in range(6) if index != asker_index]
    source_by_agent_dir = {
        str(row.get("agent_dir")): row
        for row in (packet.get("source") or {}).get("users") or []
    }
    sampling = (packet.get("preprocessing") or {}).get("sampling") or {}
    sample_fps = sampling.get("fps")
    clips: list[dict[str, Any]] = []

    for position, user_index in enumerate(order):
        user = users[user_index]
        full_frames = []
        for frame in user.get("frames") or []:
            relative = Path(str(frame.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"{packet_id}: frame path is not packet-relative: {relative}"
                )
            full_frames.append(
                {**frame, "path": str((packet_dir / relative).resolve())}
            )
        source_user = source_by_agent_dir.get(str(user.get("agent_dir"))) or {}
        source_segments = list(source_user.get("segments") or [])
        source_urls = [
            str(row["video_url"])
            for row in source_segments
            if row.get("video_url")
        ]
        role = "speaker" if position == 0 else "provider"
        clips.append(
            {
                "user_index": user_index,
                "agent_dir": user.get("agent_dir"),
                "agent_id": user.get("agent_id"),
                "agent_name": user.get("agent_name"),
                "day": packet.get("day"),
                "time_token": packet.get("time_token"),
                "clip_clock": packet.get("clip_clock"),
                "duration_seconds": packet.get("duration_seconds"),
                "segment_count": len(source_segments),
                "source_video_urls": source_urls,
                "video_url": source_urls[0] if source_urls else None,
                "media_role": role,
                "frame_mode": "full_unpruned_sampled",
                "generator_media_mode": FULL_JUDGE_MEDIA_MODE,
                "force_frame_inputs": True,
                "is_pruned": False,
                "original_frame_count": len(full_frames),
                "retained_frame_count": len(full_frames),
                "frames": full_frames,
                "full_frames": full_frames,
                "context_sampling": {
                    "policy": "complete_full_unpruned_sampled_frames",
                    "analysis_sample_fps": sample_fps,
                    "source_frame_count": len(full_frames),
                    "model_input_frame_count": len(full_frames),
                    "effective_model_input_fps": (
                        len(full_frames) / float(packet.get("duration_seconds") or 1.0)
                    ),
                },
            }
        )

    required_users = [str(clip.get("agent_name") or "") for clip in clips]
    frame_counts = [len(clip["full_frames"]) for clip in clips]
    asker = users[asker_index]
    evidence_id = f"{packet_id}__ASKER_{asker.get('agent_id') or asker_index}"
    return {
        "schema_version": packet.get("schema_version"),
        "candidate_type": "six_user_judge_full_sampled_frames",
        "evidence_id": evidence_id,
        "generation_group_id": evidence_id,
        "source_packet_id": packet_id,
        "dataset_root": str(dataset_root),
        "packet_id": packet_id,
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "clip_clock": packet.get("clip_clock"),
        "duration_seconds": packet.get("duration_seconds"),
        "selection": packet.get("selection"),
        "asker_index": asker_index,
        "asker_agent_dir": asker.get("agent_dir"),
        "asker_user": required_users[0],
        "required_users": required_users,
        "input_users": required_users,
        "speaker_user": required_users[0],
        "provider_users": required_users[1:],
        "evidence_provider_user": required_users[1],
        "evidence_provider_users": required_users[1:],
        "media_roles": {
            required_users[0]: "speaker_full_unpruned_sampled_frames",
            **{
                user: "provider_full_unpruned_sampled_frames"
                for user in required_users[1:]
            },
        },
        "source_urls": {
            str(clip.get("agent_name") or ""): list(
                clip.get("source_video_urls") or []
            )
            for clip in clips
        },
        "generator_media_mode": FULL_JUDGE_MEDIA_MODE,
        "preprocessed_generator_media_mode": FULL_JUDGE_MEDIA_MODE,
        "generator_context_budget": {
            "policy": "six_full_unpruned_sampled_frame_sets",
            "analysis_sample_fps": sample_fps,
            "source_frame_count": sum(frame_counts),
            "model_input_frame_count": sum(frame_counts),
            "per_user_source_frame_counts": frame_counts,
            "per_user_model_input_frame_counts": frame_counts,
        },
        "clips": clips,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_array(value: Any, *, field: str, location: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: {field} is not valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"{location}: {field} must be an array")
    return value


def _strict_bool(value: Any, *, field: str, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location}: {field} must be a JSON boolean")
    return value


def _nested_label(row: dict[str, Any], *path: str) -> Any:
    value: Any = row.get("labels")
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _validated_label_row(row: dict[str, Any], *, location: str) -> dict[str, Any]:
    if row.get("schema_version") != LABEL_SCHEMA:
        raise ValueError(f"{location}: unsupported label schema")
    if row.get("annotation_status") != "completed":
        raise ValueError(f"{location}: annotation is not completed")
    if row.get("candidate_skipped") is not False:
        raise ValueError(f"{location}: skipped candidates cannot be training targets")

    verdict_fields = {
        "formality_verdict": ("qa_formality", "verdict"),
        "evidence_grounding_verdict": ("evidence_groundedness", "verdict"),
        "answerability_verdict": ("answerability", "verdict"),
    }
    normalized = dict(row)
    for field, nested_path in verdict_fields.items():
        verdict = normalize_verdict(row.get(field)).value
        nested = _nested_label(row, *nested_path)
        if nested is not None and normalize_verdict(nested).value != verdict:
            raise ValueError(f"{location}: flattened and nested {field} disagree")
        normalized[field] = verdict

    asker_only = _strict_bool(
        row.get("asker_only_answerable"),
        field="asker_only_answerable",
        location=location,
    )
    all_six = _strict_bool(
        row.get("all_six_answerable"),
        field="all_six_answerable",
        location=location,
    )
    nested_asker = _nested_label(row, "answerability", "speaker_only", "answerable")
    nested_all = _nested_label(
        row, "answerability", "combined_all_six_users", "answerable"
    )
    if nested_asker is not None and nested_asker is not asker_only:
        raise ValueError(f"{location}: nested asker-only label disagrees")
    if nested_all is not None and nested_all is not all_six:
        raise ValueError(f"{location}: nested all-six label disagrees")
    gate = (not asker_only) and all_six
    if (normalized["answerability_verdict"] == "pass") is not gate:
        raise ValueError(f"{location}: aggregate answerability gate is inconsistent")

    normalized["asker_only_answerable"] = asker_only
    normalized["all_six_answerable"] = all_six
    normalized["options"] = [
        str(value) for value in _json_array(row.get("options"), field="options", location=location)
    ]
    normalized["required_users"] = [
        str(value)
        for value in _json_array(
            row.get("required_users"), field="required_users", location=location
        )
    ]
    normalized["source_video_urls"] = [
        str(value)
        for value in _json_array(
            row.get("source_video_urls"), field="source_video_urls", location=location
        )
    ]
    return normalized


def load_labels(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    seen: set[str] = set()
    fingerprints: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"label file is missing: {resolved}")
        checksums[str(resolved)] = _sha256(resolved)
        for line_number, raw in enumerate(_read_jsonl(resolved), start=1):
            row = _validated_label_row(raw, location=f"{resolved}:{line_number}")
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not candidate_id:
                raise ValueError(f"{resolved}:{line_number}: candidate_id is missing")
            if candidate_id in seen:
                raise ValueError(f"duplicate human label for candidate_id={candidate_id}")
            seen.add(candidate_id)
            fingerprints.add(str(row.get("dataset_fingerprint") or ""))
            rows.append(row)
    if not rows:
        raise ValueError("no completed human labels were loaded")
    if len(fingerprints) != 1 or "" in fingerprints:
        raise ValueError(f"label files have incompatible dataset fingerprints: {fingerprints}")
    return rows, checksums


def load_generation(
    generation_root: Path,
) -> tuple[dict[str, AttemptRecord], set[str], dict[str, Any]]:
    generation_root = generation_root.resolve()
    paths = {name: generation_root / name for name in GENERATION_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("generation artifacts are missing: " + ", ".join(missing))
    summary = _read_json(paths["run_summary.json"])
    if summary.get("schema_version") != "egolife_rlhf_qa_generation_v1":
        raise ValueError("generation run_summary has an unsupported schema")
    if "remaining_generation_attempt_count" not in summary:
        raise ValueError("generation run_summary omits remaining attempt count")
    if int(summary.get("remaining_generation_attempt_count") or 0) != 0:
        raise ValueError("generation run is incomplete")

    packet_ids = {
        str(row.get("source_packet_id") or "")
        for row in _read_jsonl(paths["loop_outcomes.jsonl"])
    }
    packet_ids.discard("")
    attempts: dict[str, AttemptRecord] = {}
    for row_number, loop in enumerate(
        _read_jsonl(paths["qa_mcq.intermediate.jsonl"]), start=1
    ):
        packet_id = str(loop.get("rlhf_source_packet_id") or "").strip()
        evidence_id = str(loop.get("evidence_id") or "").strip()
        asker_index = int(loop.get("rlhf_asker_index", -1))
        asker_user = str(loop.get("rlhf_asker_user") or "").strip()
        if not packet_id or not evidence_id or not 0 <= asker_index < 6:
            raise ValueError(f"intermediate row {row_number} lacks provenance")
        raw_attempts = loop.get("attempts")
        if not isinstance(raw_attempts, list):
            raise ValueError(f"intermediate row {row_number} has no attempts array")
        for fallback, attempt_row in enumerate(raw_attempts, start=1):
            if not isinstance(attempt_row, dict):
                raise ValueError(f"{evidence_id}: attempt is not an object")
            attempt = int(attempt_row.get("attempt") or fallback)
            generation = attempt_row.get("generation")
            qa = generation.get("parsed_qa") if isinstance(generation, dict) else None
            if not isinstance(qa, dict):
                continue
            candidate_id = f"{evidence_id}::attempt_{attempt:02d}"
            if candidate_id in attempts:
                raise ValueError(f"duplicate trajectory candidate_id={candidate_id}")
            attempts[candidate_id] = AttemptRecord(
                candidate_id=candidate_id,
                packet_id=packet_id,
                evidence_id=evidence_id,
                asker_index=asker_index,
                asker_user=asker_user,
                attempt=attempt,
                qa=qa,
                schema_errors=tuple(
                    str(error) for error in (attempt_row.get("schema_errors") or [])
                ),
            )
    if not attempts:
        raise ValueError("generation trajectory contains no parseable QA attempts")
    provenance = {
        "generation_root": str(generation_root),
        "run_summary": summary,
        "sha256": {name: _sha256(path) for name, path in paths.items()},
        "trajectory_parseable_attempts": len(attempts),
    }
    return attempts, packet_ids, provenance


def _qa_from_label(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": str(label.get("question") or ""),
        "options": list(label["options"]),
        "correct": str(label.get("correct") or ""),
        "answer": str(label.get("answer") or ""),
        "required_users": list(label["required_users"]),
    }


def _validate_qa_join(label: dict[str, Any], attempt: AttemptRecord) -> None:
    expected = _qa_from_label(label)
    for field, value in expected.items():
        if attempt.qa.get(field) != value:
            raise ValueError(
                f"{attempt.candidate_id}: human label and generation trajectory "
                f"disagree on {field}"
            )
    source_qa_id = str(label.get("source_qa_id") or "").strip()
    if attempt.qa.get("qa_id") and str(attempt.qa["qa_id"]) != source_qa_id:
        raise ValueError(f"{attempt.candidate_id}: source_qa_id does not match trajectory")
    if str(label.get("source_evidence_id") or "") != attempt.evidence_id:
        raise ValueError(f"{attempt.candidate_id}: source_evidence_id mismatch")
    if str(label.get("evidence_id") or "") != attempt.packet_id:
        raise ValueError(f"{attempt.candidate_id}: packet/evidence_id mismatch")
    if int(label.get("generation_attempt") or 0) != attempt.attempt:
        raise ValueError(f"{attempt.candidate_id}: generation_attempt mismatch")
    if str(label.get("asker_user") or "") != attempt.asker_user:
        raise ValueError(f"{attempt.candidate_id}: asker_user mismatch")


def _frame_fields(view: dict[str, Any], *, speaker_only: bool) -> dict[str, Any]:
    clips = list(view.get("clips") or [])
    selected = clips[:1] if speaker_only else clips
    labels = []
    indices = []
    for offset, clip in enumerate(selected):
        role = "speaker" if offset == 0 else f"provider_{offset}"
        labels.append(f"{role}: {clip.get('agent_name')}")
        indices.append(int(clip["user_index"]))
    return {
        "frame_packet": str(
            (Path(view["dataset_root"]) / "packets" / view["packet_id"]).resolve()
        ),
        "frame_user_indices": indices,
        "frame_order": labels,
    }


def _validate_source_video_provenance(
    actual_urls: Iterable[str],
    expected_urls: Iterable[str],
    *,
    candidate_id: str,
) -> None:
    """Require identical source videos without treating display order as semantic."""

    actual = Counter(str(url) for url in actual_urls)
    expected = Counter(str(url) for url in expected_urls)
    if actual == expected:
        return
    missing = list((expected - actual).elements())
    unexpected = list((actual - expected).elements())
    raise ValueError(
        f"{candidate_id}: source video URL set mismatch "
        f"expected_count={sum(expected.values())} actual_count={sum(actual.values())} "
        f"missing_count={len(missing)} unexpected_count={len(unexpected)} "
        f"first_missing={missing[0] if missing else None!r} "
        f"first_unexpected={unexpected[0] if unexpected else None!r}"
    )


def _validate_view(
    view: dict[str, Any],
    *,
    label: dict[str, Any],
    verify_media: bool,
) -> None:
    if float(view.get("duration_seconds") or 0.0) != 600.0:
        raise ValueError(f"{view.get('packet_id')}: expected 600-second source duration")
    if view.get("required_users") != label["required_users"]:
        raise ValueError(f"{label['candidate_id']}: required_users mismatch packet view")
    clips = list(view.get("clips") or [])
    if len(clips) != 6:
        raise ValueError(f"{label['candidate_id']}: packet view must contain six users")
    for clip in clips:
        frames = list(clip.get("full_frames") or [])
        if len(frames) != 300:
            raise ValueError(f"{label['candidate_id']}: each user must have 300 frames")
        if [int(frame.get("frame_index", -1)) for frame in frames] != list(range(300)):
            raise ValueError(f"{label['candidate_id']}: frame order is not 0..299")
        if verify_media:
            for frame in frames:
                path = Path(str(frame.get("path") or ""))
                if not path.is_file() or path.stat().st_size <= 0:
                    raise FileNotFoundError(f"sampled frame is missing or empty: {path}")
    urls = [
        url
        for user in view["required_users"]
        for url in (view.get("source_urls") or {}).get(user, [])
    ]
    _validate_source_video_provenance(
        urls,
        label["source_video_urls"],
        candidate_id=str(label["candidate_id"]),
    )


def _record(
    *,
    label: dict[str, Any],
    attempt: AttemptRecord,
    task: JudgeTask,
    prompt: str,
    verdict: str,
    suffix: str,
    frame_fields: dict[str, Any] | None = None,
    condition_type: str | None = None,
) -> dict[str, Any]:
    result = {
        "example_id": f"{attempt.candidate_id}::{suffix}",
        "group_id": attempt.packet_id,
        "task": task.value,
        "prompt": prompt,
        "verdict": verdict,
        "provenance": {
            "candidate_id": attempt.candidate_id,
            "source_qa_id": label.get("source_qa_id"),
            "source_evidence_id": attempt.evidence_id,
            "generation_attempt": attempt.attempt,
            "assignment_id": label.get("assignment_id"),
            "reviewer_id": label.get("reviewer_id"),
        },
    }
    if frame_fields:
        result.update(frame_fields)
    if condition_type:
        result["condition_type"] = condition_type
    return result


def build_records(
    *,
    labels: list[dict[str, Any]],
    attempts: dict[str, AttemptRecord],
    generation_packet_ids: set[str],
    dataset_root: Path,
    verify_media: bool,
    contradiction_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = _repo_module("prompts")
    view_cache: dict[tuple[str, int], dict[str, Any]] = {}
    validated_views: set[tuple[str, int]] = set()
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}

    for label in sorted(labels, key=lambda value: str(value["candidate_id"])):
        candidate_id = str(label["candidate_id"])
        attempt = attempts.get(candidate_id)
        if attempt is None:
            raise ValueError(f"human label has no matching generation attempt: {candidate_id}")
        _validate_qa_join(label, attempt)
        if attempt.packet_id not in generation_packet_ids:
            raise ValueError(f"{candidate_id}: packet is missing from loop outcomes")
        key = (attempt.packet_id, attempt.asker_index)
        if key not in view_cache:
            view_cache[key] = _load_full_judge_view(
                dataset_root, attempt.packet_id, attempt.asker_index
            )
        view = view_cache[key]
        if view.get("evidence_id") != attempt.evidence_id:
            raise ValueError(f"{candidate_id}: reconstructed evidence_id mismatch")
        _validate_view(
            view,
            label=label,
            verify_media=verify_media and key not in validated_views,
        )
        validated_views.add(key)

        qa = dict(attempt.qa)
        formality_errors = prompts.qa_formality_errors(
            qa,
            list(attempt.schema_errors),
            participant_names=prompts.formality_participant_names(view, qa),
        )
        formality_prompt = prompts.build_qa_formality_judge_prompt(
            qa, view, schema_errors=list(attempt.schema_errors)
        )
        if formality_errors and label["formality_verdict"] == "pass":
            exclusion = {
                "candidate_id": candidate_id,
                "task": JudgeTask.FORMALITY.value,
                "human_verdict": "pass",
                "reason": "prompt deterministic branch is FAIL",
                "deterministic_errors": formality_errors,
            }
            if contradiction_policy == "error":
                raise ValueError(json.dumps(exclusion, ensure_ascii=False))
            exclusions.append(exclusion)
        else:
            records.append(
                _record(
                    label=label,
                    attempt=attempt,
                    task=JudgeTask.FORMALITY,
                    prompt=formality_prompt,
                    verdict=label["formality_verdict"],
                    suffix="formality",
                )
            )

        all_six_fields = _frame_fields(view, speaker_only=False)
        records.append(
            _record(
                label=label,
                attempt=attempt,
                task=JudgeTask.GROUNDEDNESS,
                prompt=prompts.build_evidence_groundedness_judge_prompt(qa, view),
                verdict=label["evidence_grounding_verdict"],
                suffix="groundedness",
                frame_fields=all_six_fields,
            )
        )
        speaker_condition = {
            "condition_id": f"speaker_only::{view['required_users'][0]}",
            "condition_type": "speaker_only",
            "users": [view["required_users"][0]],
        }
        all_six_condition = {
            "condition_id": "combined_all_six_users::" + "+".join(view["required_users"]),
            "condition_type": "combined_all_six_users",
            "users": list(view["required_users"]),
        }
        records.append(
            _record(
                label=label,
                attempt=attempt,
                task=JudgeTask.ANSWERABILITY,
                prompt=prompts.build_answerability_prompt(qa, speaker_condition),
                verdict=(
                    Verdict.PASS.value
                    if label["asker_only_answerable"]
                    else Verdict.FAIL.value
                ),
                suffix="answerability::speaker-only",
                frame_fields=_frame_fields(view, speaker_only=True),
                condition_type="speaker_only",
            )
        )
        records.append(
            _record(
                label=label,
                attempt=attempt,
                task=JudgeTask.ANSWERABILITY,
                prompt=prompts.build_answerability_prompt(qa, all_six_condition),
                verdict=(
                    Verdict.PASS.value
                    if label["all_six_answerable"]
                    else Verdict.FAIL.value
                ),
                suffix="answerability::all-six",
                frame_fields=all_six_fields,
                condition_type="combined_all_six_users",
            )
        )

    for record in records:
        key = record["task"]
        if record.get("condition_type"):
            key += f"::{record['condition_type']}"
        if key not in snapshots:
            frame_order = list(record.get("frame_order") or [])
            snapshots[key] = {
                "example_id": record["example_id"],
                "frame_order": frame_order,
                "prompt_builder_output": record["prompt"],
                "model_visible_prompt": (
                    render_frame_order_blocks(
                        [(label, 300) for label in frame_order]
                    )
                    + record["prompt"]
                ),
                "assistant_prefix": VERDICT_ASSISTANT_PREFIX,
            }
    return records, exclusions, list(snapshots.values())


def _record_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    tasks: dict[str, Any] = {}
    for task in JudgeTask:
        selected = [row for row in rows if row["task"] == task.value]
        tasks[task.value] = {
            "examples": len(selected),
            "pass": sum(row["verdict"] == "pass" for row in selected),
            "fail": sum(row["verdict"] == "fail" for row in selected),
            "groups": len({row["group_id"] for row in selected}),
        }
    return {
        "examples": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "tasks": tasks,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    labels, label_checksums = load_labels(args.labels)
    if args.expected_candidates is not None and len(labels) != args.expected_candidates:
        raise ValueError(
            f"expected {args.expected_candidates} labeled candidates, got {len(labels)}"
        )
    packet_count = len({str(row["evidence_id"]) for row in labels})
    if args.expected_packets is not None and packet_count != args.expected_packets:
        raise ValueError(f"expected {args.expected_packets} labeled packets, got {packet_count}")
    attempts, generation_packets, generation = load_generation(args.generation_root)
    records, exclusions, prompt_snapshots = build_records(
        labels=labels,
        attempts=attempts,
        generation_packet_ids=generation_packets,
        dataset_root=args.dataset_root.resolve(),
        verify_media=not args.skip_media_file_check,
        contradiction_policy=args.contradiction_policy,
    )
    train = sorted(records, key=lambda row: row["example_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "train.jsonl", train)
    _write_jsonl(args.output_dir / "excluded_task_rows.jsonl", exclusions)
    (args.output_dir / "prompt_snapshots.json").write_text(
        json.dumps(prompt_snapshots, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "passed",
        "label_files_sha256": label_checksums,
        "dataset_fingerprint": labels[0]["dataset_fingerprint"],
        "labeled_candidates": len(labels),
        "labeled_packets": packet_count,
        "answerability_target_contract": (
            "two independent PASS/FAIL rows from asker_only_answerable and "
            "all_six_answerable; aggregate answerability_verdict is consistency-only"
        ),
        "media_contract": {
            "dataset_root": str(args.dataset_root.resolve()),
            "representation": "packet-owned JPEG frames",
            "duration_seconds_per_user": 600,
            "source_fps": 0.5,
            "frames_per_user": 300,
            "groundedness_users": 6,
            "speaker_only_users": 1,
            "all_six_users": 6,
        },
        "generation": generation,
        "excluded_task_rows": len(exclusions),
        "exclusion_policy": args.contradiction_policy,
        "partition": {
            "strategy": "all_supplied_rows_for_training",
            "internal_validation": None,
            "checkpoint_selection": (
                "future separately labeled standalone validation and test sets"
            ),
            "train": _record_summary(train),
        },
    }
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int)
    parser.add_argument("--expected-packets", type=int)
    parser.add_argument(
        "--contradiction-policy",
        choices=("exclude-task", "error"),
        default="exclude-task",
    )
    parser.add_argument(
        "--skip-media-file-check",
        action="store_true",
        help="Testing/debug only; production sbatches never set this flag.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
