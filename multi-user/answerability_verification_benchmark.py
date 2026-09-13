"""Benchmark one-pass external answerability verifiers against human labels."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answerability_verification import (
    MEDIA_ROLE_CHOICES,
    _source_qa_is_accepted,
    preflight_answerability_media,
    verify_answerability,
)
from .io_utils import iter_jsonl, read_json, write_json, write_jsonl
from .qwen3vl_runner import DEFAULT_OPENROUTER_BASE_URL, OPENROUTER_REASONING_EFFORTS


CONFIG_VERSION = 1
GOLD_LABELS = {"pass": True, "fail": False}
DEFAULT_MAX_NEW_TOKENS = 4096


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        raise ValueError(f"{label} must use letters, digits, dot, underscore, or hyphen: {text!r}")
    return text


def _resolved_path(base_dir: Path, value: Any, *, label: str) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        raise ValueError(f"missing {label}")
    if "$" in raw or "%" in raw:
        raise ValueError(f"{label} contains an unresolved environment variable: {raw!r}")
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_benchmark_config(
    config_path: str | Path,
    *,
    annotations_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    require_annotations: bool = True,
) -> dict[str, Any]:
    """Load and normalize a versioned benchmark config."""

    config_path = Path(config_path).resolve()
    raw = read_json(config_path)
    if not isinstance(raw, dict):
        raise ValueError("benchmark config must be a JSON object")
    if int(raw.get("version") or 0) != CONFIG_VERSION:
        raise ValueError(f"benchmark config version must be {CONFIG_VERSION}")
    base_dir = config_path.parent
    annotations_value = annotations_override or raw.get("annotations")
    output_value = output_dir_override or raw.get("output_dir")
    annotations = (
        _resolved_path(base_dir, annotations_value, label="annotations")
        if annotations_value
        else None
    )
    output_dir = _resolved_path(base_dir, output_value, label="output_dir")
    if require_annotations and annotations is None:
        raise ValueError("benchmark config is missing annotations")
    if annotations is not None and not annotations.is_file():
        raise ValueError(f"annotations file does not exist: {annotations}")

    media_role = str(raw.get("media_role") or "source_qa").strip()
    if media_role not in MEDIA_ROLE_CHOICES:
        raise ValueError(f"media_role must be one of {MEDIA_ROLE_CHOICES}")

    raw_runs = raw.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("benchmark config must contain at least one run")
    runs: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    annotation_runs: set[str] = set()
    for raw_run in raw_runs:
        if not isinstance(raw_run, dict):
            raise ValueError("each benchmark run must be an object")
        run_id = _safe_id(raw_run.get("id"), label="run id")
        annotation_run = str(raw_run.get("annotation_run") or "").strip()
        if not annotation_run:
            raise ValueError(f"{run_id}: annotation_run is required")
        if run_id in run_ids:
            raise ValueError(f"duplicate run id: {run_id}")
        if annotation_run in annotation_runs:
            raise ValueError(f"duplicate annotation_run: {annotation_run}")
        run_ids.add(run_id)
        annotation_runs.add(annotation_run)
        accepted_qa = _resolved_path(
            base_dir, raw_run.get("accepted_qa"), label=f"{run_id}.accepted_qa"
        )
        evidence = _resolved_path(base_dir, raw_run.get("evidence"), label=f"{run_id}.evidence")
        if not accepted_qa.is_file():
            raise ValueError(f"{run_id}: accepted QA file does not exist: {accepted_qa}")
        if not evidence.is_file():
            raise ValueError(f"{run_id}: evidence file does not exist: {evidence}")
        raw_include_qa_ids = raw_run.get("include_qa_ids")
        include_qa_ids: list[str] | None = None
        if raw_include_qa_ids is not None:
            if not isinstance(raw_include_qa_ids, list) or not raw_include_qa_ids:
                raise ValueError(f"{run_id}: include_qa_ids must be a non-empty list")
            include_qa_ids = [str(value or "").strip() for value in raw_include_qa_ids]
            if any(not qa_id for qa_id in include_qa_ids):
                raise ValueError(f"{run_id}: include_qa_ids contains an empty QA ID")
            if len(set(include_qa_ids)) != len(include_qa_ids):
                raise ValueError(f"{run_id}: include_qa_ids contains duplicates")
        raw_expected_counts = raw_run.get("expected_gold_label_counts")
        expected_gold_label_counts: dict[str, int] | None = None
        if raw_expected_counts is not None:
            if not isinstance(raw_expected_counts, dict):
                raise ValueError(
                    f"{run_id}: expected_gold_label_counts must be an object"
                )
            unknown_labels = sorted(set(raw_expected_counts) - {"Pass", "Fail"})
            if unknown_labels:
                raise ValueError(
                    f"{run_id}: unsupported expected gold labels: {unknown_labels}"
                )
            expected_gold_label_counts = {
                label: int(raw_expected_counts.get(label) or 0)
                for label in ("Pass", "Fail")
            }
            if any(count < 0 for count in expected_gold_label_counts.values()):
                raise ValueError(
                    f"{run_id}: expected gold label counts must be non-negative"
                )
            if include_qa_ids is not None and sum(expected_gold_label_counts.values()) != len(
                include_qa_ids
            ):
                raise ValueError(
                    f"{run_id}: expected gold counts do not sum to include_qa_ids size"
                )
        runs.append(
            {
                "id": run_id,
                "annotation_run": annotation_run,
                "accepted_qa": accepted_qa,
                "evidence": evidence,
                "include_qa_ids": include_qa_ids,
                "expected_gold_label_counts": expected_gold_label_counts,
            }
        )

    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        raise ValueError("benchmark config must contain at least one model arm")
    arms: list[dict[str, Any]] = []
    arm_ids: set[str] = set()
    for raw_arm in raw_arms:
        if not isinstance(raw_arm, dict) or raw_arm.get("enabled", True) is False:
            continue
        arm_id = _safe_id(raw_arm.get("id"), label="arm id")
        if arm_id in arm_ids:
            raise ValueError(f"duplicate arm id: {arm_id}")
        arm_ids.add(arm_id)
        model_id = str(raw_arm.get("model_id") or "").strip()
        if not model_id:
            raise ValueError(f"{arm_id}: model_id is required")
        reasoning_effort = raw_arm.get("reasoning_effort")
        if reasoning_effort is not None:
            reasoning_effort = str(reasoning_effort).strip()
            if reasoning_effort not in OPENROUTER_REASONING_EFFORTS:
                raise ValueError(
                    f"{arm_id}: unsupported reasoning_effort={reasoning_effort!r}"
                )
        max_new_tokens = int(
            raw_arm.get("max_new_tokens")
            or raw.get("max_new_tokens")
            or DEFAULT_MAX_NEW_TOKENS
        )
        if max_new_tokens <= 0:
            raise ValueError(f"{arm_id}: max_new_tokens must be positive")
        arms.append(
            {
                "id": arm_id,
                "model_id": model_id,
                "reasoning_effort": reasoning_effort,
                "max_new_tokens": max_new_tokens,
            }
        )
    if not arms:
        raise ValueError("benchmark config has no enabled model arms")

    return {
        "config_path": config_path,
        "annotations": annotations,
        "output_dir": output_dir,
        "base_url": str(raw.get("base_url") or DEFAULT_OPENROUTER_BASE_URL).rstrip("/"),
        "media_role": media_role,
        "require_all_annotations_covered": raw.get("require_all_annotations_covered", True)
        is not False,
        "allow_effort_mapping": raw.get("allow_effort_mapping", False) is True,
        "runs": runs,
        "arms": arms,
    }


def _read_annotations(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        raise ValueError("gold annotations must be a CSV export from the manual reviewer")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"run", "qa_id", "evidence_id", "review_status"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"annotation CSV is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _unique_jsonl(path: Path, *, key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        value = str(row.get(key) or "").strip()
        if not value:
            raise ValueError(f"{label} row is missing {key}: {path}")
        if value in indexed:
            raise ValueError(f"{label} has duplicate {key}={value}: {path}")
        indexed[value] = row
    if not indexed:
        raise ValueError(f"{label} contains no rows: {path}")
    return indexed


def _evidence_packet_with_clips(packet: dict[str, Any]) -> dict[str, Any]:
    """Normalize accepted-QA audit evidence into the packet shape used by judges."""

    clips = packet.get("clips")
    if isinstance(clips, list) and clips:
        return packet
    video_evidence = packet.get("video_evidence")
    if not isinstance(video_evidence, list) or not video_evidence:
        human_audit = packet.get("human_audit")
        video_evidence = (
            human_audit.get("video_evidence")
            if isinstance(human_audit, dict)
            else None
        )
    if not isinstance(video_evidence, list) or not video_evidence:
        return packet
    normalized = dict(packet)
    normalized["clips"] = [
        {
            **clip,
            "agent_name": str(
                clip.get("agent_name")
                or clip.get("user")
                or clip.get("agent_dir")
                or ""
            ),
        }
        for clip in video_evidence
        if isinstance(clip, dict)
    ]
    return normalized


def _annotation_error_tags(row: dict[str, str]) -> list[str]:
    return sorted(
        {
            tag.strip().casefold()
            for tag in str(row.get("error_tags") or "").split("|")
            if tag.strip()
        }
    )


def _annotation_manual_judge_scores(row: dict[str, str]) -> dict[str, int | None]:
    notes = str(row.get("reviewer_notes") or "")
    match = re.search(
        r"F\s*(\d+)\s*[/,\s]*[EG]\s*(\d+)\s*[/,\s]*A\s*(\d+)",
        notes,
        flags=re.IGNORECASE,
    )
    if match is None:
        return {
            "formality": None,
            "evidence_groundedness": None,
            "answerability": None,
        }
    return {
        "formality": int(match.group(1)),
        "evidence_groundedness": int(match.group(2)),
        "answerability": int(match.group(3)),
    }


def _assert_annotation_matches_qa(
    annotation: dict[str, str], qa: dict[str, Any], *, run_id: str
) -> None:
    checks = {
        "question": qa.get("question"),
        "model_correct_letter": qa.get("correct"),
        "model_answer": qa.get("answer"),
    }
    for field, qa_value in checks.items():
        annotation_value = annotation.get(field)
        if annotation_value and _normalized_text(annotation_value) != _normalized_text(qa_value):
            raise ValueError(
                f"{run_id}/{qa.get('qa_id')}: annotation {field} does not match accepted QA"
            )
    options = list(qa.get("options") or [])
    if len(options) != 5:
        raise ValueError(f"{run_id}/{qa.get('qa_id')}: accepted QA must have five options")
    for index, letter in enumerate("ABCDE"):
        annotated_option = annotation.get(f"option_{letter}")
        if annotated_option and _normalized_text(annotated_option) != _normalized_text(options[index]):
            raise ValueError(
                f"{run_id}/{qa.get('qa_id')}: annotation option_{letter} does not match"
            )


def prepare_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the gold join and all media, then write immutable selected inputs."""

    annotations_path = config.get("annotations")
    if not isinstance(annotations_path, Path):
        raise ValueError("gold-label benchmark requires an annotations CSV")
    annotations = _read_annotations(annotations_path)
    labeled: list[dict[str, Any]] = []
    ignored_counts: Counter[str] = Counter()
    seen_annotation_keys: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(annotations, start=2):
        status_text = str(row.get("review_status") or "").strip()
        status = status_text.casefold()
        if status not in GOLD_LABELS:
            ignored_counts[status_text or "blank"] += 1
            continue
        annotation_run = str(row.get("run") or "").strip()
        qa_id = str(row.get("qa_id") or "").strip()
        evidence_id = str(row.get("evidence_id") or "").strip()
        key = (annotation_run, qa_id, evidence_id)
        if not all(key):
            raise ValueError(f"annotation row {row_number} has an empty run/qa_id/evidence_id")
        if key in seen_annotation_keys:
            raise ValueError(f"duplicate labeled annotation key at row {row_number}: {key}")
        seen_annotation_keys.add(key)
        labeled.append(
            {
                "annotation_row_number": row_number,
                "annotation_run": annotation_run,
                "qa_id": qa_id,
                "evidence_id": evidence_id,
                "gold_status": status_text,
                "gold_passed": GOLD_LABELS[status],
                "error_tags": _annotation_error_tags(row),
                "reviewer_notes": str(row.get("reviewer_notes") or ""),
                "manual_judge_scores": _annotation_manual_judge_scores(row),
                "annotation": row,
            }
        )
    if not labeled:
        raise ValueError("annotation CSV contains no Pass/Fail rows")

    labels_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labeled:
        labels_by_run[label["annotation_run"]].append(label)
    configured_annotation_runs = {run["annotation_run"] for run in config["runs"]}
    uncovered_runs = sorted(set(labels_by_run) - configured_annotation_runs)
    if uncovered_runs and config["require_all_annotations_covered"]:
        raise ValueError(
            "labeled annotation runs are not configured: " + ", ".join(uncovered_runs)
        )

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected_qas"
    selected_evidence_dir = output_dir / "selected_evidence"
    run_plans: list[dict[str, Any]] = []
    gold_snapshot: list[dict[str, Any]] = []
    for run in config["runs"]:
        run_labels = labels_by_run.get(run["annotation_run"], [])
        if not run_labels:
            raise ValueError(f"{run['id']}: no Pass/Fail annotations match {run['annotation_run']!r}")
        include_qa_ids = run.get("include_qa_ids")
        if include_qa_ids is not None:
            labels_by_qa_id: dict[str, dict[str, Any]] = {}
            for label in run_labels:
                qa_id = label["qa_id"]
                if qa_id in labels_by_qa_id:
                    raise ValueError(
                        f"{run['id']}: manual annotations contain duplicate qa_id={qa_id}"
                    )
                labels_by_qa_id[qa_id] = label
            missing_selected_labels = [
                qa_id for qa_id in include_qa_ids if qa_id not in labels_by_qa_id
            ]
            if missing_selected_labels:
                raise ValueError(
                    f"{run['id']}: selected QA IDs are missing manual labels: "
                    + ", ".join(missing_selected_labels[:5])
                )
            run_labels = [labels_by_qa_id[qa_id] for qa_id in include_qa_ids]
        selected_gold_counts = Counter(label["gold_status"] for label in run_labels)
        expected_gold_counts = run.get("expected_gold_label_counts")
        if expected_gold_counts is not None:
            actual = {
                label: selected_gold_counts.get(label, 0) for label in ("Pass", "Fail")
            }
            if actual != expected_gold_counts:
                raise ValueError(
                    f"{run['id']}: selected gold label counts mismatch; "
                    f"expected={expected_gold_counts} actual={actual}"
                )
        qa_by_id = _unique_jsonl(
            run["accepted_qa"], key="qa_id", label=f"{run['id']} accepted QA"
        )
        evidence_by_id = _unique_jsonl(
            run["evidence"], key="evidence_id", label=f"{run['id']} evidence"
        )
        selected_qas: list[dict[str, Any]] = []
        run_gold: list[dict[str, Any]] = []
        for label in run_labels:
            qa = qa_by_id.get(label["qa_id"])
            if qa is None:
                raise ValueError(
                    f"{run['id']}: annotated qa_id not found in accepted QA: {label['qa_id']}"
                )
            if str(qa.get("evidence_id") or "") != label["evidence_id"]:
                raise ValueError(
                    f"{run['id']}/{label['qa_id']}: annotation evidence_id does not match"
                )
            if not _source_qa_is_accepted(qa):
                raise ValueError(f"{run['id']}/{label['qa_id']}: source QA was not accepted")
            if label["evidence_id"] not in evidence_by_id:
                raise ValueError(
                    f"{run['id']}/{label['qa_id']}: evidence packet is missing"
                )
            _assert_annotation_matches_qa(label["annotation"], qa, run_id=run["id"])
            selected_qas.append(qa)
            gold = {
                "benchmark_key": f"{run['id']}::{label['qa_id']}",
                "run_id": run["id"],
                "annotation_run": run["annotation_run"],
                "qa_id": label["qa_id"],
                "evidence_id": label["evidence_id"],
                "gold_status": label["gold_status"],
                "gold_passed": label["gold_passed"],
                "error_tags": label["error_tags"],
                "reviewer_notes": label["reviewer_notes"],
                "manual_judge_scores": label["manual_judge_scores"],
                "annotation_row_number": label["annotation_row_number"],
            }
            run_gold.append(gold)
            gold_snapshot.append(gold)
        normalized_evidence: list[dict[str, Any]] = []
        normalized_evidence_ids: set[str] = set()
        for qa in selected_qas:
            evidence_id = str(qa.get("evidence_id") or "")
            if evidence_id in normalized_evidence_ids:
                continue
            normalized_evidence_ids.add(evidence_id)
            normalized_evidence.append(
                _evidence_packet_with_clips(evidence_by_id[evidence_id])
            )
        normalized_evidence_by_id = {
            str(packet.get("evidence_id") or ""): packet
            for packet in normalized_evidence
        }
        selected_path = selected_dir / f"{run['id']}.jsonl"
        selected_evidence_path = selected_evidence_dir / f"{run['id']}.jsonl"
        write_jsonl(selected_path, selected_qas)
        write_jsonl(selected_evidence_path, normalized_evidence)
        media_preflight = preflight_answerability_media(
            accepted_rows=selected_qas,
            evidence_by_id=normalized_evidence_by_id,
            media_role=config["media_role"],
        )
        # The verifier repeats this preflight before any model call. The count is
        # retained here so validation is independently auditable.
        gold_counts = Counter(gold["gold_status"] for gold in run_gold)
        run_plans.append(
            {
                **run,
                "source_evidence": run["evidence"],
                "evidence": selected_evidence_path,
                "selected_qa": selected_path,
                "selected_qa_count": len(selected_qas),
                "gold_label_counts": dict(sorted(gold_counts.items())),
                "media_preflight": media_preflight,
                "gold": run_gold,
            }
        )

    write_jsonl(output_dir / "gold_labels.jsonl", gold_snapshot)
    plan = {
        "protocol": {
            "name": "one_pass_external_answerability_benchmark",
            "version": CONFIG_VERSION,
            "external_feedback_rounds": 1,
            "generation_loop_feedback": False,
            "conditions_per_two_user_qa": 3,
            "gold_field": "review_status",
            "gold_pass_values": ["Pass"],
            "gold_fail_values": ["Fail"],
            "unsure_or_unset_rows_scored": False,
        },
        "created_at_utc": _utc_now(),
        "config_path": str(config["config_path"]),
        "annotations": str(config["annotations"]),
        "output_dir": str(output_dir),
        "media_role": config["media_role"],
        "labeled_annotation_count": len(labeled),
        "scored_annotation_count": len(gold_snapshot),
        "ignored_annotation_counts": dict(sorted(ignored_counts.items())),
        "uncovered_annotation_runs": uncovered_runs,
        "expected_api_call_count_per_arm": len(gold_snapshot) * 3,
        "expected_api_call_count_all_arms": len(gold_snapshot) * 3 * len(config["arms"]),
        "runs": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in run.items()
                if key != "gold"
            }
            for run in run_plans
        ],
        "arms": config["arms"],
    }
    write_json(output_dir / "benchmark_plan.json", plan)
    return {**plan, "run_plans": run_plans, "gold": gold_snapshot}


def prepare_testset(config: dict[str, Any]) -> dict[str, Any]:
    """Validate every accepted QA and its media without requiring gold labels."""

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = output_dir / "selected_qas"
    selected_evidence_dir = output_dir / "selected_evidence"
    run_plans: list[dict[str, Any]] = []
    total_qas = 0
    for run in config["runs"]:
        qa_by_id = _unique_jsonl(
            run["accepted_qa"], key="qa_id", label=f"{run['id']} accepted QA"
        )
        evidence_by_id = _unique_jsonl(
            run["evidence"], key="evidence_id", label=f"{run['id']} evidence"
        )
        selected_qas = list(qa_by_id.values())
        rejected = [
            str(qa.get("qa_id") or "")
            for qa in selected_qas
            if not _source_qa_is_accepted(qa)
        ]
        if rejected:
            raise ValueError(
                f"{run['id']}: test set contains QAs that were not accepted: {rejected[:5]}"
            )
        missing_evidence = sorted(
            {
                str(qa.get("evidence_id") or "")
                for qa in selected_qas
                if str(qa.get("evidence_id") or "") not in evidence_by_id
            }
        )
        if missing_evidence:
            raise ValueError(
                f"{run['id']}: accepted QAs are missing evidence packets: "
                + ", ".join(missing_evidence[:5])
            )
        normalized_evidence: list[dict[str, Any]] = []
        normalized_evidence_ids: set[str] = set()
        for qa in selected_qas:
            evidence_id = str(qa.get("evidence_id") or "")
            if evidence_id in normalized_evidence_ids:
                continue
            normalized_evidence_ids.add(evidence_id)
            normalized_evidence.append(
                _evidence_packet_with_clips(evidence_by_id[evidence_id])
            )
        normalized_evidence_by_id = {
            str(packet.get("evidence_id") or ""): packet
            for packet in normalized_evidence
        }
        selected_path = selected_dir / f"{run['id']}.jsonl"
        selected_evidence_path = selected_evidence_dir / f"{run['id']}.jsonl"
        write_jsonl(selected_path, selected_qas)
        write_jsonl(selected_evidence_path, normalized_evidence)
        media_preflight = preflight_answerability_media(
            accepted_rows=selected_qas,
            evidence_by_id=normalized_evidence_by_id,
            media_role=config["media_role"],
        )
        run_plans.append(
            {
                **run,
                "source_evidence": run["evidence"],
                "evidence": selected_evidence_path,
                "selected_qa": selected_path,
                "selected_qa_count": len(selected_qas),
                "qa_ids": [str(qa.get("qa_id") or "") for qa in selected_qas],
                "media_preflight": media_preflight,
            }
        )
        total_qas += len(selected_qas)

    plan = {
        "protocol": {
            "name": "one_pass_external_answerability_testset",
            "version": CONFIG_VERSION,
            "external_feedback_rounds": 1,
            "generation_loop_feedback": False,
            "conditions_per_two_user_qa": 3,
            "gold_labels_used": False,
        },
        "created_at_utc": _utc_now(),
        "config_path": str(config["config_path"]),
        "output_dir": str(output_dir),
        "media_role": config["media_role"],
        "testset_qa_count": total_qas,
        "arm_count": len(config["arms"]),
        "expected_api_call_count_per_arm": total_qas * 3,
        "expected_api_call_count_all_arms": total_qas * 3 * len(config["arms"]),
        "runs": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in run.items()
                if key != "qa_ids"
            }
            for run in run_plans
        ],
        "arms": config["arms"],
    }
    write_json(output_dir / "testset_plan.json", plan)
    return {**plan, "run_plans": run_plans}


def fetch_openrouter_model_metadata(
    *, base_url: str, arms: list[dict[str, Any]], api_key: str | None = None
) -> dict[str, dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"User-Agent": "egolife-answerability-benchmark/1"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("OpenRouter model catalog returned no data list")
    by_id = {
        str(model.get("id")): model
        for model in models
        if isinstance(model, dict) and model.get("id")
    }
    missing = [arm["model_id"] for arm in arms if arm["model_id"] not in by_id]
    if missing:
        raise ValueError("OpenRouter model IDs not found: " + ", ".join(sorted(missing)))
    return {arm["model_id"]: by_id[arm["model_id"]] for arm in arms}


def validate_model_arms(
    *,
    arms: list[dict[str, Any]],
    metadata_by_id: dict[str, dict[str, Any]],
    allow_effort_mapping: bool,
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for arm in arms:
        metadata = metadata_by_id[arm["model_id"]]
        architecture = metadata.get("architecture") or {}
        modalities = list(architecture.get("input_modalities") or [])
        supported_parameters = list(metadata.get("supported_parameters") or [])
        reasoning = metadata.get("reasoning") if isinstance(metadata.get("reasoning"), dict) else {}
        effort = arm["reasoning_effort"]
        if "video" not in modalities:
            errors.append(f"{arm['id']}: model does not advertise video input")
        if effort is not None and not {"reasoning", "reasoning_effort"}.intersection(
            supported_parameters
        ):
            errors.append(f"{arm['id']}: model does not advertise reasoning control")
        supported_efforts = reasoning.get("supported_efforts")
        if (
            effort is not None
            and isinstance(supported_efforts, list)
            and effort not in supported_efforts
            and not allow_effort_mapping
        ):
            errors.append(
                f"{arm['id']}: effort={effort!r} is not an exact supported effort "
                f"{supported_efforts}; set allow_effort_mapping only if nearest-level mapping is intended"
            )
        snapshots[arm["id"]] = {
            "id": metadata.get("id"),
            "canonical_slug": metadata.get("canonical_slug"),
            "name": metadata.get("name"),
            "input_modalities": modalities,
            "context_length": metadata.get("context_length"),
            "supported_parameters": supported_parameters,
            "reasoning": reasoning,
            "pricing": metadata.get("pricing"),
        }
    if errors:
        raise ValueError("OpenRouter arm validation failed: " + "; ".join(errors))
    return snapshots


def run_benchmark(
    config: dict[str, Any],
    *,
    resume: bool,
    api_key: str | None,
    check_model_catalog: bool = True,
) -> dict[str, Any]:
    plan = prepare_benchmark(config)
    effective_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not effective_api_key:
        raise RuntimeError("benchmark run requires --api-key or OPENROUTER_API_KEY")
    if check_model_catalog:
        metadata = fetch_openrouter_model_metadata(
            base_url=config["base_url"], arms=config["arms"], api_key=effective_api_key
        )
        snapshots = validate_model_arms(
            arms=config["arms"],
            metadata_by_id=metadata,
            allow_effort_mapping=config["allow_effort_mapping"],
        )
        write_json(config["output_dir"] / "openrouter_model_snapshot.json", snapshots)

    for arm in config["arms"]:
        for run in plan["run_plans"]:
            run_output = config["output_dir"] / "arms" / arm["id"] / run["id"]
            print(
                f"benchmark_arm_run_start arm={arm['id']} run={run['id']} "
                f"model={arm['model_id']} reasoning={arm['reasoning_effort'] or 'provider_default'}",
                flush=True,
            )
            verify_answerability(
                accepted_qa_path=run["selected_qa"],
                evidence_path=run["evidence"],
                output_path=run_output / "verification.jsonl",
                prompts_path=run_output / "prompts.jsonl",
                summary_path=run_output / "summary.json",
                model_id=arm["model_id"],
                base_url=config["base_url"],
                max_new_tokens=arm["max_new_tokens"],
                reasoning_effort=arm["reasoning_effort"],
                media_role=config["media_role"],
                resume=resume,
                api_key=effective_api_key,
            )
            print(f"benchmark_arm_run_done arm={arm['id']} run={run['id']}", flush=True)
    return score_benchmark(config, prepared=plan)


def run_testset(
    config: dict[str, Any],
    *,
    resume: bool,
    api_key: str | None,
    check_model_catalog: bool = True,
) -> dict[str, Any]:
    """Run every configured arm over a test set without claiming gold accuracy."""

    plan = prepare_testset(config)
    effective_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not effective_api_key:
        raise RuntimeError("test-set run requires --api-key or OPENROUTER_API_KEY")
    if check_model_catalog:
        metadata = fetch_openrouter_model_metadata(
            base_url=config["base_url"], arms=config["arms"], api_key=effective_api_key
        )
        snapshots = validate_model_arms(
            arms=config["arms"],
            metadata_by_id=metadata,
            allow_effort_mapping=config["allow_effort_mapping"],
        )
        write_json(config["output_dir"] / "openrouter_model_snapshot.json", snapshots)

    for arm in config["arms"]:
        for run in plan["run_plans"]:
            run_output = config["output_dir"] / "arms" / arm["id"] / run["id"]
            print(
                f"testset_arm_run_start arm={arm['id']} run={run['id']} "
                f"model={arm['model_id']} reasoning={arm['reasoning_effort'] or 'provider_default'}",
                flush=True,
            )
            verify_answerability(
                accepted_qa_path=run["selected_qa"],
                evidence_path=run["evidence"],
                output_path=run_output / "verification.jsonl",
                prompts_path=run_output / "prompts.jsonl",
                summary_path=run_output / "summary.json",
                model_id=arm["model_id"],
                base_url=config["base_url"],
                max_new_tokens=arm["max_new_tokens"],
                reasoning_effort=arm["reasoning_effort"],
                media_role=config["media_role"],
                resume=resume,
                api_key=effective_api_key,
            )
            print(f"testset_arm_run_done arm={arm['id']} run={run['id']}", flush=True)
    return summarize_testset(config, prepared=plan)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["gold_passed"] and row["predicted_passed"] for row in rows)
    tn = sum(not row["gold_passed"] and not row["predicted_passed"] for row in rows)
    fp = sum(not row["gold_passed"] and row["predicted_passed"] for row in rows)
    fn = sum(row["gold_passed"] and not row["predicted_passed"] for row in rows)
    total = len(rows)
    accuracy = _safe_rate(tp + tn, total)
    pass_recall = _safe_rate(tp, tp + fn)
    failure_recall = _safe_rate(tn, tn + fp)
    pass_precision = _safe_rate(tp, tp + fp)
    failure_precision = _safe_rate(tn, tn + fn)
    balanced_accuracy = (
        (pass_recall + failure_recall) / 2
        if pass_recall is not None and failure_recall is not None
        else None
    )
    pass_f1 = (
        2 * pass_precision * pass_recall / (pass_precision + pass_recall)
        if pass_precision is not None and pass_recall is not None and pass_precision + pass_recall
        else None
    )
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else None
    return {
        "count": total,
        "gold_pass_count": tp + fn,
        "gold_fail_count": tn + fp,
        "predicted_pass_count": tp + fp,
        "predicted_fail_count": tn + fn,
        "true_pass": tp,
        "true_fail": tn,
        "false_accept": fp,
        "false_reject": fn,
        "accuracy": accuracy,
        "accuracy_wilson_95": _wilson_interval(tp + tn, total),
        "balanced_accuracy": balanced_accuracy,
        "pass_precision": pass_precision,
        "pass_recall": pass_recall,
        "pass_f1": pass_f1,
        "failure_precision": failure_precision,
        "failure_recall": failure_recall,
        "false_accept_rate": _safe_rate(fp, tn + fp),
        "false_reject_rate": _safe_rate(fn, tp + fn),
        "matthews_correlation": mcc,
    }


def _condition_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name, group_rows in (
        ("all", rows),
        ("gold_pass", [row for row in rows if row["gold_passed"]]),
        ("gold_fail", [row for row in rows if not row["gold_passed"]]),
    ):
        group: dict[str, Any] = {"count": len(group_rows)}
        for field in (
            "asker_selected_correct",
            "provider_selected_correct",
            "combined_selected_correct",
        ):
            valid = [row[field] for row in group_rows if row[field] is not None]
            group[field] = {
                "valid_count": len(valid),
                "correct_count": sum(valid),
                "correct_rate": _safe_rate(sum(valid), len(valid)),
            }
        result[group_name] = group
    return result


def _condition_choice(
    verification: dict[str, Any], *, condition_id: str, correct: str
) -> tuple[str | None, bool | None]:
    answerability = (verification.get("verification") or {}).get("answerability") or {}
    evaluations = answerability.get("evaluations") or []
    matches = [row for row in evaluations if row.get("condition_id") == condition_id]
    if len(matches) != 1:
        return None, None
    choice = str(matches[0].get("choice") or "").strip().upper()
    if choice not in "ABCDE" or len(choice) != 1:
        return None, None
    return choice, choice == correct


def _testset_prediction_row(
    *, arm: dict[str, Any], run_id: str, verification: dict[str, Any]
) -> dict[str, Any]:
    required_users = list(verification.get("required_users") or [])
    if len(required_users) < 2:
        raise ValueError(f"{run_id}/{verification.get('qa_id')}: fewer than two users")
    correct = str(verification.get("correct") or "").strip().upper()
    asker_choice, asker_correct = _condition_choice(
        verification,
        condition_id=f"single_user::{required_users[0]}",
        correct=correct,
    )
    provider_choice, provider_correct = _condition_choice(
        verification,
        condition_id=f"single_user::{required_users[1]}",
        correct=correct,
    )
    combined_choice, combined_correct = _condition_choice(
        verification,
        condition_id="combined_all_users::" + "+".join(required_users),
        correct=correct,
    )
    metadata = verification.get("verification") or {}
    answerability = metadata.get("answerability") or {}
    gate = answerability.get("gate") or {}
    qa_id = str(verification.get("qa_id") or "")
    return {
        "testset_key": f"{run_id}::{qa_id}",
        "run_id": run_id,
        "qa_id": qa_id,
        "evidence_id": verification.get("evidence_id"),
        "arm_id": arm["id"],
        "model_id": arm["model_id"],
        "reasoning_effort": arm["reasoning_effort"],
        "predicted_passed": metadata.get("passed") is True,
        "asker_user": required_users[0],
        "asker_choice": asker_choice,
        "asker_selected_correct": asker_correct,
        "provider_user": required_users[1],
        "provider_choice": provider_choice,
        "provider_selected_correct": provider_correct,
        "combined_choice": combined_choice,
        "combined_selected_correct": combined_correct,
        "gate_reason": str(gate.get("reason") or ""),
    }


def _testset_arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(row["predicted_passed"] for row in rows)
    conditions: dict[str, Any] = {}
    for field in (
        "asker_selected_correct",
        "provider_selected_correct",
        "combined_selected_correct",
    ):
        valid = [row[field] for row in rows if row[field] is not None]
        correct_count = sum(valid)
        conditions[field] = {
            "valid_count": len(valid),
            "correct_count": correct_count,
            "correct_rate": _safe_rate(correct_count, len(valid)),
        }
    return {
        "count": len(rows),
        "predicted_pass_count": passed,
        "predicted_fail_count": len(rows) - passed,
        "predicted_pass_rate": _safe_rate(passed, len(rows)),
        "condition_diagnostics": conditions,
    }


def summarize_testset(
    config: dict[str, Any], *, prepared: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Summarize model outputs and cross-model agreement without gold labels."""

    plan = prepared or prepare_testset(config)
    all_predictions: list[dict[str, Any]] = []
    predictions_by_arm: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: list[dict[str, Any]] = []
    for arm in config["arms"]:
        arm_predictions: list[dict[str, Any]] = []
        for run in plan["run_plans"]:
            verification_path = (
                config["output_dir"]
                / "arms"
                / arm["id"]
                / run["id"]
                / "verification.jsonl"
            )
            if not verification_path.is_file():
                raise ValueError(
                    f"missing verification output for arm={arm['id']} run={run['id']}: "
                    f"{verification_path}"
                )
            verification_by_id = _unique_jsonl(
                verification_path,
                key="qa_id",
                label=f"{arm['id']}/{run['id']} verification",
            )
            expected_ids = set(run["qa_ids"])
            if set(verification_by_id) != expected_ids:
                missing = sorted(expected_ids - set(verification_by_id))
                extra = sorted(set(verification_by_id) - expected_ids)
                raise ValueError(
                    f"{arm['id']}/{run['id']}: verification coverage mismatch; "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            for qa_id in run["qa_ids"]:
                verification = verification_by_id[qa_id]
                metadata = verification.get("verification") or {}
                if metadata.get("model_id") != arm["model_id"]:
                    raise ValueError(f"{run['id']}/{qa_id}: verification model mismatch")
                if metadata.get("reasoning_effort") != arm["reasoning_effort"]:
                    raise ValueError(f"{run['id']}/{qa_id}: verification reasoning mismatch")
                arm_predictions.append(
                    _testset_prediction_row(
                        arm=arm, run_id=run["id"], verification=verification
                    )
                )
        predictions_by_arm[arm["id"]] = arm_predictions
        all_predictions.extend(arm_predictions)
        arm_summaries.append(
            {
                "arm_id": arm["id"],
                "model_id": arm["model_id"],
                "reasoning_effort": arm["reasoning_effort"],
                **_testset_arm_summary(arm_predictions),
            }
        )

    predictions_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_predictions:
        predictions_by_key[row["testset_key"]].append(row)
    disagreements = [
        {
            "testset_key": key,
            "run_id": rows[0]["run_id"],
            "qa_id": rows[0]["qa_id"],
            "predictions": [
                {
                    "arm_id": row["arm_id"],
                    "predicted_passed": row["predicted_passed"],
                    "gate_reason": row["gate_reason"],
                }
                for row in rows
            ],
        }
        for key, rows in sorted(predictions_by_key.items())
        if len({row["predicted_passed"] for row in rows}) > 1
    ]
    pairwise: list[dict[str, Any]] = []
    for left, right in itertools.combinations(config["arms"], 2):
        left_rows = {row["testset_key"]: row for row in predictions_by_arm[left["id"]]}
        right_rows = {row["testset_key"]: row for row in predictions_by_arm[right["id"]]}
        keys = sorted(left_rows)
        pairwise.append(
            {
                "left_arm_id": left["id"],
                "right_arm_id": right["id"],
                "count": len(keys),
                "gate_agreement_count": sum(
                    left_rows[key]["predicted_passed"]
                    == right_rows[key]["predicted_passed"]
                    for key in keys
                ),
                "gate_disagreement_count": sum(
                    left_rows[key]["predicted_passed"]
                    != right_rows[key]["predicted_passed"]
                    for key in keys
                ),
            }
        )

    summary = {
        "protocol": plan["protocol"],
        "summarized_at_utc": _utc_now(),
        "testset_qa_count": plan["testset_qa_count"],
        "gold_accuracy_available": False,
        "recommended_arm_id": None,
        "model_disagreement_qa_count": len(disagreements),
        "arms": arm_summaries,
        "pairwise": pairwise,
    }
    output_dir: Path = config["output_dir"]
    write_json(output_dir / "testset_summary.json", summary)
    write_jsonl(output_dir / "testset_predictions.jsonl", all_predictions)
    write_jsonl(output_dir / "testset_disagreements.jsonl", disagreements)
    with (output_dir / "testset_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "arm_id",
            "model_id",
            "reasoning_effort",
            "count",
            "predicted_pass_count",
            "predicted_fail_count",
            "predicted_pass_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in arm_summaries:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return summary


def _prediction_row(
    *, arm: dict[str, Any], gold: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any]:
    required_users = list(verification.get("required_users") or [])
    if len(required_users) < 2:
        raise ValueError(f"{gold['benchmark_key']}: verification has fewer than two users")
    correct = str(verification.get("correct") or "").strip().upper()
    asker_choice, asker_correct = _condition_choice(
        verification,
        condition_id=f"single_user::{required_users[0]}",
        correct=correct,
    )
    provider_choice, provider_correct = _condition_choice(
        verification,
        condition_id=f"single_user::{required_users[1]}",
        correct=correct,
    )
    combined_id = "combined_all_users::" + "+".join(required_users)
    combined_choice, combined_correct = _condition_choice(
        verification, condition_id=combined_id, correct=correct
    )
    predicted_passed = (verification.get("verification") or {}).get("passed") is True
    return {
        **gold,
        "arm_id": arm["id"],
        "model_id": arm["model_id"],
        "reasoning_effort": arm["reasoning_effort"],
        "predicted_passed": predicted_passed,
        "prediction_correct": predicted_passed == gold["gold_passed"],
        "asker_user": required_users[0],
        "asker_choice": asker_choice,
        "asker_selected_correct": asker_correct,
        "provider_user": required_users[1],
        "provider_choice": provider_choice,
        "provider_selected_correct": provider_correct,
        "combined_choice": combined_choice,
        "combined_selected_correct": combined_correct,
        "gate_reason": str(
            ((((verification.get("verification") or {}).get("answerability") or {}).get("gate") or {}).get("reason"))
            or ""
        ),
    }


def score_benchmark(
    config: dict[str, Any], *, prepared: dict[str, Any] | None = None
) -> dict[str, Any]:
    plan = prepared or prepare_benchmark(config)
    all_predictions: list[dict[str, Any]] = []
    arm_summaries: list[dict[str, Any]] = []
    gold_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gold in plan["gold"]:
        gold_by_run[gold["run_id"]].append(gold)

    predictions_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in config["arms"]:
        arm_predictions: list[dict[str, Any]] = []
        for run in plan["run_plans"]:
            verification_path = (
                config["output_dir"]
                / "arms"
                / arm["id"]
                / run["id"]
                / "verification.jsonl"
            )
            if not verification_path.is_file():
                raise ValueError(
                    f"missing verification output for arm={arm['id']} run={run['id']}: "
                    f"{verification_path}"
                )
            verification_by_id = _unique_jsonl(
                verification_path,
                key="qa_id",
                label=f"{arm['id']}/{run['id']} verification",
            )
            expected_ids = {gold["qa_id"] for gold in gold_by_run[run["id"]]}
            if set(verification_by_id) != expected_ids:
                missing = sorted(expected_ids - set(verification_by_id))
                extra = sorted(set(verification_by_id) - expected_ids)
                raise ValueError(
                    f"{arm['id']}/{run['id']}: verification coverage mismatch; "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            for gold in gold_by_run[run["id"]]:
                verification = verification_by_id[gold["qa_id"]]
                metadata = verification.get("verification") or {}
                if metadata.get("model_id") != arm["model_id"]:
                    raise ValueError(f"{gold['benchmark_key']}: verification model mismatch")
                if metadata.get("reasoning_effort") != arm["reasoning_effort"]:
                    raise ValueError(f"{gold['benchmark_key']}: verification reasoning mismatch")
                arm_predictions.append(
                    _prediction_row(arm=arm, gold=gold, verification=verification)
                )
        predictions_by_arm[arm["id"]] = arm_predictions
        all_predictions.extend(arm_predictions)
        overall = _classification_metrics(arm_predictions)
        per_run = {
            run["id"]: _classification_metrics(
                [row for row in arm_predictions if row["run_id"] == run["id"]]
            )
            for run in config["runs"]
        }
        tags = sorted({tag for row in arm_predictions for tag in row["error_tags"]})
        per_error_tag = {
            tag: _classification_metrics(
                [row for row in arm_predictions if tag in row["error_tags"]]
            )
            for tag in tags
        }
        no_tag_rows = [row for row in arm_predictions if not row["error_tags"]]
        if no_tag_rows:
            per_error_tag["(none)"] = _classification_metrics(no_tag_rows)
        manual_answerability_scores = sorted(
            {
                row["manual_judge_scores"]["answerability"]
                for row in arm_predictions
                if (row.get("manual_judge_scores") or {}).get("answerability")
                is not None
            }
        )
        per_manual_answerability_score = {
            str(score): _classification_metrics(
                [
                    row
                    for row in arm_predictions
                    if (row.get("manual_judge_scores") or {}).get("answerability")
                    == score
                ]
            )
            for score in manual_answerability_scores
        }
        arm_summaries.append(
            {
                "arm_id": arm["id"],
                "model_id": arm["model_id"],
                "reasoning_effort": arm["reasoning_effort"],
                "overall": overall,
                "per_run": per_run,
                "per_error_tag": per_error_tag,
                "per_manual_answerability_score": per_manual_answerability_score,
                "condition_diagnostics": _condition_diagnostics(arm_predictions),
            }
        )

    ranked = sorted(
        arm_summaries,
        key=lambda row: (
            -(
                row["overall"]["balanced_accuracy"]
                if row["overall"]["balanced_accuracy"] is not None
                else -1
            ),
            -(row["overall"]["accuracy"] if row["overall"]["accuracy"] is not None else -1),
            -(
                row["overall"]["failure_recall"]
                if row["overall"]["failure_recall"] is not None
                else -1
            ),
            row["arm_id"],
        ),
    )
    rank_by_arm = {row["arm_id"]: index for index, row in enumerate(ranked, start=1)}
    for row in arm_summaries:
        row["rank_by_balanced_accuracy"] = rank_by_arm[row["arm_id"]]

    pairwise: list[dict[str, Any]] = []
    for left, right in itertools.combinations(config["arms"], 2):
        left_rows = {row["benchmark_key"]: row for row in predictions_by_arm[left["id"]]}
        right_rows = {row["benchmark_key"]: row for row in predictions_by_arm[right["id"]]}
        keys = sorted(left_rows)
        pairwise.append(
            {
                "left_arm_id": left["id"],
                "right_arm_id": right["id"],
                "count": len(keys),
                "both_correct": sum(
                    left_rows[key]["prediction_correct"] and right_rows[key]["prediction_correct"]
                    for key in keys
                ),
                "left_only_correct": sum(
                    left_rows[key]["prediction_correct"] and not right_rows[key]["prediction_correct"]
                    for key in keys
                ),
                "right_only_correct": sum(
                    not left_rows[key]["prediction_correct"] and right_rows[key]["prediction_correct"]
                    for key in keys
                ),
                "both_wrong": sum(
                    not left_rows[key]["prediction_correct"]
                    and not right_rows[key]["prediction_correct"]
                    for key in keys
                ),
                "accuracy_delta_left_minus_right": (
                    _classification_metrics(list(left_rows.values()))["accuracy"]
                    - _classification_metrics(list(right_rows.values()))["accuracy"]
                ),
            }
        )

    comparison = {
        "protocol": plan["protocol"],
        "scored_at_utc": _utc_now(),
        "gold_annotation_count": len(plan["gold"]),
        "ranking_metric": "balanced_accuracy",
        "recommended_arm_id": ranked[0]["arm_id"],
        "arms": arm_summaries,
        "pairwise": pairwise,
    }
    output_dir: Path = config["output_dir"]
    write_json(output_dir / "comparison.json", comparison)
    write_jsonl(output_dir / "predictions.jsonl", all_predictions)
    write_jsonl(
        output_dir / "disagreements.jsonl",
        [row for row in all_predictions if not row["prediction_correct"]],
    )
    csv_fields = [
        "rank",
        "arm_id",
        "model_id",
        "reasoning_effort",
        "count",
        "accuracy",
        "balanced_accuracy",
        "failure_recall",
        "failure_precision",
        "false_accept_rate",
        "pass_recall",
        "pass_precision",
        "false_reject_rate",
        "matthews_correlation",
        "true_pass",
        "true_fail",
        "false_accept",
        "false_reject",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for arm_summary in sorted(arm_summaries, key=lambda row: rank_by_arm[row["arm_id"]]):
            overall = arm_summary["overall"]
            writer.writerow(
                {
                    "rank": rank_by_arm[arm_summary["arm_id"]],
                    "arm_id": arm_summary["arm_id"],
                    "model_id": arm_summary["model_id"],
                    "reasoning_effort": arm_summary["reasoning_effort"] or "provider_default",
                    **{field: overall.get(field) for field in csv_fields if field in overall},
                }
            )
    return comparison


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same three one-pass external answerability conditions for multiple "
            "OpenRouter model arms, either scoring manual Pass/Fail labels or running "
            "an accepted-only test set without gold."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate",
        "run",
        "score",
        "validate-set",
        "run-set",
        "summarize-set",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        if command in {"validate", "run", "score"}:
            child.add_argument("--annotations", help="Override config annotations path")
        child.add_argument("--output-dir", help="Override config output_dir")
        if command in {"run", "run-set"}:
            child.add_argument("--api-key")
            child.add_argument("--resume", action="store_true")
            child.add_argument(
                "--skip-model-catalog-check",
                action="store_true",
                help="Skip the non-billable OpenRouter video/reasoning capability preflight.",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    testset_command = args.command.endswith("-set")
    config = load_benchmark_config(
        args.config,
        annotations_override=getattr(args, "annotations", None),
        output_dir_override=args.output_dir,
        require_annotations=not testset_command,
    )
    if args.command == "validate-set":
        plan = prepare_testset(config)
        print(
            f"testset_validation_passed qas={plan['testset_qa_count']} "
            f"arms={len(config['arms'])} "
            f"expected_calls={plan['expected_api_call_count_all_arms']}",
            flush=True,
        )
        return 0
    if args.command == "run-set":
        summary = run_testset(
            config,
            resume=args.resume,
            api_key=args.api_key,
            check_model_catalog=not args.skip_model_catalog_check,
        )
        print(
            f"testset_summarized qas={summary['testset_qa_count']} "
            f"model_disagreements={summary['model_disagreement_qa_count']}",
            flush=True,
        )
        return 0
    if args.command == "summarize-set":
        summary = summarize_testset(config)
        print(
            f"testset_summarized qas={summary['testset_qa_count']} "
            f"model_disagreements={summary['model_disagreement_qa_count']}",
            flush=True,
        )
        return 0
    if args.command == "validate":
        plan = prepare_benchmark(config)
        print(
            f"benchmark_validation_passed qas={plan['scored_annotation_count']} "
            f"arms={len(config['arms'])} expected_calls={plan['expected_api_call_count_all_arms']}",
            flush=True,
        )
        return 0
    if args.command == "run":
        comparison = run_benchmark(
            config,
            resume=args.resume,
            api_key=args.api_key,
            check_model_catalog=not args.skip_model_catalog_check,
        )
    else:
        comparison = score_benchmark(config)
    print(
        f"benchmark_scored qas={comparison['gold_annotation_count']} "
        f"recommended_arm={comparison['recommended_arm_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
