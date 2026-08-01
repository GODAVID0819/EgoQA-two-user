"""Post-generation answerability verification with an independent visual judge."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import append_jsonl, iter_jsonl, read_json, write_json
from .qwen3vl_runner import (
    DEFAULT_OPENROUTER_BASE_URL,
    OPENROUTER_REASONING_EFFORTS,
    make_runner,
)
from .video_qa_loop import JUDGE_VIDEO_SOURCES, clips_for_users, run_answerability_eval


DEFAULT_VERIFIER_MODEL_ID = "google/gemini-3.5-flash"
MEDIA_ROLE_CHOICES = ("source_qa", *JUDGE_VIDEO_SOURCES)


def _unique_rows_by_id(
    rows: list[dict[str, Any]],
    *,
    id_key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get(id_key) or "").strip()
        if not row_id:
            raise ValueError(f"{label} row is missing {id_key}")
        if row_id in indexed:
            raise ValueError(f"{label} contains duplicate {id_key}={row_id}")
        indexed[row_id] = row
    return indexed


def _source_answerability(qa: dict[str, Any]) -> dict[str, Any]:
    review = qa.get("review")
    if not isinstance(review, dict):
        return {}
    answerability = review.get("answerability")
    return answerability if isinstance(answerability, dict) else {}


def _source_qa_is_accepted(qa: dict[str, Any]) -> bool:
    review = qa.get("review")
    if not isinstance(review, dict):
        return False
    final_decision = review.get("final_decision")
    return (
        review.get("review_passed") is True
        and isinstance(final_decision, dict)
        and final_decision.get("accepted") is True
        and (_source_answerability(qa).get("gate") or {}).get("passed") is True
    )


def _media_role_for_qa(qa: dict[str, Any], requested_role: str) -> str:
    if requested_role != "source_qa":
        return requested_role
    source_role = str(qa.get("judge_video_source") or "full").strip()
    if source_role not in JUDGE_VIDEO_SOURCES:
        raise ValueError(
            f"{qa.get('qa_id')}: unsupported source judge_video_source={source_role!r}"
        )
    return source_role


def _exact_condition_video_path(clip: dict[str, Any], media_role: str) -> Path | None:
    """Resolve the requested judge video without silently downgrading full to pruned."""

    keys = (
        ("full_local_video", "original_local_video", "source_local_video")
        if media_role == "full"
        else ("local_video",)
    )
    for key in keys:
        raw_path = str(clip.get(key) or "").strip()
        if raw_path:
            path = Path(raw_path)
            if path.is_file():
                return path
    return None


def preflight_answerability_media(
    *,
    accepted_rows: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    media_role: str,
) -> dict[str, Any]:
    """Validate all condition media before the first potentially billable call."""

    resolved_paths: set[str] = set()
    role_counts: Counter[str] = Counter()
    errors: list[str] = []
    for qa in accepted_rows:
        qa_id = str(qa.get("qa_id") or "")
        evidence_id = str(qa.get("evidence_id") or "")
        packet = evidence_by_id[evidence_id]
        active_role = _media_role_for_qa(qa, media_role)
        role_counts[active_role] += 1
        required_users = list(qa.get("required_users") or [])
        if len(required_users) < 2:
            errors.append(f"{qa_id}: expected at least two required_users")
            continue
        for user in required_users:
            clips = clips_for_users(packet, [user])
            if len(clips) != 1:
                errors.append(
                    f"{qa_id}: expected exactly one evidence clip for user={user!r}, "
                    f"found {len(clips)}"
                )
                continue
            path = _exact_condition_video_path(clips[0], active_role)
            if path is None:
                required_keys = (
                    "full_local_video/original_local_video/source_local_video"
                    if active_role == "full"
                    else "local_video"
                )
                errors.append(
                    f"{qa_id}: no existing {active_role} video for user={user!r}; "
                    f"checked {required_keys}"
                )
                continue
            resolved_paths.add(str(path.resolve()))
    if errors:
        preview = "; ".join(errors[:10])
        suffix = f"; ... {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ValueError(f"answerability media preflight failed: {preview}{suffix}")
    return {
        "passed": True,
        "qa_count": len(accepted_rows),
        "unique_video_count": len(resolved_paths),
        "media_role_counts": dict(sorted(role_counts.items())),
    }


def _verification_record(
    *,
    qa: dict[str, Any],
    answerability: dict[str, Any],
    model_id: str,
    media_role: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    source_answerability = _source_answerability(qa)
    verification_passed = (answerability.get("gate") or {}).get("passed") is True
    return {
        "qa_id": qa.get("qa_id"),
        "evidence_id": qa.get("evidence_id"),
        "question": qa.get("question"),
        "options": qa.get("options"),
        "correct": qa.get("correct"),
        "answer": qa.get("answer"),
        "required_users": qa.get("required_users"),
        "source_generation_model_id": qa.get("model_id"),
        "source_answerability_model_id": (qa.get("review_model_ids") or {}).get(
            "answerability"
        ),
        "source_judge_video_source": qa.get("judge_video_source"),
        "source_answerability": source_answerability,
        "verification": {
            "backend": "openrouter",
            "model_id": model_id,
            "reasoning_effort": reasoning_effort,
            "media_role": media_role,
            "answerability": answerability,
            "passed": verification_passed,
            "agrees_with_original_pass": verification_passed,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def verify_answerability(
    *,
    accepted_qa_path: str | Path,
    evidence_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path,
    summary_path: str | Path,
    model_id: str = DEFAULT_VERIFIER_MODEL_ID,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    max_new_tokens: int = 512,
    reasoning_effort: str | None = None,
    media_role: str = "source_qa",
    generation_summary_path: str | Path | None = None,
    resume: bool = False,
    api_key: str | None = None,
    runner: Any | None = None,
) -> dict[str, Any]:
    if media_role not in MEDIA_ROLE_CHOICES:
        raise ValueError(f"media_role must be one of {MEDIA_ROLE_CHOICES}")

    accepted_rows = list(iter_jsonl(accepted_qa_path))
    if not accepted_rows:
        raise ValueError("accepted QA input contains no rows")
    _unique_rows_by_id(accepted_rows, id_key="qa_id", label="accepted QA input")
    for qa in accepted_rows:
        if not _source_qa_is_accepted(qa):
            raise ValueError(
                f"{qa.get('qa_id')}: input row did not pass the original generation loop"
            )

    evidence_rows = list(iter_jsonl(evidence_path))
    evidence_by_id = _unique_rows_by_id(
        evidence_rows,
        id_key="evidence_id",
        label="evidence input",
    )
    missing_evidence = [
        str(qa.get("evidence_id") or "")
        for qa in accepted_rows
        if str(qa.get("evidence_id") or "") not in evidence_by_id
    ]
    if missing_evidence:
        raise ValueError(
            "accepted QAs are missing evidence packets: " + ", ".join(missing_evidence[:5])
        )

    media_preflight = preflight_answerability_media(
        accepted_rows=accepted_rows,
        evidence_by_id=evidence_by_id,
        media_role=media_role,
    )

    generation_summary: dict[str, Any] = {}
    if generation_summary_path is not None:
        generation_summary = read_json(generation_summary_path)
        if generation_summary.get("complete_evidence_coverage") is not True:
            raise ValueError("generation summary does not report complete evidence coverage")
        expected_accepted = int(generation_summary.get("accepted_qa_count") or 0)
        if expected_accepted != len(accepted_rows):
            raise ValueError(
                "accepted QA count differs from generation summary: "
                f"rows={len(accepted_rows)} summary={expected_accepted}"
            )

    output_path = Path(output_path)
    prompts_path = Path(prompts_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("", encoding="utf-8")
        prompts_path.write_text("", encoding="utf-8")

    existing_rows = list(iter_jsonl(output_path)) if output_path.exists() else []
    existing_by_qa = _unique_rows_by_id(
        existing_rows,
        id_key="qa_id",
        label="verification output",
    )
    source_qa_ids = {str(qa["qa_id"]) for qa in accepted_rows}
    unexpected_existing = sorted(set(existing_by_qa) - source_qa_ids)
    if unexpected_existing:
        raise ValueError(
            "verification output contains QAs outside the source input: "
            + ", ".join(unexpected_existing[:5])
        )
    for row in existing_rows:
        verification = row.get("verification") or {}
        if verification.get("model_id") != model_id:
            raise ValueError(
                "cannot resume verification output created with another model: "
                f"{verification.get('model_id')!r}"
            )
        if verification.get("reasoning_effort") != reasoning_effort:
            raise ValueError(
                "cannot resume verification output created with another reasoning effort: "
                f"{verification.get('reasoning_effort')!r}"
            )

    active_runner = runner or make_runner(
        "openrouter",
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        api_key=api_key,
        allow_openai_video_input=True,
        reasoning_effort=reasoning_effort,
    )

    completed_qa_ids = set(existing_by_qa)
    for qa in accepted_rows:
        qa_id = str(qa["qa_id"])
        if qa_id in completed_qa_ids:
            print(f"verification_skip qa_id={qa_id} reason=already_complete", flush=True)
            continue
        evidence_id = str(qa.get("evidence_id") or "")
        packet = evidence_by_id[evidence_id]
        active_media_role = _media_role_for_qa(qa, media_role)
        prompt_rows: list[dict[str, Any]] = []
        print(
            f"verification_start qa_id={qa_id} evidence_id={evidence_id} "
            f"model={model_id} media_role={active_media_role}",
            flush=True,
        )
        answerability = run_answerability_eval(
            qa_item=qa,
            packet=packet,
            runner=active_runner,
            media_backend="openrouter",
            allow_openai_video_input=True,
            prompt_rows=prompt_rows,
            judge_media_role=active_media_role,
        )
        for prompt_row in prompt_rows:
            append_jsonl(
                prompts_path,
                {
                    **prompt_row,
                    "verification_backend": "openrouter",
                    "verification_model_id": model_id,
                    "source_evidence_id": evidence_id,
                },
            )
        record = _verification_record(
            qa=qa,
            answerability=answerability,
            model_id=model_id,
            media_role=active_media_role,
            reasoning_effort=reasoning_effort,
        )
        append_jsonl(output_path, record)
        existing_rows.append(record)
        completed_qa_ids.add(qa_id)
        print(
            f"verification_done qa_id={qa_id} "
            f"passed={record['verification']['passed']}",
            flush=True,
        )

    final_rows = list(iter_jsonl(output_path))
    final_by_qa = _unique_rows_by_id(
        final_rows,
        id_key="qa_id",
        label="final verification output",
    )
    missing_verifications = sorted(source_qa_ids - set(final_by_qa))
    if missing_verifications:
        raise ValueError(
            "verification output is incomplete: " + ", ".join(missing_verifications[:5])
        )

    passed_count = sum(
        (row.get("verification") or {}).get("passed") is True for row in final_rows
    )
    media_counts = Counter(
        str((row.get("verification") or {}).get("media_role") or "")
        for row in final_rows
    )
    condition_count = sum(
        len(
            (
                ((row.get("verification") or {}).get("answerability") or {}).get(
                    "evaluations"
                )
                or []
            )
        )
        for row in final_rows
    )
    summary = {
        "source_accepted_qa_path": str(accepted_qa_path),
        "source_evidence_path": str(evidence_path),
        "source_generation_summary_path": (
            str(generation_summary_path) if generation_summary_path is not None else None
        ),
        "source_accepted_qa_count": len(accepted_rows),
        "verification_backend": "openrouter",
        "verification_model_id": model_id,
        "verification_reasoning_effort": reasoning_effort,
        "requested_media_role": media_role,
        "media_role_counts": dict(sorted(media_counts.items())),
        "verified_qa_count": len(final_rows),
        "answerability_condition_count": condition_count,
        "verification_pass_count": passed_count,
        "verification_fail_count": len(final_rows) - passed_count,
        "verification_pass_rate": passed_count / len(final_rows),
        "complete_accepted_qa_coverage": True,
        "media_preflight": media_preflight,
        "cpu_only_job": True,
        "output_path": str(output_path),
        "prompts_path": str(prompts_path),
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun answerability for accepted generation-loop QAs using an independent "
            "OpenRouter visual model."
        )
    )
    parser.add_argument("--accepted-qa", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--generation-summary")
    parser.add_argument("--model-id", default=DEFAULT_VERIFIER_MODEL_ID)
    parser.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", choices=OPENROUTER_REASONING_EFFORTS)
    parser.add_argument("--media-role", choices=MEDIA_ROLE_CHOICES, default="source_qa")
    parser.add_argument("--api-key")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    summary = verify_answerability(
        accepted_qa_path=args.accepted_qa,
        evidence_path=args.evidence,
        output_path=args.output,
        prompts_path=args.prompts_output,
        summary_path=args.summary_output,
        model_id=args.model_id,
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
        reasoning_effort=args.reasoning_effort,
        media_role=args.media_role,
        generation_summary_path=args.generation_summary,
        resume=args.resume,
        api_key=args.api_key,
    )
    print(
        f"verified_qa_count={summary['verified_qa_count']} "
        f"verification_pass_count={summary['verification_pass_count']} "
        f"verification_fail_count={summary['verification_fail_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
