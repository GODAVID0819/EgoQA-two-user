"""Run a paired all-six versus minimal-answerable-set video evaluation.

Both arms use full original videos from a native six-user evidence packet.  The
minimal condition contains the speaker plus the provider users named by the
accepted QA's ``supporting_user_claims``.  Both conditions are validated in full
before the first billable OpenRouter request.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl, write_json, write_jsonl
from .qwen3vl_runner import (
    DEFAULT_OPENROUTER_BASE_URL,
    OPENROUTER_REASONING_EFFORTS,
)
from .six_video_qa_tester import (
    _minimal_answerable_users,
    _prepare_cases,
    _unique_index,
    load_evidence_rows,
    load_qa_rows,
    run_six_video_qa_test,
)


DEFAULT_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_EXPECTED_QA_COUNT = 44
DEFAULT_MAX_NEW_TOKENS = 64


def _accepted_marker_error(qa: dict[str, Any]) -> str | None:
    """Reject explicit failure markers while allowing accepted-only exports."""

    qa_id = str(qa.get("qa_id") or "<missing qa_id>")
    if "accepted" in qa and qa.get("accepted") is not True:
        return f"{qa_id}: top-level accepted marker is not true"
    review = qa.get("review")
    if not isinstance(review, dict):
        return None
    status = str(review.get("status") or "").strip().casefold()
    if status and status not in {"pass", "passed", "accepted"}:
        return f"{qa_id}: review.status={status!r} is not accepted"
    if "review_passed" in review and review.get("review_passed") is not True:
        return f"{qa_id}: review.review_passed is not true"
    result = review.get("result")
    if (
        isinstance(result, dict)
        and "accepted" in result
        and result.get("accepted") is not True
    ):
        return f"{qa_id}: review.result.accepted is not true"
    return None


def _case_media_by_agent(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_agent = {}
    for media in case["media"]:
        name = (
            str(media.get("agent_name") or media.get("user") or "")
            .strip()
            .casefold()
        )
        if not name:
            raise ValueError(f"{case['qa_id']}: media row is missing a participant name")
        if name in by_agent:
            raise ValueError(f"{case['qa_id']}: duplicate media participant {name!r}")
        by_agent[name] = media
    return by_agent


def validate_evaluation_plan(
    *,
    qa_paths: list[str | Path],
    evidence_path: str | Path,
    expected_qa_count: int = DEFAULT_EXPECTED_QA_COUNT,
) -> dict[str, Any]:
    """Validate the frozen cohort and both media arms without model calls."""

    if expected_qa_count <= 0:
        raise ValueError("expected_qa_count must be positive")
    qa_rows = load_qa_rows(qa_paths)
    if len(qa_rows) != expected_qa_count:
        raise ValueError(
            f"expected exactly {expected_qa_count} accepted QAs, found {len(qa_rows)}"
        )

    qa_ids = [str(qa["qa_id"]) for qa in qa_rows]
    duplicate_qa_ids = sorted(
        {qa_id for qa_id in qa_ids if qa_ids.count(qa_id) > 1}
    )
    if duplicate_qa_ids:
        raise ValueError(f"accepted set contains duplicate qa_id values: {duplicate_qa_ids}")
    marker_errors = [error for qa in qa_rows if (error := _accepted_marker_error(qa))]
    if marker_errors:
        raise ValueError("accepted-QA validation failed: " + "; ".join(marker_errors))

    evidence_rows = load_evidence_rows(evidence_path)
    evidence_index = _unique_index(
        evidence_rows,
        key="evidence_id",
        source=evidence_path,
    )
    all_six_cases = _prepare_cases(
        qa_rows=qa_rows,
        evidence_index=evidence_index,
        six_view_index=None,
        video_input_scope="six",
        pair_video_source="full",
    )
    minimal_cases = _prepare_cases(
        qa_rows=qa_rows,
        evidence_index=evidence_index,
        six_view_index=None,
        video_input_scope="minimal",
        pair_video_source="full",
    )
    six_by_evaluation = {case["evaluation_id"]: case for case in all_six_cases}
    minimal_by_evaluation = {case["evaluation_id"]: case for case in minimal_cases}
    if set(six_by_evaluation) != set(minimal_by_evaluation):
        raise ValueError("all-six and minimal conditions resolved different QA cases")

    plan_cases = []
    for qa, minimal_case in zip(qa_rows, minimal_cases):
        required_users = [str(user).strip() for user in qa.get("required_users") or []]
        if len(required_users) != 6:
            raise ValueError(f"{qa['qa_id']}: expected six required_users")
        minimal_users = _minimal_answerable_users(qa)
        required = {user.casefold() for user in minimal_users}
        minimal_by_agent = _case_media_by_agent(minimal_case)
        if set(minimal_by_agent) != required:
            raise ValueError(
                f"{qa['qa_id']}: minimal users {sorted(required)} do not match "
                f"minimal media participants {sorted(minimal_by_agent)}"
            )

        six_case = six_by_evaluation[minimal_case["evaluation_id"]]
        six_media_by_agent = _case_media_by_agent(six_case)
        for agent, minimal_media in minimal_by_agent.items():
            six_media = six_media_by_agent.get(agent)
            if six_media is None:
                raise ValueError(f"{qa['qa_id']}: {agent} is absent from all-six media")
            if Path(minimal_media["source_video"]).resolve() != Path(
                six_media["source_video"]
            ).resolve():
                raise ValueError(
                    f"{qa['qa_id']}: {agent} uses different video files across conditions"
                )
        minimal_order = [media["agent_name"].casefold() for media in minimal_case["media"]]
        filtered_six_order = [
            media["agent_name"].casefold()
            for media in six_case["media"]
            if media["agent_name"].casefold() in required
        ]
        if filtered_six_order != minimal_order:
            raise ValueError(
                f"{qa['qa_id']}: all-six ordering does not preserve minimal media order"
            )

        plan_cases.append(
            {
                "evaluation_id": minimal_case["evaluation_id"],
                "qa_id": minimal_case["qa_id"],
                "evidence_id": minimal_case["evidence_id"],
                "asker": minimal_case["asker"],
                "required_users": list(required_users),
                "minimal_answerable_users": minimal_users,
                "all_six_video_count": len(six_case["media"]),
                "minimal_video_count": len(minimal_case["media"]),
                "minimal_full_original_videos": [
                    media["source_video"] for media in minimal_case["media"]
                ],
                "all_six_full_or_context_videos": [
                    media["source_video"] for media in six_case["media"]
                ],
            }
        )

    return {
        "schema_version": "six_vs_minimal_plan_v1",
        "qa_count": len(qa_rows),
        "expected_qa_count": expected_qa_count,
        "condition_count": 2,
        "logical_model_call_count": len(qa_rows) * 2,
        "model_id": DEFAULT_MODEL_ID,
        "conditions": {
            "all_six": {
                "video_count_per_question": 6,
                "selected_pair_video_source": "full",
            },
            "minimal_answerable": {
                "video_count_per_question": "variable",
                "participants": "speaker plus supporting_user_claims providers",
                "video_source": "full",
            },
        },
        "qa_paths": [str(Path(path).resolve()) for path in qa_paths],
        "evidence_path": str(Path(evidence_path).resolve()),
        "cases": plan_cases,
    }


def fetch_openrouter_model_snapshot(*, model_id: str, base_url: str) -> dict[str, Any]:
    """Confirm the requested live model still supports video before paid calls."""

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"User-Agent": "egolife-six-vs-minimal/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("OpenRouter model catalog returned no data list")
    model = next((row for row in models if row.get("id") == model_id), None)
    if model is None:
        raise ValueError(f"OpenRouter model is unavailable: {model_id}")
    architecture = model.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    output_modalities = architecture.get("output_modalities") or []
    if "video" not in input_modalities or "text" not in output_modalities:
        raise ValueError(
            f"{model_id} does not advertise video-to-text support: "
            f"inputs={input_modalities}, outputs={output_modalities}"
        )
    supported_parameters = set(model.get("supported_parameters") or [])
    required_parameters = {"max_tokens", "temperature"}
    missing = sorted(required_parameters - supported_parameters)
    if missing:
        raise ValueError(f"{model_id} is missing required parameters: {missing}")
    return model


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_paired_comparison(
    *,
    all_six_results_path: Path,
    minimal_results_path: Path,
    output_dir: Path,
    model_id: str,
) -> dict[str, Any]:
    all_six_rows = list(iter_jsonl(all_six_results_path))
    minimal_rows = list(iter_jsonl(minimal_results_path))
    all_six = {row["evaluation_id"]: row for row in all_six_rows}
    minimal = {row["evaluation_id"]: row for row in minimal_rows}
    if len(all_six) != len(all_six_rows) or len(minimal) != len(minimal_rows):
        raise ValueError("a condition result file contains duplicate evaluation IDs")
    if set(all_six) != set(minimal):
        raise ValueError("condition result files contain different evaluation IDs")

    rows = []
    for evaluation_id in all_six:
        six_row = all_six[evaluation_id]
        minimal_row = minimal[evaluation_id]
        six_correct = six_row.get("is_correct") is True
        minimal_correct = minimal_row.get("is_correct") is True
        rows.append(
            {
                "evaluation_id": evaluation_id,
                "qa_id": six_row.get("qa_id"),
                "evidence_id": six_row.get("evidence_id"),
                "correct_choice": six_row.get("correct_choice"),
                "all_six_status": six_row.get("status"),
                "all_six_choice": six_row.get("predicted_choice"),
                "all_six_correct": six_correct,
                "minimal_status": minimal_row.get("status"),
                "minimal_choice": minimal_row.get("predicted_choice"),
                "minimal_correct": minimal_correct,
                "prediction_agreement": (
                    six_row.get("predicted_choice") == minimal_row.get("predicted_choice")
                    and six_row.get("predicted_choice") is not None
                ),
                "outcome": (
                    "both_correct"
                    if six_correct and minimal_correct
                    else "all_six_only_correct"
                    if six_correct
                    else "minimal_only_correct"
                    if minimal_correct
                    else "neither_correct"
                ),
            }
        )

    count = len(rows)
    summary = {
        "schema_version": "six_vs_minimal_comparison_v1",
        "model_id": model_id,
        "qa_count": count,
        "all_six_correct": sum(row["all_six_correct"] for row in rows),
        "minimal_correct": sum(row["minimal_correct"] for row in rows),
        "all_six_accuracy": (
            sum(row["all_six_correct"] for row in rows) / count if count else None
        ),
        "minimal_accuracy": (
            sum(row["minimal_correct"] for row in rows) / count if count else None
        ),
        "both_correct": sum(row["outcome"] == "both_correct" for row in rows),
        "all_six_only_correct": sum(
            row["outcome"] == "all_six_only_correct" for row in rows
        ),
        "minimal_only_correct": sum(
            row["outcome"] == "minimal_only_correct" for row in rows
        ),
        "neither_correct": sum(row["outcome"] == "neither_correct" for row in rows),
        "prediction_agreement_count": sum(row["prediction_agreement"] for row in rows),
    }
    write_jsonl(output_dir / "per_question_comparison.jsonl", rows)
    _write_csv(
        output_dir / "per_question_comparison.csv",
        rows,
        list(rows[0]) if rows else ["evaluation_id"],
    )
    write_json(output_dir / "comparison.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    plan = validate_evaluation_plan(
        qa_paths=args.qa,
        evidence_path=args.evidence,
        expected_qa_count=args.expected_qa_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan.update(
        {
            "backend": args.backend,
            "model_id": args.model_id,
            "base_url": args.base_url,
            "reasoning_effort": args.reasoning_effort,
            "turbo": args.turbo,
            "max_new_tokens": args.max_new_tokens,
            "decoding_mode": "greedy",
            "temperature": 0,
            "automatic_retry_count": 0,
        }
    )
    write_json(output_dir / "evaluation_plan.json", plan)
    if args.validate_only:
        return plan

    if args.backend == "openrouter":
        snapshot = fetch_openrouter_model_snapshot(
            model_id=args.model_id,
            base_url=args.base_url,
        )
        reasoning = snapshot.get("reasoning") or {}
        if args.reasoning_effort == "none" and reasoning.get("mandatory") is True:
            raise ValueError(
                f"{args.model_id} now requires reasoning and cannot use effort=none"
            )
        if args.reasoning_effort != "none" and "reasoning" not in set(
            snapshot.get("supported_parameters") or []
        ):
            raise ValueError(
                f"{args.model_id} does not advertise the reasoning request parameter"
            )
        write_json(output_dir / "openrouter_model_snapshot.json", snapshot)
        # This paired benchmark has an exact 88-logical-call contract. A failed
        # request becomes an error row and is retried only by a resumed job.
        os.environ["OPENROUTER_MAX_RETRIES"] = "0"

    common = {
        "qa_paths": args.qa,
        "evidence_path": args.evidence,
        "six_view_manifest_path": None,
        "backend": args.backend,
        "model_id": args.model_id,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_effort": args.reasoning_effort,
        "openrouter_turbo": args.turbo,
        "decoding_mode": "greedy",
        "resume": args.resume,
        "fail_fast": False,
        "media_alias_dir": output_dir / "anonymous_media",
    }
    all_six_dir = output_dir / "all_six"
    minimal_dir = output_dir / "minimal_answerable"
    all_six_summary = run_six_video_qa_test(
        **common,
        video_input_scope="six",
        pair_video_source="full",
        output_path=all_six_dir / "results.jsonl",
        summary_path=all_six_dir / "summary.json",
        prompts_path=all_six_dir / "prompts.jsonl",
    )
    minimal_summary = run_six_video_qa_test(
        **common,
        video_input_scope="minimal",
        pair_video_source="full",
        output_path=minimal_dir / "results.jsonl",
        summary_path=minimal_dir / "summary.json",
        prompts_path=minimal_dir / "prompts.jsonl",
    )
    comparison = write_paired_comparison(
        all_six_results_path=all_six_dir / "results.jsonl",
        minimal_results_path=minimal_dir / "results.jsonl",
        output_dir=output_dir,
        model_id=args.model_id,
    )
    run_summary = {
        "plan_path": str(output_dir / "evaluation_plan.json"),
        "all_six": all_six_summary,
        "minimal_answerable": minimal_summary,
        "comparison": comparison,
    }
    write_json(output_dir / "run_summary.json", run_summary)
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gemini 2.5 Flash answers from all six full videos against "
            "the speaker-plus-supporting-providers minimal answerable set"
        )
    )
    parser.add_argument("--qa", action="append", required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help="Packet evidence JSONL, or the accepted QA JSONL when it embeds video_evidence",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-qa-count", type=int, default=DEFAULT_EXPECTED_QA_COUNT)
    parser.add_argument("--backend", choices=("openrouter", "dry-run"), default="openrouter")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--api-key")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        choices=OPENROUTER_REASONING_EFFORTS,
        default="none",
        help="Default disables thinking for the short A-E answer task",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Prioritize OpenRouter providers by throughput (Nitro routing)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    if args.validate_only:
        print(
            f"validated={summary['qa_count']} logical_calls="
            f"{summary['logical_model_call_count']}"
        )
    else:
        comparison = summary["comparison"]
        print(
            f"compared={comparison['qa_count']} "
            f"all_six_accuracy={comparison['all_six_accuracy']} "
            f"minimal_accuracy={comparison['minimal_accuracy']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
