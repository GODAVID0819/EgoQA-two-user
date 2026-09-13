"""Run the curated all-six versus required-pair test through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl, write_json
from .qwen3vl_runner import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL_ID,
    OPENROUTER_REASONING_EFFORTS,
    make_runner,
)
from .six_video_qa_tester import (
    _prepare_cases,
    _unique_index,
    load_qa_rows,
    run_six_video_qa_test,
)
from .summarize_curated_ablation import summarize


DEFAULT_EXPECTED_COUNT = 5
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_MODEL_ID = f"{DEFAULT_OPENROUTER_MODEL_ID}:nitro"


def _resolved_media(case: dict[str, Any]) -> list[str]:
    return [str(Path(row["source_video"]).resolve()) for row in case["media"]]


def validate_evaluation_plan(
    *,
    qa_path: str | Path,
    evidence_path: str | Path,
    six_view_manifest_path: str | Path,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
) -> dict[str, Any]:
    """Resolve every case and prove the paired arm is the six-arm prefix."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    qa_rows = load_qa_rows([qa_path])
    if len(qa_rows) != expected_count:
        raise ValueError(f"expected {expected_count} curated QAs, found {len(qa_rows)}")
    qa_ids = [str(row.get("qa_id") or "") for row in qa_rows]
    if any(not qa_id for qa_id in qa_ids) or len(set(qa_ids)) != len(qa_ids):
        raise ValueError("curated smoke QAs must have distinct non-empty qa_id values")
    for row in qa_rows:
        qa_id = str(row["qa_id"])
        if str(row.get("review_status") or "").casefold() != "pass":
            raise ValueError(f"{qa_id}: review_status is not pass")
        required_users = row.get("required_users")
        if (
            not isinstance(required_users, list)
            or len(required_users) != 2
            or len({str(user).strip().casefold() for user in required_users}) != 2
        ):
            raise ValueError(f"{qa_id}: expected two distinct required_users")

    evidence_index = _unique_index(
        list(iter_jsonl(evidence_path)),
        key="evidence_id",
        source=evidence_path,
    )
    six_view_index = _unique_index(
        list(iter_jsonl(six_view_manifest_path)),
        key="evidence_id",
        source=six_view_manifest_path,
    )
    six_cases = _prepare_cases(
        qa_rows=qa_rows,
        evidence_index=evidence_index,
        six_view_index=six_view_index,
        video_input_scope="six",
        pair_video_source="full",
    )
    pair_cases = _prepare_cases(
        qa_rows=qa_rows,
        evidence_index=evidence_index,
        six_view_index=None,
        video_input_scope="pair",
        pair_video_source="full",
    )

    cases = []
    for six_case, pair_case in zip(six_cases, pair_cases, strict=True):
        if six_case["evaluation_id"] != pair_case["evaluation_id"]:
            raise ValueError("six-user and required-pair cases resolved in different orders")
        six_media = _resolved_media(six_case)
        pair_media = _resolved_media(pair_case)
        if len(six_media) != 6 or len(pair_media) != 2:
            raise ValueError(f"{six_case['qa_id']}: expected 6-versus-2 media")
        if six_media[:2] != pair_media:
            raise ValueError(
                f"{six_case['qa_id']}: required pair is not the exact first-two "
                "prefix of the six-user arm"
            )
        cases.append(
            {
                "evaluation_id": six_case["evaluation_id"],
                "qa_id": six_case["qa_id"],
                "evidence_id": six_case["evidence_id"],
                "asker": six_case["asker"],
                "required_users": list(six_case["qa"]["required_users"]),
                "six_video_count": len(six_media),
                "required_video_count": len(pair_media),
                "required_pair_is_six_prefix": True,
                "six_media": six_media,
                "required_media": pair_media,
            }
        )

    return {
        "schema_version": "curated_openrouter_ablation_plan_v1",
        "qa_count": len(qa_rows),
        "logical_model_call_count": len(qa_rows) * 2,
        "condition_order": ["six_users", "required_users"],
        "model_id": DEFAULT_MODEL_ID,
        "qa_path": str(Path(qa_path).resolve()),
        "evidence_path": str(Path(evidence_path).resolve()),
        "six_view_manifest_path": str(Path(six_view_manifest_path).resolve()),
        "cases": cases,
    }


def fetch_openrouter_model_snapshot(*, model_id: str, base_url: str) -> dict[str, Any]:
    """Confirm live video-to-text support before the first billable request."""

    catalog_model_id = model_id.removesuffix(":nitro")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"User-Agent": "egolife-curated-openrouter-smoke/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("OpenRouter model catalog returned no data list")
    model = next((row for row in models if row.get("id") == catalog_model_id), None)
    if model is None:
        raise ValueError(f"OpenRouter model is unavailable: {catalog_model_id}")
    architecture = model.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    output_modalities = architecture.get("output_modalities") or []
    if "video" not in input_modalities or "text" not in output_modalities:
        raise ValueError(
            f"{catalog_model_id} does not advertise video-to-text support: "
            f"inputs={input_modalities}, outputs={output_modalities}"
        )
    return {
        **model,
        "requested_model_id": model_id,
        "catalog_model_id": catalog_model_id,
        "routing_variant": "nitro" if model_id.endswith(":nitro") else None,
    }


def _completed_condition(summary: dict[str, Any], *, expected_count: int) -> None:
    if summary.get("completed_count") != expected_count:
        raise RuntimeError(
            f"condition completed {summary.get('completed_count')} of {expected_count} calls"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    plan = validate_evaluation_plan(
        qa_path=args.qa,
        evidence_path=args.evidence,
        six_view_manifest_path=args.six_view_manifest,
        expected_count=args.expected_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan.update(
        {
            "backend": args.backend,
            "model_id": args.model_id,
            "base_url": args.base_url,
            "reasoning_effort": args.reasoning_effort,
            "max_new_tokens": args.max_new_tokens,
            "automatic_retry_count": 0,
        }
    )
    write_json(output_dir / "evaluation_plan.json", plan)
    if args.validate_only:
        return plan

    snapshot = fetch_openrouter_model_snapshot(
        model_id=args.model_id,
        base_url=args.base_url,
    )
    reasoning = snapshot.get("reasoning") or {}
    if args.reasoning_effort == "none" and reasoning.get("mandatory") is True:
        raise ValueError(f"{args.model_id} now requires reasoning and cannot use effort=none")
    if args.reasoning_effort != "none" and "reasoning" not in set(
        snapshot.get("supported_parameters") or []
    ):
        raise ValueError(
            f"{args.model_id} does not advertise the reasoning request parameter"
        )
    write_json(output_dir / "openrouter_model_snapshot.json", snapshot)
    os.environ["OPENROUTER_MAX_RETRIES"] = "0"
    runner = make_runner(
        args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        max_new_tokens=args.max_new_tokens,
        allow_openai_video_input=True,
        reasoning_effort=args.reasoning_effort,
    )
    common = {
        "qa_paths": [args.qa],
        "evidence_path": args.evidence,
        "backend": args.backend,
        "model_id": args.model_id,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_effort": args.reasoning_effort,
        "decoding_mode": "greedy",
        "resume": False,
        "fail_fast": True,
        "runner": runner,
        "media_alias_dir": output_dir / "anonymous_media",
        "pair_video_source": "full",
    }
    six_summary = run_six_video_qa_test(
        **common,
        six_view_manifest_path=args.six_view_manifest,
        video_input_scope="six",
        output_path=output_dir / "six_users.jsonl",
        summary_path=output_dir / "six_users.summary.json",
        prompts_path=output_dir / "six_users.prompts.jsonl",
    )
    _completed_condition(six_summary, expected_count=args.expected_count)
    required_summary = run_six_video_qa_test(
        **common,
        six_view_manifest_path=None,
        video_input_scope="pair",
        output_path=output_dir / "required_users.jsonl",
        summary_path=output_dir / "required_users.summary.json",
        prompts_path=output_dir / "required_users.prompts.jsonl",
    )
    _completed_condition(required_summary, expected_count=args.expected_count)

    paired_summary = summarize(
        argparse.Namespace(
            six=output_dir / "six_users.jsonl",
            required=output_dir / "required_users.jsonl",
            output=output_dir / "paired_summary.json",
            per_question_output=output_dir / "per_question_comparison.jsonl",
            expected_count=args.expected_count,
        )
    )
    result = {
        "status": "completed",
        "qa_count": args.expected_count,
        "logical_model_call_count": args.expected_count * 2,
        "condition_order": ["six_users", "required_users"],
        "six_users": six_summary,
        "required_users": required_summary,
        "paired": paired_summary,
    }
    write_json(output_dir / "run_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--six-view-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--backend", choices=("openrouter",), default="openrouter")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument("--api-key")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        choices=OPENROUTER_REASONING_EFFORTS,
        default="none",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
