"""Validation helpers for EgoLife two-user QA outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import write_json
from .qwen3vl_runner import DEFAULT_MODEL_ID
from .schema import load_and_validate, write_human_review_sheet, write_qa_csv


def validate_outputs(
    *,
    qa_path: str | Path,
    report_path: str | Path,
    csv_path: str | Path | None = None,
    human_review_path: str | Path | None = None,
    strict_review: bool = False,
) -> int:
    count, errors = load_and_validate(qa_path, strict_review=strict_review)
    if csv_path:
        write_qa_csv(qa_path, csv_path)
    if human_review_path:
        write_human_review_sheet(qa_path, human_review_path)
    report = {
        "qa_path": str(qa_path),
        "qa_count": count,
        "strict_review": strict_review,
        "passed": not errors,
        "errors": errors,
    }
    report_path = Path(report_path)
    if report_path.suffix.lower() in {".md", ".markdown"}:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# EgoLife Two-User QA Generation Report",
            "",
            f"- QA path: `{qa_path}`",
            f"- QA count: {count}",
            f"- Strict review: {strict_review}",
            f"- Passed: {not errors}",
            "",
            "## Validation Errors",
            "",
        ]
        lines.extend([f"- {error}" for error in errors] or ["- none"])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        write_json(report_path, report)
    return 0 if not errors else 1


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "openai-compatible-local"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EgoLife two-user QA validation helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    val = sub.add_parser("validate_outputs")
    val.add_argument("--qa", required=True)
    val.add_argument("--report", required=True)
    val.add_argument("--csv-output")
    val.add_argument("--human-review-output")
    val.add_argument("--strict-review", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "validate_outputs":
        return validate_outputs(
            qa_path=args.qa,
            report_path=args.report,
            csv_path=args.csv_output,
            human_review_path=args.human_review_output,
            strict_review=args.strict_review,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
