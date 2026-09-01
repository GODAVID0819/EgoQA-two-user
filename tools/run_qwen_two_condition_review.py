"""运行 Qwen minimum-set 与 all-six 成对视频 QA 评审。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from egolife_two_user_qa.qwen3vl_runner import (
    GenerationCallProfile,
    MEMORY_SAFE_BACKEND,
    make_runner,
)
from egolife_two_user_qa.qwen_two_condition_review import (
    deduplicate_items,
    finalize_review,
    load_approved_markdown,
    load_curated_jsonl,
    prepare_review,
    run_items,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"
BASE_CONTRACT_KEYS = (
    "approved_markdown",
    "curated_jsonl",
    "media_root",
    "output_dir",
    "model_id",
    "backend",
    "decoding_mode",
    "qa_limit",
)
MODEL_CONTRACT_KEYS = (
    "max_new_tokens",
    "max_image_pixels",
    "disable_thinking",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-markdown", required=True)
    parser.add_argument("--curated-jsonl", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-image-pixels", type=int)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
    )
    thinking.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
    )
    parser.set_defaults(disable_thinking=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rerun-nonvalid", action="store_true")
    parser.add_argument("--qa-limit", type=int)
    return parser


def _write_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_existing_manifest(
    path: Path,
    requested: dict,
) -> dict | None:
    if not path.is_file():
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing_mode = existing.get("mode")
    requested_mode = requested.get("mode")
    if existing_mode == "model_review" and requested_mode == "prepare_only":
        raise SystemExit(
            "existing run contract differs: model_review directory "
            "cannot be reused for prepare_only"
        )
    keys = list(BASE_CONTRACT_KEYS)
    if existing_mode == requested_mode == "model_review":
        keys.extend(MODEL_CONTRACT_KEYS)
    differences = [
        key
        for key in keys
        if existing.get(key) != requested.get(key)
    ]
    if differences:
        raise SystemExit(
            "existing run contract differs: " + ", ".join(differences)
        )
    return existing


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    requested_contract = {
        "mode": "prepare_only" if args.prepare_only else "model_review",
        "approved_markdown": str(Path(args.approved_markdown).resolve()),
        "curated_jsonl": str(Path(args.curated_jsonl).resolve()),
        "media_root": str(Path(args.media_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "model_id": args.model_id,
        "backend": MEMORY_SAFE_BACKEND,
        "max_new_tokens": args.max_new_tokens,
        "max_image_pixels": args.max_image_pixels,
        "disable_thinking": args.disable_thinking,
        "decoding_mode": "greedy",
        "qa_limit": args.qa_limit,
    }
    existing_manifest = _validate_existing_manifest(
        output_dir / "run_manifest.json",
        requested_contract,
    )
    approved = load_approved_markdown(args.approved_markdown)
    curated = load_curated_jsonl(args.curated_jsonl)
    input_items = [*approved, *curated]
    if args.qa_limit is not None:
        if args.qa_limit <= 0:
            raise SystemExit("--qa-limit must be positive")
        input_items = list(
            deduplicate_items(input_items).items[: args.qa_limit]
        )
    prepared = prepare_review(
        input_items,
        args.media_root,
        output_dir,
    )
    media_report = json.loads(
        (output_dir / "media_preflight.json").read_text(encoding="utf-8")
    )
    missing_media_qa_ids = {
        row["qa_id"]
        for row in media_report["items"]
        if not row["media_ready"]
    }
    manifest = {
        "created_at_utc": (
            existing_manifest.get("created_at_utc")
            if existing_manifest is not None
            and existing_manifest.get("mode") == requested_contract["mode"]
            else datetime.now(timezone.utc).isoformat()
        ),
        **requested_contract,
        "selected_count": len(prepared.items),
        "media_ready_count": media_report["media_ready_count"],
    }
    if args.prepare_only:
        _write_manifest(output_dir / "run_manifest.json", manifest)
        finalize_review(
            prepared.items,
            output_dir,
            missing_media_qa_ids=missing_media_qa_ids,
        )
        print(f"SELECTED_COUNT={len(prepared.items)}")
        print(f"MEDIA_READY_COUNT={media_report['media_ready_count']}")
        print(f"OUTPUT_DIR={output_dir}")
        return 0
    if (
        args.max_new_tokens is None
        or args.max_image_pixels is None
        or args.disable_thinking is None
    ):
        raise SystemExit(
            "model review requires --max-new-tokens, --max-image-pixels, "
            "and exactly one of --enable-thinking/--disable-thinking"
        )
    runner = make_runner(
        MEMORY_SAFE_BACKEND,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        disable_thinking=args.disable_thinking,
    )
    manifest.update(
        {
            "effective_video_fps": getattr(runner, "video_fps", None),
            "effective_min_video_pixels": getattr(
                runner,
                "min_video_pixels",
                None,
            ),
            "effective_max_input_tokens": getattr(
                runner,
                "max_input_tokens",
                None,
            ),
        }
    )
    _write_manifest(output_dir / "run_manifest.json", manifest)
    call_profile = GenerationCallProfile(
        max_new_tokens=args.max_new_tokens,
        disable_thinking=args.disable_thinking,
    )
    run_items(
        prepared.items,
        args.media_root,
        output_dir,
        runner,
        call_profile=call_profile,
        rerun_nonvalid=args.rerun_nonvalid,
    )
    summary = finalize_review(
        prepared.items,
        output_dir,
        missing_media_qa_ids=missing_media_qa_ids,
    )
    print(f"PAIRED_COUNT={summary['paired_count']}")
    print(f"OUTPUT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
