"""Zero-GPU data audit and Qwen3-VL module-structure probe."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from io_utils import download_file

from .data import (
    SPLIT_MODES,
    build_external_holdout_manifest,
    build_split_manifest,
    build_train_only_manifest,
    load_annotation_csv,
)
from .lora import expected_lora_targets, locate_shared_language_layers


def build_media_map(csv_path: str | Path, dataset_root: str | Path) -> dict[str, str]:
    audit = load_annotation_csv(csv_path)
    root = Path(dataset_root)
    sources = {
        source
        for evidence in audit.eligible_evidence
        for source in (evidence.video_a_source, evidence.video_b_source)
    }
    result: dict[str, str] = {}
    marker = "/resolve/main/"
    for source in sorted(sources):
        if marker not in source:
            raise ValueError(f"unsupported media URL; expected Hugging Face resolve/main path: {source}")
        relative = source.split(marker, 1)[1]
        local = root.joinpath(*relative.split("/"))
        if not local.is_file() or local.stat().st_size <= 0:
            raise ValueError(f"materialized video is missing or empty: {local}")
        result[source] = str(local.resolve())
    return result


def _scan_reward_output_media(
    csv_path: str | Path,
    reward_output_root: str | Path,
    *,
    excluded_manifests: set[Path] | None = None,
) -> tuple[
    dict[str, tuple[str, str]],
    set[str],
    dict[str, str],
    dict[str, set[str]],
    list[Path],
]:
    """Return the usable subset of media recorded by candidate collection."""
    audit = load_annotation_csv(csv_path)
    root = Path(reward_output_root)
    if not root.is_dir():
        raise ValueError(f"reward output root is missing: {root}")

    expected_by_evidence = {
        evidence.evidence_id: (evidence.video_a_source, evidence.video_b_source)
        for evidence in audit.eligible_evidence
    }
    excluded = {path.resolve() for path in (excluded_manifests or set())}
    manifests = [
        path
        for path in sorted(root.rglob("evidence_manifest.jsonl"))
        if path.resolve() not in excluded
    ]
    seen_evidence: set[str] = set()
    candidates: dict[str, set[str]] = {
        source: set()
        for sources in expected_by_evidence.values()
        for source in sources
    }
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {manifest}:{line_number}: {error}"
                    ) from error
                evidence_id = str(record.get("evidence_id") or "").strip()
                if evidence_id not in expected_by_evidence:
                    continue
                clips = record.get("clips")
                if not isinstance(clips, list) or len(clips) != 2:
                    raise ValueError(
                        f"annotated evidence {evidence_id} must have exactly two clips in "
                        f"{manifest}:{line_number}"
                    )
                urls = tuple(str(clip.get("video_url") or "").strip() for clip in clips)
                if urls != expected_by_evidence[evidence_id]:
                    raise ValueError(
                        f"video URL/order mismatch for annotated evidence {evidence_id} in "
                        f"{manifest}:{line_number}"
                    )
                seen_evidence.add(evidence_id)
                for clip, source in zip(clips, urls):
                    full_video = str(clip.get("full_local_video") or "").strip()
                    if full_video:
                        candidates[source].add(full_video)
                        raw_path = Path(full_video)
                        if not raw_path.is_file():
                            # Candidate-collection outputs are sometimes copied between
                            # HPC users. In that case the manifest keeps the original
                            # /scratch/<user>/... absolute path even though the same
                            # root-relative media file exists below reward_output_root.
                            # Prefer the longest matching suffix to avoid basename-only
                            # collisions between different packets.
                            parts = raw_path.parts
                            for start in range(len(parts)):
                                rebased = root.joinpath(*parts[start:])
                                if rebased.is_file() and rebased.stat().st_size > 0:
                                    candidates[source].add(str(rebased))
                                    break

    result: dict[str, str] = {}
    for source, paths in sorted(candidates.items()):
        existing = sorted(
            path.resolve()
            for raw_path in paths
            if (path := Path(raw_path)).is_file() and path.stat().st_size > 0
        )
        if existing:
            result[source] = str(existing[0])
    return expected_by_evidence, seen_evidence, result, candidates, manifests


def build_media_map_from_reward_outputs(
    csv_path: str | Path, reward_output_root: str | Path
) -> dict[str, str]:
    """Strictly resolve every annotated URL through collection manifests."""
    root = Path(reward_output_root)
    expected, seen, result, candidates, manifests = _scan_reward_output_media(
        csv_path, root
    )
    if not manifests:
        raise ValueError(f"no evidence_manifest.jsonl files found under reward output root: {root}")
    missing_evidence = sorted(set(expected) - seen)
    if missing_evidence:
        preview = ", ".join(missing_evidence[:5])
        suffix = "..." if len(missing_evidence) > 5 else ""
        raise ValueError(
            f"missing candidate-collection manifests for {len(missing_evidence)} annotated "
            f"evidence IDs under {root}: {preview}{suffix}"
        )
    missing_videos = [
        f"{source} (manifest paths: {', '.join(sorted(candidates[source])) or 'none'})"
        for source in sorted(candidates)
        if source not in result
    ]
    if missing_videos:
        preview = "; ".join(missing_videos[:3])
        suffix = "..." if len(missing_videos) > 3 else ""
        raise ValueError(
            f"{len(missing_videos)} annotated full videos are missing or empty: {preview}{suffix}"
        )
    return result


def _source_identity(source: str) -> tuple[str, str, str]:
    """Return ``(day, time_token, agent_dir)`` from an EgoLife source URL."""
    marker = "/resolve/main/"
    if marker not in source:
        raise ValueError(
            f"unsupported media URL; expected Hugging Face resolve/main path: {source}"
        )
    relative = source.split(marker, 1)[1]
    parts = tuple(part for part in relative.split("/") if part)
    if len(parts) < 3 or any(part in {".", ".."} for part in parts):
        raise ValueError(f"unsafe materialized-media relative path: {relative!r}")
    agent_dir, day, filename = parts[-3:]
    match = re.fullmatch(r"(DAY\d+)_(.+)_(\d{8})\.mp4", filename, flags=re.IGNORECASE)
    if not match or match.group(1).upper() != day.upper() or match.group(2) != agent_dir:
        raise ValueError(f"unexpected EgoLife source layout: {source}")
    return day.upper(), match.group(3), agent_dir


def _import_video_target(source: str, import_run_root: Path, side: str) -> Path:
    day, time_token, agent_dir = _source_identity(source)
    return (
        import_run_root
        / "evidence_build"
        / "clip_pruned_pairs"
        / f"{day}_{time_token}"
        / "benchmark_video_pairs"
        / "0-1"
        / f"{side}_{agent_dir}_original.mp4"
    )


def materialize_media_map_from_reward_outputs(
    csv_path: str | Path,
    reward_output_root: str | Path,
    import_run_root: str | Path,
    *,
    download_missing: bool = True,
    download_workers: int = 4,
    downloader: Callable[[str, Path], Path] = download_file,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Add missing annotation media using the candidate-collection layout.

    Existing candidate-collection files always win. Missing files are placed in a
    dedicated run below ``reward_output_root`` using the same
    ``evidence_build/clip_pruned_pairs/...`` hierarchy as the collection job. A
    minimal evidence manifest makes those additions discoverable on later runs.
    """
    if download_workers <= 0:
        raise ValueError("download_workers must be positive")
    reward_root = Path(reward_output_root).resolve()
    import_root = Path(import_run_root).resolve()
    if reward_root not in import_root.parents:
        raise ValueError(
            f"import run must be a child of reward output root: {import_root} not under {reward_root}"
        )
    import_manifest = import_root / "evidence_manifest.jsonl"
    expected, seen, collected, _, manifests = _scan_reward_output_media(
        csv_path,
        reward_root,
        excluded_manifests={import_manifest},
    )
    audit = load_annotation_csv(csv_path)
    evidence_records = {item.evidence_id: item for item in audit.eligible_evidence}
    all_sources = sorted({source for pair in expected.values() for source in pair})
    import_root.mkdir(parents=True, exist_ok=True)
    result = dict(collected)
    reused: list[str] = []
    targets: dict[str, Path] = {}
    import_evidence_ids = sorted(
        evidence_id
        for evidence_id, sources in expected.items()
        if evidence_id not in seen or any(source not in collected for source in sources)
    )
    target_sources: dict[Path, str] = {}
    manifest_records: list[dict[str, Any]] = []
    for evidence_id in import_evidence_ids:
        evidence = evidence_records[evidence_id]
        clips: list[dict[str, Any]] = []
        for source, side, user in (
            (evidence.video_a_source, "left", evidence.video_a_user),
            (evidence.video_b_source, "right", evidence.video_b_user),
        ):
            if source in collected:
                local_path = Path(collected[source])
            else:
                local_path = _import_video_target(source, import_root, side)
                prior_source = target_sources.setdefault(local_path, source)
                if prior_source != source:
                    raise ValueError(
                        f"two source URLs would overwrite the same imported video: "
                        f"{prior_source} and {source} -> {local_path}"
                    )
                if local_path.is_file() and local_path.stat().st_size > 0:
                    result[source] = str(local_path.resolve())
                    reused.append(source)
                else:
                    targets[source] = local_path
            resolved_path = str(local_path.resolve())
            clips.append(
                {
                    "agent_name": user,
                    "video_url": source,
                    "source_local_video": resolved_path,
                    "original_local_video": resolved_path,
                    "full_local_video": resolved_path,
                    "benchmark_media": {
                        "judge_video": resolved_path,
                        "answerability_video": resolved_path,
                    },
                }
            )
        day, time_token, _ = _source_identity(evidence.video_a_source)
        manifest_records.append(
            {
                "schema_version": "reviewer_imported_annotation_media_v1",
                "evidence_id": evidence_id,
                "day": day,
                "time_token": time_token,
                "clips": clips,
            }
        )

    if targets and not download_missing:
        raise ValueError(
            f"{len(targets)} annotated videos are absent from collection outputs and "
            f"the imported collection run; rerun with download_missing enabled"
        )

    downloaded: list[str] = []
    errors: list[str] = []
    if targets:
        workers = min(download_workers, len(targets))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(downloader, source, target): (source, target)
                for source, target in targets.items()
            }
            for future in as_completed(futures):
                source, target = futures[future]
                try:
                    local = Path(future.result())
                    if not local.is_file() or local.stat().st_size <= 0:
                        raise ValueError(f"downloaded file is missing or empty: {local}")
                    result[source] = str(local.resolve())
                    downloaded.append(source)
                except Exception as error:
                    errors.append(f"{source}: {type(error).__name__}: {error}")
    if errors:
        preview = "; ".join(sorted(errors)[:3])
        suffix = "..." if len(errors) > 3 else ""
        raise ValueError(
            f"failed to materialize {len(errors)} annotated videos: {preview}{suffix}"
        )

    unresolved = sorted(set(all_sources) - set(result))
    if unresolved:
        raise ValueError(
            f"media materialization left {len(unresolved)} unresolved sources: "
            f"{', '.join(unresolved[:3])}"
        )
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in manifest_records
    )
    temporary_manifest = import_manifest.with_suffix(".jsonl.tmp")
    temporary_manifest.write_text(manifest_text, encoding="utf-8")
    temporary_manifest.replace(import_manifest)

    # Validate through the same strict resolver used by training. This also proves
    # that the newly written manifest follows the candidate-collection contract.
    strict_result = build_media_map_from_reward_outputs(csv_path, reward_root)
    if strict_result != dict(sorted(result.items())):
        raise ValueError("strict media-map validation disagrees with imported media map")

    missing_evidence = sorted(set(expected) - seen)
    report = {
        "status": "passed",
        "csv_path": str(Path(csv_path).resolve()),
        "reward_output_root": str(reward_root),
        "import_run_root": str(import_root),
        "import_manifest": str(import_manifest),
        "prior_manifest_count": len(manifests),
        "imported_manifest_evidence_count": len(manifest_records),
        "evidence_count": len(expected),
        "missing_manifest_evidence_count": len(missing_evidence),
        "missing_manifest_evidence_ids": missing_evidence,
        "source_count": len(all_sources),
        "resolved_from_prior_jobs_count": len(collected),
        "reused_import_count": len(reused),
        "downloaded_count": len(downloaded),
        "download_workers": min(download_workers, max(1, len(targets))),
        "reused_import_sources": sorted(reused),
        "downloaded_sources": sorted(downloaded),
    }
    return strict_result, report


def annotation_audit_report(
    csv_path: str | Path,
    *,
    train_count: int,
    validation_count: int,
    locked_test_count: int,
    seed: int = 42,
    split_mode: str = "train_validation_test",
) -> dict[str, Any]:
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"unsupported split_mode: {split_mode}")
    audit = load_annotation_csv(csv_path)
    eligible = len(audit.eligible_evidence)
    report = audit.to_dict()
    if split_mode == "train_only":
        requested_counts = (eligible, 0, 0)
    elif split_mode == "external_holdout":
        requested_counts = (0, 0, eligible)
    else:
        requested_counts = (train_count, validation_count, locked_test_count)
        if split_mode == "train_validation" and locked_test_count != 0:
            raise ValueError("train_validation requires locked_test_count=0")
        if split_mode == "train_validation_test" and locked_test_count <= 0:
            raise ValueError("train_validation_test requires a positive locked_test_count")
    required = sum(requested_counts)
    report["formal_split_gate"] = {
        "split_mode": split_mode,
        "train_evidence_count": requested_counts[0],
        "validation_evidence_count": requested_counts[1],
        "locked_test_evidence_count": requested_counts[2],
        "required_evidence_count": required,
        "eligible_evidence_count": eligible,
        "missing_evidence_count": max(0, required - eligible),
        "status": "passed" if eligible >= required else "failed",
    }
    report["status"] = "passed" if eligible >= required else "insufficient_data"
    if eligible >= required:
        if split_mode == "train_only":
            report["split_manifest"] = build_train_only_manifest(
                audit.eligible_evidence, csv_sha256=audit.csv_sha256
            )
        elif split_mode == "external_holdout":
            report["split_manifest"] = build_external_holdout_manifest(
                audit.eligible_evidence, csv_sha256=audit.csv_sha256
            )
        else:
            report["split_manifest"] = build_split_manifest(
                audit.eligible_evidence,
                train_count=train_count,
                validation_count=validation_count,
                locked_test_count=locked_test_count,
                seed=seed,
                csv_sha256=audit.csv_sha256,
                require_full_class_support=(
                    split_mode == "train_validation"
                    or min(train_count, validation_count, locked_test_count) >= 10
                ),
            )
    return report


def structure_report(model_name_or_path: str, *, expected_layer_count: int = 36) -> dict[str, Any]:
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForImageTextToText
    except ImportError as error:
        raise RuntimeError("transformers and accelerate are required for the structure probe") from error
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(config, "model_type", None) != "qwen3_vl":
        raise RuntimeError(
            f'expected model_type="qwen3_vl"; found {getattr(config, "model_type", None)!r}'
        )
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
    path, layers = locate_shared_language_layers(model)
    targets = expected_lora_targets(model, last_n=2, projections=("q_proj", "v_proj"))
    if len(layers) != expected_layer_count:
        raise RuntimeError(
            f"expected {expected_layer_count} shared language blocks; found {len(layers)} at {path}"
        )
    target_shapes = {}
    for target in targets:
        current: Any = model
        for component in target.split("."):
            current = current[int(component)] if component.isdigit() else getattr(current, component)
        target_shapes[target] = {
            "in_features": int(current.in_features),
            "out_features": int(current.out_features),
        }
    return {
        "status": "passed",
        "model_class": type(model).__name__,
        "model_type": config.model_type,
        "shared_stack_path": path,
        "shared_layer_count": len(layers),
        "target_layer_indices": [len(layers) - 2, len(layers) - 1],
        "lora_targets": list(targets),
        "target_shapes": target_shapes,
        "vision_module_present": bool(getattr(model, "visual", None) is not None or getattr(getattr(model, "model", None), "visual", None) is not None),
    }


def _write(path: str | None, value: dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    annotation = subparsers.add_parser("annotation-csv", help="audit the human annotation CSV")
    annotation.add_argument("--csv", required=True)
    annotation.add_argument("--output")
    annotation.add_argument("--split-output")
    annotation.add_argument("--train-evidence-count", type=int, default=60)
    annotation.add_argument("--validation-evidence-count", type=int, default=10)
    annotation.add_argument("--locked-test-evidence-count", type=int, default=0)
    annotation.add_argument("--seed", type=int, default=42)
    annotation.add_argument("--split-mode", choices=SPLIT_MODES, default="train_validation")
    annotation.add_argument("--require-formal-split", action="store_true")
    structure = subparsers.add_parser("structure", help="inspect Qwen3-VL shared blocks without loading weights")
    structure.add_argument("--model", required=True)
    structure.add_argument("--expected-layer-count", type=int, default=36)
    structure.add_argument("--output")
    media = subparsers.add_parser("media-map", help="resolve annotated Hugging Face URLs to local videos")
    media.add_argument("--csv", required=True)
    media_source = media.add_mutually_exclusive_group(required=True)
    media_source.add_argument("--dataset-root")
    media_source.add_argument(
        "--reward-output-root",
        help="candidate-collection output tree containing evidence_manifest.jsonl files",
    )
    media.add_argument(
        "--import-run-root",
        help=(
            "dedicated child of --reward-output-root where missing videos are written "
            "using the candidate-collection directory layout"
        ),
    )
    media.add_argument("--download-missing", action="store_true")
    media.add_argument("--download-workers", type=int, default=4)
    media.add_argument("--report-output")
    media.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "annotation-csv":
        report = annotation_audit_report(
            args.csv,
            train_count=args.train_evidence_count,
            validation_count=args.validation_evidence_count,
            locked_test_count=args.locked_test_evidence_count,
            seed=args.seed,
            split_mode=args.split_mode,
        )
        _write(args.output, report)
        if args.split_output and "split_manifest" in report:
            _write(args.split_output, report["split_manifest"])
        return 2 if args.require_formal_split and report["status"] != "passed" else 0
    if args.command == "media-map":
        if args.reward_output_root:
            if args.import_run_root:
                mapping, report = materialize_media_map_from_reward_outputs(
                    args.csv,
                    args.reward_output_root,
                    args.import_run_root,
                    download_missing=args.download_missing,
                    download_workers=args.download_workers,
                )
                if args.report_output:
                    _write(args.report_output, report)
            else:
                mapping = build_media_map_from_reward_outputs(
                    args.csv, args.reward_output_root
                )
        else:
            if args.import_run_root or args.download_missing or args.report_output:
                raise ValueError(
                    "media import options require --reward-output-root"
                )
            mapping = build_media_map(args.csv, args.dataset_root)
        _write(args.output, mapping)
        return 0
    _write(args.output, structure_report(args.model, expected_layer_count=args.expected_layer_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
