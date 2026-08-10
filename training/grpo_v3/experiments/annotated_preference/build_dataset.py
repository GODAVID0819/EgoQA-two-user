"""从严格人工 F/E/A 标注构建 Pareto DPO 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.grpo_v3.experiments.annotated_preference.pareto import (
    PairAudit,
    PreferencePair,
    build_pareto_pairs,
    compact_fingerprint,
    dominates,
)
from training.grpo_v3.experiments.annotated_preference.prompting import (
    COMPACT_QA_CONTRACT,
    PROMPT_REVISION,
    build_compact_generation_prompt,
    prompt_sha256,
    serialize_compact_completion,
)
from training.grpo_v3.experiments.human_preference_reviewer.v1.data import (
    CandidateRecord,
    EvidenceRecord,
    load_annotation_csv,
    validate_split_manifest,
)

PREFERENCE_SOURCE = "human_fea_pareto_v1"
OUTPUT_FILENAMES = (
    "train_dpo.jsonl",
    "validation_dpo.jsonl",
    "train_pair_index.jsonl",
    "validation_pair_index.jsonl",
    "overfit_4_dpo.jsonl",
    "pareto_audit.json",
    "dataset_manifest.json",
)


@dataclass(frozen=True)
class BuildOutputs:
    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
    train_index: tuple[dict[str, Any], ...]
    validation_index: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def _score_vector(candidate: CandidateRecord) -> tuple[int, int, int]:
    labels = candidate.labels()
    return (
        labels["qa_formality"],
        labels["evidence_quality"],
        labels["answerability"],
    )


def _validated_local_media_path(value: object, *, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"media map value for {source!r} must be a path string")
    if "://" in value or not Path(value).is_absolute():
        raise ValueError(f"media map value for {source!r} must be a local absolute path")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"media file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"media file is empty: {path}")
    return str(path.resolve())


def _pair_is_valid(pair: PreferencePair, evidence: EvidenceRecord) -> None:
    if pair.evidence_id != evidence.evidence_id:
        raise ValueError("Pareto pair evidence_id does not match its evidence")
    if pair.chosen.evidence_id != evidence.evidence_id or pair.rejected.evidence_id != evidence.evidence_id:
        raise ValueError("Pareto candidate evidence_id does not match its evidence")
    if pair.chosen.candidate_id == pair.rejected.candidate_id:
        raise ValueError("chosen and rejected candidate IDs must differ")
    if pair.chosen_fingerprint == pair.rejected_fingerprint:
        raise ValueError("chosen and rejected fingerprints must differ")
    if pair.chosen_fingerprint != compact_fingerprint(evidence.evidence_id, pair.chosen):
        raise ValueError("chosen fingerprint does not match candidate content")
    if pair.rejected_fingerprint != compact_fingerprint(evidence.evidence_id, pair.rejected):
        raise ValueError("rejected fingerprint does not match candidate content")
    if not dominates(_score_vector(pair.chosen), _score_vector(pair.rejected)):
        raise ValueError("chosen score vector must Pareto-dominate rejected score vector")


def build_dpo_row(
    pair: PreferencePair,
    evidence: EvidenceRecord,
    media_map: Mapping[str, str],
) -> dict[str, Any]:
    """生成一个只含 ms-swift 官方 DPO 字段的训练行。"""
    _pair_is_valid(pair, evidence)
    if evidence.video_a_user != "A / Speaker" or evidence.video_b_user != "B / Provider":
        raise ValueError(
            f"evidence {evidence.evidence_id} roles must be A / Speaker then B / Provider"
        )
    try:
        speaker_value = media_map[evidence.video_a_source]
        provider_value = media_map[evidence.video_b_source]
    except KeyError as error:
        raise ValueError(f"media map is missing source: {error.args[0]}") from error
    speaker_path = _validated_local_media_path(speaker_value, source=evidence.video_a_source)
    provider_path = _validated_local_media_path(provider_value, source=evidence.video_b_source)
    if Path(speaker_path) == Path(provider_path):
        raise ValueError(f"evidence {evidence.evidence_id} speaker/provider media paths must differ")
    return {
        "messages": [
            {"role": "user", "content": build_compact_generation_prompt()},
            {"role": "assistant", "content": serialize_compact_completion(pair.chosen)},
        ],
        "rejected_response": serialize_compact_completion(pair.rejected),
        "videos": [speaker_path, provider_path],
    }


def pair_index_row(pair: PreferencePair) -> dict[str, Any]:
    """保存不进入模型输入的偏好对审计字段。"""
    if pair.chosen.candidate_id == pair.rejected.candidate_id:
        raise ValueError("chosen and rejected candidate IDs must differ")
    if pair.chosen_fingerprint == pair.rejected_fingerprint:
        raise ValueError("chosen and rejected fingerprints must differ")
    return {
        "evidence_id": pair.evidence_id,
        "chosen_candidate_id": pair.chosen.candidate_id,
        "rejected_candidate_id": pair.rejected.candidate_id,
        "chosen_scores": list(_score_vector(pair.chosen)),
        "rejected_scores": list(_score_vector(pair.rejected)),
        "chosen_fingerprint": pair.chosen_fingerprint,
        "rejected_fingerprint": pair.rejected_fingerprint,
    }


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_and_resolve_media_map(
    raw_media_map: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
) -> dict[str, str]:
    sources = {
        source
        for item in evidence
        for source in (item.video_a_source, item.video_b_source)
    }
    if len(sources) != 140:
        raise ValueError(f"eligible evidence must reference exactly 140 distinct media sources; found {len(sources)}")
    if set(raw_media_map) != sources:
        missing = sorted(sources - set(raw_media_map))
        extra = sorted(set(raw_media_map) - sources)
        raise ValueError(f"media map keys must exactly cover sources; missing={missing}, extra={extra}")
    resolved = {
        source: _validated_local_media_path(raw_media_map[source], source=source)
        for source in sorted(sources)
    }
    for item in evidence:
        if item.video_a_user != "A / Speaker" or item.video_b_user != "B / Provider":
            raise ValueError(
                f"evidence {item.evidence_id} roles must be A / Speaker then B / Provider"
            )
        if Path(resolved[item.video_a_source]) == Path(resolved[item.video_b_source]):
            raise ValueError(f"evidence {item.evidence_id} speaker/provider media paths must differ")
    return resolved


def _sum_pair_audits(audits: Sequence[PairAudit]) -> dict[str, int]:
    return {
        "total_combinations": sum(item.total_combinations for item in audits),
        "dominance_pair_count": sum(item.dominance_pair_count for item in audits),
        "equal_vector_pair_count": sum(item.equal_vector_pair_count for item in audits),
        "incomparable_pair_count": sum(item.incomparable_pair_count for item in audits),
        "duplicate_candidate_count": sum(item.duplicate_candidate_count for item in audits),
    }


def _build_split(
    evidence_ids: Sequence[str],
    evidence_by_id: Mapping[str, EvidenceRecord],
    media_map: Mapping[str, str],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    pair_audits: list[PairAudit] = []
    per_evidence: list[dict[str, Any]] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id[evidence_id]
        pairs, pair_audit = build_pareto_pairs(evidence)
        if pair_audit.total_combinations != (
            pair_audit.dominance_pair_count
            + pair_audit.equal_vector_pair_count
            + pair_audit.incomparable_pair_count
        ):
            raise ValueError(f"evidence {evidence_id} Pareto audit combinations do not conserve")
        if len(pairs) != pair_audit.dominance_pair_count:
            raise ValueError(f"evidence {evidence_id} Pareto pair count does not match audit")
        pair_audits.append(pair_audit)
        for pair in pairs:
            _pair_is_valid(pair, evidence)
            rows.append(build_dpo_row(pair, evidence, media_map))
            index.append(pair_index_row(pair))
        per_evidence.append({
            "evidence_id": evidence_id,
            "pair_count": len(pairs),
            **asdict(pair_audit),
        })
    totals = _sum_pair_audits(pair_audits)
    if totals["total_combinations"] != (
        totals["dominance_pair_count"]
        + totals["equal_vector_pair_count"]
        + totals["incomparable_pair_count"]
    ):
        raise ValueError("Pareto audit combinations do not conserve")
    if totals["dominance_pair_count"] != len(rows) or len(rows) != len(index):
        raise ValueError("Pareto rows, index, and audit pair counts do not agree")
    summary = {
        "evidence_count": len(evidence_ids),
        "evidence_with_pairs_count": sum(item["pair_count"] > 0 for item in per_evidence),
        "pair_count": len(rows),
        **totals,
        "zero_pair_evidence_ids": [item["evidence_id"] for item in per_evidence if item["pair_count"] == 0],
        "per_evidence": per_evidence,
    }
    return tuple(rows), tuple(index), summary


def _validate_outputs(outputs: BuildOutputs, manifest: Mapping[str, Any]) -> None:
    expected = {
        "train": set(manifest["train_evidence_ids"]),
        "validation": set(manifest["validation_evidence_ids"]),
    }
    actual: dict[str, set[str]] = {}
    for name, rows, index in (
        ("train", outputs.train_rows, outputs.train_index),
        ("validation", outputs.validation_rows, outputs.validation_index),
    ):
        if len(rows) != len(index):
            raise ValueError(f"{name} row/index counts do not agree")
        if any(set(row) != {"messages", "rejected_response", "videos"} for row in rows):
            raise ValueError(f"{name} contains a non-official training row schema")
        actual[name] = {str(row["evidence_id"]) for row in index}
        if not actual[name] <= expected[name]:
            raise ValueError(f"{name} pair index contains evidence outside its split")
        summary = outputs.audit["splits"][name]
        if summary["pair_count"] != len(rows):
            raise ValueError(f"{name} audit pair count does not agree with rows")
        expected_with_pairs = {
            item["evidence_id"] for item in summary["per_evidence"] if item["pair_count"] > 0
        }
        if actual[name] != expected_with_pairs:
            raise ValueError(f"{name} pair index does not cover every evidence with pairs")
    if actual["train"] & actual["validation"]:
        raise ValueError("train and validation pair indexes overlap")


def build_outputs(
    csv_path: str | Path,
    split_path: str | Path,
    media_map_path: str | Path,
) -> BuildOutputs:
    """严格验证三项输入并在内存中构建 train/validation 输出。"""
    annotation_audit = load_annotation_csv(csv_path)
    manifest = _load_json_object(split_path, label="split manifest")
    validate_split_manifest(manifest, expected_counts=(60, 10, 0), require_contract=True)
    if manifest["reserve_evidence_ids"]:
        raise ValueError("reserve_evidence_ids must be empty")
    if str(manifest["csv_sha256"]).lower() != annotation_audit.csv_sha256.lower():
        raise ValueError("split manifest csv_sha256 does not match annotation CSV")

    eligible = annotation_audit.eligible_evidence
    if len(eligible) != 70:
        raise ValueError(f"expected exactly 70 eligible evidence; found {len(eligible)}")
    evidence_by_id = {item.evidence_id: item for item in eligible}
    if len(evidence_by_id) != 70:
        raise ValueError("eligible evidence IDs must be unique")
    all_split_ids = (
        manifest["train_evidence_ids"]
        + manifest["validation_evidence_ids"]
        + manifest["locked_test_evidence_ids"]
        + manifest["reserve_evidence_ids"]
    )
    if set(all_split_ids) != set(evidence_by_id) or len(all_split_ids) != len(set(all_split_ids)):
        missing = sorted(set(evidence_by_id) - set(all_split_ids))
        extra = sorted(set(all_split_ids) - set(evidence_by_id))
        raise ValueError(f"split IDs must exactly cover eligible evidence; missing={missing}, extra={extra}")

    raw_media_map = _load_json_object(media_map_path, label="media map")
    media_map = _validate_and_resolve_media_map(raw_media_map, eligible)
    train_rows, train_index, train_audit = _build_split(
        manifest["train_evidence_ids"], evidence_by_id, media_map
    )
    validation_rows, validation_index, validation_audit = _build_split(
        manifest["validation_evidence_ids"], evidence_by_id, media_map
    )
    overfit_evidence_ids = sorted(
        item["evidence_id"] for item in train_audit["per_evidence"] if item["pair_count"] > 0
    )[:4]
    if len(overfit_evidence_ids) != 4:
        raise ValueError("train split must contain at least four evidence IDs with Pareto pairs")
    total_audit = {
        key: train_audit[key] + validation_audit[key]
        for key in (
            "total_combinations", "dominance_pair_count", "equal_vector_pair_count",
            "incomparable_pair_count", "duplicate_candidate_count", "pair_count",
        )
    }
    total_audit["evidence_count"] = 70
    total_audit["evidence_with_pairs_count"] = (
        train_audit["evidence_with_pairs_count"] + validation_audit["evidence_with_pairs_count"]
    )
    audit = {
        "compact_qa_contract": COMPACT_QA_CONTRACT,
        "preference_source": PREFERENCE_SOURCE,
        "eligible_evidence_count": 70,
        "splits": {"train": train_audit, "validation": validation_audit},
        "totals": total_audit,
        "overfit_evidence_ids": overfit_evidence_ids,
    }
    outputs = BuildOutputs(train_rows, validation_rows, train_index, validation_index, audit)
    _validate_outputs(outputs, manifest)
    return outputs


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    )
    return (text + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overfit_rows(outputs: BuildOutputs) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row, index in zip(outputs.train_rows, outputs.train_index):
        grouped.setdefault(str(index["evidence_id"]), []).append(row)
    return tuple(
        row
        for evidence_id in outputs.audit["overfit_evidence_ids"]
        for row in grouped[evidence_id]
    )


def _publish_staging(staging: Path, output_dir: Path) -> None:
    backup = Path(tempfile.mkdtemp(prefix=".annotated_preference-backup-", dir=output_dir))
    backed_up: list[str] = []
    published: list[str] = []
    cleanup_backup = False
    try:
        for name in OUTPUT_FILENAMES:
            target = output_dir / name
            if target.exists():
                target.replace(backup / name)
                backed_up.append(name)
        for name in OUTPUT_FILENAMES:
            (staging / name).replace(output_dir / name)
            published.append(name)
        cleanup_backup = True
    except BaseException as publish_error:
        rollback_errors: list[Exception] = []
        for name in reversed(published):
            target = output_dir / name
            try:
                if target.exists():
                    target.unlink()
            except Exception as error:  # pragma: no cover - exceptional filesystem damage
                rollback_errors.append(error)
        for name in backed_up:
            old = backup / name
            target = output_dir / name
            try:
                if target.exists():
                    target.unlink()
                if old.exists():
                    old.replace(target)
            except Exception as error:  # pragma: no cover - exceptional filesystem damage
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                "dataset publish failed and rollback was incomplete; "
                f"backup preserved at {backup.resolve()}: {rollback_errors}"
            ) from publish_error
        cleanup_backup = True
        raise
    finally:
        if cleanup_backup:
            shutil.rmtree(backup, ignore_errors=True)


def publish_dataset(
    csv_path: str | Path,
    split_path: str | Path,
    media_map_path: str | Path,
    output_dir: str | Path,
) -> None:
    """构建七个文件并以可回滚事务发布到输出目录。"""
    csv_input = Path(csv_path).resolve()
    split_input = Path(split_path).resolve()
    media_map_input = Path(media_map_path).resolve()
    outputs = build_outputs(csv_input, split_input, media_map_input)
    overfit_rows = _overfit_rows(outputs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".annotated_preference-staging-", dir=output))
    try:
        payloads = {
            "train_dpo.jsonl": _jsonl_bytes(outputs.train_rows),
            "validation_dpo.jsonl": _jsonl_bytes(outputs.validation_rows),
            "train_pair_index.jsonl": _jsonl_bytes(outputs.train_index),
            "validation_pair_index.jsonl": _jsonl_bytes(outputs.validation_index),
            "overfit_4_dpo.jsonl": _jsonl_bytes(overfit_rows),
            "pareto_audit.json": _json_bytes(outputs.audit),
        }
        manifest = {
            "compact_qa_contract": COMPACT_QA_CONTRACT,
            "prompt_revision": PROMPT_REVISION,
            "prompt_sha256": prompt_sha256(),
            "preference_source": PREFERENCE_SOURCE,
            "inputs": {
                "csv": {"path": str(csv_input), "sha256": _sha256_file(csv_input)},
                "split": {"path": str(split_input), "sha256": _sha256_file(split_input)},
                "media_map": {"path": str(media_map_input), "sha256": _sha256_file(media_map_input)},
            },
            "counts": {
                "train_evidence_count": outputs.audit["splits"]["train"]["evidence_count"],
                "train_pair_count": len(outputs.train_rows),
                "validation_evidence_count": outputs.audit["splits"]["validation"]["evidence_count"],
                "validation_pair_count": len(outputs.validation_rows),
                "overfit_evidence_count": len(outputs.audit["overfit_evidence_ids"]),
                "overfit_pair_count": len(overfit_rows),
            },
            "outputs": {
                name: {"sha256": _sha256_bytes(payloads[name])}
                for name in OUTPUT_FILENAMES[:-1]
            },
        }
        payloads["dataset_manifest.json"] = _json_bytes(manifest)
        for name in OUTPUT_FILENAMES:
            (staging / name).write_bytes(payloads[name])
        for name in OUTPUT_FILENAMES:
            path = staging / name
            if _sha256_file(path) != _sha256_bytes(payloads[name]):
                raise OSError(f"staged output hash mismatch: {name}")
        expected_line_counts = {
            "train_dpo.jsonl": len(outputs.train_rows),
            "validation_dpo.jsonl": len(outputs.validation_rows),
            "train_pair_index.jsonl": len(outputs.train_index),
            "validation_pair_index.jsonl": len(outputs.validation_index),
            "overfit_4_dpo.jsonl": len(overfit_rows),
        }
        for name, expected_count in expected_line_counts.items():
            actual_count = len((staging / name).read_text(encoding="utf-8").splitlines())
            if actual_count != expected_count:
                raise OSError(f"staged output row count mismatch: {name}")
        _publish_staging(staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="构建并发布 Pareto DPO 数据集")
    build.add_argument("--csv", required=True)
    build.add_argument("--split", required=True)
    build.add_argument("--media-map", required=True)
    build.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        publish_dataset(args.csv, args.split, args.media_map, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
