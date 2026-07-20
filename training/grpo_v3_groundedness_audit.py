"""从 GRPO v3 reward trace 导出 groundedness 人工审计包并汇总人工结论。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from training.grpo_v3_json_format import validate_completion_json


SCHEMA_VERSION = "grpo_v3_groundedness_audit_v1"
HUMAN_CHOICES = {"PASS", "FAIL", "UNCERTAIN"}
HUMAN_FIELDS = (
    "human_groundedness",
    "human_combined_answerability",
    "human_speaker_leakage",
    "human_provider_answerability",
    "human_qa_formality",
    "human_shallow_activity",
)
MANUAL_AUXILIARY_FIELDS = ("claim_visible", "answer_supported", "reviewer_agreement", "notes")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON object")
            rows.append(value)
    return rows


def _judge_check(record: dict[str, Any], name: str) -> dict[str, Any]:
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    judger = review.get("judger") if isinstance(review.get("judger"), dict) else {}
    if isinstance(judger.get("merged"), dict):
        judger = judger["merged"]
    checks = judger.get("checks") if isinstance(judger.get("checks"), dict) else {}
    value = checks.get(name)
    return value if isinstance(value, dict) else {}


def _groundedness_check(record: dict[str, Any]) -> dict[str, Any]:
    return _judge_check(record, "evidence_groundedness")


def _boolean_label(value: Any, *, true_label: str, false_label: str) -> str:
    if value is True:
        return true_label
    if value is False:
        return false_label
    return ""


def _status_label(value: Any, mapping: dict[str, str]) -> str:
    return mapping.get(str(value or "").strip().upper(), "")


def _answerability_details(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answerability = record.get("answerability") if isinstance(record.get("answerability"), dict) else {}
    gate = answerability.get("gate") if isinstance(answerability.get("gate"), dict) else {}
    evaluations: list[dict[str, Any]] = []
    for value in answerability.get("evaluations") or []:
        if not isinstance(value, dict):
            continue
        uncertainty = value.get("choice_uncertainty") if isinstance(value.get("choice_uncertainty"), dict) else {}
        evaluations.append({
            "condition_id": str(value.get("condition_id") or ""),
            "condition_type": str(value.get("condition_type") or ""),
            "users": [str(user) for user in value.get("users") or []],
            "choice": str(value.get("choice") or ""),
            "answer_text": str(value.get("answer_text") or ""),
            "evidence_used": str(value.get("evidence_used") or ""),
            "insufficient_reason": str(value.get("insufficient_reason") or ""),
            "normalized_entropy": uncertainty.get("normalized_entropy"),
        })
    return gate, evaluations


def _qa(record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(record.get("qa"), dict):
        return dict(record["qa"])
    raw = str(record.get("raw_qa") or "")
    parsed = validate_completion_json(raw)
    return dict(parsed.value) if isinstance(parsed.value, dict) else {}


def _video_paths(record: dict[str, Any]) -> list[str]:
    prompts = record.get("judge_prompts") if isinstance(record.get("judge_prompts"), list) else []
    matches = [
        row for row in prompts
        if isinstance(row, dict) and row.get("stage") == "evidence_groundedness_judge"
    ]
    if not matches:
        return []
    values = matches[-1].get("video_paths")
    return [str(item) for item in values] if isinstance(values, list) else []


def _seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    match = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        hours = float(match.group(1) or 0)
        return hours * 3600 + float(match.group(2)) * 60 + float(match.group(3))
    return float(text) if re.fullmatch(r"\d+(?:\.\d+)?", text) else None


def _video_windows(qa: dict[str, Any], video_paths: list[str]) -> list[dict[str, Any]]:
    users = [str(item) for item in qa.get("required_users") or []]
    evidence = qa.get("evidence") if isinstance(qa.get("evidence"), list) else []
    timestamps = qa.get("referred_timestamps") if isinstance(qa.get("referred_timestamps"), list) else []
    windows: list[dict[str, Any]] = []
    for index, video in enumerate(video_paths):
        user = users[index] if index < len(users) else f"user_{index + 1}"
        values: list[float] = []
        for claim in evidence:
            if not isinstance(claim, dict) or str(claim.get("user") or "") != user:
                continue
            timeframe = str(claim.get("timeframe") or "")
            match = re.fullmatch(r"\s*([^\s-]+)\s*-\s*([^\s-]+)\s*", timeframe)
            if match:
                values.extend(item for item in (_seconds(match.group(1)), _seconds(match.group(2))) if item is not None)
        for timestamp in timestamps:
            if isinstance(timestamp, dict) and str(timestamp.get("user") or "") == user:
                value = _seconds(timestamp.get("timestamp_seconds"))
                if value is not None:
                    values.append(value)
        relative_values = [value for value in values if value <= 600.0]
        if relative_values:
            values = relative_values
        elif values:
            absolute_start = min(values)
            values = [value - absolute_start for value in values]
        start = max(0.0, min(values) - 2.0) if values else 0.0
        end = max(values) + 2.0 if values else 10.0
        if end <= start:
            end = start + 5.0
        windows.append({"user": user, "video_path": video, "start_seconds": round(start, 3), "end_seconds": round(end, 3)})
    return windows


def _case(trace: dict[str, Any]) -> dict[str, Any] | None:
    record = trace.get("record") if isinstance(trace.get("record"), dict) else {}
    check = _groundedness_check(record)
    status = str(record.get("groundedness_status") or check.get("status") or "").upper()
    if status not in {"PASS", "FAIL"}:
        return None
    qa = _qa(record)
    formality_check = _judge_check(record, "qa_formality")
    answerability_gate, answerability_evaluations = _answerability_details(record)
    required_users = [str(user) for user in qa.get("required_users") or []]
    speaker_user = str(answerability_gate.get("speaker_user") or (required_users[0] if required_users else ""))
    provider_user = str(answerability_gate.get("evidence_provider_user") or (required_users[1] if len(required_users) > 1 else ""))
    evidence_id = str(trace.get("evidence_id") or record.get("evidence_id") or "")
    call_index = int(trace.get("reward_call_index") or 0)
    candidate_index = int(trace.get("candidate_index") or 0)
    video_paths = _video_paths(record)
    return {
        "case_id": f"{evidence_id}__g{call_index:04d}__c{candidate_index}",
        "evidence_id": evidence_id,
        "reward_call_index": call_index,
        "candidate_index": candidate_index,
        "phase": str(trace.get("phase") or "train"),
        "reward": trace.get("reward"),
        "question_type": str(qa.get("question_type") or ""),
        "question": str(qa.get("question") or ""),
        "options": list(qa.get("options") or []),
        "correct": qa.get("correct"),
        "answer": qa.get("answer"),
        "required_users": required_users,
        "speaker_user": speaker_user,
        "provider_user": provider_user,
        "evidence_claims": list(qa.get("evidence") or qa.get("per_user_evidence_claims") or []),
        "referred_timestamps": list(qa.get("referred_timestamps") or []),
        "reviewer_groundedness": status,
        "reviewer_reason": str(check.get("reason") or ""),
        "reviewer_fix": str(check.get("fix") or ""),
        "reviewer_combined_answerability": _boolean_label(
            record.get("combined_correct"), true_label="PASS", false_label="FAIL"
        ),
        "reviewer_speaker_leakage": _boolean_label(
            record.get("speaker_only_correct"), true_label="LEAK", false_label="NO_LEAK"
        ),
        "reviewer_provider_answerability": _boolean_label(
            record.get("provider_only_correct"), true_label="ANSWERABLE", false_label="NOT_ANSWERABLE"
        ),
        "reviewer_qa_formality": str(
            record.get("qa_formality_status") or formality_check.get("status") or ""
        ).upper(),
        "reviewer_qa_formality_reason": str(formality_check.get("reason") or ""),
        "reviewer_qa_formality_fix": str(formality_check.get("fix") or ""),
        "reviewer_shallow_activity": _status_label(
            record.get("shallow_activity_status"),
            {"PASS": "NO_SHALLOW", "FAIL": "SHALLOW", "UNCERTAIN": "UNCERTAIN"},
        ),
        "reviewer_answerability_gate_passed": answerability_gate.get("passed"),
        "reviewer_answerability_gate_reason": str(answerability_gate.get("reason") or ""),
        "answerability_evaluations": answerability_evaluations,
        "reward_components": dict(record.get("reward_components") or {}),
        "format_status": str((record.get("format_validation") or {}).get("status") or ""),
        "video_paths": video_paths,
        "video_windows": _video_windows(qa, video_paths),
        "raw_completion": str(record.get("raw_qa") or ""),
    }


def _diverse_pick(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["question_type"],
            row["format_status"],
            row["reward_call_index"] % 2,
            float(row["reward"]) if isinstance(row.get("reward"), (int, float)) else 0.0,
            row["case_id"],
        ),
    )
    if count == len(ordered):
        return ordered
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)] if count > 1 else [0]
    return [ordered[index] for index in indices]


def select_audit_cases(
    traces: Iterable[dict[str, Any]], *, pass_count: int = 12, fail_count: int = 12
) -> list[dict[str, Any]]:
    cases = [case for trace in traces if (case := _case(trace)) is not None]
    selected: list[dict[str, Any]] = []
    for status, count in (("PASS", pass_count), ("FAIL", fail_count)):
        stratum = [row for row in cases if row["reviewer_groundedness"] == status]
        if len(stratum) < count:
            raise ValueError(f"groundedness {status} 可用案例 {len(stratum)}，少于要求的 {count}")
        selected.extend(_diverse_pick(stratum, count))
    return sorted(selected, key=lambda row: (row["reviewer_groundedness"], row["case_id"]))


def build_review_rows(cases: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "case_id": str(case["case_id"]),
            "evidence_id": str(case["evidence_id"]),
            "reviewer_groundedness": str(case["reviewer_groundedness"]),
            "reviewer_combined_answerability": str(case.get("reviewer_combined_answerability") or ""),
            "reviewer_speaker_leakage": str(case.get("reviewer_speaker_leakage") or ""),
            "reviewer_provider_answerability": str(case.get("reviewer_provider_answerability") or ""),
            "reviewer_qa_formality": str(case.get("reviewer_qa_formality") or ""),
            "reviewer_shallow_activity": str(case.get("reviewer_shallow_activity") or ""),
            "human_groundedness": "",
            "human_combined_answerability": "",
            "human_speaker_leakage": "",
            "human_provider_answerability": "",
            "human_qa_formality": "",
            "human_shallow_activity": "",
            "claim_visible": "",
            "answer_supported": "",
            "reviewer_agreement": "",
            "notes": "",
        }
        for case in cases
    ]


def merge_existing_reviews(
    new_rows: list[dict[str, str]], old_rows: list[dict[str, Any]]
) -> list[dict[str, str]]:
    new_by_id = {str(row.get("case_id") or ""): row for row in new_rows}
    old_by_id = {str(row.get("case_id") or ""): row for row in old_rows}
    if set(new_by_id) != set(old_by_id):
        added = sorted(set(new_by_id) - set(old_by_id))
        missing = sorted(set(old_by_id) - set(new_by_id))
        raise ValueError(f"case_id 集合不一致：新增={added}；旧表多出={missing}")
    standard_fields = set().union(*(row.keys() for row in new_rows)) if new_rows else set()
    preserved_fields = set(HUMAN_FIELDS) | set(MANUAL_AUXILIARY_FIELDS)
    merged: list[dict[str, str]] = []
    for new_row in new_rows:
        old_row = old_by_id[str(new_row["case_id"])]
        result = dict(new_row)
        for key, value in old_row.items():
            if key in preserved_fields or key not in standard_fields:
                result[str(key)] = str(value or "")
        merged.append(result)
    return merged


def summarize_reviews(
    rows: Iterable[dict[str, Any]], *, approved_for_weight_change: bool
) -> dict[str, Any]:
    completed = [
        dict(row) for row in rows
        if str(row.get("human_groundedness") or "").strip().upper() in HUMAN_CHOICES
    ]
    if len(completed) < 20:
        raise ValueError(f"至少完成 20 个案例，当前完成 {len(completed)}")
    reviewer_counts = Counter(str(row.get("reviewer_groundedness") or "").upper() for row in completed)
    if reviewer_counts["PASS"] < 8 or reviewer_counts["FAIL"] < 8:
        raise ValueError("已完成案例中 reviewer PASS 和 FAIL 必须分别至少 8 个")
    uncertain = sum(str(row["human_groundedness"]).upper() == "UNCERTAIN" for row in completed)
    agreements = sum(
        str(row["human_groundedness"]).upper() == str(row.get("reviewer_groundedness") or "").upper()
        for row in completed
    )
    by_status: dict[str, dict[str, Any]] = {}
    for status in ("PASS", "FAIL"):
        subset = [row for row in completed if str(row.get("reviewer_groundedness") or "").upper() == status]
        agree = sum(str(row["human_groundedness"]).upper() == status for row in subset)
        by_status[status] = {
            "completed": len(subset),
            "agreement_count": agree,
            "agreement_rate": agree / len(subset) if subset else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_count": len(completed),
        "reviewer_pass_completed": reviewer_counts["PASS"],
        "reviewer_fail_completed": reviewer_counts["FAIL"],
        "agreement_count": agreements,
        "agreement_rate": agreements / len(completed),
        "uncertain_count": uncertain,
        "uncertain_rate": uncertain / len(completed),
        "by_reviewer_status": by_status,
        "approved_for_weight_change": bool(approved_for_weight_change),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# GRPO v3 多信号人工审计指南",
        "",
        "本审计同时复核 groundedness、combined answerability、speaker leakage、provider-only answerability、QA formality 与 shallow activity。各信号必须分别判断，不得用一个总体印象代替。",
        "",
        "操作顺序：先阅读 QA 和观看两段视频，先独立判断，再阅读 reviewer 结论，最后填写 CSV。这样可以减少 reviewer 对人工判断的锚定。",
        "",
        "## CSV 人工字段",
        "",
        "| 字段 | 允许值 | 含义 |",
        "|---|---|---|",
        "| `human_groundedness` | PASS / FAIL / UNCERTAIN | claims 与正确答案是否有视频依据 |",
        "| `human_combined_answerability` | PASS / FAIL / UNCERTAIN | 两段视频合并后是否唯一支持正确项 |",
        "| `human_speaker_leakage` | LEAK / NO_LEAK / UNCERTAIN | speaker 单独是否已经能答对 |",
        "| `human_provider_answerability` | ANSWERABLE / NOT_ANSWERABLE / UNCERTAIN | provider 单独是否能答对；ANSWERABLE 本身不算失败 |",
        "| `human_qa_formality` | PASS / FAIL / UNCERTAIN | 问题自然度与五选项结构是否合格 |",
        "| `human_shallow_activity` | PASS / FAIL / UNCERTAIN | PASS 表示不是浅层活动问题，FAIL 表示问题过浅 |",
        "",
        "人工 answerability gate 仅在 combined=PASS 且 speaker=NO_LEAK 时通过；provider 单独可回答是允许的诊断现象。JSON format 只展示机器结果。",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        lines.extend([
            f"## {index}. {case['case_id']}",
            "",
            f"- evidence：`{case['evidence_id']}`；类型：`{case['question_type']}`；reward：`{case['reward']}`；格式：`{case['format_status']}`",
            f"- 角色：speaker=`{case.get('speaker_user') or '未记录'}`；provider=`{case.get('provider_user') or '未记录'}`",
            f"- 问题：{case['question']}",
            f"- 选项：{json.dumps(case['options'], ensure_ascii=False)}",
            f"- 正确项：`{case['correct']}`；答案：{case['answer']}",
            f"- evidence claims：{json.dumps(case['evidence_claims'], ensure_ascii=False)}",
            f"- timestamps：{json.dumps(case['referred_timestamps'], ensure_ascii=False)}",
            "",
            "### A. 先独立观看与判断",
            "",
            "先不要阅读下方 reviewer 结论。分别判断：claim 是否可见、合并视频能否唯一作答、speaker 是否泄漏答案、provider 是否能单独作答。",
            "",
        ])
        for video_index, window in enumerate(case["video_windows"], start=1):
            video = window["video_path"]
            start = window["start_seconds"]
            end = window["end_seconds"]
            lines.extend([
                f"视频 {video_index}（{window['user']}，建议窗口 {start:.1f}s–{end:.1f}s）：`{video}`",
                "",
                f"- 播放窗口：`ffplay -autoexit -ss {start} -t {end - start:.3f} -i {video}`",
                f"- 或：`mpv --start={start} --end={end} {video}`",
                f"- 截取审计副本：`ffmpeg -y -ss {start} -i {video} -t {end - start:.3f} -c:v libx264 -preset veryfast -crf 23 -an {case['case_id']}_u{video_index}.mp4`",
                "",
            ])
        lines.extend([
            "### B. Reviewer 信号（完成独立判断后再看）",
            "",
            f"- Groundedness：`{case.get('reviewer_groundedness') or '未记录'}`",
            f"  - 理由：{case.get('reviewer_reason') or '未记录'}",
            f"  - 建议：{case.get('reviewer_fix') or '未记录'}",
            f"- Combined answerability：`{case.get('reviewer_combined_answerability') or '未记录'}`",
            f"- Speaker leakage：`{case.get('reviewer_speaker_leakage') or '未记录'}`",
            f"- Provider-only answerability：`{case.get('reviewer_provider_answerability') or '未记录'}`",
            f"- Answerability gate：`{case.get('reviewer_answerability_gate_passed') if case.get('reviewer_answerability_gate_passed') is not None else '未记录'}`；理由：{case.get('reviewer_answerability_gate_reason') or '未记录'}",
            f"- QA formality：`{case.get('reviewer_qa_formality') or '未记录'}`；理由：{case.get('reviewer_qa_formality_reason') or '未记录'}；建议：{case.get('reviewer_qa_formality_fix') or '未记录'}",
            f"- Shallow activity：`{case.get('reviewer_shallow_activity') or '未记录'}`",
            f"- Reward components：`{json.dumps(case.get('reward_components') or {}, ensure_ascii=False)}`",
            "",
            "Answerability 条件明细：",
            "",
        ])
        evaluations = case.get("answerability_evaluations") or []
        if not evaluations:
            lines.extend(["- 未记录", ""])
        for evaluation in evaluations:
            lines.extend([
                f"- `{evaluation.get('condition_id') or '未记录'}`；用户：`{json.dumps(evaluation.get('users') or [], ensure_ascii=False)}`；选择：`{evaluation.get('choice') or '未记录'}`",
                f"  - answer text：{evaluation.get('answer_text') or '未记录'}",
                f"  - evidence：{evaluation.get('evidence_used') or '未记录'}",
                f"  - insufficient reason：{evaluation.get('insufficient_reason') or '无'}",
                f"  - normalized entropy：`{evaluation.get('normalized_entropy') if evaluation.get('normalized_entropy') is not None else '未记录'}`",
            ])
        lines.extend([
            "",
            "### C. 填写 CSV",
            "",
            "依次填写 `human_groundedness`、`human_combined_answerability`、`human_speaker_leakage`、`human_provider_answerability`、`human_qa_formality`、`human_shallow_activity`，并在 `notes` 中写明用户、时间和支持或反驳依据。",
            "",
        ])
    return "\n".join(lines) + "\n"


def export_audit(trace: Path, output_dir: Path, *, pass_count: int, fail_count: int) -> dict[str, Any]:
    cases = select_audit_cases(read_jsonl(trace), pass_count=pass_count, fail_count=fail_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "groundedness_audit_cases.jsonl"
    csv_path = output_dir / "groundedness_audit_review.csv"
    guide_path = output_dir / "groundedness_audit_guide_cn.md"
    clips_script = output_dir / "extract_audit_clips.sh"
    review_rows = build_review_rows(cases)
    backup_path: Path | None = None
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            review_rows = merge_existing_reviews(review_rows, list(csv.DictReader(handle)))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = csv_path.with_name(f"{csv_path.stem}.backup_{timestamp}{csv_path.suffix}")
        shutil.copy2(csv_path, backup_path)
    _write_jsonl(cases_path, cases)
    _write_csv(csv_path, review_rows)
    guide_path.write_text(_markdown(cases), encoding="utf-8")
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", 'OUT_DIR="${1:-groundedness_audit_clips}"', 'mkdir -p "${OUT_DIR}"']
    for case in cases:
        for index, window in enumerate(case["video_windows"], start=1):
            output_name = f"{case['case_id']}__u{index}.mp4"
            commands.append(
                "ffmpeg -hide_banner -loglevel error -y "
                f"-ss {window['start_seconds']} -i {shlex.quote(window['video_path'])} "
                f"-t {window['end_seconds'] - window['start_seconds']:.3f} -c:v libx264 -preset veryfast -crf 23 -an "
                f'"${{OUT_DIR}}/{output_name}"'
            )
    clips_script.write_text("\n".join(commands) + "\n", encoding="utf-8", newline="\n")
    result = {
        "cases": len(cases),
        "cases_path": str(cases_path),
        "review_csv": str(csv_path),
        "review_csv_backup": str(backup_path) if backup_path else None,
        "guide": str(guide_path),
        "extract_clips_script": str(clips_script),
    }
    (output_dir / "groundedness_audit_export.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="导出或汇总 GRPO v3 groundedness 人工审计")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--trace", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--pass-count", type=int, default=12)
    export.add_argument("--fail-count", type=int, default=12)
    summary = sub.add_parser("summarize")
    summary.add_argument("--review-csv", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--approve-weight-change", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        result = export_audit(args.trace, args.output_dir, pass_count=args.pass_count, fail_count=args.fail_count)
    else:
        with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            result = summarize_reviews(csv.DictReader(handle), approved_for_weight_change=args.approve_weight_change)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
