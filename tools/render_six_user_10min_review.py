"""将十分钟六用户 QA 运行的结构化产物重排为中文人工审核 Markdown。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _short(value: Any, limit: int = 1200) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value: Any, limit: int = 800) -> str:
    return _short(value, limit).replace("|", "\\|").replace("\n", " ")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("generation_slot_id") or row.get("evidence_id") or "missing-slot"),
        str(row.get("evidence_id") or "missing-evidence"),
    )


def _qa_from_row(row: dict[str, Any]) -> dict[str, Any]:
    qa = row.get("qa")
    if isinstance(qa, dict):
        return qa
    attempts = row.get("attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and isinstance(attempt.get("qa"), dict):
                return attempt["qa"]
    return {}


def _status_from_qa(qa: dict[str, Any], fallback: str) -> str:
    decision = ((qa.get("review") or {}).get("final_decision") or {})
    if decision.get("accepted") is True:
        return "accepted"
    if decision.get("accepted") is False:
        return "rejected"
    return fallback


def _sanitized(value: Any) -> Any:
    """保留结构化 QA/评审信息，去掉重复媒体路径和模型原始长文本。"""

    omitted = {
        "raw_output",
        "initial_raw_output",
        "condition_media",
        "image_paths",
        "video_paths",
        "full_image_paths",
        "full_video_paths",
        "judge_image_paths",
        "judge_video_paths",
        "prepared_video_uploads",
        "media",
    }
    if isinstance(value, dict):
        return {
            str(key): _sanitized(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [_sanitized(item) for item in value]
    return value


def _check_rows(qa: dict[str, Any]) -> list[tuple[str, Any]]:
    checks = (((qa.get("review") or {}).get("judger") or {}).get("checks") or {})
    answerability = ((qa.get("review") or {}).get("answerability") or {}).get("gate") or {}
    return [
        ("题面形式", checks.get("qa_formality") or {}),
        ("Evidence", checks.get("evidence_groundedness") or {}),
        ("Answerability", checks.get("answerability") or answerability),
    ]


def _render_qa_card(
    *,
    number: int,
    slot_id: str,
    group_id: str,
    status: str,
    qa: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[str]:
    review = qa.get("review") or {}
    checks = (((review.get("judger") or {}).get("checks") or {}))
    answerability = review.get("answerability") or {}
    gate = answerability.get("gate") or checks.get("answerability") or {}
    evidence_check = checks.get("evidence_groundedness") or {}
    vote_summary = evidence_check.get("vote_summary") or {}
    lines = [
        f"### QA {number:03d} · {status} · `{_cell(slot_id, 180)}`",
        "",
        "| 字段 | 内容 |",
        "|---|---|",
        f"| generation group | `{_cell(group_id, 180)}` |",
        f"| evidence id | `{_cell(qa.get('evidence_id'), 180)}` |",
        f"| question type | `{_cell(qa.get('question_type'))}` |",
        f"| attempt count | `{_cell(qa.get('attempt_count') or len(attempts))}` |",
        f"| speaker | `{_cell(qa.get('speaker_user'))}` |",
        f"| question | {_cell(qa.get('question'), 1600)} |",
        f"| declared answer | `{_cell(qa.get('correct'))}` {_cell(qa.get('answer'), 900)} |",
        "",
        "**选项**",
        "",
    ]
    options = qa.get("options") or []
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        marker = " ← 声明答案" if letter == str(qa.get("correct") or "") else ""
        lines.append(f"- **{letter}.** {_cell(option, 1000)}{marker}")
    lines.extend(
        [
            "",
            "#### 三个 judge 与确定性投票",
            "",
            "| 评审项 | 状态 | 原因 |",
            "|---|---|---|",
        ]
    )
    for name, check in _check_rows(qa):
        if not isinstance(check, dict):
            check = {}
        lines.append(
            f"| {name} | **{_cell(check.get('status') or '未记录', 80)}** | "
            f"{_cell(check.get('reason'), 1000)} |"
        )
    lines.extend(
        [
            "",
            f"- Evidence 可见用户数：`{_cell(vote_summary.get('visible_user_count'))}`；"
            f"支持计数：`{_cell(vote_summary.get('option_support_counts'))}`；"
            f"阈值选项：`{_cell(vote_summary.get('threshold_options'))}`。",
            f"- Answerability：speaker-only=`{_cell(gate.get('speaker_only_answerable'))}`；"
            f"combined-all-six=`{_cell(gate.get('all_six_answerable'))}`；"
            f"条件数=`{_cell(gate.get('answerability_evaluated_condition_count'))}`。",
        ]
    )
    rationale = qa.get("generator_rationale") or qa.get("why_two_users_needed")
    if rationale:
        lines.extend([f"- 生成/多视角理由：{_cell(rationale, 1600)}", ""])
    lines.extend(
        [
            "",
            f"<details><summary>展开该 QA 的结构化生成与评审信息（已去除重复媒体路径和原始长输出）</summary>",
            "",
            "```json",
            json.dumps(_sanitized(qa), ensure_ascii=False, indent=2),
            "```",
            "",
            "</details>",
            "",
        ]
    )
    return lines


def build_report(output_dir: str | Path, output_path: str | Path) -> Path:
    root = Path(output_dir)
    output = Path(output_path)
    result = _read_json(root / "six_user_qa_result.json")
    manifest = _read_json(root / "job_manifest.json")
    accepted = _read_jsonl(root / "qa_mcq.jsonl")
    rejected = _read_jsonl(root / "qa_mcq.rejected.jsonl")
    intermediate = _read_jsonl(root / "qa_mcq.intermediate.jsonl")
    attempt_rows = _read_jsonl(root / "qa_mcq.attempts.jsonl")
    prompt_rows = _read_jsonl(root / "video_first_prompts.jsonl")
    candidates = _read_jsonl(root / "six_user_candidates.jsonl")

    attempts_by_slot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        attempts_by_slot[_identity(row)].append(row)

    final_by_slot: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for row in accepted:
        final_by_slot[_identity(row)] = ("accepted", row)
    for row in rejected:
        final_by_slot.setdefault(_identity(row), ("rejected", row))
    for row in intermediate:
        status = str(row.get("status") or "")
        if status in {"accepted", "rejected", "time_budget_partial"}:
            final_by_slot.setdefault(_identity(row), (status, row))
    for identity, rows in attempts_by_slot.items():
        if identity not in final_by_slot and rows:
            final_by_slot[identity] = (str(rows[-1].get("status") or "unknown"), rows[-1])

    def group_for(identity: tuple[str, str], row: dict[str, Any], qa: dict[str, Any]) -> str:
        return str(
            row.get("generation_group_id")
            or qa.get("generation_group_id")
            or "missing-group"
        )

    ordered_slots = sorted(
        final_by_slot.items(),
        key=lambda item: (
            group_for(item[0], item[1][1], _qa_from_row(item[1][1])),
            str(item[0][0]),
        ),
    )
    group_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for identity, (status, row) in ordered_slots:
        qa = _qa_from_row(row)
        group_counts[group_for(identity, row, qa)] += 1
        status_counts[_status_from_qa(qa, status)] += 1

    stage_counts = Counter(str(row.get("stage") or "missing") for row in prompt_rows)
    duration_value = result.get("evidence_duration_seconds") or manifest.get(
        "evidence_duration_seconds"
    )
    segment_value = result.get("expected_segments_per_user")
    if segment_value is None and duration_value:
        segment_value = float(duration_value) / 30
    if segment_value is None:
        segment_value = 20
    cross_gap_value = (
        result.get("pruning_max_cross_gap_seconds")
        or manifest.get("pruning_max_cross_gap_seconds")
        or manifest.get("cross_gap_seconds")
    )
    lines = [
        "# 六用户十分钟 QA 生成与人工审核报告",
        "",
        "> 本文由 Torch 端结构化运行产物生成，保留全部 generation slot 的最终 QA/评审信息。原始 JSONL、视频和 prompt 记录仍在同一 JobID 目录；本 Markdown 去除重复媒体路径和原始模型长输出，便于人工审阅。",
        "",
        "## 1. 结论摘要",
        "",
        f"- JobID：`{_cell(result.get('job_id') or manifest.get('job_id'))}`；运行状态：`{_cell(result.get('status') or manifest.get('status'))}`。",
        f"- 候选视频组：`{len(candidates)}`；generation group：`{_cell(result.get('generation_group_ids') or sorted(group_counts))}`。",
        f"- QA slot 总数：**{len(ordered_slots)}**；accepted：**{status_counts.get('accepted', 0)}**；rejected：**{status_counts.get('rejected', 0)}**；partial：**{status_counts.get('time_budget_partial', 0)}**。",
        f"- 目标：3 个不同六用户视频组，每组 20 个 QA；实际按 group 统计如下。",
        "",
        "## 2. 运行配置与统计",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 时长 | `{_cell(duration_value)}` 秒 |",
        f"| 每用户 segment | `{_cell(segment_value)}` |",
        f"| cross-gap | `{_cell(cross_gap_value)}` 秒 |",
        f"| Generator token 上限 | `{_cell(manifest.get('max_new_tokens'))}` |",
        f"| Formality/repair token 上限 | `{_cell(manifest.get('formality_max_new_tokens'))}` |",
        f"| reasoning profile | `{_cell(manifest.get('ten_minute_reasoning_profile') or manifest.get('reasoning_profile'))}` |",
        f"| 生成 prompt 行数 | `{stage_counts.get('generation', 0)}` |",
        f"| Formality 行数 | `{stage_counts.get('qa_formality_judge', 0)}` |",
        f"| Evidence segment 行数 | `{stage_counts.get('evidence_segment_observation', 0)}` |",
        f"| Evidence aggregation 行数 | `{stage_counts.get('evidence_groundedness_aggregation', 0)}` |",
        f"| Answerability 行数 | `{stage_counts.get('answerability', 0)}` |",
        "",
        "### 按 generation group 统计",
        "",
        "| generation group | slot 数 |",
        "|---|---:|",
    ]
    for group_id, count in sorted(group_counts.items()):
        lines.append(f"| `{_cell(group_id, 220)}` | {count} |")
    lines.extend(
        [
            "",
            "## 3. QA 逐条审核卡片",
            "",
            "每张卡片对应一个 generation slot；若该 slot 发生重试，卡片保留最后一次 QA，并在结构化展开区保留该 QA 的完整评审字段。",
            "",
        ]
    )
    for number, (identity, (status, row)) in enumerate(ordered_slots, start=1):
        qa = _qa_from_row(row)
        if not qa:
            qa = {"qa_id": identity[1], "evidence_id": identity[1]}
        slot_attempts = attempts_by_slot.get(identity, [])
        lines.extend(
            _render_qa_card(
                number=number,
                slot_id=identity[0],
                group_id=group_for(identity, row, qa),
                status=_status_from_qa(qa, status),
                qa=qa,
                attempts=slot_attempts,
            )
        )
    lines.extend(
        [
            "## 4. 原始产物索引",
            "",
            "| 文件 | 用途 |",
            "|---|---|",
            "| `six_user_qa_result.json` | 作业状态、计数和运行配置 |",
            "| `job_manifest.json` | JobID、资源、模型与阶段合同 |",
            "| `six_user_candidates.jsonl` | 三个六用户视频组及媒体映射 |",
            "| `qa_mcq.jsonl` | accepted QA 原始结构化记录 |",
            "| `qa_mcq.rejected.jsonl` | 达到最大尝试后拒绝的 slot |",
            "| `qa_mcq.intermediate.jsonl` | 每个 slot 的最终中间记录 |",
            "| `qa_mcq.attempts.jsonl` | 每一次生成/评审尝试 |",
            "| `video_first_prompts.jsonl` | 全部生成与三个 judge 的 prompt 元数据 |",
            "",
            "## 5. 人工审核边界",
            "",
            "- 本文可以逐条查看题目、选项、声明答案、三个 judge 结果、Evidence 可见用户投票摘要和 Answerability 双条件。",
            "- 本文不替代原始视频；需要判断视觉事实时，请打开对应用户的完整十分钟视频或本地拼接成片。",
            "- accepted/rejected 是自动评审结果，不等同于人工最终正确率；人工标注仍是研究结论的终点评估。",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = build_report(args.output_dir, args.output)
    print(f"HUMAN_REVIEW_MARKDOWN={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
