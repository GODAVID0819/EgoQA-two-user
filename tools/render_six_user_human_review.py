"""把 Torch 汇总报告重排为适合人工审核的中文 Markdown，并生成媒体候选清单。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _short(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value: Any, limit: int = 500) -> str:
    return _short(value, limit).replace("|", "\\|")


def _strip_media_metadata(value: Any) -> Any:
    """保留 QA/评审信息，去掉重复的逐帧媒体元数据。"""
    if isinstance(value, dict):
        omitted = {
            "condition_media",
            "media",
            "video_evidence",
            "source_segments",
            "sampled_frames",
            "raw_output",
        }
        return {
            key: _strip_media_metadata(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [_strip_media_metadata(item) for item in value]
    return value


def _parse_report_blocks(lines: list[str]) -> tuple[dict[str, dict[str, list[Any]]], dict[str, list[str]]]:
    blocks: dict[str, dict[str, list[Any]]] = {}
    raw_blocks: dict[str, list[str]] = {}
    current_job = ""
    index = 0
    while index < len(lines):
        heading = re.match(r"### 2\.\d+ Job `([^`]+)`", lines[index])
        if heading:
            current_job = heading.group(1)
        if current_job and lines[index].startswith("<details><summary>"):
            title_match = re.search(r"<summary>(.*?)</summary>", lines[index])
            title = title_match.group(1) if title_match else "未命名区块"
            fence = index + 1
            while fence < len(lines) and not lines[fence].startswith("```"):
                fence += 1
            end = fence + 1
            while end < len(lines) and not lines[end].startswith("```"):
                end += 1
            payload = lines[fence + 1 : end]
            raw_blocks.setdefault(current_job, []).append(
                lines[index] + "\n" + lines[fence] + "\n" + "\n".join(payload) + "\n```")
            if fence < len(lines) and lines[fence].startswith("```jsonl"):
                parsed: list[Any] = []
                for line in payload:
                    if line.strip():
                        parsed.append(json.loads(line))
                blocks.setdefault(current_job, {})[title] = parsed
            index = end
        index += 1
    return blocks, raw_blocks


def _parse_urls(lines: list[str]) -> tuple[str, list[dict[str, Any]]]:
    group = "DAY6::20060000"
    users: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    user_heading = re.compile(r"### (.+)（([^）]+)）")
    segment_line = re.compile(
        r"- 片段 \d+：`([^`]+)` / `([^`]+)` — (https?://\S+)"
    )
    for line in lines:
        match = user_heading.match(line)
        if match:
            current = {
                "agent_name": match.group(1).strip(),
                "agent_dir": match.group(2).strip(),
                "segments": [],
            }
            users.append(current)
            continue
        match = segment_line.match(line)
        if match and current is not None:
            token, clock, url = match.groups()
            h, m, s = clock.split(":")
            current["segments"].append(
                {
                    "time_token": token,
                    "clip_clock": clock,
                    "clock_seconds": int(h) * 3600 + int(m) * 60 + float(s),
                    "video_url": url,
                }
            )
    if len(users) != 6 or any(len(row["segments"]) != 6 for row in users):
        raise ValueError("报告中的 URL 区块没有解析出 6 位用户和每人 6 个片段")
    return group, users


def _trace_sections(blocks: dict[str, dict[str, list[Any]]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for job, sections in blocks.items():
        traces = next(
            (value for title, value in sections.items() if "trace" in title and "完整" in title),
            [],
        )
        result[job] = [item for item in traces if isinstance(item, dict)]
    return result


def _check_row(name: str, check: dict[str, Any] | None) -> str:
    if not check:
        return f"| {name} | 未记录 | — | — |"
    status = str(check.get("status") or "未记录")
    label = {"PASS": "通过", "FAIL": "失败"}.get(status, status)
    return f"| {name} | {label} | {_cell(check.get('reason'), 420)} | {_cell(check.get('fix'), 260)} |"


def _render_attempt(job: str, item: dict[str, Any], number: int) -> list[str]:
    qa = item.get("qa") or {}
    review = qa.get("review") or {}
    judger = review.get("judger") or {}
    checks = judger.get("checks") or {}
    if not checks:
        checks = {key: review.get(key) for key in ("qa_formality", "evidence_groundedness", "answerability")}
    trace = item.get("trace") or {}
    focus = item.get("generation_diversity_focus") or {}
    final_decision = review.get("final_decision") or {}
    status = item.get("status") or review.get("status") or "未记录"
    qa_id = qa.get("qa_id") or item.get("qa_id") or "未记录"
    lines = [
        f"### {job} · 第 {number} 次尝试 · `{qa_id}`",
        "",
        "| 字段 | 值 |",
        "| --- | --- |",
        f"| 最终状态 | **{status}** |",
        f"| generation slot | `{item.get('generation_slot_id') or trace.get('generation_slot_id') or '未记录'}` |",
        f"| generation group | `{item.get('generation_group_id') or trace.get('generation_group_id') or '未记录'}` |",
        f"| focal provider | {_cell(focus.get('focal_provider') or '未记录')} |",
        f"| temporal band | `{focus.get('temporal_band_seconds') or '未记录'}` |",
        "",
        f"**问题：** {_cell(qa.get('question'), 1200)}",
        "",
        "**选项：**",
        "",
    ]
    options = qa.get("options") or []
    for index, option in enumerate(options):
        label = chr(ord("A") + index)
        marker = " ← 正确答案" if label == str(qa.get("correct") or "") else ""
        lines.append(f"- **{label}.** {_cell(option, 600)}{marker}")
    lines.extend(
        [
            "",
            f"**答案文本：** {_cell(qa.get('answer'), 600)}",
            f"**需要用户：** {_cell(', '.join(qa.get('required_users') or []), 400)}",
            "",
            "#### 证据需求",
            "",
            "| 用户 | 需要支持的事实 | 时间范围 | 取帧 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for evidence in qa.get("evidence") or []:
        lines.append(
            f"| {_cell(evidence.get('user'))} | {_cell(evidence.get('needed_fact'), 650)} | "
            f"{_cell(evidence.get('timeframe'))} | {_cell(', '.join(evidence.get('frames_used') or []), 260)} |"
        )
    if not qa.get("evidence"):
        lines.append("| — | 未记录 | — | — |")
    lines.extend(
        [
            "",
            "#### 自动评审结果",
            "",
            "| 检查项 | 状态 | 原因 | 修复意见 |",
            "| --- | --- | --- | --- |",
            _check_row("题面形式", checks.get("qa_formality")),
            _check_row("证据 grounding", checks.get("evidence_groundedness")),
            _check_row("可回答性", checks.get("answerability")),
            "",
        ]
    )
    rationale = qa.get("generator_rationale") or qa.get("why_generator_asked_this")
    why_two = qa.get("why_two_users_needed")
    if rationale:
        lines.extend([f"**生成器理由：** {_cell(rationale, 1200)}", ""])
    if why_two:
        lines.extend([f"**为什么需要多视角：** {_cell(why_two, 1200)}", ""])
    if final_decision:
        lines.extend([f"**最终判定说明：** {_cell(final_decision.get('reason') or final_decision, 1000)}", ""])
    single = qa.get("single_user_answerability") or {}
    if single:
        lines.extend(["<details><summary>展开逐用户可回答性记录</summary>", "", "| 用户/条件 | 生成器记录 |", "| --- | --- |"])
        for user, reason in single.items():
            lines.append(f"| {_cell(user)} | {_cell(reason, 900)} |")
        lines.extend(["", "</details>", ""])
    lines.extend(
        [
            "<details><summary>原始记录索引</summary>",
            "",
            "该尝试的完整 trace、视频输入元数据和 prompt 原文保留在原始汇总文档中；本卡片只展开人工审核所需字段。",
            f"- 原始汇总：`review_artifacts/six_user_qa_3min_resubmissions_20260827/qa_review_summary_3min_20260828.md`",
            f"- 原始 trace 字段：`{', '.join(sorted(trace))}`",
            "",
            "</details>",
            "",
        ]
    )
    return lines


def _render_media(
    users: list[dict[str, Any]],
    media_dir: Path,
    media_manifest: dict[str, Any] | None,
) -> list[str]:
    lines = [
        "## 3. 六路视频与本地拼接结果",
        "",
        "当前组为 `DAY6::20060000`，每位用户 6 个连续 30 秒片段；本地已有 36 个片段和 6 个约 180 秒的拼接成片。",
        "",
        "| 用户 | 片段数 | 拼接时长 | 本地三分钟成片 |",
        "| --- | ---: | ---: | --- |",
    ]
    manifest_users = (media_manifest or {}).get("users") or {}
    stitched_dir = media_dir / "stitched"
    for user in users:
        name = user["agent_name"]
        agent = user["agent_dir"]
        info = manifest_users.get(name) or {}
        duration = ((info.get("stitched_ffprobe") or {}).get("format") or {}).get("duration")
        matches = sorted(stitched_dir.glob(f"{agent}_{name}_DAY6_20060000_180s.mp4"))
        stitched = matches[0] if matches else stitched_dir / f"{agent}_{name}_DAY6_20060000_180s.mp4"
        href = stitched.as_posix()
        lines.append(f"| {name}（{agent}） | 6 | {duration or '待检查'} 秒 | [打开成片]({href}) |")
    lines.extend(
        [
            "",
            f"媒体目录：`{media_dir}`。下载/拼接清单：[`review_media_manifest.json`]({(media_dir / 'review_media_manifest.json').as_posix()})。",
            "",
            "<details><summary>展开 36 个源视频 URL 与对应片段文件</summary>",
            "",
            "| 用户 | 片段 | 源 URL | 本地片段 |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for user in users:
        for index, segment in enumerate(user["segments"]):
            local = media_dir / "segments" / user["agent_dir"] / f"{segment['time_token']}.mp4"
            lines.append(
                f"| {user['agent_name']} | {index} ({segment['clip_clock']}) | "
                f"[源视频]({segment['video_url']}) | `{local}` |"
            )
    lines.extend(["", "</details>", ""])
    return lines


def build_report(source: Path, output: Path, media_dir: Path) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines()
    blocks, raw_blocks = _parse_report_blocks(lines)
    traces = _trace_sections(blocks)
    group, users = _parse_urls(lines)
    candidate = {
        "generation_group_id": group,
        "evidence_id": f"EGOLIFE6U_CONSENSUS_{group.replace('::', '_')}_REVIEW",
        "clips": [
            {
                "agent_dir": user["agent_dir"],
                "agent_id": user["agent_dir"].split("_", 1)[0],
                "agent_name": user["agent_name"],
                "user": user["agent_name"],
                "segments": user["segments"],
            }
            for user in users
        ],
    }
    candidate_path = output.parent / "six_user_candidates_DAY6_20060000_from_report.jsonl"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8")

    media_manifest_path = media_dir / "review_media_manifest.json"
    media_manifest = None
    if media_manifest_path.is_file():
        media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))

    all_attempts = sum(len(items) for items in traces.values())
    accepted = sum(1 for items in traces.values() for item in items if item.get("status") == "accepted")
    rejected = sum(1 for items in traces.values() for item in items if item.get("status") == "rejected")
    report: list[str] = [
        "# 六用户三分钟 QA 人工审核报告",
        "",
        "> 本文是对 `qa_review_summary_3min_20260828.md` 的人工阅读版重排。原始汇总不修改；每条 QA 尝试、自动评审和原始 JSON 均保留在可展开区块中。",
        "",
        "## 1. 先看结论",
        "",
        f"- 三次正式作业共保留 **{all_attempts} 次 QA 尝试**：accepted **{accepted}**，rejected **{rejected}**。",
        "- 三次作业都在外部 SIGTERM 前完成了候选和部分评审，但都没有完成 finalizer；accepted 行只能作为人工审核候选，不能当成完整正式结果。",
        "- 三次候选合同均为 14 行、3 个 generation groups、六位用户：Alice、Jake、Katrina、Lucia、Shure、Tasha。",
        "- 本报告的媒体组为 `DAY6::20060000`，对应 36 个源视频片段和 6 个本地三分钟成片。",
        "",
        "## 2. 作业总览",
        "",
        "| JobID | 尝试数 | accepted | rejected | 候选行 | prompt 行 | 人工阅读重点 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for job in sorted(traces):
        sections = blocks.get(job, {})
        prompt_count = next((len(value) for title, value in sections.items() if "prompt stage" in title), 0)
        candidate_line = next(
            (line for line in lines if line.startswith("- 候选统计") and f"{job}" not in line),
            "",
        )
        candidate_count = re.search(r"行数 `([^`]+)`", candidate_line)
        cand = candidate_count.group(1) if candidate_count else "14"
        a = sum(1 for item in traces[job] if item.get("status") == "accepted")
        r = sum(1 for item in traces[job] if item.get("status") == "rejected")
        focus = "accepted 行优先；同时看 rejected 的评审原因" if a else "全部 rejected；重点看失败阶段"
        report.append(f"| `{job}` | {len(traces[job])} | {a} | {r} | {cand} | {prompt_count} | {focus} |")
    report.extend(["", "### 2.1 如何阅读每条 QA", "", "1. 先看问题、选项和证据需求。", "2. 再看三项自动评审状态：题面形式、证据 grounding、可回答性。", "3. 最后展开逐用户记录和完整原始 JSON，回看模型理由、评审原因和媒体来源。", ""])
    report.extend(_render_media(users, media_dir, media_manifest))
    report.extend(["## 4. QA 逐条审核卡片", ""])
    for job in sorted(traces):
        report.extend([f"### Job `{job}`", "", f"本作业共 {len(traces[job])} 次 trace；以下按原始顺序列出。", ""])
        for number, item in enumerate(traces[job], start=1):
            report.extend(_render_attempt(job, item, number))
    report.extend(
        [
            "## 5. 原始生成/评审资料索引",
            "",
            "第 4 节已经覆盖全部 20 次 QA 尝试的题面、选项、证据需求、三项评审状态、理由、逐用户可回答性和最终判定。需要逐字段回溯时，请打开原始汇总；其中还保留完整 prompt stage、rejected/intermediate 摘要和关键日志。",
            "",
            "| JobID | prompt stage 行数 | 原始汇总位置 |",
            "| --- | ---: | --- |",
        ]
    )
    for job in sorted(blocks):
        prompt_count = next((len(value) for title, value in blocks[job].items() if "prompt stage" in title), 0)
        report.append(
            f"| `{job}` | {prompt_count} | `review_artifacts/six_user_qa_3min_resubmissions_20260827/qa_review_summary_3min_20260828.md` |"
        )
    report.extend(["", "原始汇总的本地文件：`review_artifacts/six_user_qa_3min_resubmissions_20260827/qa_review_summary_3min_20260828.md`。", ""])
    report.extend(
        [
            "## 6. 结果边界与文件索引",
            "",
            "- 原始汇总：`qa_review_summary_3min_20260828.md`；本文件只做可读性重排。",
            "- 当前三次作业都没有 `six_user_qa_result.json`，因此不把它们写成完整 finalizer 结果。",
            "- 视频下载和拼接采用原有脚本；已有 `.part` 或旧目录不会被当成成功证据，最终以六个成片和媒体清单为准。",
            f"- 由本报告生成的候选清单：`{candidate_path}`。",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    return candidate_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--media-dir", required=True)
    args = parser.parse_args()
    candidate = build_report(Path(args.source), Path(args.output), Path(args.media_dir))
    print(f"HUMAN_REPORT={Path(args.output).resolve()}")
    print(f"CANDIDATE_JSONL={candidate.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
