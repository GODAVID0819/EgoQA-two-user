"""把 EgoQA evidence packet 转成 ms-swift 原生双视频 GRPO 数据。"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable


PromptBuilder = Callable[..., str]


def _default_prompt_builder() -> PromptBuilder:
    return importlib.import_module("prompts").build_video_generation_prompt


def _video_for_user(packet: dict[str, Any], user: str) -> str:
    matches = [clip for clip in packet.get("clips", []) if str(clip.get("agent_name") or "") == user]
    if len(matches) != 1:
        raise ValueError(f"用户 {user} 必须恰好对应一段视频，实际 clips={len(matches)}")
    clip = matches[0]
    if clip.get("generator_media_mode") == "frames_only" or clip.get("force_frame_inputs"):
        raise ValueError(f"用户 {user} 使用 frames_only/sampled_frames，v3 正式入口禁止")
    value = clip.get("local_video")
    if isinstance(value, (list, tuple, dict)):
        raise ValueError(f"用户 {user} 的视频字段不是原生 .mp4 路径")
    path = Path(str(value or "")).expanduser()
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"用户 {user} 的视频不是 .mp4: {path}")
    if not path.is_file():
        raise ValueError(f"用户 {user} 的视频不存在: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"用户 {user} 的视频为空: {path}")
    return str(path.resolve())


def packet_to_swift_row(
    packet: dict[str, Any],
    *,
    question_type: str,
    generation_mode: str = "baseline",
    prompt_builder: PromptBuilder | None = None,
) -> dict[str, Any]:
    evidence_id = str(packet.get("evidence_id") or "").strip()
    if not evidence_id:
        raise ValueError("packet 缺少 evidence_id")
    required_users = [str(item).strip() for item in packet.get("required_users") or []]
    if len(required_users) != 2 or any(not item for item in required_users) or len(set(required_users)) != 2:
        raise ValueError(f"required_users 必须是恰好两个不同用户: {required_users}")
    videos = [_video_for_user(packet, user) for user in required_users]
    builder = prompt_builder or _default_prompt_builder()
    prompt = builder(packet, question_type, generation_mode=generation_mode)
    row = {
        "messages": [{"role": "user", "content": f"<video><video>\n{prompt}"}],
        "videos": videos,
        "evidence_id": evidence_id,
        "packet_json": json.dumps(packet, ensure_ascii=False),
        "question_type": question_type,
        "generation_mode": generation_mode,
        "required_users": required_users,
        "video_order": required_users,
    }
    validate_swift_row(row)
    return row


def validate_swift_row(row: dict[str, Any], *, require_files: bool = True) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("messages 必须只含一条 user 消息")
    content = str(messages[0].get("content") or "")
    if content.count("<video>") != 2:
        raise ValueError("messages 必须恰好包含两个 <video> 占位符")
    videos = row.get("videos")
    if not isinstance(videos, list) or len(videos) != 2 or any(isinstance(item, list) for item in videos):
        raise ValueError("videos 必须恰好包含两个原生视频路径，禁止帧列表")
    required_users = row.get("required_users")
    if row.get("video_order") != required_users or not isinstance(required_users, list) or len(required_users) != 2:
        raise ValueError("video_order 必须与 required_users 完全一致")
    if require_files:
        for value in videos:
            path = Path(str(value))
            if path.suffix.lower() != ".mp4" or not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"无效原生视频: {path}")


def convert_packets(
    packets: Iterable[dict[str, Any]],
    *,
    question_type: str,
    generation_mode: str,
    max_prompts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet in packets:
        if len(rows) >= max_prompts:
            break
        rows.append(
            packet_to_swift_row(
                packet,
                question_type=question_type,
                generation_mode=generation_mode,
            )
        )
    if not rows:
        raise ValueError("没有可转换的 evidence packet")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 ms-swift v3 原生双视频 GRPO JSONL")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-type", default="commonality")
    parser.add_argument("--generation-mode", default="baseline")
    parser.add_argument("--max-prompts", type=int, default=1)
    args = parser.parse_args()
    rows = convert_packets(
        read_jsonl(args.evidence),
        question_type=args.question_type,
        generation_mode=args.generation_mode,
        max_prompts=args.max_prompts,
    )
    write_jsonl(args.output, rows)
    preview = args.output.with_name("dataset_preview.json")
    preview.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output), "preview": str(preview)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
