"""复用正式 asker 视角与 prompt，构造 ms-swift 的独立图片输入。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.judge_sft.prepare_real_data import _repo_module


def checked_packet(dataset_root: str | Path, packet_id: str, asker_index: int) -> Path:
    if not isinstance(packet_id, str) or not packet_id or Path(packet_id).name != packet_id:
        raise ValueError("source_packet_id 必须是单个目录名")
    if packet_id in {".", ".."} or "/" in packet_id or "\\" in packet_id:
        raise ValueError("source_packet_id 不得包含路径跳转")
    if isinstance(asker_index, bool) or not isinstance(asker_index, int) or not 0 <= asker_index < 6:
        raise ValueError("asker_index 必须为 0..5")
    path = Path(dataset_root).expanduser().resolve() / "packets" / packet_id
    if not (path / "packet.json").is_file():
        raise FileNotFoundError(path / "packet.json")
    return path


def make_row(dataset_root: str | Path, packet_id: str, asker_index: int, *, generation_profile=None) -> dict[str, Any]:
    packet_dir = checked_packet(dataset_root, packet_id, asker_index)
    view = _repo_module("rlhf_evidence_preprocessing").load_asker_view(
        dataset_root, packet_id, asker_index
    )
    users = view["required_users"]
    if len(users) != 6 or len(set(users)) != 6 or any(not u or u == "None" for u in users):
        raise ValueError("每个 packet 必须有六个不同的真实用户")
    paths, counts = [], []
    for index, clip in enumerate(view["clips"]):
        frames, full = clip["frames"], clip["full_frames"]
        indices = [f["frame_index"] for f in full]
        if indices != list(range(300)):
            raise ValueError("完整用户时间线必须按 0..299 排序")
        if index == 0 and frames != full:
            raise ValueError("speaker 的生成输入必须保留完整采样帧")
        for frame in full:
            path = Path(frame["path"]).resolve()
            if not path.is_relative_to(packet_dir):
                raise ValueError("媒体路径越出所属 packet")
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        paths.extend(str(Path(f["path"]).resolve()) for f in frames)
        counts.append(len(frames))
    if len(set(paths)) != len(paths):
        raise ValueError("生成输入存在重复媒体路径")
    prompt = _repo_module("prompts").build_video_generation_prompt(
        view, "neutral", generation_mode="baseline"
    )
    result = {
        "messages": [{"role": "user", "content": "\n".join(["<image>"] * len(paths)) + "\n" + prompt}],
        "images": paths, "generator_image_paths": list(paths),
        "dataset_root": str(Path(dataset_root).resolve()), "source_packet_id": packet_id,
        "asker_index": asker_index, "evidence_id": view["evidence_id"],
        "required_users": users, "generator_frame_counts": counts,
        "generation_mode": "baseline", "question_type": "neutral",
    }
    if generation_profile is not None:
        from .generation import apply_profile
        result=apply_profile(result,generation_profile)
    return result


def validate_row(row: dict[str, Any], *, require_prompt: bool = True) -> None:
    expected = make_row(row["dataset_root"], row["source_packet_id"], row["asker_index"],
                        generation_profile=row.get("generation_profile"))
    if "videos" in row:
        raise ValueError("六用户 GRPO 使用独立采样图片，不接受 videos")
    for key, value in expected.items():
        if not require_prompt and key not in {"dataset_root", "source_packet_id", "asker_index", "evidence_id", "required_users", "generator_image_paths"}:
            continue
        if row.get(key) != value:
            raise ValueError(f"{key} 与当前 packet / prompt 不一致")


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    owners: dict[str, str] = {}
    for split, rows in splits.items():
        keys = set()
        for row in rows:
            packet_id = row["source_packet_id"]
            if packet_id in owners and owners[packet_id] != split:
                raise ValueError(f"source packet 跨集合泄漏：{packet_id}")
            owners[packet_id] = split
            key = (packet_id, row["asker_index"])
            if key in keys:
                raise ValueError(f"重复 packet/asker：{key}")
            keys.add(key)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="构建或检查六用户在线 GRPO 数据")
    parser.add_argument("--selection", type=Path, help="JSON 对象：各 split 下为 source_packet_id/asker_index 列表")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate", type=Path, nargs="+")
    args = parser.parse_args()
    if args.validate:
        resolved = [path.resolve() for path in args.validate]
        if len(set(resolved)) != len(resolved):
            raise ValueError("训练和验证不能重复引用同一数据文件")
        splits = {str(path): read_rows(path) for path in resolved}
        if any(not rows for rows in splits.values()):
            raise ValueError("训练和验证数据不得为空")
        validate_splits(splits)
        for rows in splits.values():
            for row in rows:
                validate_row(row)
        print(json.dumps({"status": "passed", "rows": {k: len(v) for k, v in splits.items()}}))
        return
    if not args.selection or not args.dataset_root or not args.output_dir:
        parser.error("构建需要 --selection、--dataset-root 和 --output-dir")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or not selection or set(selection) - {"train", "validation", "test"}:
        raise ValueError("selection 只允许 train / validation / test")
    validate_splits(selection)
    splits = {name: [make_row(args.dataset_root, r["source_packet_id"], r["asker_index"]) for r in rows]
              for name, rows in selection.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in splits:
        if (args.output_dir / f"{name}.jsonl").exists():
            raise FileExistsError("不覆盖已存在的训练数据")
    for name, rows in splits.items():
        with (args.output_dir / f"{name}.jsonl").open("x", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": {k: len(v) for k, v in splits.items()}}))


if __name__ == "__main__":
    main()
