"""从成片均匀抽帧，并将已审核的题目按固定划分写成训练清单。"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from .aria12 import CONDITION, render_prompt, validate_split


def extract_window(window, media_root, output, ffmpeg, ffprobe):
    video = media_root / window["video_relative_path"]
    label = window["window_id"]
    duration = float(window["parts"][-1]["window_end_seconds"])
    stream = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-of", "json", str(video),
    ]))["streams"][0]
    fps = Fraction(stream["avg_frame_rate"])
    if fps.denominator != 1 or fps != Fraction(stream["r_frame_rate"]):
        raise ValueError(f"{label}: 需要已拼接的恒定帧率成片")
    if abs(float(stream["start_time"])) > 1e-6 or abs(float(stream["duration"]) - duration) > .1:
        raise ValueError(f"{label}: 成片时间轴与来源映射不符")
    count = math.ceil(duration / 2)
    target = output / label
    target.mkdir()
    # 233.9秒/10fps源片取第0、20、40帧……，与窗口时刻精确对应。
    stride = int(fps * 2)
    subprocess.run([
        ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-i", str(video),
        "-an", "-filter_threads", "2", "-vf", f"select=not(mod(n\\,{stride})),scale=1024:-2",
        "-fps_mode", "vfr", "-threads", "2", "-q:v", "2", str(target / "%04d.jpg"),
    ], check=True)
    frames = sorted(target.glob("*.jpg"))
    if len(frames) != count or any(f.stat().st_size == 0 for f in frames):
        raise ValueError(f"{label}: 帧数错误，预期 {count}，实际 {len(frames)}")
    cuts = sorted({math.ceil(p["window_start_seconds"] / 2) for p in window["parts"][1:]})
    return {"label": label, "frames": [f.relative_to(output).as_posix() for f in frames],
            "times_seconds": [2*i for i in range(count)], "cuts_before": cuts}


def write_manifests(dataset: Path, output: Path) -> None:
    packet = json.loads((output / "media/packet.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (dataset / "qa.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["example_id"]: row for row in rows}
    if len(rows) != 200 or len(by_id) != 200:
        raise ValueError("训练池必须恰好包含200条不同的源题")
    windows = sorted(w["label"] for w in packet["windows"])
    sets = {}
    for name, count, positives in [("train", 160, 40), ("test", 40, 10)]:
        ids = json.loads((dataset / f"{name}_ids.json").read_text(encoding="utf-8"))
        selected = [by_id[key] for key in ids]
        if len(ids) != count or len(set(ids)) != count or sum(r["verdict"] == "pass" for r in selected) != positives:
            raise ValueError(f"{name} 划分与已批准的数量不符")
        sets[name] = []
        for row in selected:
            speaker = row["speaker_window_id"]
            if speaker not in windows or row["group_id"] != packet["media_group_id"]:
                raise ValueError(f"{row['example_id']}: 媒体映射不一致")
            sets[name].append({
                "example_id": row["example_id"], "group_id": row["group_id"],
                "split_group_id": row["split_group_id"], "speaker_window_id": speaker,
                "task": "answerability", "condition_type": CONDITION, "verdict": row["verdict"],
                "frame_packet": "media/packet.json", "frame_order": [speaker] + [w for w in windows if w != speaker],
                "prompt": render_prompt(row["question"], row["options"]),
            })
    validate_split(sets["train"], sets["test"])
    for name, selected in sets.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.source_map:
        source = json.loads(args.source_map.read_text(encoding="utf-8"))
        media = args.output / "media"
        media.mkdir(parents=True)
        windows = source["windows"]
        if len(windows) != 12 or any(abs(w["parts"][-1]["window_end_seconds"]-233.9)>1e-6 for w in windows):
            raise ValueError("输入应为12个233.9秒窗口")
        with ThreadPoolExecutor(max_workers=4) as executor:
            groups = list(executor.map(lambda w: extract_window(
                w, args.source_map.parent, media, args.ffmpeg, args.ffprobe), windows))
        (media / "packet.json").write_text(json.dumps({
            "schema_version": "aria12_sampled_frames_v1", "media_group_id": "ARIA12_SCENE_PACKET_01",
            "duration_seconds": 233.9, "fps": 0.5, "windows": groups,
        }, indent=2), encoding="utf-8")
    write_manifests(args.dataset, args.output)


if __name__ == "__main__":
    main()
