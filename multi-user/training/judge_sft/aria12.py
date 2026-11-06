"""12窗口联合C：显式新模式，旧六用户可答性语义保持不变。"""
from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from .contracts import JudgeTask, normalize_verdict
from .data import FrameSet, JudgeExample, SUPPORTED_FRAME_SUFFIXES

CONDITION = "cross_view_requirement"


def render_prompt(question: str, options: list[str]) -> str:
    """只接受模型应看到的问题与选项，不接收gold或审核理由。"""
    if not question.strip() or len(options) != 5 or any(not x.strip() for x in options):
        raise ValueError("question and exactly five nonempty options are required")
    choices = "\n".join(f"{chr(65+i)}. {text}" for i, text in enumerate(options))
    return f"""You judge the cross-view answerability requirement for one multiple-choice question.
There are twelve independent egocentric windows. The FIRST image group is the assigned speaker's window; the other eleven are providers.
PASS if and only if BOTH hold:
1. The speaker's complete window alone does NOT contain enough evidence to determine a unique correct option.
2. All twelve windows together contain enough evidence to determine exactly one correct option.
FAIL if either requirement fails. A question answerable from the first group alone is FAIL even when the combined input also answers it.
Check the whole first group, not only a convenient frame. Unsupported factual premises, unresolved references, no supported option, or multiple supported options are FAIL.
Every first-person event or experience asserted in the question must belong to the assigned speaker window. A provider's experiences do not become the speaker's memories.
Windows are NOT assumed synchronized. Equal window timestamps do not establish simultaneity. EDIT CUT marks an omitted interval, not an observed transition. Match people or objects only using distinguishing visual evidence; do not invent handoffs, routes or temporal relations.
Do not use outside knowledge, the apparent plausibility of an option, or the question wording as a substitute for missing visual evidence. No declared answer is provided.
Judge answerability only; do not impose unrelated stylistic requirements. Do not answer the multiple-choice question.
Question:
{question}
Options:
{choices}
Return only a JSON object with verdict equal to pass or fail.
"""


@lru_cache(maxsize=16)
def _packet(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "aria12_sampled_frames_v1":
        raise ValueError("unsupported Aria frame packet")
    return value


def cross_view_record_to_example(row, *, location, require_frame_files):
    if row.get("task") != "answerability":
        raise ValueError(f"{location}: joint C is an answerability task")
    for key in ("example_id", "group_id", "split_group_id", "speaker_window_id", "prompt"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{location}: missing {key}")
    if "videos" in row or "video_order" in row:
        raise ValueError("native videos must be prepared into sampled images before training")
    if "frame_packet" in row:
        if "frame_sets" in row:
            raise ValueError("choose packet references or inline frames, not both")
        packet_path = Path(row["frame_packet"]).expanduser().resolve()
        packet = _packet(str(packet_path))
        if packet.get("media_group_id") != row["group_id"]:
            raise ValueError("media group mismatch")
        windows = packet["windows"]
        if len(windows) != 12 or len({w["label"] for w in windows}) != 12:
            raise ValueError("frame packet must contain twelve distinct windows")
        lookup = {w["label"]: w for w in windows}
        order = row.get("frame_order", [])
        if len(order) != 12 or set(order) != set(lookup):
            raise ValueError("frame_order must be an exact permutation of packet windows")
        raw_sets = [dict(lookup[w], frames=[str(packet_path.parent / f) for f in lookup[w]["frames"]]) for w in order]
        duration, fps = packet["duration_seconds"], packet["fps"]
    else:
        raw_sets = row.get("frame_sets", [])
        duration, fps = row.get("duration_seconds"), row.get("fps")
    if len(raw_sets) != 12 or len({w["label"] for w in raw_sets}) != 12:
        raise ValueError("cross-view requirement needs twelve distinct windows")
    if raw_sets[0]["label"] != row["speaker_window_id"]:
        raise ValueError("speaker window must be the first group")
    duration, fps = float(duration), float(fps)
    if not math.isfinite(duration) or duration <= 0 or fps != 0.5:
        raise ValueError("Aria mode requires positive duration and approved 0.5 FPS")
    expected = math.ceil(duration * fps)
    groups = []
    all_paths = set()
    for raw in raw_sets:
        frames = tuple(str(x) for x in raw["frames"])
        times = tuple(float(x) for x in raw.get("times_seconds", []))
        cuts = tuple(raw.get("cuts_before", []))
        if len(frames) != expected or len(times) != expected:
            raise ValueError(f"{raw['label']}: expected {expected} frames and timestamps")
        if any(not math.isclose(t, i/fps, abs_tol=1e-6) for i,t in enumerate(times)):
            raise ValueError("timestamps must retain the complete uniform timeline")
        if any(type(i) is not int or not 0 < i < expected for i in cuts) or list(cuts) != sorted(set(cuts)):
            raise ValueError("invalid cut indices")
        for frame in frames:
            path = Path(frame)
            if not frame or path.suffix.lower() not in SUPPORTED_FRAME_SUFFIXES:
                raise ValueError("invalid frame path")
            canonical = str(path.resolve())
            if canonical in all_paths:
                raise ValueError("duplicate frame path across windows")
            all_paths.add(canonical)
            if require_frame_files and (not path.is_file() or path.stat().st_size == 0):
                raise ValueError(f"missing or empty frame: {path}")
        groups.append(FrameSet(raw["label"],frames,times,cuts))
    return JudgeExample(row["example_id"],row["group_id"],JudgeTask.ANSWERABILITY,
        row["prompt"],normalize_verdict(row["verdict"]),tuple(groups),CONDITION,
        split_group_id=row["split_group_id"])


def visual_messages(example: JudgeExample, *, min_pixels: int, max_pixels: int):
    """窗口和时刻放在对应图片旁边；不添加标签相关提示。"""
    content = []
    for group, fs in enumerate(example.frame_sets,1):
        role = "SPEAKER" if group == 1 else "PROVIDER"
        content.append({"type":"text","text":f"WINDOW_BEGIN group={group} {fs.label} {role}; independent window; 0.5 FPS\n"})
        for i,(path,t) in enumerate(zip(fs.frames,fs.times_seconds)):
            cut = "EDIT CUT: omitted interval before this frame. " if i in fs.cuts_before else ""
            content.append({"type":"text","text":f"{cut}{fs.label} F{i:03d} t={t:g}s\n"})
            content.append({"type":"image","image":path,"min_pixels":min_pixels,"max_pixels":max_pixels})
        content.append({"type":"text","text":f"WINDOW_END {fs.label}\n"})
    content.append({"type":"text","text":example.prompt})
    return [{"role":"user","content":content}]


def grouped_split(rows, *, test_count=40, test_pass=10, seed=17):
    """按已人工归并的事实组做精确分层；无解时报错，不拆同义组。"""
    groups = {}
    ids = [r["example_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example_id")
    for row in rows:
        key = row.get("split_group_id")
        if not key:
            raise ValueError("missing split_group_id")
        groups.setdefault(key,[]).append(row)
    names = sorted(groups);random.Random(seed).shuffle(names)
    reachable = {(0,0):()}
    for name in names:
        members=groups[name];n=len(members);p=sum(r["verdict"]=="pass" for r in members)
        for (count,passed), chosen in list(reachable.items()):
            state=(count+n,passed+p)
            if state[0]<=test_count and state[1]<=test_pass and state not in reachable:
                reachable[state]=chosen+(name,)
    if (test_count,test_pass) not in reachable:
        raise ValueError("cannot meet requested split without splitting a fact group")
    held=set(reachable[(test_count,test_pass)])
    train=[r for r in rows if r["split_group_id"] not in held]
    test=[r for r in rows if r["split_group_id"] in held]
    validate_split(train,test)
    return train,test


def validate_split(train, test):
    if not train or not test:
        raise ValueError("both splits must be nonempty")
    for field in ("example_id","split_group_id"):
        if {r[field] for r in train}&{r[field] for r in test}:
            raise ValueError(f"train/test leakage in {field}")
