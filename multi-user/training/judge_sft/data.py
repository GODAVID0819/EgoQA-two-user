"""Normalized manifest boundary for verdict-token judge training.

Visual examples reference the packet-owned 0.5 FPS JPEG timelines produced by
``rlhf_evidence_preprocessing``. They never reconstruct or decode native video
during training. A compact manifest stores the packet directory plus ordered
user indices; this module resolves those references to the exact 300 frames per
selected user when the manifest is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .contracts import JudgeTask, TASK_IDS, Verdict, normalize_verdict


FRAMES_PER_USER = 300
USER_COUNT = 6
SUPPORTED_FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class FrameSet:
    label: str
    frames: tuple[str, ...]


@dataclass(frozen=True)
class JudgeExample:
    example_id: str
    group_id: str
    task: JudgeTask
    prompt: str
    verdict: Verdict
    frame_sets: tuple[FrameSet, ...] = ()
    condition_type: str | None = None

    @property
    def target(self) -> int:
        return self.verdict.target

    @property
    def task_id(self) -> int:
        return TASK_IDS[self.task]

    @property
    def frame_count(self) -> int:
        return sum(len(frame_set.frames) for frame_set in self.frame_sets)

    @property
    def frames(self) -> tuple[str, ...]:
        return tuple(
            frame for frame_set in self.frame_sets for frame in frame_set.frames
        )


class JudgeDataset:
    def __init__(self, examples: Iterable[JudgeExample]) -> None:
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("judge dataset is empty")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> JudgeExample:
        return self.examples[index]


def _nonempty_string(value: Any, *, field: str, location: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return result


def _validate_frame_paths(
    frames: tuple[str, ...],
    *,
    location: str,
    require_frame_files: bool,
) -> None:
    if len(frames) != FRAMES_PER_USER:
        raise ValueError(
            f"{location}: every user timeline must contain exactly "
            f"{FRAMES_PER_USER} frames, got {len(frames)}"
        )
    if len(set(frames)) != len(frames):
        raise ValueError(f"{location}: frame paths must be distinct within one timeline")
    for value in frames:
        path = Path(value).expanduser()
        if path.suffix.casefold() not in SUPPORTED_FRAME_SUFFIXES:
            raise ValueError(f"{location}: unsupported sampled-frame suffix: {path}")
        if require_frame_files and (not path.is_file() or path.stat().st_size <= 0):
            raise ValueError(f"{location}: sampled frame is missing or empty: {path}")


def _inline_frame_sets(
    value: Any,
    *,
    location: str,
    require_frame_files: bool,
) -> tuple[FrameSet, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{location}: frame_sets must be an array")
    result: list[FrameSet] = []
    for index, raw in enumerate(value):
        item_location = f"{location}.frame_sets[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{item_location}: expected an object")
        label = _nonempty_string(raw.get("label"), field="label", location=item_location)
        raw_frames = raw.get("frames")
        if not isinstance(raw_frames, list) or any(
            not isinstance(frame, str) for frame in raw_frames
        ):
            raise ValueError(f"{item_location}: frames must be an array of paths")
        frames = tuple(str(frame).strip() for frame in raw_frames)
        if any(not frame for frame in frames):
            raise ValueError(f"{item_location}: frame paths must be non-empty")
        _validate_frame_paths(
            frames,
            location=item_location,
            require_frame_files=require_frame_files,
        )
        result.append(FrameSet(label=label, frames=frames))
    return tuple(result)


@lru_cache(maxsize=256)
def _packet_users(packet_dir_value: str) -> tuple[dict[str, Any], ...]:
    packet_dir = Path(packet_dir_value)
    packet_path = packet_dir / "packet.json"
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"frame packet metadata is missing: {packet_path}") from exc
    users = packet.get("users") if isinstance(packet, dict) else None
    if not isinstance(users, list) or len(users) != USER_COUNT:
        raise ValueError(f"{packet_path}: expected exactly {USER_COUNT} packet users")
    if float(packet.get("duration_seconds") or 0.0) != 600.0:
        raise ValueError(f"{packet_path}: expected a 600-second packet")
    sampling = ((packet.get("preprocessing") or {}).get("sampling") or {})
    fps = sampling.get("fps")
    interval = sampling.get("interval_seconds")
    if fps is not None and float(fps) != 0.5:
        raise ValueError(f"{packet_path}: expected preprocessing FPS 0.5, got {fps}")
    if interval is not None and float(interval) != 2.0:
        raise ValueError(
            f"{packet_path}: expected a 2-second sampling interval, got {interval}"
        )
    return tuple(users)


@lru_cache(maxsize=1_536)
def _packet_frames(
    packet_dir_value: str,
    user_index: int,
    require_frame_files: bool,
) -> tuple[str, ...]:
    packet_dir = Path(packet_dir_value)
    users = _packet_users(packet_dir_value)
    user = users[user_index]
    stored_index = int(user.get("user_index", user_index))
    if stored_index != user_index:
        raise ValueError(
            f"{packet_dir}: packet user index mismatch at position {user_index}"
        )
    raw_frames = user.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError(f"{packet_dir}: packet user {user_index} has no frames")
    ordered_rows = sorted(raw_frames, key=lambda item: int(item.get("frame_index", -1)))
    expected_indices = list(range(FRAMES_PER_USER))
    actual_indices = [int(item.get("frame_index", -1)) for item in ordered_rows]
    if actual_indices != expected_indices:
        raise ValueError(
            f"{packet_dir}: packet user {user_index} frame indices are not 0..299"
        )
    frames = tuple(
        str((packet_dir / Path(str(item.get("path") or ""))).resolve())
        for item in ordered_rows
    )
    _validate_frame_paths(
        frames,
        location=f"{packet_dir}.packet_user[{user_index}]",
        require_frame_files=require_frame_files,
    )
    return frames


def _referenced_frame_sets(
    row: dict[str, Any],
    *,
    location: str,
    require_frame_files: bool,
) -> tuple[FrameSet, ...]:
    packet_dir = Path(
        _nonempty_string(row.get("frame_packet"), field="frame_packet", location=location)
    ).expanduser().resolve()
    raw_indices = row.get("frame_user_indices")
    raw_order = row.get("frame_order")
    if not isinstance(raw_indices, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_indices
    ):
        raise ValueError(f"{location}: frame_user_indices must be an array of integers")
    if not isinstance(raw_order, list) or any(
        not isinstance(value, str) for value in raw_order
    ):
        raise ValueError(f"{location}: frame_order must be an array of labels")
    if len(raw_indices) != len(raw_order):
        raise ValueError(
            f"{location}: frame_user_indices and frame_order must have equal length"
        )
    if len(set(raw_indices)) != len(raw_indices):
        raise ValueError(f"{location}: frame_user_indices must be distinct")
    users = _packet_users(str(packet_dir))
    result: list[FrameSet] = []
    for offset, (user_index, raw_label) in enumerate(zip(raw_indices, raw_order)):
        if not 0 <= user_index < len(users):
            raise ValueError(f"{location}: invalid packet user index {user_index}")
        label = _nonempty_string(
            raw_label,
            field="frame_order label",
            location=f"{location}[{offset}]",
        )
        frames = _packet_frames(str(packet_dir), user_index, require_frame_files)
        result.append(FrameSet(label=label, frames=frames))
    return tuple(result)


def _validate_task_media(
    *,
    task: JudgeTask,
    condition_type: str | None,
    frame_sets: tuple[FrameSet, ...],
    location: str,
) -> None:
    labels = [frame_set.label for frame_set in frame_sets]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{location}: frame_order labels must be unique")
    flattened = [frame for frame_set in frame_sets for frame in frame_set.frames]
    if len(set(flattened)) != len(flattened):
        raise ValueError(f"{location}: frame paths overlap across user timelines")
    if task is JudgeTask.FORMALITY:
        if frame_sets:
            raise ValueError(f"{location}: formality is text-only and must not contain frames")
        return
    if task is JudgeTask.GROUNDEDNESS:
        if len(frame_sets) != USER_COUNT:
            raise ValueError(
                f"{location}: groundedness requires all six full frame timelines"
            )
        return
    expected = {"speaker_only": 1, "combined_all_six_users": USER_COUNT}
    if condition_type not in expected:
        raise ValueError(
            f"{location}: answerability condition_type must be speaker_only or "
            "combined_all_six_users"
        )
    if len(frame_sets) != expected[condition_type]:
        raise ValueError(
            f"{location}: {condition_type} requires {expected[condition_type]} "
            f"frame timeline(s), got {len(frame_sets)}"
        )


def normalized_record_to_example(
    row: dict[str, Any],
    *,
    location: str = "record",
    require_frame_files: bool = True,
) -> JudgeExample:
    if not isinstance(row, dict):
        raise ValueError(f"{location}: expected a JSON object")
    if "videos" in row or "video_order" in row:
        raise ValueError(
            f"{location}: native-video manifests are obsolete; use packet-owned "
            "0.5 FPS frame timelines"
        )
    example_id = _nonempty_string(row.get("example_id"), field="example_id", location=location)
    group_id = _nonempty_string(
        row.get("group_id") or example_id,
        field="group_id",
        location=location,
    )
    try:
        task = JudgeTask(str(row.get("task") or "").strip())
    except ValueError as exc:
        raise ValueError(
            f"{location}: task must be one of {[task.value for task in JudgeTask]}"
        ) from exc
    prompt = _nonempty_string(row.get("prompt"), field="prompt", location=location)
    verdict = normalize_verdict(row.get("verdict"))
    condition_type = row.get("condition_type")
    if condition_type is not None:
        condition_type = _nonempty_string(
            condition_type,
            field="condition_type",
            location=location,
        )

    has_inline = "frame_sets" in row
    has_reference = any(
        field in row for field in ("frame_packet", "frame_user_indices", "frame_order")
    )
    if has_inline and has_reference:
        raise ValueError(f"{location}: choose inline or packet-referenced frames, not both")
    if has_inline:
        frame_sets = _inline_frame_sets(
            row["frame_sets"],
            location=location,
            require_frame_files=require_frame_files,
        )
    elif has_reference:
        frame_sets = _referenced_frame_sets(
            row,
            location=location,
            require_frame_files=require_frame_files,
        )
    else:
        frame_sets = ()
    _validate_task_media(
        task=task,
        condition_type=condition_type,
        frame_sets=frame_sets,
        location=location,
    )
    return JudgeExample(
        example_id=example_id,
        group_id=group_id,
        task=task,
        prompt=prompt,
        verdict=verdict,
        frame_sets=frame_sets,
        condition_type=condition_type,
    )


def iter_normalized_manifest(
    path: Path,
    *,
    require_frame_files: bool = True,
) -> Iterator[JudgeExample]:
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            example = normalized_record_to_example(
                row,
                location=f"{path}:{line_number}",
                require_frame_files=require_frame_files,
            )
            if example.example_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate example_id={example.example_id}")
            seen_ids.add(example.example_id)
            yield example


def load_normalized_manifest(
    path: Path,
    *,
    require_frame_files: bool = True,
) -> list[JudgeExample]:
    examples = list(
        iter_normalized_manifest(path, require_frame_files=require_frame_files)
    )
    if not examples:
        raise ValueError(f"{path}: manifest is empty")
    return examples


def assert_group_disjoint(
    train_examples: Iterable[JudgeExample],
    eval_examples: Iterable[JudgeExample],
) -> None:
    train_groups = {example.group_id for example in train_examples}
    eval_groups = {example.group_id for example in eval_examples}
    overlap = sorted(train_groups & eval_groups)
    if overlap:
        raise ValueError(
            "train/eval group leakage detected; group_id overlap=" + ",".join(overlap[:10])
        )


def manifest_summary(examples: Iterable[JudgeExample]) -> dict[str, Any]:
    rows = list(examples)
    tasks: dict[str, dict[str, int]] = {}
    for task in JudgeTask:
        selected = [row for row in rows if row.task is task]
        tasks[task.value] = {
            "examples": len(selected),
            "pass": sum(row.verdict is Verdict.PASS for row in selected),
            "fail": sum(row.verdict is Verdict.FAIL for row in selected),
            "groups": len({row.group_id for row in selected}),
            "frames": sum(row.frame_count for row in selected),
        }
    return {
        "examples": len(rows),
        "groups": len({row.group_id for row in rows}),
        "frames": sum(row.frame_count for row in rows),
        "tasks": tasks,
    }
