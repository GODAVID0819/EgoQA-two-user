"""将线上 completion 路由到同版本 Judge prompt 和完整采样帧。"""
from __future__ import annotations

from typing import Any
from copy import deepcopy
from functools import lru_cache

from training.judge_sft.data import normalized_record_to_example
from training.judge_sft.prepare_real_data import _frame_fields, _load_full_judge_view, _repo_module
from .data import checked_packet


class InvalidCompletion(ValueError):
    """仅用于生成内容错误；媒体或模型故障不得归入此类。"""


@lru_cache(maxsize=32)
def _cached_full_view(dataset_root, packet_id, asker_index, metadata_version):
    return _load_full_judge_view(dataset_root, packet_id, asker_index)


def full_judge_view(dataset_root, packet_id, asker_index):
    packet = checked_packet(dataset_root, packet_id, asker_index)
    metadata = (packet / 'packet.json').stat()
    view = _cached_full_view(str(packet.parent.parent), packet_id, asker_index,
                            (metadata.st_mtime_ns, metadata.st_size))
    return deepcopy(view)


def parse_completion(raw: str) -> dict[str, Any]:
    try:
        qa = _repo_module("schema").extract_json_object(raw)
    except (ValueError, TypeError) as exc:
        raise InvalidCompletion("生成结果不是合法 QA JSON") from exc
    if not isinstance(qa, dict):
        raise InvalidCompletion("QA 必须是对象")
    options = qa.get("options")
    if not isinstance(options, list) or len(options) != 5 or any(not isinstance(x, str) or not x.strip() for x in options):
        raise InvalidCompletion("必须有五个非空选项")
    correct = qa.get("correct")
    if not isinstance(correct, str) or correct not in ("A", "B", "C", "D", "E"):
        raise InvalidCompletion("correct 必须为 A–E")
    if qa.get("answer") != options[ord(correct) - ord("A")]:
        raise InvalidCompletion("answer 与 correct 不一致")
    if not isinstance(qa.get("question"), str) or not qa["question"].strip():
        raise InvalidCompletion("缺少问题")
    return qa


def build_examples(request: dict[str, Any]) -> dict[str, Any]:
    view = full_judge_view(request["dataset_root"], request["source_packet_id"], request["asker_index"])
    if request["evidence_id"] != view["evidence_id"]:
        raise ValueError("evidence_id 与真实 packet/asker 不一致")
    qa = parse_completion(request["completion"])
    # 用户身份与顺序来自真实输入，不允许模型生成内容改写媒体路由。
    qa = {**qa, "required_users": list(view["required_users"]),
          "speaker_user": view["required_users"][0], "question_type": "neutral"}
    prompts = _repo_module("prompts")
    result = {}
    for key, task, speaker_only in (("formality", "formality", False),
                                    ("groundedness", "groundedness", False),
                                    ("speaker_only", "answerability", True),
                                    ("all_six", "answerability", False)):
        row = {"example_id": request["request_id"] + "::" + key,
               "group_id": view["source_packet_id"], "task": task, "verdict": "fail"}
        if key == "formality":
            row["prompt"] = prompts.build_qa_formality_judge_prompt(qa, view, schema_errors=[])
        else:
            row.update(_frame_fields(view, speaker_only=speaker_only))
            if key == "groundedness":
                row["prompt"] = prompts.build_evidence_groundedness_judge_prompt(qa, view)
            else:
                condition = "speaker_only" if speaker_only else "combined_all_six_users"
                users = view["required_users"][:1] if speaker_only else view["required_users"]
                row["condition_type"] = condition
                row["prompt"] = prompts.build_answerability_prompt(qa, {
                    "condition_type": condition, "condition_id": condition + "::" + "+".join(users),
                    "users": users})
        result[key] = normalized_record_to_example(row)
    return result
