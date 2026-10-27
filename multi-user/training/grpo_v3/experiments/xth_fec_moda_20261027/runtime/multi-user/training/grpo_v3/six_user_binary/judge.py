"""将线上 completion 路由到同版本 Judge prompt 和完整采样帧。"""
from __future__ import annotations

from typing import Any
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json
import re

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


@lru_cache(maxsize=3)
def prompt_template(name):
    return (Path(__file__).resolve().parents[4] / 'prompts' / name).read_text(encoding='utf-8-sig')


def render_template(name, values):
    template = prompt_template(name)
    if any(template.count(key) != 1 for key in values):
        raise ValueError('用户Judge模板占位符数量改变')
    return re.sub('|'.join(re.escape(key) for key in values), lambda match: values[match.group(0)], template)


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
    for key, task in (("formality", "formality"), ("groundedness", "groundedness"),
                      ("combined_answerability", "answerability")):
        row = {"example_id": request["request_id"] + "::" + key,
               "group_id": view["source_packet_id"], "task": task, "verdict": "fail"}
        if key == "formality":
            names = list(view['required_users'])
            errors = prompts.qa_formality_errors(qa, [], participant_names=names)
            row["prompt"] = render_template('formality_prompt.txt', {
                '<DETERMINISTIC_SCHEMA_RESULT_JSON>': json.dumps({'status':'FAIL' if errors else 'PASS','errors':errors},ensure_ascii=False,indent=2),
                '<PARTICIPANT_NAMES_JSON>': json.dumps({'participant_names':names},ensure_ascii=False,indent=2),
                '<QUESTION_ANSWER_ITEM_JSON>': json.dumps(prompts.formality_qa_item_brief(qa),ensure_ascii=False,indent=2),
            })
        else:
            row.update(_frame_fields(view, speaker_only=False))
            if key == "groundedness":
                row["prompt"] = render_template('groundedness_model_prompt.txt', {
                    '{video_packet_brief(packet)}': prompts.video_packet_brief(view),
                    '{qa_item JSON}': json.dumps(qa,ensure_ascii=False,indent=2),
                })
            else:
                row["condition_type"] = "combined_all_six_users"
                row["prompt"] = render_template('combined_answerability_judge_prompt.txt', {'{question}':qa['question']})
        result[key] = normalized_record_to_example(row)
    return result
