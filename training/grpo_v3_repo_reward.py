"""GRPO v3 对仓库原生 reviewer/judge reward 的薄封装。"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable

from training.grpo_v3_json_format import (
    UNRECOVERABLE_FORMAT_REWARD,
    validate_completion_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT_REWARD_REVISIONS = {"repo_native_v1", "ground_answer_gap_v1"}


def validate_groundedness_audit_approval(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 groundedness 人工审计 summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in {
        "grpo_v3_groundedness_audit_v1",
        "grpo_v3_multisignal_audit_v2",
    }:
        raise ValueError("groundedness 人工审计 summary schema_version 不匹配")
    if int(data.get("completed_count") or 0) < 20:
        raise ValueError("groundedness 人工审计少于 20 个已完成案例")
    if int(data.get("reviewer_pass_completed") or 0) < 8 or int(data.get("reviewer_fail_completed") or 0) < 8:
        raise ValueError("groundedness 人工审计 PASS/FAIL 子集不足 8 个")
    if data.get("approved_for_weight_change") is not True:
        raise ValueError("groundedness 人工审计未批准 reward 权重修改")
    return data


def apply_content_reward_revision(record: dict[str, Any], revision: str) -> dict[str, Any]:
    if revision not in CONTENT_REWARD_REVISIONS:
        raise ValueError(f"未知 content reward revision: {revision}")
    result = dict(record)
    components = dict(result.get("reward_components") or {})
    if revision == "repo_native_v1":
        result["content_reward_revision"] = revision
        return result
    groundedness_status = str(result.get("groundedness_status") or "").upper()
    components["groundedness"] = {"PASS": 1.5, "UNCERTAIN": -1.0, "FAIL": -2.0}.get(groundedness_status, -2.0)
    components["combined_answerability"] = 1.5 if result.get("combined_correct") is True else -2.0
    additive = (
        "groundedness", "combined_answerability", "grounded_answerable_bonus",
        "subset_leakage", "qa_formality", "shallow_activity_query",
    )
    total = sum(float(components.get(name) or 0.0) for name in additive)
    components["provider_only_cap"] = 0.0
    components["shallow_activity_cap"] = 0.0
    components["speaker_leakage_cap"] = 0.0
    if result.get("provider_only_correct") is True:
        total = min(total, 2.0)
        components["provider_only_cap"] = 2.0
    if str(result.get("shallow_activity_status") or "").upper() == "FAIL":
        total = min(total, 1.5)
        components["shallow_activity_cap"] = 1.5
    if result.get("speaker_only_correct") is True:
        total = min(total, 0.5)
        components["speaker_leakage_cap"] = 0.5
    result["reward_components"] = components
    result["reward_total"] = round(total, 6)
    result["content_reward_revision"] = revision
    return result


def _repo_modules() -> dict[str, Any]:
    """延迟导入仓库模块，使源码根目录中的相对导入保持有效。"""

    try:
        video_loop = importlib.import_module("egolife_two_user_qa.video_qa_loop")
        schema = importlib.import_module("egolife_two_user_qa.schema")
        runner = importlib.import_module("egolife_two_user_qa.qwen3vl_runner")
    except ModuleNotFoundError:
        package_name = "_egoqa_repo_v3"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(PROJECT_ROOT)]
            package.__package__ = package_name
            sys.modules[package_name] = package
        video_loop = importlib.import_module(f"{package_name}.video_qa_loop")
        schema = importlib.import_module(f"{package_name}.schema")
        runner = importlib.import_module(f"{package_name}.qwen3vl_runner")
    scoring = importlib.import_module("grpo_judge_reward.scoring")
    return {
        "media_for_clips": video_loop.media_for_clips,
        "complete_generator_metadata": video_loop.complete_generator_metadata,
        "video_evidence_for_packet": video_loop.video_evidence_for_packet,
        "human_audit_packet": video_loop.human_audit_packet,
        "run_parallel_review_judges": video_loop.run_parallel_review_judges,
        "build_review_from_gates": video_loop.build_review_from_gates,
        "extract_json_object": schema.extract_json_object,
        "validate_qa_item": schema.validate_qa_item,
        "OpenAICompatibleLocalRunner": runner.OpenAICompatibleLocalRunner,
        "compute_judge_reward": scoring.compute_judge_reward,
    }


def make_repo_score_fn(
    *,
    review_model_id: str,
    review_base_url: str,
    policy_model_id: str,
    review_max_new_tokens: int,
    modules: dict[str, Any] | None = None,
    content_reward_revision: str | None = None,
    groundedness_audit_summary: str | Path | None = None,
) -> Callable[..., dict[str, Any]]:
    revision = content_reward_revision or os.environ.get("EGOQA_CONTENT_REWARD_REVISION", "repo_native_v1")
    if revision not in CONTENT_REWARD_REVISIONS:
        raise ValueError(f"未知 content reward revision: {revision}")
    if revision == "ground_answer_gap_v1":
        summary_value = groundedness_audit_summary or os.environ.get("EGOQA_GROUNDEDNESS_AUDIT_SUMMARY")
        if not summary_value:
            raise ValueError("ground_answer_gap_v1 需要 EGOQA_GROUNDEDNESS_AUDIT_SUMMARY")
        validate_groundedness_audit_approval(Path(summary_value))
    repo_modules = _repo_modules() if modules is None else modules
    reviewer = repo_modules["OpenAICompatibleLocalRunner"](
        model_id=review_model_id,
        base_url=review_base_url,
        max_new_tokens=review_max_new_tokens,
        timeout=900,
        allow_video_input=True,
    )

    def score(
        *,
        raw_completion: str,
        packet: dict[str, Any],
        evidence_id: str,
        question_type: str,
        generation_mode: str,
        candidate_index: int,
    ) -> dict[str, Any]:
        candidate_id = f"{evidence_id}::grpo_v3::{candidate_index}"
        packet_evidence_id = packet.get("evidence_id")
        if packet_evidence_id is not None and str(packet_evidence_id) != evidence_id:
            return {
                "reward": None,
                "record": {
                    "candidate_id": candidate_id,
                    "evidence_id": evidence_id,
                    "masked": True,
                    "eligible_for_grpo": False,
                    "mask_reason": (
                        "evidence_id_mismatch: "
                        f"completion={evidence_id} packet={packet_evidence_id}"
                    ),
                    "raw_qa": raw_completion,
                },
            }

        full_images, full_videos = repo_modules["media_for_clips"](
            packet.get("clips", []),
            backend="openai-compatible-local",
            allow_openai_video_input=True,
            media_role="full",
        )
        if len(full_videos) != 2:
            return {
                "reward": None,
                "record": {
                    "candidate_id": candidate_id,
                    "evidence_id": evidence_id,
                    "masked": True,
                    "eligible_for_grpo": False,
                    "mask_reason": (
                        "reviewer_full_video_count_mismatch: "
                        f"expected=2 actual={len(full_videos)}"
                    ),
                    "raw_qa": raw_completion,
                },
            }

        format_result = validate_completion_json(raw_completion)
        format_validation = format_result.to_dict()
        if format_result.status == "unrecoverable":
            return {
                "reward": UNRECOVERABLE_FORMAT_REWARD,
                "record": {
                    "candidate_id": candidate_id,
                    "group_id": evidence_id,
                    "evidence_id": evidence_id,
                    "masked": False,
                    "eligible_for_grpo": True,
                    "mask_reason": None,
                    "raw_qa": raw_completion,
                    "qa": None,
                    "reward_components": {"format": UNRECOVERABLE_FORMAT_REWARD},
                    "reward_total": UNRECOVERABLE_FORMAT_REWARD,
                    "content_reward_revision": revision,
                    "format_validation": format_validation,
                },
            }

        qa = dict(format_result.value or {})

        qa["qa_id"] = str(qa.get("qa_id") or f"GRPO_V3_{evidence_id}_{candidate_index}")
        qa["question_type"] = question_type
        qa["generation_mode"] = generation_mode
        qa["required_users"] = list(packet.get("required_users") or qa.get("required_users") or [])
        qa["model_id"] = policy_model_id
        qa["source_urls"] = packet.get("source_urls", {})
        qa.setdefault("review", {})
        qa["video_evidence"] = repo_modules["video_evidence_for_packet"](packet)
        qa["human_audit"] = repo_modules["human_audit_packet"](packet)
        qa.setdefault("generation_trace", [])
        repo_modules["complete_generator_metadata"](qa, packet=packet, question_type=question_type)
        schema_errors = repo_modules["validate_qa_item"](qa)
        prompt_rows: list[dict[str, Any]] = []
        judge, answerability, judge_trace = repo_modules["run_parallel_review_judges"](
            qa_item=qa,
            packet=packet,
            schema_errors=schema_errors,
            runner=reviewer,
            media_backend="openai-compatible-local",
            allow_openai_video_input=True,
            prompt_rows=prompt_rows,
            full_image_paths=full_images,
            full_video_paths=full_videos,
            attempt=candidate_index + 1,
        )
        judge_passed = bool((judge.get("gate") or {}).get("passed"))
        answerability_passed = bool((answerability.get("gate") or {}).get("passed"))
        accepted = not schema_errors and judge_passed and answerability_passed
        rejection_stage = (
            "schema" if schema_errors else "judger" if not judge_passed else "answerability" if not answerability_passed else None
        )
        review = repo_modules["build_review_from_gates"](
            judge=judge,
            answerability=answerability,
            schema_errors=schema_errors,
            accepted=accepted,
            rejection_stage=rejection_stage,
            final_reason=(judge.get("gate") or {}).get("reason"),
        )
        qa["review"] = review
        data = {
            "candidate_id": candidate_id,
            "group_id": evidence_id,
            "evidence_id": evidence_id,
            "qa_id": qa.get("qa_id"),
            "attempt": candidate_index + 1,
            "raw_qa": raw_completion,
            "qa": qa,
            "schema_errors": schema_errors,
            "review": review,
            "answerability": answerability,
        }
        record = repo_modules["compute_judge_reward"](data)
        record_dict = record.to_dict()
        record_dict = apply_content_reward_revision(record_dict, revision)
        content_reward = record_dict.get("reward_total")
        format_penalty = format_result.format_penalty
        reward_components = dict(record_dict.get("reward_components") or {})
        reward_components["format"] = format_penalty
        final_reward = None if content_reward is None else float(content_reward) + format_penalty
        record_dict["reward_components"] = reward_components
        record_dict["reward_total"] = final_reward
        record_dict["format_validation"] = format_validation
        record_dict["review_model_id"] = review_model_id
        record_dict["judge_trace"] = judge_trace
        record_dict["judge_prompts"] = prompt_rows
        return {"reward": final_reward, "record": record_dict}

    return score
