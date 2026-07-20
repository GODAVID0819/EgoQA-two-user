"""仅使用 qa_formality judge 置信度的 GRPO v3 reward。"""

from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path
from typing import Any, Callable

from training.grpo_v3_json_format import validate_completion_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMALITY_REWARD_REVISION = "qa_formality_confidence_v1"
FORMALITY_MARGIN_CLIP = 32.0
FORMALITY_COMPONENT = "qa_formality_confidence"
UNJUDGEABLE_FORMALITY_REWARD = -1.0


def confidence_reward(pass_logprob: float, fail_logprob: float) -> float:
    """将 PASS/FAIL logprob margin 截断并缩放到 [-1, 1]。"""

    values = (float(pass_logprob), float(fail_logprob))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("qa_formality PASS/FAIL logprob 包含非有限值")
    raw_margin = values[0] - values[1]
    clipped_margin = max(-FORMALITY_MARGIN_CLIP, min(FORMALITY_MARGIN_CLIP, raw_margin))
    return clipped_margin / FORMALITY_MARGIN_CLIP


def _formality_modules() -> dict[str, Any]:
    try:
        video_loop = importlib.import_module("egolife_two_user_qa.video_qa_loop")
        prompts = importlib.import_module("egolife_two_user_qa.prompts")
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
        prompts = importlib.import_module(f"{package_name}.prompts")
        schema = importlib.import_module(f"{package_name}.schema")
        runner = importlib.import_module(f"{package_name}.qwen3vl_runner")
    return {
        "OpenAICompatibleLocalRunner": runner.OpenAICompatibleLocalRunner,
        "build_qa_formality_judge_prompt": prompts.build_qa_formality_judge_prompt,
        "run_model_judge_branch": video_loop.run_model_judge_branch,
        "qa_for_judger_prompt": video_loop.qa_for_judger_prompt,
        "validate_qa_item": schema.validate_qa_item,
        "complete_generator_metadata": video_loop.complete_generator_metadata,
    }


def make_formality_score_fn(
    *,
    review_model_id: str,
    review_base_url: str,
    policy_model_id: str,
    review_max_new_tokens: int,
    modules: dict[str, Any] | None = None,
) -> Callable[..., dict[str, Any]]:
    repo_modules = _formality_modules() if modules is None else modules
    reviewer = repo_modules["OpenAICompatibleLocalRunner"](
        model_id=review_model_id,
        base_url=review_base_url,
        max_new_tokens=review_max_new_tokens,
        timeout=900,
        allow_video_input=False,
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
        candidate_id = f"{evidence_id}::grpo_v3_formality::{candidate_index}"
        packet_evidence_id = packet.get("evidence_id")
        if packet_evidence_id is not None and str(packet_evidence_id) != evidence_id:
            raise ValueError(
                "evidence_id 不一致: "
                f"completion={evidence_id} packet={packet_evidence_id}"
            )

        format_result = validate_completion_json(raw_completion)
        format_validation = format_result.to_dict()
        if format_result.status == "unrecoverable":
            return {
                "reward": UNJUDGEABLE_FORMALITY_REWARD,
                "record": {
                    "candidate_id": candidate_id,
                    "group_id": evidence_id,
                    "evidence_id": evidence_id,
                    "masked": False,
                    "eligible_for_grpo": True,
                    "mask_reason": None,
                    "raw_qa": raw_completion,
                    "qa": None,
                    "qa_formality_status": "FAIL",
                    "reward_source": "deterministic_unjudgeable_floor",
                    "judge_called": False,
                    "reward_components": {
                        FORMALITY_COMPONENT: UNJUDGEABLE_FORMALITY_REWARD,
                    },
                    "reward_total": UNJUDGEABLE_FORMALITY_REWARD,
                    "reward_revision": FORMALITY_REWARD_REVISION,
                    "format_validation": format_validation,
                    "judge_trace": {},
                },
            }

        qa = dict(format_result.value or {})
        qa["qa_id"] = str(
            qa.get("qa_id") or f"GRPO_V3_FORMALITY_{evidence_id}_{candidate_index}"
        )
        qa["question_type"] = question_type
        qa["generation_mode"] = generation_mode
        qa["required_users"] = list(
            packet.get("required_users") or qa.get("required_users") or []
        )
        qa["model_id"] = policy_model_id
        qa["source_urls"] = packet.get("source_urls", {})
        repo_modules["complete_generator_metadata"](
            qa,
            packet=packet,
            question_type=question_type,
        )
        schema_errors = repo_modules["validate_qa_item"](qa)
        prompt = repo_modules["build_qa_formality_judge_prompt"](
            repo_modules["qa_for_judger_prompt"](qa),
            packet,
            schema_errors=schema_errors,
        )
        judge = repo_modules["run_model_judge_branch"](
            check_name="qa_formality",
            prompt=prompt,
            runner=reviewer,
            image_paths=[],
            video_paths=[],
            evidence_id=evidence_id,
            qa_id=qa["qa_id"],
            attempt=candidate_index + 1,
        )
        checks = judge.get("checks")
        check = checks.get("qa_formality") if isinstance(checks, dict) else None
        status = str(check.get("status") if isinstance(check, dict) else "").strip().upper()
        if status not in {"PASS", "FAIL"}:
            raise ValueError("qa_formality judge 未返回有效 PASS/FAIL status")
        signal = judge.get("choice_logit_signal")
        if not isinstance(signal, dict) or signal.get("available") is not True:
            raise ValueError("qa_formality judge 缺少可用 PASS/FAIL logprob")
        choice_logprobs = signal.get("choice_logprobs")
        if not isinstance(choice_logprobs, dict):
            raise ValueError("qa_formality judge 缺少 PASS/FAIL logprob")
        if "PASS" not in choice_logprobs or "FAIL" not in choice_logprobs:
            raise ValueError("qa_formality judge 缺少 PASS/FAIL logprob")
        pass_logprob = float(choice_logprobs["PASS"])
        fail_logprob = float(choice_logprobs["FAIL"])
        raw_margin = pass_logprob - fail_logprob
        clipped_margin = max(
            -FORMALITY_MARGIN_CLIP,
            min(FORMALITY_MARGIN_CLIP, raw_margin),
        )
        reward = confidence_reward(pass_logprob, fail_logprob)
        return {
            "reward": reward,
            "record": {
                "candidate_id": candidate_id,
                "group_id": evidence_id,
                "evidence_id": evidence_id,
                "qa_id": qa["qa_id"],
                "masked": False,
                "eligible_for_grpo": True,
                "mask_reason": None,
                "raw_qa": raw_completion,
                "qa": qa,
                "schema_errors": schema_errors,
                "qa_formality_status": status,
                "reward_source": "judge_pass_fail_logprob_margin",
                "judge_called": True,
                "pass_logprob": pass_logprob,
                "fail_logprob": fail_logprob,
                "logprob_margin_raw": raw_margin,
                "logprob_margin_clipped": clipped_margin,
                "qa_formality_confidence": reward,
                "reward_components": {FORMALITY_COMPONENT: reward},
                "reward_total": reward,
                "reward_revision": FORMALITY_REWARD_REVISION,
                "format_validation": format_validation,
                "review_model_id": review_model_id,
                "judge_trace": {
                    "qa_formality": {
                        "prompt": prompt,
                        "raw_output": judge.get("raw_output"),
                        "parsed": judge,
                    }
                },
            },
        }

    return score
