"""Combined-video answer-margin 的单 completion reward 契约。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training.grpo_v3_answer_margin import (
    ANSWER_MARGIN_REWARD_REVISION,
    LABELS,
    PermutationKey,
    compute_answer_margin,
    extract_core_qa,
    permute_options,
)
from training.grpo_v3_answer_scorer import PromptAuditMaterial, ScoreRequest, ScoreResponse


EXPERIMENT_REVISION = "combined_video_answer_margin_convergence_v1"
EXPERIMENT_CONDITION_ID = "t05"
TRACE_SCHEMA_VERSION = 1


def resolve_ordered_videos(packet: Mapping[str, Any], evidence_id: str) -> tuple[str, str]:
    """按 required_users 解析且验证两段有序原生 MP4。"""

    if not isinstance(packet, Mapping):
        raise ValueError("packet 必须是 object")
    packet_evidence = str(packet.get("evidence_id") or "").strip()
    if not packet_evidence or packet_evidence != str(evidence_id):
        raise ValueError("packet evidence_id 与展开字段不一致")
    users = packet.get("required_users")
    if (
        not isinstance(users, list)
        or len(users) != 2
        or any(not isinstance(user, str) or not user.strip() for user in users)
        or len(set(users)) != 2
    ):
        raise ValueError("required_users 必须恰好包含两个不同用户")
    clips = packet.get("clips")
    if not isinstance(clips, list):
        raise ValueError("clips 必须是列表")
    if len(clips) != 2:
        raise ValueError("packet 必须恰好包含两段视频")

    videos: list[str] = []
    for user in users:
        matches = [
            clip for clip in clips
            if isinstance(clip, Mapping) and str(clip.get("agent_name") or "") == user
        ]
        if len(matches) != 1:
            raise ValueError(f"用户 {user} 必须恰好映射一段视频")
        clip = matches[0]
        if clip.get("generator_media_mode") == "frames_only" or clip.get("force_frame_inputs"):
            raise ValueError(f"用户 {user} 禁止 sampled frames")
        value = clip.get("local_video")
        if isinstance(value, (list, tuple, Mapping)):
            raise ValueError(f"用户 {user} 的媒体映射不是原生 mp4 路径")
        path = Path(str(value or "")).expanduser()
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"用户 {user} 的视频不是 mp4")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"用户 {user} 的视频不存在或为空")
        videos.append(str(path.resolve()))
    return videos[0], videos[1]


def summarize_packet(packet: Any) -> dict[str, Any]:
    """生成不依赖 packet 完整合法性的可审计身份摘要。"""

    try:
        serialized = json.dumps(packet, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        serialized = repr(packet)
    summary: dict[str, Any] = {
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "value_type": type(packet).__name__,
        "evidence_id": None,
        "required_users": None,
        "clips": None,
    }
    if not isinstance(packet, Mapping):
        return summary
    users = packet.get("required_users")
    clips = packet.get("clips")
    summary["evidence_id"] = packet.get("evidence_id")
    summary["required_users"] = list(users) if isinstance(users, list) else users
    if isinstance(clips, list):
        summary["clips"] = [
            {
                "agent_name": clip.get("agent_name"),
                "local_video": clip.get("local_video"),
            }
            if isinstance(clip, Mapping)
            else {"invalid_clip_type": type(clip).__name__}
            for clip in clips
        ]
    else:
        summary["clips"] = clips
    return summary


def video_input_summary(
    packet: Mapping[str, Any], videos: tuple[str, str]
) -> list[dict[str, str]]:
    users = list(packet["required_users"])
    return [
        {"user": str(users[index]), "path": path, "basename": Path(path).name}
        for index, path in enumerate(videos)
    ]


def _audit_material(extracted: Any) -> PromptAuditMaterial:
    value = extracted.format_validation.value
    excluded: dict[str, str] = {}
    if isinstance(value, Mapping):
        for field, field_value in value.items():
            if field in {"question", "options"}:
                continue
            if isinstance(field, str) and isinstance(field_value, str):
                excluded[field] = field_value
    return PromptAuditMaterial(excluded)


def _format_record(extracted: Any) -> dict[str, Any]:
    return {
        **extracted.format_validation.to_dict(),
        "status": extracted.status,
        "inner_format_status": extracted.inner_format_status,
    }


def _key_payload(key: PermutationKey) -> dict[str, Any]:
    return {
        "reward_revision": key.reward_revision,
        "experiment_condition_id": key.experiment_condition_id,
        "phase": key.phase,
        "evidence_id": key.evidence_id,
        "generation_seed_or_call_index": key.generation_seed_or_call_index,
        "candidate_index": key.candidate_index,
    }


def _floor_record(
    extracted: Any,
    *,
    evidence_id: str,
    candidate_index: int,
    key: PermutationKey,
    global_step: int,
    reward_call_index: int,
    packet_summary: dict[str, Any],
    video_inputs: list[dict[str, str]],
    question_type: str,
    generation_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "reward_revision": ANSWER_MARGIN_REWARD_REVISION,
        "experiment_revision": EXPERIMENT_REVISION,
        "experiment_version": EXPERIMENT_REVISION,
        "phase": key.phase,
        "question_type": str(question_type),
        "generation_mode": str(generation_mode),
        "evidence_id": str(evidence_id),
        "candidate_index": candidate_index,
        "experiment_condition_id": EXPERIMENT_CONDITION_ID,
        "global_step": global_step,
        "reward_call_index": reward_call_index,
        "packet_summary": packet_summary,
        "video_inputs": video_inputs,
        "permutation_key": _key_payload(key),
        "raw_completion": extracted.raw_completion,
        "core_qa": {
            "ok": False,
            "question": extracted.question,
            "options": extracted.options,
            "correct": extracted.correct,
            "failure_reason": extracted.failure_reason,
        },
        "format_validation": _format_record(extracted),
        "reward_source": "core_qa_unrecoverable_floor",
        "masked": False,
        "eligible_for_grpo": True,
        "normalized_reward": -1.0,
        "infrastructure_error": None,
    }


def score_completion(
    raw_completion: str,
    packet: Mapping[str, Any],
    evidence_id: str,
    candidate_index: int,
    *,
    scorer: Any,
    key: PermutationKey,
    global_step: int,
    reward_call_index: int,
    question_type: str = "",
    generation_mode: str = "",
) -> dict[str, Any]:
    """为一条 completion 评分；基础设施异常原样抛出。"""

    if (
        key.evidence_id != str(evidence_id)
        or key.candidate_index != candidate_index
        or key.experiment_condition_id != EXPERIMENT_CONDITION_ID
        or key.reward_revision != ANSWER_MARGIN_REWARD_REVISION
        or key.generation_seed_or_call_index != reward_call_index
    ):
        raise ValueError("permutation key 与 completion 元数据错位")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step 必须是非负整数")
    if isinstance(reward_call_index, bool) or not isinstance(reward_call_index, int) or reward_call_index < 0:
        raise ValueError("reward_call_index 必须是非负整数")
    videos = resolve_ordered_videos(packet, evidence_id)
    packet_summary = summarize_packet(packet)
    video_inputs = video_input_summary(packet, videos)
    extracted = extract_core_qa(raw_completion)
    if not extracted.ok:
        return {
            "reward": -1.0,
            "record": _floor_record(
                extracted,
                evidence_id=evidence_id,
                candidate_index=candidate_index,
                key=key,
                global_step=global_step,
                reward_call_index=reward_call_index,
                packet_summary=packet_summary,
                video_inputs=video_inputs,
                question_type=question_type,
                generation_mode=generation_mode,
            ),
        }
    assert extracted.options is not None and extracted.correct is not None
    permutation = permute_options(extracted.options, extracted.correct, key)
    request = ScoreRequest(
        videos=videos,
        question=str(extracted.question),
        options=tuple(permutation.options),
    )
    response = scorer.score(request, audit_material=_audit_material(extracted))
    if not isinstance(response, ScoreResponse):
        raise RuntimeError("answer scorer 返回了非 ScoreResponse")
    sequence_scores = {
        label: response.scores[label].sequence_logprob for label in LABELS
    }
    margin = compute_answer_margin(sequence_scores, permutation.correct)
    label_scores = {
        label: {
            **response.scores[label].to_payload(),
            "canonical_label": label,
            "token_count": len(response.scores[label].token_ids),
            "local_log_probability": margin.log_probabilities[label],
        }
        for label in LABELS
    }
    record = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "reward_revision": ANSWER_MARGIN_REWARD_REVISION,
        "experiment_revision": EXPERIMENT_REVISION,
        "experiment_version": EXPERIMENT_REVISION,
        "phase": key.phase,
        "question_type": str(question_type),
        "generation_mode": str(generation_mode),
        "evidence_id": str(evidence_id),
        "candidate_index": candidate_index,
        "experiment_condition_id": EXPERIMENT_CONDITION_ID,
        "global_step": global_step,
        "reward_call_index": reward_call_index,
        "packet_summary": packet_summary,
        "video_inputs": video_inputs,
        "raw_completion": extracted.raw_completion,
        "core_qa": {
            "ok": True,
            "question": extracted.question,
            "options": list(extracted.options),
            "correct": extracted.correct,
            "failure_reason": None,
        },
        "format_validation": _format_record(extracted),
        "permutation_key": _key_payload(key),
        "original_options": list(extracted.options),
        "original_correct": extracted.correct,
        "permutation": list(permutation.permutation),
        "inverse_permutation": list(permutation.inverse),
        "permutation_digests": list(permutation.digests),
        "permuted_options": list(permutation.options),
        "mapped_correct": permutation.correct,
        "prompt_audit": response.prompt_audit.to_payload(),
        "label_scores": label_scores,
        "raw_margin": margin.raw_margin,
        "clipped_margin": margin.clipped_margin,
        "normalized_reward": margin.reward,
        "top1": margin.unique_top1,
        "tie": margin.tie,
        "reward_source": ANSWER_MARGIN_REWARD_REVISION,
        "masked": False,
        "eligible_for_grpo": True,
        "infrastructure_error": None,
    }
    return {"reward": margin.reward, "record": record}


def make_answer_margin_score_fn(*, base_url: str, timeout_seconds: float):
    from training.grpo_v3_answer_scorer_service import AnswerScorerClient

    client = AnswerScorerClient(base_url, timeout_seconds=timeout_seconds)

    def score_fn(**kwargs: Any) -> dict[str, Any]:
        return score_completion(scorer=client, **kwargs)

    return score_fn
