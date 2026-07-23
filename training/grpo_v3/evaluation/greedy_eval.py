"""在固定 native-video 数据集上对单个 LoRA 做一次性原 repo greedy 评估。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from training.grpo_v3.shared.adapter_reload import discover_adapter_dir
from training.grpo_v3.shared.data import read_jsonl
from training.grpo_v3.baseline.repo_reward import make_repo_score_fn


def _prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("eval row messages 必须恰好一条")
    content = str(messages[0].get("content") or "")
    marker = "<video><video>\n"
    if not content.startswith(marker):
        raise ValueError("eval row prompt 必须以两个有序 <video> 占位符开头")
    return content[len(marker):]


def evaluate_rows(
    rows: Iterable[dict[str, Any]], *, runner: Any,
    score_fn: Callable[..., dict[str, Any]], adapter_label: str,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    keys = [(str(row.get("evidence_id") or ""), str(row.get("question_type") or "")) for row in materialized]
    if any(not all(key) for key in keys):
        raise ValueError("eval row 缺少 evidence_id/question_type")
    if len(set(keys)) != len(keys):
        raise ValueError("固定评估集存在重复 evidence_id/question_type")
    results: list[dict[str, Any]] = []
    for index, row in enumerate(materialized):
        videos = row.get("videos")
        if not isinstance(videos, list) or len(videos) != 2:
            raise ValueError(f"{keys[index]} 必须恰好包含两段有序视频")
        packet_value = row.get("packet_json")
        packet = json.loads(packet_value) if isinstance(packet_value, str) else packet_value
        if not isinstance(packet, dict):
            raise ValueError(f"{keys[index]} packet_json 不是 object")
        raw = runner.generate(
            _prompt(row), image_paths=[], video_paths=[str(item) for item in videos], decoding_mode="greedy"
        )
        scored = score_fn(
            raw_completion=str(raw), packet=packet, evidence_id=keys[index][0],
            question_type=keys[index][1], generation_mode=str(row.get("generation_mode") or "baseline"),
            candidate_index=0,
        )
        reward = scored.get("reward")
        record = scored.get("record") if isinstance(scored.get("record"), dict) else {}
        if reward is None or record.get("masked") is True or record.get("eligible_for_grpo") is False:
            raise RuntimeError(f"{keys[index]} greedy 评估产生 masked reward: {record.get('mask_reason')}")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(f"{keys[index]} greedy reward 不是有限值: {reward}")
        results.append({
            "schema_version": "grpo_v3_greedy_eval_v1",
            "adapter_label": adapter_label,
            "policy_model_id": str(getattr(runner, "model_id", "")),
            "evidence_id": keys[index][0],
            "question_type": keys[index][1],
            "generation_mode": str(row.get("generation_mode") or "baseline"),
            "video_paths": [str(item) for item in videos],
            "decode_config": {"mode": "greedy", "do_sample": False},
            "raw_completion": str(raw),
            "reward": reward,
            "record": record,
        })
    return results


def load_lora_runner(
    *, model_path: str, adapter_dir: Path, max_new_tokens: int,
    max_image_pixels: int, dtype: str = "bfloat16",
) -> Any:
    from peft import PeftModel
    from qwen3vl_runner import Qwen3VLTransformersRunner

    runner = Qwen3VLTransformersRunner(
        model_id=model_path, max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels, dtype=dtype,
    )
    runner.model = PeftModel.from_pretrained(runner.model, str(adapter_dir), is_trainable=False)
    runner.model.eval()
    runner.device = next(runner.model.parameters()).device
    lora_parameters = sum(
        parameter.numel() for name, parameter in runner.model.named_parameters()
        if "lora_" in name.lower()
    )
    if lora_parameters <= 0:
        raise RuntimeError("LoRA adapter 加载后没有 lora 参数")
    return runner


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="固定集单 LoRA 原 repo greedy 评估")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--adapter-dir", type=Path)
    group.add_argument("--adapter-search-root", type=Path)
    parser.add_argument("--adapter-label", required=True)
    parser.add_argument("--review-model", required=True)
    parser.add_argument("--review-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--review-max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-image-pixels", type=int, default=50176)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter_dir = args.adapter_dir or discover_adapter_dir(args.adapter_search_root)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter 不存在: {adapter_dir}")
    runner = load_lora_runner(
        model_path=args.model_path, adapter_dir=adapter_dir,
        max_new_tokens=args.max_new_tokens, max_image_pixels=args.max_image_pixels,
    )
    scorer = make_repo_score_fn(
        review_model_id=args.review_model, review_base_url=args.review_base_url,
        policy_model_id=f"{args.model_path}+{adapter_dir}",
        review_max_new_tokens=args.review_max_new_tokens,
        content_reward_revision="repo_native_v1",
    )
    results = evaluate_rows(
        read_jsonl(args.dataset), runner=runner, score_fn=scorer,
        adapter_label=args.adapter_label,
    )
    _write_jsonl(args.output, results)
    summary = {
        "schema_version": "grpo_v3_greedy_eval_summary_v1",
        "adapter_label": args.adapter_label,
        "adapter_dir": str(adapter_dir.resolve()),
        "dataset": str(args.dataset.resolve()),
        "row_count": len(results),
        "decode_config": {"mode": "greedy", "do_sample": False},
        "reward_mean": sum(row["reward"] for row in results) / len(results),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
