"""固定种子比较 GRPO v3 formality Probe 的 step 0 与 step 40。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from training.grpo_v3.experiments.archived.formality.reward import FORMALITY_COMPONENT, make_formality_score_fn


CHECKPOINT_STEPS = (0, 40)
FIXED_SEEDS = tuple(range(2026072000, 2026072016))
FIXED_EVAL_MIN_VIDEO_PIXELS = 4 * 32 * 32
BOOTSTRAP_SEED = 20260720
BOOTSTRAP_REPLICATES = 10000
SCHEMA_VERSION = "grpo_v3_formality_fixed_eval_v1"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("均值至少需要一个值")
    return float(sum(values) / len(values))


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile 至少需要一个值")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _paired_bootstrap(
    differences: Sequence[float], *, seed: int, replicates: int
) -> list[float]:
    if not differences:
        raise ValueError("配对 bootstrap 至少需要一个差值")
    if replicates <= 0:
        raise ValueError("bootstrap_replicates 必须为正整数")
    generator = random.Random(seed)
    sample_size = len(differences)
    means = sorted(
        _mean([differences[generator.randrange(sample_size)] for _ in range(sample_size)])
        for _ in range(replicates)
    )
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _validate_record(row: Mapping[str, Any]) -> tuple[float, bool]:
    reward = float(row.get("reward"))
    if not math.isfinite(reward):
        raise ValueError("fixed eval reward 必须为有限值")
    record = row.get("record")
    if not isinstance(record, Mapping):
        raise ValueError("fixed eval record 必须为 object")
    if record.get("masked") is True:
        raise RuntimeError("fixed eval 不允许基础设施 masked reward")
    components = record.get("reward_components")
    if not isinstance(components, Mapping) or set(components) != {FORMALITY_COMPONENT}:
        raise ValueError("fixed eval 只能包含 qa_formality_confidence reward")
    component_reward = float(components[FORMALITY_COMPONENT])
    if not math.isfinite(component_reward) or not math.isclose(
        component_reward, reward, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("reward_total 与 qa_formality_confidence 分量不一致")
    trace = record.get("judge_trace")
    if not isinstance(trace, Mapping) or not set(trace).issubset({"qa_formality"}):
        raise ValueError("fixed eval 出现非 qa_formality judge trace")
    unjudgeable = record.get("reward_source") == "deterministic_unjudgeable_floor"
    if unjudgeable:
        if not math.isclose(reward, -1.0) or record.get("judge_called") is not False or trace:
            raise ValueError("不可判定候选不符合 -1 floor 契约")
    elif record.get("judge_called") is not True or set(trace) != {"qa_formality"}:
        raise ValueError("可判定候选缺少 qa_formality judge 证据")
    return reward, unjudgeable


def analyze_fixed_eval(
    rows: Iterable[dict[str, Any]],
    *,
    checkpoint_steps: Sequence[int] = CHECKPOINT_STEPS,
    seeds: Sequence[int] = FIXED_SEEDS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """验收严格的 step×seed 键空间并生成配对端点统计。"""

    steps = tuple(int(step) for step in checkpoint_steps)
    seed_values = tuple(int(seed) for seed in seeds)
    if steps != CHECKPOINT_STEPS:
        raise ValueError("fixed eval checkpoint 必须严格为 step 0 和 step 40")
    if len(seed_values) != 16 or len(set(seed_values)) != 16:
        raise ValueError("fixed eval 必须提供 16 个唯一 seed")
    materialized = list(rows)
    expected_keys = {(step, seed) for step in steps for seed in seed_values}
    actual_keys: list[tuple[int, int]] = []
    validated: dict[tuple[int, int], tuple[float, bool]] = {}
    for row in materialized:
        key = (int(row.get("checkpoint_step")), int(row.get("seed")))
        actual_keys.append(key)
        if key in validated:
            raise ValueError(f"fixed eval 存在重复键: {key}")
        validated[key] = _validate_record(row)
    if set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys):
        missing = sorted(expected_keys - set(actual_keys))
        extra = sorted(set(actual_keys) - expected_keys)
        raise ValueError(f"fixed eval 键空间不完整: missing={missing} extra={extra}")

    checkpoint_summaries: dict[str, Any] = {}
    for step in steps:
        values = [validated[(step, seed)][0] for seed in seed_values]
        judgeable = [
            validated[(step, seed)][0]
            for seed in seed_values
            if not validated[(step, seed)][1]
        ]
        unjudgeable_count = len(values) - len(judgeable)
        checkpoint_summaries[str(step)] = {
            "checkpoint_step": step,
            "candidate_count": len(values),
            "reward_mean": _mean(values),
            "reward_std": float(statistics.pstdev(values)),
            "judgeable_count": len(judgeable),
            "judgeable_reward_mean": _mean(judgeable) if judgeable else None,
            "unjudgeable_count": unjudgeable_count,
            "unjudgeable_rate": unjudgeable_count / len(values),
        }

    differences = [
        validated[(40, seed)][0] - validated[(0, seed)][0] for seed in seed_values
    ]
    delta = _mean(differences)
    comparison = {
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }
    interval = _paired_bootstrap(
        differences, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    unjudgeable_not_increased = (
        checkpoint_summaries["40"]["unjudgeable_rate"]
        <= checkpoint_summaries["0"]["unjudgeable_rate"]
    )
    if delta <= 0.0 or not unjudgeable_not_increased:
        conclusion = "not_improved"
    elif interval[0] > 0.0:
        conclusion = "improved"
    else:
        conclusion = "inconclusive"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_status": "passed",
        "experiment_conclusion": conclusion,
        "row_count": len(materialized),
        "checkpoint_count": len(steps),
        "seed_count": len(seed_values),
        "seeds": list(seed_values),
        "checkpoints": checkpoint_summaries,
        "reward_delta": delta,
        "paired_comparison": comparison,
        "paired_bootstrap_95_ci": interval,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "checks": {
            "reward_delta_positive": delta > 0.0,
            "unjudgeable_rate_not_increased": unjudgeable_not_increased,
            "paired_ci_lower_positive": interval[0] > 0.0,
        },
    }


def _adapter_entry(step: int, adapter_dir: Path) -> dict[str, Any]:
    resolved = adapter_dir.resolve()
    config = resolved / "adapter_config.json"
    weights = [resolved / "adapter_model.safetensors", resolved / "adapter_model.bin"]
    if not config.is_file() or not any(path.is_file() for path in weights):
        raise FileNotFoundError(f"step {step} LoRA adapter 不完整: {resolved}")
    return {
        "checkpoint_step": step,
        "checkpoint_label": f"step_{step}",
        "adapter_dir": str(resolved),
        "adapter_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "weight_file": str(next(path for path in weights if path.is_file())),
    }


def build_checkpoint_inventory(probe_dir: Path) -> dict[str, Any]:
    """从 Probe 证据链解析 parent adapter 和唯一 checkpoint-40。"""

    probe = probe_dir.resolve()
    resolved_config_path = probe / "resolved_config.json"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"缺少 Probe resolved_config.json: {resolved_config_path}")
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    parent_run = Path(str(resolved_config.get("parent_run") or ""))
    parent_manifest_path = parent_run / "run_manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(f"缺少 parent run_manifest.json: {parent_manifest_path}")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_adapter = Path(str(parent_manifest.get("adapter_dir") or ""))
    final_configs = [
        path
        for path in probe.rglob("adapter_config.json")
        if path.parent.name == "checkpoint-40"
    ]
    if len(final_configs) != 1:
        raise ValueError(f"Probe 必须恰好保留一个 checkpoint-40，实际为 {len(final_configs)}")
    checkpoints = [
        _adapter_entry(0, parent_adapter),
        _adapter_entry(40, final_configs[0].parent),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "probe_dir": str(probe),
        "parent_run": str(parent_run.resolve()),
        "checkpoints": checkpoints,
    }


def select_probe_row(rows: Sequence[dict[str, Any]], probe_dir: Path) -> dict[str, Any]:
    """用 Probe trace 的唯一 evidence_id 绑定原训练 evidence。"""

    trace_path = probe_dir / "reward_trace.jsonl"
    if not trace_path.is_file():
        raise FileNotFoundError(f"缺少 Probe reward_trace.jsonl: {trace_path}")
    trace_rows = _read_jsonl(trace_path)
    evidence_ids = set()
    for trace in trace_rows:
        record = trace.get("record") if isinstance(trace.get("record"), Mapping) else {}
        evidence_id = trace.get("evidence_id") or record.get("evidence_id") or record.get("group_id")
        if evidence_id:
            evidence_ids.add(str(evidence_id))
    if len(evidence_ids) != 1:
        raise ValueError(f"Probe trace 必须恰好包含唯一 evidence，实际为 {sorted(evidence_ids)}")
    evidence_id = next(iter(evidence_ids))
    matches = [row for row in rows if str(row.get("evidence_id") or "") == evidence_id]
    if len(matches) != 1:
        raise ValueError(
            f"dataset 必须恰好匹配 Probe evidence_id={evidence_id}，实际为 {len(matches)} 行"
        )
    return matches[0]


def set_generation_seed(seed: int, *, torch_module: Any | None = None) -> None:
    """在每个候选生成前重设所有可控随机源。"""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ModuleNotFoundError:
        pass
    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError:
            torch_module = None
    if torch_module is not None:
        torch_module.manual_seed(seed)
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.manual_seed_all(seed)


def _prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("fixed eval 数据必须恰好包含一条 user message")
    content = str(messages[0].get("content") or "")
    marker = "<video><video>\n"
    if not content.startswith(marker):
        raise ValueError("fixed eval prompt 必须以两个有序 <video> 占位符开头")
    return content[len(marker) :]


def evaluate_adapter(
    *,
    row: dict[str, Any],
    runner: Any,
    score_fn: Callable[..., dict[str, Any]],
    checkpoint_step: int,
    adapter_dir: Path,
    seeds: Sequence[int] = FIXED_SEEDS,
    temperature: float = 0.7,
) -> list[dict[str, Any]]:
    """使用一个 adapter 和固定 seeds 生成 formality-only 记录。"""

    videos = row.get("videos")
    if not isinstance(videos, list) or len(videos) != 2:
        raise ValueError("fixed eval 必须恰好包含两段有序视频")
    packet_value = row.get("packet_json")
    packet = json.loads(packet_value) if isinstance(packet_value, str) else packet_value
    if not isinstance(packet, dict):
        raise ValueError("fixed eval packet_json 必须为 object")
    evidence_id = str(row.get("evidence_id") or "")
    question_type = str(row.get("question_type") or "")
    if not evidence_id or not question_type:
        raise ValueError("fixed eval 数据缺少 evidence_id/question_type")
    label = f"step_{int(checkpoint_step)}"
    runner.model.set_adapter(label)
    results = []
    for candidate_index, seed in enumerate(seeds):
        set_generation_seed(int(seed), torch_module=getattr(runner, "torch", None))
        raw = runner.generate(
            _prompt(row),
            image_paths=[],
            video_paths=[str(item) for item in videos],
            decoding_mode="sampling",
            temperature=temperature,
            top_p=1.0,
        )
        scored = score_fn(
            raw_completion=str(raw),
            packet=packet,
            evidence_id=evidence_id,
            question_type=question_type,
            generation_mode=str(row.get("generation_mode") or "baseline"),
            candidate_index=candidate_index,
        )
        reward = scored.get("reward")
        record = scored.get("record")
        result = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_step": int(checkpoint_step),
            "checkpoint_label": label,
            "adapter_dir": str(adapter_dir.resolve()),
            "seed": int(seed),
            "policy_model_id": str(getattr(runner, "model_id", "")),
            "evidence_id": evidence_id,
            "question_type": question_type,
            "generation_mode": str(row.get("generation_mode") or "baseline"),
            "video_paths": [str(item) for item in videos],
            "decode_config": {
                "mode": "sampling",
                "do_sample": True,
                "temperature": temperature,
                "top_p": 1.0,
            },
            "raw_completion": str(raw),
            "reward": reward,
            "record": record,
            "judge_trace": record.get("judge_trace") if isinstance(record, Mapping) else None,
            "unjudgeable": (
                isinstance(record, Mapping)
                and record.get("reward_source") == "deterministic_unjudgeable_floor"
            ),
        }
        _validate_record(result)
        results.append(result)
    return results


def load_multi_adapter_runner(
    *,
    model_path: str,
    adapters: Mapping[int, Path],
    max_new_tokens: int,
    max_image_pixels: int,
    dtype: str = "bfloat16",
) -> Any:
    from peft import PeftModel
    from qwen3vl_runner import Qwen3VLTransformersRunner

    if set(adapters) != set(CHECKPOINT_STEPS):
        raise ValueError("runner adapters 必须严格为 step 0 和 step 40")
    if max_image_pixels < FIXED_EVAL_MIN_VIDEO_PIXELS:
        raise ValueError("fixed eval max_image_pixels must cover the Qwen3-VL video minimum")
    runner = Qwen3VLTransformersRunner(
        model_id=model_path,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        min_video_pixels=FIXED_EVAL_MIN_VIDEO_PIXELS,
        dtype=dtype,
    )
    runner.model = PeftModel.from_pretrained(
        runner.model,
        str(adapters[0]),
        adapter_name="step_0",
        is_trainable=False,
    )
    runner.model.load_adapter(
        str(adapters[40]), adapter_name="step_40", is_trainable=False
    )
    runner.model.eval()
    runner.device = next(runner.model.parameters()).device
    return runner


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是 object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_run_manifest(
    *,
    summary: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    storage_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """把运行完整性与实验三态结论分开记录。"""

    storage_status = str(storage_preflight.get("status") or "")
    if storage_status != "passed":
        raise RuntimeError(f"storage preflight 未通过: {storage_status or 'missing'}")
    if summary.get("run_status") != "passed":
        raise RuntimeError("fixed eval summary 运行状态未通过")
    conclusion = str(summary.get("experiment_conclusion") or "")
    if conclusion not in {"improved", "not_improved", "inconclusive"}:
        raise ValueError(f"未知实验结论: {conclusion}")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_status": "passed",
        "experiment_conclusion": conclusion,
        "row_count": int(summary.get("row_count") or 0),
        "checkpoint_count": int(summary.get("checkpoint_count") or 0),
        "seed_count": int(summary.get("seed_count") or 0),
        "storage_preflight_status": storage_status,
        "reward_components": [FORMALITY_COMPONENT],
        "calls_video_reviewer": False,
        "artifact_paths": dict(artifact_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO v3 formality step 0/40 固定种子配对评估")
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--review-model", required=True)
    parser.add_argument("--review-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--review-max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-image-pixels", type=int, default=50176)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.temperature != 0.7:
        raise ValueError("fixed eval temperature 必须保持为 0.7")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = args.output_dir / "storage_preflight.json"
    if not storage_path.is_file():
        raise FileNotFoundError(f"fixed eval 启动前必须存在 storage_preflight.json: {storage_path}")
    storage_preflight = json.loads(storage_path.read_text(encoding="utf-8"))
    inventory = build_checkpoint_inventory(args.probe_dir)
    rows = _read_jsonl(args.dataset)
    eval_row = select_probe_row(rows, args.probe_dir)
    adapters = {
        int(item["checkpoint_step"]): Path(item["adapter_dir"])
        for item in inventory["checkpoints"]
    }
    runner = load_multi_adapter_runner(
        model_path=args.model_path,
        adapters=adapters,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
    )
    results = []
    for step in CHECKPOINT_STEPS:
        scorer = make_formality_score_fn(
            review_model_id=args.review_model,
            review_base_url=args.review_base_url,
            policy_model_id=f"{args.model_path}+{adapters[step]}",
            review_max_new_tokens=args.review_max_new_tokens,
        )
        results.extend(
            evaluate_adapter(
                row=eval_row,
                runner=runner,
                score_fn=scorer,
                checkpoint_step=step,
                adapter_dir=adapters[step],
                seeds=FIXED_SEEDS,
                temperature=args.temperature,
            )
        )
    summary = analyze_fixed_eval(results)
    resolved_config = {
        "schema_version": SCHEMA_VERSION,
        "probe_dir": str(args.probe_dir.resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "evidence_id": str(eval_row["evidence_id"]),
        "policy_model": args.model_path,
        "reviewer_model": args.review_model,
        "reward_revision": "qa_formality_confidence_v1",
        "reward_components": [FORMALITY_COMPONENT],
        "calls_video_reviewer": False,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "seeds": list(FIXED_SEEDS),
        "decode_config": {
            "mode": "sampling",
            "temperature": args.temperature,
            "top_p": 1.0,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    artifact_paths = {
        "storage_preflight": str(storage_path.resolve()),
        "checkpoint_inventory": str((args.output_dir / "checkpoint_inventory.json").resolve()),
        "resolved_config": str((args.output_dir / "resolved_config.json").resolve()),
        "results": str((args.output_dir / "fixed_eval_results.jsonl").resolve()),
        "summary": str((args.output_dir / "fixed_eval_summary.json").resolve()),
    }
    manifest = build_run_manifest(
        summary=summary,
        artifact_paths=artifact_paths,
        storage_preflight=storage_preflight,
    )
    _write_json(args.output_dir / "checkpoint_inventory.json", inventory)
    _write_json(args.output_dir / "resolved_config.json", resolved_config)
    _write_jsonl(args.output_dir / "fixed_eval_results.jsonl", results)
    _write_json(args.output_dir / "fixed_eval_summary.json", summary)
    _write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
