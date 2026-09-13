"""Score aligned base/LoRA generations and build blinded review artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..experiments.human_preference_reviewer.v1.grpo_reward import (
    REWARD_REVISION,
    prepare_completion_for_review,
    score_completion,
)
from ..experiments.human_preference_reviewer.v1.service import (
    ScoreOnlyReviewerClient,
)
from ..shared.data import read_jsonl, validate_swift_row


COMPARISON_CONTRACT = "base_vs_three_grpo_lora_greedy_and_sampling_heldout_v4"
SPLIT_RATIONALES = {
    "eval": (
        "The eval split is used because the comparison selects a learning rate. "
        "The untouched test split is reserved for one final base-versus-selected-policy run."
    ),
    "test": (
        "The locked test split is used for a broader four-policy diagnostic. Because "
        "the result compares all learning rates and may influence policy selection, "
        "this run consumes the test split for model-selection purposes and must not "
        "later be presented as an untouched final estimate."
    ),
    "heldout": (
        "The 10-packet eval and 20-packet test splits are concatenated in locked "
        "manifest order for a 30-packet exploratory learning-rate comparison. This "
        "combination has been used for model selection and is not an untouched final "
        "evaluation set."
    ),
}


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) if parts else None
    return None


def extract_inference_completion(row: Mapping[str, Any]) -> str:
    """Accept the result shapes emitted by supported ms-swift infer backends."""

    for field in ("response", "completion", "generated_text", "output"):
        text = _content_text(row.get(field))
        if text is not None:
            return text

    messages = row.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                text = _content_text(message.get("content"))
                if text is not None:
                    return text

    for container_name in ("response", "infer_response"):
        container = row.get(container_name)
        if not isinstance(container, Mapping):
            continue
        choices = container.get("choices")
        if isinstance(choices, Sequence) and choices:
            choice = choices[0]
            if isinstance(choice, Mapping):
                message = choice.get("message")
                if isinstance(message, Mapping):
                    text = _content_text(message.get("content"))
                    if text is not None:
                        return text

    choices = row.get("choices")
    if isinstance(choices, Sequence) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping):
            message = choice.get("message")
            if isinstance(message, Mapping):
                text = _content_text(message.get("content"))
                if text is not None:
                    return text
            text = _content_text(choice.get("text"))
            if text is not None:
                return text

    raise ValueError(
        "ms-swift inference row has no supported generated-completion field; "
        f"keys={sorted(row)}"
    )


def _read_inference_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"inference output is empty: {path}")
    return rows


def _parse_generation_spec(value: str) -> tuple[str, str, Path]:
    identity, separator, raw_path = value.partition("=")
    decoding, decoding_separator, name = identity.partition(":")
    decoding = decoding.strip()
    name = name.strip()
    path = Path(raw_path.strip()) if separator else Path()
    if (
        not decoding
        or not decoding_separator
        or not name
        or not separator
        or not raw_path.strip()
    ):
        raise argparse.ArgumentTypeError(
            "--generation must use DECODING:POLICY=/absolute/result.jsonl"
        )
    return decoding, name, path


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _population_std(values: Sequence[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else None


def _bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, samples: int = 10_000
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def _summarize(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    policy_order: Sequence[str],
    baseline_policy: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows_by_policy = {
        policy: [row for row in scored_rows if row["policy"] == policy]
        for policy in policy_order
    }
    policies: dict[str, Any] = {}
    for policy, rows in rows_by_policy.items():
        rewards = [float(row["reward"]) for row in rows]
        valid = [row for row in rows if row["reward_source"] != "deterministic_rejection"]
        head_means = {}
        for field in ("evidence_quality", "answerability", "qa_formality"):
            values = [
                float(row["expected_scores"][field])
                for row in valid
                if isinstance(row.get("expected_scores"), Mapping)
            ]
            head_means[field] = _mean(values)
        policies[policy] = {
            "count": len(rows),
            "reward_mean": _mean(rewards),
            "reward_std": _population_std(rewards),
            "reward_median": statistics.median(rewards) if rewards else None,
            "reward_min": min(rewards) if rewards else None,
            "reward_max": max(rewards) if rewards else None,
            "valid_reviewer_count": len(valid),
            "deterministic_rejection_count": len(rows) - len(valid),
            "deterministic_rejection_rate": (
                (len(rows) - len(valid)) / len(rows) if rows else None
            ),
            "expected_score_means_over_valid": head_means,
        }

    baseline = {
        str(row["evidence_id"]): float(row["reward"])
        for row in rows_by_policy[baseline_policy]
    }
    paired: dict[str, Any] = {}
    for policy in policy_order:
        if policy == baseline_policy:
            continue
        candidate = {
            str(row["evidence_id"]): float(row["reward"])
            for row in rows_by_policy[policy]
        }
        if set(candidate) != set(baseline):
            raise ValueError(f"policy {policy!r} is not aligned with the baseline")
        deltas = [candidate[evidence_id] - baseline[evidence_id] for evidence_id in baseline]
        paired[policy] = {
            "count": len(deltas),
            "mean_reward_delta": _mean(deltas),
            "median_reward_delta": statistics.median(deltas) if deltas else None,
            "wins": sum(delta > 1e-12 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
            "losses": sum(delta < -1e-12 for delta in deltas),
            "packet_bootstrap_95pct_ci_for_mean_delta": _bootstrap_mean_ci(
                deltas, seed=bootstrap_seed
            ),
        }

    ranking = sorted(
        policy_order,
        key=lambda policy: (
            float("-inf")
            if policies[policy]["reward_mean"] is None
            else float(policies[policy]["reward_mean"])
        ),
        reverse=True,
    )
    return {
        "policies": policies,
        "paired_vs_baseline": paired,
        "ranking_by_mean_reviewer_reward": ranking,
    }


def _blind_rows(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
    decoding_order: Sequence[str],
    policy_order: Sequence[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (str(row["decoding"]), str(row["evidence_id"]), str(row["policy"])): row
        for row in scored_rows
    }
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for dataset_index, dataset_row in enumerate(dataset_rows):
        evidence_id = str(dataset_row["evidence_id"])
        for decoding in decoding_order:
            shuffled = list(policy_order)
            random.Random(f"{seed}:{decoding}:{evidence_id}").shuffle(shuffled)
            candidates = []
            mappings = []
            for index, policy in enumerate(shuffled):
                label = chr(ord("A") + index)
                scored = by_key[(decoding, evidence_id, policy)]
                candidates.append({
                    "blind_label": label,
                    "completion": scored["completion"],
                    "qa": scored["qa"],
                })
                mappings.append({
                    "blind_label": label,
                    "policy": policy,
                    "reward": scored["reward"],
                    "reward_source": scored["reward_source"],
                    "expected_scores": scored["expected_scores"],
                })
            blind_rows.append({
                "dataset_index": dataset_index,
                "decoding": decoding,
                "evidence_id": evidence_id,
                "required_users": dataset_row["required_users"],
                "generator_image_paths": dataset_row["generator_image_paths"],
                "reviewer_video_paths": dataset_row["reviewer_video_paths"],
                "candidates": candidates,
            })
            key_rows.append({
                "decoding": decoding,
                "evidence_id": evidence_id,
                "mapping": mappings,
            })
    return blind_rows, {"blind_seed": seed, "rows": key_rows}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def _write_blind_html(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sections: list[str] = []
    for row in rows:
        evidence_id = str(row["evidence_id"])
        decoding = str(row["decoding"])
        retained_paths = "".join(
            f"<li><code>{html.escape(str(value))}</code></li>"
            for value in row.get("generator_image_paths", [])
        )
        reviewer_paths = "".join(
            f"<li><code>{html.escape(str(value))}</code></li>"
            for value in row.get("reviewer_video_paths", [])
        )
        media = (
            "<details><summary>Evidence media paths</summary>"
            "<p><b>Retained generator frames</b></p><ul>" + retained_paths + "</ul>"
            "<p><b>Full reviewer videos</b></p><ul>" + reviewer_paths + "</ul>"
            "</details>"
        )
        cards = []
        for candidate in row["candidates"]:
            qa = candidate.get("qa") or {}
            options = "".join(
                f"<li><b>{chr(ord('A') + index)}.</b> {html.escape(str(option))}</li>"
                for index, option in enumerate(qa.get("options") or [])
            )
            cards.append(
                "<article class='candidate'>"
                f"<h3>Candidate {html.escape(candidate['blind_label'])}</h3>"
                f"<p class='question'>{html.escape(str(qa.get('question') or '[invalid JSON completion]'))}</p>"
                f"<ol>{options}</ol>"
                f"<p><b>Correct:</b> {html.escape(str(qa.get('correct') or ''))} &nbsp; "
                f"<b>Answer:</b> {html.escape(str(qa.get('answer') or ''))}</p>"
                "<details><summary>Raw completion</summary>"
                f"<pre>{html.escape(str(candidate['completion']))}</pre></details>"
                "</article>"
            )
        choices = "".join(
            f"<option value='{label}'>{label}</option>" for label in ("A", "B", "C", "D")
        )
        sections.append(
            f"<section class='packet' data-evidence='{html.escape(evidence_id)}' "
            f"data-decoding='{html.escape(decoding)}'>"
            f"<h2>{html.escape(evidence_id)}</h2>"
            f"<p><b>Decoding condition:</b> {html.escape(decoding)}</p>"
            + media +
            "<div class='candidates'>" + "".join(cards) + "</div>"
            "<div class='label-row'><label>Best candidate: <select class='best'>"
            "<option value=''>Unlabeled</option>" + choices +
            "<option value='tie'>Tie</option><option value='all_bad'>All bad</option>"
            "</select></label>"
            "<label>Confidence: <select class='confidence'><option value=''>-</option>"
            "<option>low</option><option>medium</option><option>high</option></select></label>"
            "<label>Notes: <input class='notes' size='60'></label></div>"
            "</section>"
        )
    document = """<!doctype html>
<html><head><meta charset="utf-8"><title>Blind policy comparison</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;background:#f6f7f9;color:#17191c}
.packet{background:white;padding:1.2rem;margin:0 0 1.5rem;border:1px solid #ccd1d8;border-radius:10px}
.candidates{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
.candidate{border:1px solid #d9dde3;border-radius:8px;padding:1rem}.question{font-size:1.05rem;font-weight:600}
.label-row{display:flex;flex-wrap:wrap;gap:1rem;margin-top:1rem;padding-top:1rem;border-top:1px solid #ddd}
pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:.65rem 1rem;font-weight:700}
@media(max-width:900px){.candidates{grid-template-columns:1fr}}
</style></head><body>
<h1>Blind base-versus-GRPO question review</h1>
<p>Judge question quality and evidence support. Policy identities and reviewer scores are hidden.</p>
<button id="download">Download labels JSON</button>
""" + "".join(sections) + """
<script>
document.getElementById('download').addEventListener('click', () => {
  const rows=[...document.querySelectorAll('.packet')].map(p => ({
    evidence_id:p.dataset.evidence,
    decoding:p.dataset.decoding,
    best_label:p.querySelector('.best').value,
    confidence:p.querySelector('.confidence').value,
    notes:p.querySelector('.notes').value
  }));
  const blob=new Blob([JSON.stringify(rows,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='manual_labels.json';a.click();
  URL.revokeObjectURL(a.href);
});
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def compare(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    dataset_rows = read_jsonl(dataset_path)
    if len(dataset_rows) != args.expected_rows:
        raise ValueError(
            f"comparison dataset row count {len(dataset_rows)} != {args.expected_rows}"
        )
    for row in dataset_rows:
        # Cached split rows retain immutable provenance for the training run. A
        # later, unrelated edit anywhere in prompts.py changes its file hash.
        # Permit that provenance drift only because validate_swift_row also
        # regenerates and compares the complete prompt text byte-for-byte.
        validate_swift_row(row, require_current_prompt_source=False)
    evidence_ids = [str(row["evidence_id"]) for row in dataset_rows]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("comparison dataset evidence_id values are not unique")

    generation_specs = [_parse_generation_spec(value) for value in args.generation]
    spec_keys = [(decoding, policy) for decoding, policy, _ in generation_specs]
    if len(spec_keys) != len(set(spec_keys)):
        raise ValueError("comparison generation decoding/policy pairs are not unique")
    decoding_order = list(dict.fromkeys(decoding for decoding, _, _ in generation_specs))
    if set(decoding_order) != {"greedy", "sampling"} or len(decoding_order) != 2:
        raise ValueError("comparison requires exactly greedy and sampling decoding")
    policies_by_decoding = {
        decoding: [
            policy
            for spec_decoding, policy, _ in generation_specs
            if spec_decoding == decoding
        ]
        for decoding in decoding_order
    }
    policy_order = policies_by_decoding[decoding_order[0]]
    if len(policy_order) != 4 or len(set(policy_order)) != 4:
        raise ValueError("comparison requires exactly four distinct policies")
    for decoding, policies in policies_by_decoding.items():
        if len(policies) != 4 or set(policies) != set(policy_order):
            raise ValueError(
                f"decoding {decoding!r} does not contain the same four policies"
            )
    if args.baseline_policy not in policy_order:
        raise ValueError("baseline policy is missing from --generation")

    client = ScoreOnlyReviewerClient(
        args.reviewer_base_url, timeout_seconds=args.reviewer_timeout_seconds
    )
    client.health()
    scored_rows: list[dict[str, Any]] = []
    for generation_index, (decoding, policy, result_path) in enumerate(
        generation_specs
    ):
        result_path = result_path.resolve()
        inference_rows = _read_inference_rows(result_path)
        if len(inference_rows) != len(dataset_rows):
            raise ValueError(
                f"policy {policy!r} produced {len(inference_rows)} rows; "
                f"expected {len(dataset_rows)}"
            )
        for dataset_index, (dataset_row, inference_row) in enumerate(
            zip(dataset_rows, inference_rows)
        ):
            evidence_id = str(dataset_row["evidence_id"])
            result_evidence_id = str(inference_row.get("evidence_id") or "").strip()
            if result_evidence_id and result_evidence_id != evidence_id:
                raise ValueError(
                    f"policy {policy!r} inference row {dataset_index} evidence mismatch: "
                    f"{result_evidence_id!r} != {evidence_id!r}"
                )
            completion = extract_inference_completion(inference_row)
            packet = json.loads(dataset_row["packet_json"])
            prepared = prepare_completion_for_review(
                completion,
                packet,
                evidence_id=evidence_id,
                candidate_index=generation_index,
                dataset_generator_images=dataset_row["generator_image_paths"],
                dataset_reviewer_videos=dataset_row["reviewer_video_paths"],
            )
            result = score_completion(
                completion,
                packet,
                evidence_id=evidence_id,
                candidate_index=generation_index,
                client=client,
                dataset_generator_images=dataset_row["generator_image_paths"],
                dataset_reviewer_videos=dataset_row["reviewer_video_paths"],
            )
            record = result["record"]
            reviewer_score = record.get("reviewer_score") or {}
            candidate = prepared.get("candidate") or {}
            qa = None if prepared["status"] != "eligible" else {
                "question": candidate["question"],
                "options": candidate["options"],
                "correct": candidate["correct"],
                "answer": candidate["answer"],
            }
            scored_rows.append({
                "comparison_contract": COMPARISON_CONTRACT,
                "reward_revision": REWARD_REVISION,
                "dataset_index": dataset_index,
                "evidence_id": evidence_id,
                "decoding": decoding,
                "policy": policy,
                "completion": completion,
                "qa": qa,
                "format_valid": prepared["status"] == "eligible",
                "rejection_reason": prepared.get("reason"),
                "reward": float(result["reward"]),
                "reward_source": record["reward_source"],
                "expected_scores": reviewer_score.get("expected_scores"),
                "hard_scores": reviewer_score.get("hard_scores"),
                "class_probabilities": reviewer_score.get("class_probabilities"),
            })

    decoding_results = {
        decoding: _summarize(
            [row for row in scored_rows if row["decoding"] == decoding],
            policy_order=policy_order,
            baseline_policy=args.baseline_policy,
            bootstrap_seed=args.blind_seed,
        )
        for decoding in decoding_order
    }
    summary = {
        "status": "passed",
        "comparison_contract": COMPARISON_CONTRACT,
        "reward_revision": REWARD_REVISION,
        "dataset": str(dataset_path),
        "split": args.split,
        "split_rationale": SPLIT_RATIONALES[args.split],
        "decoding_conditions": {
            "greedy": {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
            },
            "sampling": {
                "temperature": args.sampling_temperature,
                "top_p": args.sampling_top_p,
                "top_k": args.sampling_top_k,
                "seed": args.sampling_seed,
                "max_new_tokens": args.max_new_tokens,
            },
        },
        "decoding_results": decoding_results,
        "baseline_policy": args.baseline_policy,
        "policy_order": policy_order,
        "expected_rows_per_policy": args.expected_rows,
        "generation_count": len(scored_rows),
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "scored_generations.jsonl", scored_rows)
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    blind_rows, blind_key = _blind_rows(
        scored_rows,
        dataset_rows=dataset_rows,
        decoding_order=decoding_order,
        policy_order=policy_order,
        seed=args.blind_seed,
    )
    _write_jsonl(output / "manual_review_blind.jsonl", blind_rows)
    (output / "manual_review_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_blind_html(output / "manual_review_blind.html", blind_rows)
    with (output / "manual_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "decoding", "evidence_id", "best_label", "ranking", "confidence", "notes"
        ])
        for decoding in decoding_order:
            for evidence_id in evidence_ids:
                writer.writerow([decoding, evidence_id, "", "", "", ""])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("eval", "test", "heldout"), default="eval"
    )
    parser.add_argument("--generation", action="append", required=True)
    parser.add_argument("--baseline-policy", default="baseline")
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--sampling-temperature", type=float, default=0.85)
    parser.add_argument("--sampling-top-p", type=float, default=0.95)
    parser.add_argument("--sampling-top-k", type=int, default=40)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--reviewer-base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--reviewer-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--blind-seed", type=int, default=20260816)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.expected_rows <= 0
        or args.max_new_tokens <= 0
        or args.sampling_temperature <= 0
        or not 0 < args.sampling_top_p <= 1
        or args.sampling_top_k <= 0
    ):
        raise ValueError(
            "expected rows, max new tokens, sampling temperature/top-k must be "
            "positive, and sampling top-p must be in (0, 1]"
        )
    summary = compare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
