from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NORMAL_SOURCES = {"qa_cross_view_relation_v2", "qa_cross_view_relation_v3"}
TEMPLATE_KEEP_WORDS = frozenset(
    {
        "after", "before", "while", "during", "when", "where", "what", "who",
        "which", "how", "why", "i", "me", "my", "we", "our", "the", "a", "an",
        "did", "was", "were", "is", "are", "do", "does", "had", "has", "have",
        "left", "returned", "return", "see", "saw", "tell", "find", "found",
        "end", "ended", "up", "changed", "change", "missing", "same", "different",
        "other", "person", "happened", "happen", "compared", "between", "to",
        "from", "in", "on", "at", "with", "and", "but", "not", "could", "couldn",
        "t", "wasn", "there",
    }
)
RELATION_TERMS = (
    "after", "before", "while", "during", "end up", "changed", "missing",
    "same", "different", "other person", "compared",
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_delta_ci(first: list[float], last: list[float], *, n: int = 20_000) -> list[float | None]:
    if not first or not last:
        return [None, None]
    random.seed(0)
    deltas: list[float] = []
    for _ in range(n):
        a = [first[random.randrange(len(first))] for _ in first]
        b = [last[random.randrange(len(last))] for _ in last]
        deltas.append((sum(b) / len(b)) - (sum(a) / len(a)))
    deltas.sort()
    return [deltas[int(0.025 * n)], deltas[int(0.975 * n) - 1]]


def _record(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("record") or {}


def _deterministic(row: dict[str, Any]) -> dict[str, Any]:
    return _record(row).get("deterministic") or {}


def _qa(row: dict[str, Any]) -> dict[str, Any]:
    return _deterministic(row).get("qa") or {}


def _judge_trace(row: dict[str, Any]) -> dict[str, Any]:
    return _record(row).get("judge_trace") or {}


def _score(row: dict[str, Any]) -> dict[str, Any]:
    cid = row.get("candidate_id")
    for item in _judge_trace(row).get("candidate_scores") or []:
        if item.get("candidate_id") == cid:
            return item
    return {}


def _normalize_question_template(question: str) -> str:
    tokens = re.findall(r"[a-z]+|\d+", question.lower())
    normalized: list[str] = []
    for token in tokens:
        value = token if token in TEMPLATE_KEEP_WORDS else "<x>"
        if not normalized or value != "<x>" or normalized[-1] != "<x>":
            normalized.append(value)
    return " ".join(normalized)


def analyze_question_patterns(
    rows: list[dict[str, Any]],
    *,
    _include_slices: bool = True,
) -> dict[str, Any]:
    questions = [str(_qa(row).get("question") or "").strip() for row in rows]
    questions = [question for question in questions if question]
    if not questions:
        empty: dict[str, Any] = {
            "question_count": 0,
            "exact_question_unique_rate": None,
            "normalized_template_unique_rate": None,
            "top_template": None,
            "top_template_fraction": None,
            "question_prefix_top5": [],
            "question_word_distribution": {},
            "relation_term_distribution": {},
            "object_word_distribution": {},
            "specified_template_fraction": None,
            "per_evidence_metrics": {},
        }
        if _include_slices:
            empty["high_reward"] = analyze_question_patterns([], _include_slices=False)
            empty["by_phase"] = {}
        return empty
    templates = [_normalize_question_template(question) for question in questions]
    template_counts = Counter(templates)
    top_template, top_count = template_counts.most_common(1)[0]
    prefix_counts = Counter(" ".join(question.lower().split()[:5]) for question in questions)
    question_words = Counter()
    relation_terms = Counter()
    object_words = Counter()
    specified_template_count = 0
    for question in questions:
        lowered = question.lower()
        match = re.search(r"\b(who|what|where|when|which|how|why)\b", lowered)
        question_words[match.group(1) if match else "<none>"] += 1
        for term in RELATION_TERMS:
            if term in lowered:
                relation_terms[term] += 1
        for token in re.findall(r"[a-z]+", lowered):
            if token not in TEMPLATE_KEEP_WORDS and len(token) > 2:
                object_words[token] += 1
        if re.search(r"\bi was\b.*\bbut\b.*\bcould(?:n['’]?t| not)?\b", lowered):
            specified_template_count += 1

    per_evidence: dict[str, dict[str, Any]] = {}
    evidence_questions: dict[str, list[str]] = defaultdict(list)
    evidence_types: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        question = str(_qa(row).get("question") or "").strip()
        if not question:
            continue
        evidence_id = str(row.get("evidence_id") or "<unknown>")
        evidence_questions[evidence_id].append(question)
        evidence_types[evidence_id][str(row.get("question_type") or "<unknown>")] += 1
    for evidence_id, items in sorted(evidence_questions.items()):
        normalized = [_normalize_question_template(item) for item in items]
        per_evidence[evidence_id] = {
            "question_count": len(items),
            "exact_unique_rate": len(set(items)) / len(items),
            "normalized_template_unique_rate": len(set(normalized)) / len(normalized),
            "question_type_counts": dict(sorted(evidence_types[evidence_id].items())),
        }
    result = {
        "question_count": len(questions),
        "exact_question_unique_rate": len(set(questions)) / len(questions),
        "normalized_template_unique_rate": len(template_counts) / len(templates),
        "top_template": top_template,
        "top_template_fraction": top_count / len(templates),
        "question_prefix_top5": [
            {"prefix": prefix, "count": count}
            for prefix, count in prefix_counts.most_common(5)
        ],
        "question_word_distribution": dict(sorted(question_words.items())),
        "relation_term_distribution": dict(sorted(relation_terms.items())),
        "object_word_distribution": dict(object_words.most_common(30)),
        "specified_template_fraction": specified_template_count / len(questions),
        "per_evidence_metrics": per_evidence,
    }
    if _include_slices:
        high_reward_rows = [
            row
            for row in rows
            if row.get("reward") is not None and float(row["reward"]) >= 0.8
        ]
        result["high_reward"] = analyze_question_patterns(
            high_reward_rows,
            _include_slices=False,
        )
        phase_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            phase_rows[str(row.get("phase") or "<unknown>")].append(row)
        result["by_phase"] = {
            phase: analyze_question_patterns(items, _include_slices=False)
            for phase, items in sorted(phase_rows.items())
        }
    return result


def _load_rows(probe_dir: Path) -> list[dict[str, Any]]:
    trace = probe_dir / "reward_trace.jsonl"
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(trace.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            row["trace_line"] = line_number
            rows.append(row)
    return rows


def analyze(probe_dir: Path) -> dict[str, Any]:
    rows = _load_rows(probe_dir)
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[int(row["reward_call_index"])].append(row)
    group_ids = sorted(by_group)
    all_group_means = [_mean([float(row["reward"]) for row in by_group[group]]) for group in group_ids]
    all_group_means = [item for item in all_group_means if item is not None]

    judge_group_means_by_id: dict[int, float] = {}
    normal_counts_by_group: dict[int, int] = {}
    for group in group_ids:
        values = [
            float(row["reward"])
            for row in by_group[group]
            if _record(row).get("reward_source") in NORMAL_SOURCES
        ]
        normal_counts_by_group[group] = len(values)
        if values:
            judge_group_means_by_id[group] = sum(values) / len(values)

    judged_group_ids = sorted(judge_group_means_by_id)
    first10_ids = set(judged_group_ids[:10])
    last10_ids = set(judged_group_ids[-10:])
    first20_ids = set(judged_group_ids[:20])
    last20_ids = set(judged_group_ids[-20:])
    first10_judge_groups = [judge_group_means_by_id[i] for i in judged_group_ids if i in first10_ids]
    last10_judge_groups = [judge_group_means_by_id[i] for i in judged_group_ids if i in last10_ids]
    first20_judge_groups = [judge_group_means_by_id[i] for i in judged_group_ids if i in first20_ids]
    last20_judge_groups = [judge_group_means_by_id[i] for i in judged_group_ids if i in last20_ids]

    first10_judge_candidates = [
        float(row["reward"])
        for row in rows
        if int(row["reward_call_index"]) in first10_ids and _record(row).get("reward_source") in NORMAL_SOURCES
    ]
    last10_judge_candidates = [
        float(row["reward"])
        for row in rows
        if int(row["reward_call_index"]) in last10_ids and _record(row).get("reward_source") in NORMAL_SOURCES
    ]

    blocking_errors: Counter[str] = Counter()
    for row in rows:
        errors = _deterministic(row).get("blocking_errors") or []
        blocking_errors.update(errors or ["<none>"])

    score_dist = {
        "relation": Counter(),
        "naturalness": Counter(),
        "consistency": Counter(),
        "anchor": Counter(),
    }
    for row in rows:
        score = _score(row)
        if score:
            score_dist["relation"][str(score.get("cross_view_relation_score"))] += 1
            score_dist["naturalness"][str(score.get("semantic_naturalness_score"))] += 1
            score_dist["consistency"][str(score.get("internal_consistency_score"))] += 1
            score_dist["anchor"][str(score.get("anchor_tier"))] += 1
        else:
            for counter in score_dist.values():
                counter["<not_judged>"] += 1

    return {
        "probe_dir": str(probe_dir),
        "row_count": len(rows),
        "group_count": len(group_ids),
        "all_rows": {
            "first10_mean": _mean(all_group_means[:10]),
            "last10_mean": _mean(all_group_means[-10:]),
            "delta": (_mean(all_group_means[-10:]) or 0.0) - (_mean(all_group_means[:10]) or 0.0),
            "first20_mean": _mean(all_group_means[:20]),
            "last20_mean": _mean(all_group_means[-20:]),
            "delta20": (_mean(all_group_means[-20:]) or 0.0) - (_mean(all_group_means[:20]) or 0.0),
            "bootstrap_ci_first10_last10": _bootstrap_delta_ci(all_group_means[:10], all_group_means[-10:]),
        },
        "judge_only_group_mean": {
            "first10_group_count": len(first10_judge_groups),
            "last10_group_count": len(last10_judge_groups),
            "first10_mean": _mean(first10_judge_groups),
            "last10_mean": _mean(last10_judge_groups),
            "delta": (_mean(last10_judge_groups) or 0.0) - (_mean(first10_judge_groups) or 0.0),
            "first20_mean": _mean(first20_judge_groups),
            "last20_mean": _mean(last20_judge_groups),
            "delta20": (_mean(last20_judge_groups) or 0.0) - (_mean(first20_judge_groups) or 0.0),
            "bootstrap_ci_first10_last10": _bootstrap_delta_ci(first10_judge_groups, last10_judge_groups),
            "normal_counts_by_group": normal_counts_by_group,
        },
        "judge_only_candidate_mean": {
            "first10_n": len(first10_judge_candidates),
            "last10_n": len(last10_judge_candidates),
            "first10_mean": _mean(first10_judge_candidates),
            "last10_mean": _mean(last10_judge_candidates),
            "delta": (_mean(last10_judge_candidates) or 0.0) - (_mean(first10_judge_candidates) or 0.0),
            "bootstrap_ci_first10_last10": _bootstrap_delta_ci(first10_judge_candidates, last10_judge_candidates),
        },
        "reward_source": dict(Counter(_record(row).get("reward_source") for row in rows)),
        "format_status": dict(Counter(_record(row).get("format_status") for row in rows)),
        "blocking_errors": dict(blocking_errors),
        "score_dist": {key: dict(value) for key, value in score_dist.items()},
        "order_instability_groups": sum(
            1 for group in group_ids if _judge_trace(by_group[group][0]).get("order_instability")
        ),
        "question_patterns": analyze_question_patterns(rows),
    }


def write_slim_csv(probe_dir: Path, rows: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "bucket",
        "reward_call_index",
        "candidate_id",
        "reward",
        "reward_source",
        "format_status",
        "blocking_errors",
        "question",
        "options",
        "correct",
        "answer",
        "judge_relation",
        "judge_naturalness",
        "judge_consistency",
        "judge_anchor",
        "judge_reason",
        "order_instability",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = _record(row)
            det = _deterministic(row)
            qa = _qa(row)
            score = _score(row)
            source = record.get("reward_source")
            bucket = (
                "normal_high_judge"
                if source in NORMAL_SOURCES and float(row["reward"]) >= 0.8
                else "normal_judge"
                if source in NORMAL_SOURCES
                else "blocked_penalty"
                if source == "deterministic_blocking_penalty"
                else "skipped_zero_group"
            )
            writer.writerow(
                {
                    "bucket": bucket,
                    "reward_call_index": row.get("reward_call_index"),
                    "candidate_id": row.get("candidate_id"),
                    "reward": row.get("reward"),
                    "reward_source": source,
                    "format_status": record.get("format_status"),
                    "blocking_errors": ";".join(det.get("blocking_errors") or []),
                    "question": qa.get("question"),
                    "options": json.dumps(qa.get("options"), ensure_ascii=False),
                    "correct": qa.get("correct"),
                    "answer": qa.get("answer"),
                    "judge_relation": score.get("cross_view_relation_score"),
                    "judge_naturalness": score.get("semantic_naturalness_score"),
                    "judge_consistency": score.get("internal_consistency_score"),
                    "judge_anchor": score.get("anchor_tier"),
                    "judge_reason": (score.get("reasons") or {}).get("summary"),
                    "order_instability": _judge_trace(row).get("order_instability"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    summary = analyze(args.probe_dir)
    summary_path = args.summary or args.probe_dir / "qa_cross_view_relation_probe_audit_summary.json"
    csv_path = args.csv or args.probe_dir / "qa_cross_view_relation_probe_slim_audit.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_slim_csv(args.probe_dir, _load_rows(args.probe_dir), csv_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"SUMMARY={summary_path}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
