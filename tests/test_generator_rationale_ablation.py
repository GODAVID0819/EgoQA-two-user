from __future__ import annotations

import json
from pathlib import Path

import pytest

from egolife_two_user_qa.generator_rationale_ablation import (
    build_judge_content,
    compare_acceptance_summaries,
    compact_production_outputs,
    compact_qa_payload,
    condition_order,
    load_generated_questions,
    parse_pass_fail,
    run_paired_rationale_ablation,
    summarize_baseline_packets,
    summarize_production_outputs,
)


def test_production_launcher_keeps_scoring_and_quota_archived() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "hpc"
        / "run_generator_rationale_ablation_qwen.sbatch"
    ).read_text(encoding="utf-8")

    assert "--judge-quality-quota" not in script
    assert "PRODUCTION_QUALITY_QUOTA" not in script
    assert 'production_judge_contract=binary_PASS_FAIL_only' in script
    assert 'production_point_scoring=legacy_archived_not_active' in script
    assert 'production_quality_quota=legacy_archived_not_active' in script
    assert 'production_rationale_binary_50' in script
    assert 'pass_fail_only=True' in script
    assert 'pass_fail_only=False' not in script


def _qa(index: int) -> dict:
    return {
        "qa_id": f"QA_{index}",
        "question": "Which flavor were the chips in the bowl?",
        "options": ["Salted", "Barbecue", "Cheese", "Chili", "Vinegar"],
        "correct": "A",
        "answer": "Salted",
        "required_users": ["Alice", "Bob"],
        "question_type": "neutral",
        "generator_rationale": (
            f"rationale sentinel {index}: Bob's view confirms that the bowl contains chips."
        ),
        "why_two_users_needed": "Alice sees the bowl while Bob supposedly sees its contents.",
        "per_user_evidence_claims": [{"user": "Bob", "claim": "The bowl has chips."}],
        "referred_timestamps": [12.0],
    }


def _trace(
    index: int,
    videos: list[str],
    *,
    accepted: bool = False,
    attempt: int | None = None,
) -> dict:
    qa = _qa(index)
    return {
        "attempt": index if attempt is None else attempt,
        "media": {
            "full_video_paths": videos,
            "human_audit": {
                "video_evidence": [
                    {"user": "Alice", "full_local_video": videos[0]},
                    {"user": "Bob", "full_local_video": videos[1]},
                ]
            },
        },
        "generation": {
            "raw_output": json.dumps(qa),
            "parsed_qa": qa,
        },
        "result": {"accepted": accepted, "reason": "old pipeline decision"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_loads_only_parseable_attempt_one_from_both_row_layouts(tmp_path: Path) -> None:
    videos = [str(tmp_path / "alice.mp4"), str(tmp_path / "bob.mp4")]
    path = tmp_path / "intermediate.jsonl"
    broken = {
        "attempt": 2,
        "generation": {"raw_output": "not valid JSON"},
        "media": {"full_video_paths": videos},
    }
    _write_jsonl(
        path,
        [
            {
                "status": "accepted",
                "evidence_id": "E1",
                "attempts": [_trace(1, videos, accepted=True)],
            },
            {
                "status": "rejected",
                "evidence_id": "E2",
                "attempts": [{"attempt": 1, "qa": {}}, {"attempt": 2, "qa": {}}],
                "generation_trace": [_trace(2, videos, attempt=1), broken],
            },
            {
                "status": "rejected",
                "evidence_id": "E3",
                "generation_trace": [broken, _trace(3, videos, attempt=2)],
            },
        ],
    )

    candidates = load_generated_questions(path)

    assert [candidate["candidate_id"] for candidate in candidates] == ["Q000001", "Q000002"]
    assert [candidate["qa"]["qa_id"] for candidate in candidates] == ["QA_1", "QA_2"]
    assert candidates[0]["prior_attempt_accepted"] is True
    assert candidates[1]["source_row_status"] == "rejected"
    assert [candidate["attempt_number"] for candidate in candidates] == [1, 1]
    assert [row["user"] for row in candidates[0]["video_evidence"]] == ["Alice", "Bob"]


def test_pair_payload_diff_is_only_generator_rationale(tmp_path: Path) -> None:
    videos = [tmp_path / "alice.mp4", tmp_path / "bob.mp4"]
    for video in videos:
        video.touch()
    candidate = {
        "candidate_id": "Q000001",
        "qa": _qa(1),
        "video_evidence": [
            {"user": "Alice", "path": str(videos[0])},
            {"user": "Bob", "path": str(videos[1])},
        ],
    }

    without = compact_qa_payload(candidate, include_rationale=False)
    with_rationale = compact_qa_payload(candidate, include_rationale=True)
    assert set(with_rationale) - set(without) == {"generator_rationale"}
    assert {key: with_rationale[key] for key in without} == without
    assert "why_two_users_needed" not in with_rationale
    assert "per_user_evidence_claims" not in with_rationale
    assert "referred_timestamps" not in with_rationale

    content_without, _ = build_judge_content(
        candidate,
        include_rationale=False,
        media_root=None,
        max_image_pixels=131072,
        fps=1.0,
    )
    content_with, _ = build_judge_content(
        candidate,
        include_rationale=True,
        media_root=None,
        max_image_pixels=131072,
        fps=1.0,
    )
    assert content_without[:-1] == content_with[:-1]
    assert "bowl of dough" in content_without[-1]["text"]
    assert "rationale sentinel 1" not in content_without[-1]["text"]
    assert "rationale sentinel 1" in content_with[-1]["text"]
    assert "score, rank" in content_without[-1]["text"]


def test_condition_order_is_deterministic_and_counterbalanced() -> None:
    orders = [condition_order(index, seed=1729) for index in range(1, 5)]
    assert orders[0] != orders[1]
    assert orders[0] == orders[2]
    assert orders[1] == orders[3]
    assert all(set(order) == {"with_rationale", "without_rationale"} for order in orders)


def test_pass_fail_parser_rejects_scores() -> None:
    assert parse_pass_fail('{"status":"fail","reason":"The bowl contains dough."}') == {
        "status": "FAIL",
        "reason": "The bowl contains dough.",
    }
    with pytest.raises(ValueError, match="unexpected fields"):
        parse_pass_fail('{"status":"FAIL","reason":"wrong object","score":1}')


def test_acceptance_comparison_uses_50_packet_style_aggregate_counts(tmp_path: Path) -> None:
    videos = [str(tmp_path / "alice.mp4"), str(tmp_path / "bob.mp4")]
    intermediate = tmp_path / "intermediate.jsonl"
    _write_jsonl(
        intermediate,
        [
            {
                "status": "accepted",
                "evidence_id": "E1",
                "attempts": [_trace(1, videos, accepted=True, attempt=1)],
            },
            {
                "status": "rejected",
                "evidence_id": "E2",
                "generation_trace": [
                    _trace(2, videos, attempt=1),
                    _trace(3, videos, attempt=2),
                ],
            },
        ],
    )
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    _write_jsonl(
        accepted,
        [
            {"evidence_id": "E1", "attempt_count": 1},
            {"evidence_id": "E2", "attempt_count": 2},
        ],
    )
    _write_jsonl(rejected, [])

    baseline = summarize_baseline_packets(intermediate, packet_count=2)
    production = summarize_production_outputs(accepted, rejected, packet_count=2)
    comparison = compare_acceptance_summaries(baseline, production)

    assert baseline["accepted_count"] == 1
    assert baseline["total_generation_attempts"] == 3
    assert production["accepted_count"] == 2
    assert production["total_generation_attempts"] == 3
    assert comparison["acceptance_rate_delta"] == 0.5
    assert comparison["acceptance_rate_delta_percentage_points"] == 50.0
    assert comparison["no_rationale_acceptance_direction"] == "higher"
    assert comparison["same_evidence_id_set"] is True


def test_compacts_production_attempt_history_without_losing_attempt_counts(
    tmp_path: Path,
) -> None:
    videos = [str(tmp_path / "alice.mp4"), str(tmp_path / "bob.mp4")]
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"

    def production_trace(index: int) -> dict:
        trace = _trace(index, videos, attempt=index)
        trace["judge"] = {
            "generator_rationale_included": False,
            "pass_fail_only": True,
        }
        return trace

    _write_jsonl(
        accepted,
        [{"evidence_id": "E1", "generation_trace": [production_trace(1)]}],
    )
    _write_jsonl(
        rejected,
        [
            {
                "evidence_id": "E2",
                "attempts": [{"attempt": 1, "qa": _qa(1)}],
                "generation_trace": [production_trace(1), production_trace(2)],
            }
        ],
    )

    counts = compact_production_outputs(accepted, rejected)
    accepted_row = json.loads(accepted.read_text(encoding="utf-8"))
    rejected_row = json.loads(rejected.read_text(encoding="utf-8"))
    summary = summarize_production_outputs(accepted, rejected, packet_count=2)

    assert counts == {"accepted_rows": 1, "rejected_rows": 1}
    assert "generation_trace" not in accepted_row
    assert "generation_trace" not in rejected_row
    assert "attempts" not in rejected_row
    assert accepted_row["attempt_count"] == 1
    assert rejected_row["attempt_count"] == 2
    assert rejected_row["production_judge_config"] == {
        "generator_rationale_included": False,
        "pass_fail_only": True,
    }
    assert summary["total_generation_attempts"] == 3

    # Recompaction is safe for a resumed job whose dense traces are already gone.
    compact_production_outputs(accepted, rejected)
    assert json.loads(rejected.read_text(encoding="utf-8"))["attempt_count"] == 2


class _FakeJudge:
    model_id = "fake-paired-judge"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_content(self, content: list[dict]) -> str:
        prompt = str(content[-1]["text"])
        self.prompts.append(prompt)
        if "rationale sentinel" in prompt:
            return json.dumps(
                {
                    "status": "PASS",
                    "reason": "The rationale anchors me to the generator's chips interpretation.",
                }
            )
        return json.dumps(
            {
                "status": "FAIL",
                "reason": "The bowl visibly contains dough, not chips.",
            }
        )


def test_end_to_end_runs_exactly_two_compact_binary_calls_per_question_and_resumes(
    tmp_path: Path,
) -> None:
    videos = [tmp_path / "alice.mp4", tmp_path / "bob.mp4"]
    for video in videos:
        video.touch()
    intermediate = tmp_path / "intermediate.jsonl"
    _write_jsonl(
        intermediate,
        [
            {
                "status": "rejected",
                "evidence_id": "E1",
                "generation_trace": [_trace(1, [str(path) for path in videos])],
            },
            {
                "status": "accepted",
                "evidence_id": "E2",
                "attempts": [
                    _trace(2, [str(path) for path in videos], accepted=True, attempt=1)
                ],
            },
        ],
    )
    output = tmp_path / "paired_pass_fail.jsonl"
    runner = _FakeJudge()

    summary = run_paired_rationale_ablation(
        intermediate,
        output_path=output,
        runner=runner,
        max_image_pixels=131072,
        seed=1729,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(runner.prompts) == 4
    assert len(rows) == 2
    assert all(row["status_pair"] == "without_FAIL__with_PASS" for row in rows)
    assert summary["question_count"] == 2
    assert summary["judge_call_count"] == 4
    assert summary["removal_changed_pass_to_fail"] == 2
    assert summary["scoring"] is False
    assert summary["quota"] is False
    assert not any(
        forbidden in row
        for row in rows
        for forbidden in ("score", "quality_score", "ranking", "prompt", "raw_output", "history")
    )

    resumed_runner = _FakeJudge()
    resumed = run_paired_rationale_ablation(
        intermediate,
        output_path=output,
        runner=resumed_runner,
        max_image_pixels=131072,
        seed=1729,
        resume=True,
    )
    assert resumed_runner.prompts == []
    assert resumed["question_count"] == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_malformed_judge_json_gets_bounded_format_repair_without_dense_trace(
    tmp_path: Path,
) -> None:
    videos = [tmp_path / "alice.mp4", tmp_path / "bob.mp4"]
    for video in videos:
        video.touch()
    intermediate = tmp_path / "intermediate.jsonl"
    _write_jsonl(
        intermediate,
        [
            {
                "status": "rejected",
                "evidence_id": "E1",
                "generation_trace": [
                    _trace(1, [str(path) for path in videos], attempt=1)
                ],
            }
        ],
    )

    class RepairingJudge:
        model_id = "format-repair-judge"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_content(self, content: list[dict]) -> str:
            prompt = str(content[-1]["text"])
            self.prompts.append(prompt)
            if "FORMAT REPAIR RETRY" not in prompt:
                return "PASS, because the question looks plausible"
            status = "PASS" if "rationale sentinel" in prompt else "FAIL"
            return json.dumps(
                {
                    "status": status,
                    "reason": "The repaired response follows the binary JSON contract.",
                }
            )

    output = tmp_path / "paired.jsonl"
    runner = RepairingJudge()
    summary = run_paired_rationale_ablation(
        intermediate,
        output_path=output,
        runner=runner,
        max_format_attempts=2,
    )
    row = json.loads(output.read_text(encoding="utf-8"))

    assert len(runner.prompts) == 4
    assert row["format_attempts"] == {
        "without_rationale": 2,
        "with_rationale": 2,
    }
    assert row["status_pair"] == "without_FAIL__with_PASS"
    assert summary["judge_decision_count"] == 2
    assert summary["judge_call_count"] == 4
    assert summary["format_retry_count"] == 2
    assert "raw_output" not in row
    assert "prompt" not in row
