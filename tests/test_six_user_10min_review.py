from __future__ import annotations

import json
from pathlib import Path

from egolife_two_user_qa.tools.render_six_user_10min_review import build_report


def qa_row(slot: str, group: str, *, accepted: bool) -> dict[str, object]:
    status = "accepted" if accepted else "rejected"
    qa = {
        "qa_id": f"qa-{slot}",
        "evidence_id": f"evidence-{slot}",
        "generation_slot_id": slot,
        "generation_group_id": group,
        "question_type": "neutral",
        "speaker_user": "Jake",
        "question": f"Which object was visible in {slot}?",
        "options": ["A object", "B object", "C object", "D object", "E object"],
        "correct": "A",
        "answer": "A object",
        "generator_rationale": "",
        "why_two_users_needed": "legacy field should not be rendered",
        "minimum_required_users": ["Katrina", "Ron"],
        "review": {
            "final_decision": {"accepted": accepted, "reason": status},
            "judger": {
                "checks": {
                    "qa_formality": {"status": "PASS", "reason": "natural", "fix": ""},
                    "evidence_groundedness": {
                        "status": "PASS" if accepted else "FAIL",
                        "reason": "visible" if accepted else "not enough",
                        "vote_summary": {
                            "visible_user_count": 3,
                            "option_support_counts": {"A": 3, "B": 0, "C": 0, "D": 0, "E": 0},
                            "threshold_options": ["A"],
                        },
                    },
                    "answerability": {"status": "PASS" if accepted else "FAIL", "reason": status},
                }
            },
            "answerability": {
                "gate": {
                    "speaker_only_answerable": False,
                    "all_six_answerable": True,
                    "answerability_evaluated_condition_count": 2,
                    "minimum_required_users": ["Katrina", "Ron"],
                    "minimum_required_user_count": 2,
                }
            },
        },
        "raw_output": "large model output should not be rendered",
    }
    return {
        "generation_slot_id": slot,
        "generation_group_id": group,
        "evidence_id": f"evidence-{slot}",
        "status": status,
        "qa": qa,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_report_contains_all_slot_cards_and_statistics(tmp_path: Path) -> None:
    (tmp_path / "six_user_qa_result.json").write_text(
        json.dumps(
            {
                "job_id": 123,
                "status": "completed",
                "evidence_duration_seconds": 600,
                "pruning_max_cross_gap_seconds": 30,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "job_manifest.json").write_text(
        json.dumps(
            {
                "job_id": 123,
                "evidence_duration_seconds": 600,
                "max_new_tokens": 8192,
                "formality_max_new_tokens": 2048,
                "ten_minute_reasoning_profile": True,
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "qa_mcq.jsonl",
        [qa_row("group-a::round_0000", "group-a", accepted=True)],
    )
    write_jsonl(
        tmp_path / "qa_mcq.rejected.jsonl",
        [qa_row("group-b::round_0000", "group-b", accepted=False)],
    )
    write_jsonl(
        tmp_path / "qa_mcq.attempts.jsonl",
        [
            {
                **qa_row("group-a::round_0000", "group-a", accepted=True),
                "attempt": 1,
            },
            {
                **qa_row("group-b::round_0000", "group-b", accepted=False),
                "attempt": 3,
            },
        ],
    )
    write_jsonl(
        tmp_path / "video_first_prompts.jsonl",
        [
            {"stage": "generation"},
            {"stage": "qa_formality_judge"},
            {"stage": "evidence_segment_observation"},
            {"stage": "evidence_groundedness_aggregation"},
            {"stage": "answerability"},
        ],
    )
    write_jsonl(tmp_path / "six_user_candidates.jsonl", [{"generation_group_id": "group-a"}])

    output = build_report(tmp_path, tmp_path / "human_review.md")
    text = output.read_text(encoding="utf-8")

    assert "QA slot 总数：**2**" in text
    assert "group-a" in text and "group-b" in text
    assert "QA 001" in text and "QA 002" in text
    assert "Evidence 可见用户数" in text
    assert "speaker-only" in text
    assert "最小回答视角集" in text
    assert "Katrina" in text and "Ron" in text
    assert "large model output should not be rendered" not in text
    assert "legacy field should not be rendered" not in text
    assert "```json" not in text
    assert "<details>" not in text
