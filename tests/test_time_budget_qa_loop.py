from __future__ import annotations

import json
import argparse
from collections import Counter
import sys
import types
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa import video_qa_loop
from egolife_two_user_qa import qa_generation_schedule
from egolife_two_user_qa.prompts import build_video_generation_prompt
from egolife_two_user_qa.qa_generation_schedule import (
    deadline_reached,
    generation_slot_id,
    round_robin_generation_slots,
)


def packets() -> list[dict]:
    return [
        {
            "evidence_id": f"group-speaker-{speaker}",
            "generation_group_id": "DAY1::12000000",
            "speaker_index": speaker,
        }
        for speaker in (0, 2, 5)
    ]


def test_round_robin_uses_only_available_speakers() -> None:
    slots = list(round_robin_generation_slots(packets(), max_slots=8))

    assert [row["speaker_index"] for row in slots] == [0, 2, 5, 0, 2, 5, 0, 2]
    assert [row["generation_round_index"] for row in slots] == list(range(8))


def test_round_robin_balances_groups_before_speakers() -> None:
    group_speakers = {
        "group-a": [0, 1, 2, 3, 4, 5],
        "group-b": [0, 2, 5],
        "group-c": [1, 4],
    }
    source_packets = [
        {
            "evidence_id": f"{group_id}-speaker-{speaker}",
            "generation_group_id": group_id,
            "speaker_index": speaker,
        }
        for group_id, speakers in group_speakers.items()
        for speaker in speakers
    ]

    slots = list(round_robin_generation_slots(source_packets, max_slots=60))

    assert Counter(slot["generation_group_id"] for slot in slots) == {
        "group-a": 20,
        "group-b": 20,
        "group-c": 20,
    }
    for group_id in group_speakers:
        per_speaker = Counter(
            slot["speaker_index"]
            for slot in slots
            if slot["generation_group_id"] == group_id
        )
        assert set(per_speaker) == set(group_speakers[group_id])
        assert min(per_speaker.values()) >= 2


def test_generation_slot_id_is_stable_per_evidence_and_round() -> None:
    assert generation_slot_id("group-speaker-2", 3) == "group-speaker-2::round_0003"


def test_deadline_reached_uses_explicit_clock() -> None:
    assert deadline_reached(100.0, now_epoch_seconds=99.9) is False
    assert deadline_reached(100.0, now_epoch_seconds=100.0) is True


def test_rounds_rotate_focus_without_changing_three_minute_media() -> None:
    assert hasattr(qa_generation_schedule, "diversity_focus_for_round")
    packet = {
        **packets()[0],
        "required_users": [f"user-{index}" for index in range(6)],
        "clips": [
            {"agent_name": f"user-{index}", "full_local_video": f"user-{index}.mp4"}
            for index in range(6)
        ],
    }

    focuses = [
        qa_generation_schedule.diversity_focus_for_round(packet, round_index)
        for round_index in range(6)
    ]

    assert [row["temporal_band_seconds"] for row in focuses] == [
        [0, 30],
        [30, 60],
        [60, 90],
        [90, 120],
        [120, 150],
        [150, 180],
    ]
    assert len({row["focal_provider"] for row in focuses}) == 5
    assert packet["clips"] == [
        {"agent_name": f"user-{index}", "full_local_video": f"user-{index}.mp4"}
        for index in range(6)
    ]


def test_video_loop_cli_accepts_time_budget_arguments() -> None:
    parser = argparse.ArgumentParser()
    video_qa_loop.add_video_loop_args(parser)

    args = parser.parse_args(
        [
            "--deadline-epoch-seconds",
            "123.5",
            "--repeat-evidence",
            "--max-generation-slots",
            "7",
            "--attempts-output",
            "attempts.jsonl",
        ]
    )

    assert args.deadline_epoch_seconds == 123.5
    assert args.repeat_evidence is True
    assert args.max_generation_slots == 7
    assert args.attempts_output == "attempts.jsonl"


def test_prompt_lists_only_previous_questions_when_present() -> None:
    packet = {
        "required_users": [f"user-{index}" for index in range(6)],
        "previous_questions_to_avoid": ["Where was the cup?", "Who moved the chair?"],
    }

    prompt = build_video_generation_prompt(packet, "neutral")

    assert "Where was the cup?" in prompt
    assert "Who moved the chair?" in prompt
    assert "Do not repeat these questions" in prompt


class TimeBudgetRunner:
    model_id = "test-generator"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.review_calls = 0
        self.questions: list[str] | None = None

    def generate(self, prompt, *, image_paths, video_paths):
        self.prompts.append(prompt)
        index = len(self.prompts)
        question = (
            self.questions[index - 1]
            if self.questions is not None and index <= len(self.questions)
            else f"Generated question {index}?"
        )
        return json.dumps(
            {
                "qa_id": f"candidate-{index}",
                "question": question,
                "options": [f"Option {letter}" for letter in "ABCDE"],
                "correct": "A",
                "answer": "Option A",
                "required_users": [f"user-{user}" for user in range(6)],
            }
        )


def _run_time_budget_loop(
    tmp_path: Path,
    *,
    max_generation_slots: int,
    deadline_side_effect=None,
    judge_passed: bool = True,
    generated_questions: list[str] | None = None,
):
    evidence_path = tmp_path / "evidence.jsonl"
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    prompts_path = tmp_path / "prompts.jsonl"
    intermediate_path = tmp_path / "intermediate.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    evidence_path.write_text(
        "".join(
            json.dumps(
                {
                    "evidence_id": f"speaker-{speaker}",
                    "generation_group_id": "DAY1::12000000",
                    "speaker_index": speaker,
                    "required_users": [f"user-{user}" for user in range(6)],
                    "clips": [],
                    "source_urls": {},
                }
            )
            + "\n"
            for speaker in (0, 2, 5)
        ),
        encoding="utf-8",
    )
    runner = TimeBudgetRunner()
    runner.questions = generated_questions

    def fake_review(**kwargs):
        runner.review_calls += 1
        judge = {
            "gate": {"passed": judge_passed, "reason": "passed" if judge_passed else "retry"},
            "review_passed": judge_passed,
            "feedback_to_generator": "" if judge_passed else "retry",
            "blocking_failures": [] if judge_passed else ["qa_formality"],
            "checks": {},
        }
        return judge, {"gate": {"passed": True}}, {"attempt": kwargs["attempt"]}

    deadline_patch = (
        mock.patch.object(video_qa_loop, "deadline_reached", side_effect=deadline_side_effect)
        if deadline_side_effect is not None
        else mock.patch.object(video_qa_loop, "deadline_reached", return_value=False)
    )
    with (
        deadline_patch,
        mock.patch.object(video_qa_loop, "make_runner", return_value=runner),
        mock.patch.object(video_qa_loop, "run_parallel_review_judges", side_effect=fake_review),
        mock.patch.object(video_qa_loop, "validate_qa_item", return_value=[]),
        mock.patch.object(video_qa_loop, "qa_formality_errors", return_value=[]),
        mock.patch.object(video_qa_loop, "complete_generator_metadata"),
        mock.patch.object(video_qa_loop, "human_audit_packet", return_value={}),
        mock.patch.object(video_qa_loop, "video_evidence_for_packet", return_value=[]),
        mock.patch.object(video_qa_loop, "prepare_runner_video_uploads", return_value={}),
        mock.patch.object(
            video_qa_loop,
            "build_review_from_gates",
            side_effect=lambda **kwargs: {"accepted": kwargs["accepted"]},
        ),
    ):
        accepted = video_qa_loop.generate_video_qa_loop(
            evidence_path=evidence_path,
            output_path=accepted_path,
            prompts_path=prompts_path,
            rejected_path=rejected_path,
            intermediate_path=intermediate_path,
            attempts_path=attempts_path,
            backend="test",
            target_count=1,
            max_attempts=3,
            question_types=("neutral",),
            deadline_epoch_seconds=100.0,
            repeat_evidence=True,
            max_generation_slots=max_generation_slots,
        )

    def rows(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    return accepted, runner, rows(attempts_path), rows(intermediate_path), rows(prompts_path)


def test_time_budget_loop_round_robins_partial_speaker_set_and_ignores_accept_target(
    tmp_path: Path,
) -> None:
    accepted, runner, attempts, _, prompt_rows = _run_time_budget_loop(
        tmp_path,
        max_generation_slots=5,
    )

    assert [row["speaker_index"] for row in accepted] == [0, 2, 5, 0, 2]
    assert len(runner.prompts) == 5
    assert [row["generation_slot_id"] for row in attempts] == [
        "speaker-0::round_0000",
        "speaker-2::round_0001",
        "speaker-5::round_0002",
        "speaker-0::round_0003",
        "speaker-2::round_0004",
    ]
    assert "Generated question 1?" in runner.prompts[1]
    assert [row["generation_slot_id"] for row in prompt_rows if row["stage"] == "generation"] == [
        "speaker-0::round_0000",
        "speaker-2::round_0001",
        "speaker-5::round_0002",
        "speaker-0::round_0003",
        "speaker-2::round_0004",
    ]
    generation_rows = [row for row in prompt_rows if row["stage"] == "generation"]
    assert "generation_diversity_focus" in generation_rows[0]
    assert "generation_diversity_focus" in attempts[0]
    assert generation_rows[0]["generation_diversity_focus"]["temporal_band_seconds"] == [0, 30]
    assert attempts[0]["generation_diversity_focus"]["relation_focus"] == "missing detail"
    assert [row["correct"] for row in accepted] == ["A", "B", "C", "D", "E"]
    assert all(
        row["answer"] == row["options"]["ABCDE".index(row["correct"])]
        for row in accepted
    )


def test_normalized_duplicate_skips_all_expensive_judges_and_uses_attempt_budget(
    tmp_path: Path,
) -> None:
    accepted, runner, attempts, _intermediate, _prompt_rows = _run_time_budget_loop(
        tmp_path,
        max_generation_slots=1,
        judge_passed=False,
        generated_questions=[
            "Where was the cup?",
            "  WHERE   was the cup! ",
            "Where was the cup...",
        ],
    )

    assert accepted == []
    assert len(runner.prompts) == 3
    assert runner.review_calls == 1
    assert len(attempts) == 3
    assert attempts[1]["trace"]["result"]["failure_label"] == "normalized_duplicate_question"
    assert attempts[2]["trace"]["result"]["failure_label"] == "normalized_duplicate_question"


def test_deadline_before_retry_persists_attempt_and_time_budget_partial(tmp_path: Path) -> None:
    calls = 0

    def deadline_after_first_attempt(*args, **kwargs):
        nonlocal calls
        calls += 1
        return calls >= 3

    accepted, runner, attempts, intermediate, _ = _run_time_budget_loop(
        tmp_path,
        max_generation_slots=5,
        deadline_side_effect=deadline_after_first_attempt,
        judge_passed=False,
    )

    assert accepted == []
    assert len(runner.prompts) == 1
    assert len(attempts) == 1
    assert attempts[0]["status"] == "rejected"
    assert intermediate[-1]["status"] == "time_budget_partial"
    assert intermediate[-1]["generation_slot_id"] == "speaker-0::round_0000"
