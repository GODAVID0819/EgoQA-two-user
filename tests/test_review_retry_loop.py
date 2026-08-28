from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa import video_qa_loop  # noqa: E402


class RecordingRunner:
    model_id = "test-generator"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt, *, image_paths, video_paths):
        self.prompts.append(prompt)
        attempt = len(self.prompts)
        return json.dumps(
            {
                "qa_id": f"candidate-{attempt}",
                "question": f"What did I place on the table in attempt {attempt}?",
                "options": [f"Option {letter}" for letter in "ABCDE"],
                "correct": "A",
                "answer": "Option A",
                "required_users": ["speaker", "provider"],
            }
        )


def run_loop_with_judgments(tmp_path: Path, judgments: list[dict[str, object]]):
    evidence_path = tmp_path / "evidence.jsonl"
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    prompts_path = tmp_path / "prompts.jsonl"
    intermediate_path = tmp_path / "intermediate.jsonl"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence_id": "retry-packet",
                "required_users": ["speaker", "provider"],
                "clips": [],
                "source_urls": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = RecordingRunner()
    judgment_iter = iter(judgments)

    def fake_review(**kwargs):
        judgment = next(judgment_iter)
        passed = bool(judgment["passed"])
        feedback = str(judgment.get("feedback") or "")
        judge = {
            "gate": {"passed": passed, "reason": feedback or "passed"},
            "review_passed": passed,
            "feedback_to_generator": feedback,
            "blocking_failures": [] if passed else ["qa_formality"],
            "checks": {},
        }
        return judge, {"gate": {"passed": True}}, {"attempt": kwargs["attempt"]}

    with (
        mock.patch.object(video_qa_loop, "make_runner", return_value=runner),
        mock.patch.object(
            video_qa_loop,
            "run_parallel_review_judges",
            side_effect=fake_review,
        ),
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
            backend="test",
            target_count=1,
            max_attempts=3,
            question_types=("neutral",),
        )

    rejected_rows = [
        json.loads(line)
        for line in rejected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return accepted, rejected_rows, runner


def test_gate_feedback_is_returned_and_second_attempt_can_be_accepted(tmp_path: Path) -> None:
    feedback = "qa_formality: remove the ambiguous reference"

    accepted, rejected, runner = run_loop_with_judgments(
        tmp_path,
        [
            {"passed": False, "feedback": feedback},
            {"passed": True},
        ],
    )

    assert len(runner.prompts) == 2
    assert feedback in runner.prompts[1]
    assert "What did I place on the table in attempt 1?" in runner.prompts[1]
    assert len(accepted) == 1
    assert accepted[0]["attempt_count"] == 2
    assert rejected == []


def test_item_is_rejected_only_after_all_three_attempts_fail(tmp_path: Path) -> None:
    accepted, rejected, runner = run_loop_with_judgments(
        tmp_path,
        [
            {"passed": False, "feedback": "first gate feedback"},
            {"passed": False, "feedback": "second gate feedback"},
            {"passed": False, "feedback": "third gate feedback"},
        ],
    )

    assert len(runner.prompts) == 3
    assert "first gate feedback" in runner.prompts[1]
    assert "second gate feedback" in runner.prompts[2]
    assert accepted == []
    assert len(rejected) == 1
    assert len(rejected[0]["attempts"]) == 3
    assert rejected[0]["review"]["accepted"] is False


def test_ten_minute_profile_reaches_generator_and_review_without_second_runner(
    tmp_path: Path,
) -> None:
    class ProfileRunner:
        model_id = "profile-runner"

        def __init__(self) -> None:
            self.call_profiles = []

        def generate(self, prompt, *, image_paths, video_paths, call_profile=None):
            self.call_profiles.append(call_profile)
            return json.dumps(
                {
                    "qa_id": "profile-qa",
                    "question": "What object could I not identify from where I was sitting?",
                    "options": [f"Option {letter}" for letter in "ABCDE"],
                    "correct": "A",
                    "answer": "Option A",
                    "required_users": [
                        "speaker",
                        "provider_one",
                        "provider_two",
                        "provider_three",
                        "provider_four",
                        "provider_five",
                    ],
                }
            )

    evidence_path = tmp_path / "evidence.jsonl"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence_id": "ten-minute-profile-packet",
                "required_users": [
                    "speaker",
                    "provider_one",
                    "provider_two",
                    "provider_three",
                    "provider_four",
                    "provider_five",
                ],
                "clips": [],
                "source_urls": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runner = ProfileRunner()
    observed_stage_profiles = []

    def fake_review(**kwargs):
        observed_stage_profiles.append(kwargs["stage_profiles"])
        return (
            {"gate": {"passed": True}, "review_passed": True, "checks": {}},
            {"gate": {"passed": True}},
            {"attempt": kwargs["attempt"]},
        )

    with (
        mock.patch.object(video_qa_loop, "make_runner", return_value=runner) as factory,
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
            return_value={"accepted": True},
        ),
    ):
        accepted = video_qa_loop.generate_video_qa_loop(
            evidence_path=evidence_path,
            output_path=tmp_path / "accepted.jsonl",
            prompts_path=tmp_path / "prompts.jsonl",
            rejected_path=tmp_path / "rejected.jsonl",
            backend="test",
            target_count=1,
            max_attempts=1,
            question_types=("neutral",),
            six_user_ten_minute_reasoning_profile=True,
        )

    profiles = video_qa_loop.six_user_ten_minute_reasoning_profiles()
    assert len(accepted) == 1
    assert factory.call_count == 1
    assert runner.call_profiles == [profiles["generator"]]
    assert observed_stage_profiles == [profiles]


def test_review_summary_counts_each_gate_by_attempt() -> None:
    def trace(attempt: int, statuses: dict[str, str] | None):
        merged = (
            {"checks": {name: {"status": status} for name, status in statuses.items()}}
            if statuses is not None
            else {}
        )
        return {"attempt": attempt, "judge": {"merged": merged}}

    accepted = [
        {
            "attempt_count": 2,
            "generation_trace": [
                trace(
                    1,
                    {
                        "qa_formality": "FAIL",
                        "evidence_groundedness": "PASS",
                        "answerability": "PASS",
                    },
                ),
                trace(
                    2,
                    {
                        "qa_formality": "PASS",
                        "evidence_groundedness": "PASS",
                        "answerability": "PASS",
                    },
                ),
            ],
        }
    ]
    rejected = [
        {
            "generation_trace": [
                trace(
                    1,
                    {
                        "qa_formality": "FAIL",
                        "evidence_groundedness": "FAIL",
                        "answerability": "FAIL",
                    },
                ),
                trace(
                    2,
                    {
                        "qa_formality": "PASS",
                        "evidence_groundedness": "FAIL",
                        "answerability": "PASS",
                    },
                ),
                trace(3, None),
            ]
        }
    ]

    summary = video_qa_loop.summarize_review_gate_attempts(
        accepted,
        rejected,
        max_attempts=3,
    )

    assert summary["attempted_group_count"] == 2
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["acceptance_rate"] == 0.5
    assert summary["generator_attempt_count"] == 5
    assert summary["accepted_by_attempt"] == {"1": 0, "2": 1, "3": 0}
    assert summary["rejected_after_max_attempts"] == 1
    assert summary["gate_review_by_attempt"]["1"]["qa_formality"] == {
        "pass_count": 0,
        "reject_count": 2,
        "not_run_count": 0,
        "reviewed_count": 2,
        "pass_rate": 0.0,
        "reject_rate": 1.0,
    }
    assert summary["gate_review_by_attempt"]["2"]["evidence_groundedness"][
        "reject_count"
    ] == 1
    assert summary["gate_review_by_attempt"]["3"]["answerability"][
        "not_run_count"
    ] == 1
    assert summary["gate_review_overall"]["answerability"] == {
        "pass_count": 3,
        "reject_count": 1,
        "not_run_count": 1,
        "reviewed_count": 4,
        "pass_rate": 0.75,
        "reject_rate": 0.25,
    }
