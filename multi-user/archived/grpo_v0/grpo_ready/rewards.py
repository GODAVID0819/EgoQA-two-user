"""基于冻结历史 evaluator 观测计算 Reward v0。"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

try:
    from egolife_two_user_qa.schema import normalize_correct
except ModuleNotFoundError:
    package_name = "_egoqa_repo_v0"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[3])]
        package.__package__ = package_name
        sys.modules[package_name] = package
    normalize_correct = importlib.import_module(f"{package_name}.schema").normalize_correct

from .records import AttemptRecord, RewardRecord


REWARD_VERSION = "v0"


def _status_from_judge(judge: dict[str, Any] | None, branch_name: str) -> bool | None:
    if not isinstance(judge, dict):
        return None
    branch = judge.get(branch_name)
    if not isinstance(branch, dict):
        return None
    parsed = branch.get("parsed")
    if not isinstance(parsed, dict):
        parsed = branch
    checks = parsed.get("checks")
    check = checks.get(branch_name) if isinstance(checks, dict) else None
    if isinstance(check, dict):
        status = str(check.get("status") or "").upper()
        return status == "PASS" if status else None
    status = str(parsed.get("status") or "").upper()
    return status == "PASS" if status else None


def _choice_is_correct(row: dict[str, Any] | None, correct: str) -> bool | None:
    if not isinstance(row, dict) or "choice" not in row:
        return None
    try:
        return normalize_correct(row.get("choice")) == correct
    except ValueError:
        return False


def _answerability_observations(
    attempt: AttemptRecord,
) -> tuple[bool | None, bool | None, bool | None]:
    qa = attempt.parsed_qa
    answerability = attempt.answerability
    if not isinstance(qa, dict) or not isinstance(answerability, dict):
        return None, None, None
    try:
        correct = normalize_correct(qa.get("correct"))
    except ValueError:
        return None, None, None
    users = list(qa.get("required_users") or [])
    if len(users) < 2:
        return None, None, None
    evaluations = answerability.get("evaluations")
    if not isinstance(evaluations, list):
        return None, None, None

    combined_row = None
    speaker_row = None
    provider_row = None
    for row in evaluations:
        if not isinstance(row, dict):
            continue
        condition_type = row.get("condition_type")
        row_users = list(row.get("users") or [])
        if condition_type == "combined_all_users":
            combined_row = row
        elif condition_type == "single_user" and row_users == [users[0]]:
            speaker_row = row
        elif condition_type == "single_user" and row_users == [users[1]]:
            provider_row = row
    return (
        _choice_is_correct(combined_row, correct),
        _choice_is_correct(speaker_row, correct),
        _choice_is_correct(provider_row, correct),
    )


def compute_reward(attempt: AttemptRecord) -> RewardRecord:
    parse_success = attempt.parsed_qa is not None
    parse_reward = 0.5 if parse_success else -2.0

    if not parse_success:
        schema_pass = None
        schema_error_count = None
        formality_pass = None
        groundedness_pass = None
        combined_correct = None
        speaker_correct = None
        provider_correct = None
    else:
        schema_error_count = len(attempt.schema_errors)
        schema_pass = schema_error_count == 0
        formality_pass = _status_from_judge(attempt.judge, "qa_formality")
        groundedness_pass = _status_from_judge(attempt.judge, "evidence_groundedness")
        combined_correct, speaker_correct, provider_correct = _answerability_observations(attempt)

    schema_reward = None if schema_pass is None else (1.0 if schema_pass else -0.5)
    formality_reward = None if formality_pass is None else (0.5 if formality_pass else -0.5)
    groundedness_reward = None if groundedness_pass is None else (2.0 if groundedness_pass else -2.0)
    combined_reward = None if combined_correct is None else (1.0 if combined_correct else -1.0)
    speaker_reward = None if speaker_correct is None else (-2.0 if speaker_correct else 0.0)
    provider_reward = None if provider_correct is None else 0.0

    components = {
        "schema": schema_reward,
        "formality": formality_reward,
        "groundedness": groundedness_reward,
        "combined": combined_reward,
        "speaker_leakage": speaker_reward,
        "provider_alone": provider_reward,
    }
    missing = tuple(name for name, value in components.items() if value is None)
    total = parse_reward + sum(value for value in components.values() if value is not None)
    return RewardRecord(
        attempt_id=attempt.attempt_id,
        parse_success=parse_success,
        schema_pass=schema_pass,
        schema_error_count=schema_error_count,
        formality_pass=formality_pass,
        groundedness_pass=groundedness_pass,
        combined_correct=combined_correct,
        speaker_alone_correct=speaker_correct,
        provider_alone_correct=provider_correct,
        parse_reward=parse_reward,
        schema_reward=schema_reward,
        formality_reward=formality_reward,
        groundedness_reward=groundedness_reward,
        combined_reward=combined_reward,
        speaker_leakage_reward=speaker_reward,
        provider_alone_reward=provider_reward,
        total=total,
        reward_version=REWARD_VERSION,
        missing_components=missing,
        is_complete_reward=not missing,
    )
