"""真实 reviewer v1 数据类的 Pareto 测试夹具。"""

from __future__ import annotations

from training.grpo_v3.experiments.human_preference_reviewer.v1.data import (
    CandidateRecord,
    EvidenceRecord,
)


def candidate(
    candidate_id: str,
    *,
    evidence_id: str = "evidence-1",
    display_order: int = 1,
    qa_formality: int | None = 3,
    evidence_quality: int | None = 3,
    answerability: int | None = 3,
    question: str = "Who performed the action?",
    options: tuple[str, ...] = ("A", "B", "C"),
    correct: str = "A",
    answer: str = "A",
    overall_rank: int | None = 1,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        evidence_id=evidence_id,
        display_order=display_order,
        question=question,
        options=options,
        correct=correct,
        answer=answer,
        evidence_quality=evidence_quality,
        answerability=answerability,
        qa_formality=qa_formality,
        overall_rank=overall_rank,
    )


def evidence(
    *candidates: CandidateRecord,
    evidence_id: str = "evidence-1",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        annotation_status="scored",
        video_a_user="speaker",
        video_a_source="source-a",
        video_b_user="provider",
        video_b_source="source-b",
        candidates=tuple(candidates),
    )
