from __future__ import annotations

import unittest

from training.grpo_v3.experiments.human_preference_reviewer.v1.audit import annotation_audit_report
from tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_data import rows_for, AnnotationDataTests


class AuditTests(AnnotationDataTests):
    def test_report_exposes_formal_split_blocker_without_discarding_audit(self) -> None:
        path = self._write(rows_for("e1"))

        report = annotation_audit_report(path, train_count=40, validation_count=10, locked_test_count=10)

        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["formal_split_gate"]["required_evidence_count"], 60)
        self.assertEqual(report["formal_split_gate"]["eligible_evidence_count"], 1)
        self.assertEqual(report["formal_split_gate"]["missing_evidence_count"], 59)


if __name__ == "__main__":
    unittest.main()
