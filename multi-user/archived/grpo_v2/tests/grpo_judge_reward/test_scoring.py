import unittest

from grpo_judge_reward.scoring import compute_judge_reward


def reward_input(**overrides):
    qa = {
        "qa_id": "qa1",
        "correct": "B",
        "required_users": ["Alice", "Bob"],
    }
    review = {
        "judger": {
            "checks": {
                "qa_formality": {
                    "status": "PASS",
                    "semantic_subchecks": {
                        "other_person_activity_query": {"status": "PASS"}
                    },
                    "schema_branch": {"status": "PASS", "errors": []},
                },
                "evidence_groundedness": {"status": "PASS"},
            },
            "feedback_to_generator": "",
            "gate": {"passed": True},
        },
        "answerability": {
            "evaluations": [
                {"condition_type": "single_user", "condition_id": "single_user::Alice", "users": ["Alice"], "choice": "insufficient"},
                {"condition_type": "single_user", "condition_id": "single_user::Bob", "users": ["Bob"], "choice": "B"},
                {"condition_type": "combined_all_users", "condition_id": "combined_all_users::Alice+Bob", "users": ["Alice", "Bob"], "choice": "B"},
            ],
            "gate": {
                "passed": True,
                "speaker_user": "Alice",
                "evidence_provider_user": "Bob",
            },
        },
        "schema_validation": {"passed": True, "errors": []},
        "review_passed": True,
        "final_decision": {"accepted": True, "rejection_stage": None},
    }
    data = {
        "candidate_id": "g1::0",
        "group_id": "g1",
        "evidence_id": "e1",
        "qa_id": "qa1",
        "attempt": 1,
        "qa": qa,
        "review": review,
        "answerability": review["answerability"],
        "schema_errors": [],
    }
    data.update(overrides)
    return data


class JudgeRewardScoringTests(unittest.TestCase):
    def test_linearly_adds_reward_table_components(self):
        data = reward_input()
        data["answerability"]["evaluations"][1]["choice"] = "A"
        record = compute_judge_reward(data)

        self.assertFalse(record.masked)
        self.assertEqual(record.reward_components["groundedness"], 1.0)
        self.assertEqual(record.reward_components["combined_answerability"], 1.0)
        self.assertEqual(record.reward_components["grounded_answerable_bonus"], 0.5)
        self.assertEqual(record.reward_components["qa_formality"], 0.5)
        self.assertEqual(record.reward_total, 3.0)
        self.assertTrue(record.eligible_for_grpo)

    def test_provider_only_correct_removes_bonus_and_caps_reward(self):
        record = compute_judge_reward(reward_input())

        self.assertTrue(record.provider_only_correct)
        self.assertEqual(record.reward_components["grounded_answerable_bonus"], 0.0)
        self.assertEqual(record.reward_components["provider_only_cap"], 2.0)
        self.assertEqual(record.reward_total, 2.0)

    def test_schema_fail_masks_candidate(self):
        review = reward_input()["review"]
        review["schema_validation"] = {"passed": False, "errors": ["missing fields: review"]}

        record = compute_judge_reward(reward_input(review=review, schema_errors=["missing fields: review"]))

        self.assertTrue(record.masked)
        self.assertEqual(record.mask_reason, "schema_fail")
        self.assertIsNone(record.reward_total)
        self.assertFalse(record.eligible_for_grpo)

    def test_groundedness_uncertain_and_combined_wrong_are_negative(self):
        data = reward_input()
        data["review"]["judger"]["checks"]["evidence_groundedness"]["status"] = "UNCERTAIN"
        data["answerability"]["evaluations"][-1]["choice"] = "A"

        record = compute_judge_reward(data)

        self.assertEqual(record.reward_components["groundedness"], -0.7)
        self.assertEqual(record.reward_components["combined_answerability"], -1.2)
        self.assertEqual(record.reward_components["grounded_answerable_bonus"], 0.0)
        self.assertEqual(record.reward_total, -1.4)

    def test_speaker_leakage_caps_final_reward_at_half_point(self):
        data = reward_input()
        data["answerability"]["evaluations"][0]["choice"] = "B"

        record = compute_judge_reward(data)

        self.assertTrue(record.speaker_only_correct)
        self.assertEqual(record.reward_total, 0.5)
        self.assertEqual(record.reward_components["speaker_leakage_cap"], 0.5)

    def test_subset_and_shallow_activity_penalties_apply(self):
        data = reward_input()
        data["answerability"]["evaluations"].append(
            {"condition_type": "proper_subset", "condition_id": "proper_subset::Alice", "users": ["Alice"], "choice": "B"}
        )
        data["review"]["judger"]["checks"]["qa_formality"]["semantic_subchecks"]["other_person_activity_query"]["status"] = "FAIL"

        record = compute_judge_reward(data)

        self.assertTrue(record.proper_subset_correct)
        self.assertEqual(record.reward_components["subset_leakage"], -0.8)
        self.assertEqual(record.reward_components["shallow_activity_query"], -0.8)
        self.assertEqual(record.reward_components["shallow_activity_cap"], 1.5)
        self.assertEqual(record.reward_total, 0.9)

    def test_shallow_activity_caps_otherwise_high_reward(self):
        data = reward_input()
        data["answerability"]["evaluations"][1]["choice"] = "A"
        data["review"]["judger"]["checks"]["qa_formality"]["semantic_subchecks"]["other_person_activity_query"]["status"] = "FAIL"

        record = compute_judge_reward(data)

        self.assertFalse(record.provider_only_correct)
        self.assertEqual(record.reward_components["shallow_activity_query"], -0.8)
        self.assertEqual(record.reward_components["shallow_activity_cap"], 1.5)
        self.assertEqual(record.reward_total, 1.5)


if __name__ == "__main__":
    unittest.main()
