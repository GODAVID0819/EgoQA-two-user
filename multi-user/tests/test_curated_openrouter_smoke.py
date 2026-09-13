from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from egolife_two_user_qa import curated_openrouter_ablation, six_video_qa_tester


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
QA_17 = PACKAGE_ROOT / "qa_curated_17_trace_review_v3.jsonl"
QA_5 = PACKAGE_ROOT / "qa_curated_5_gemini_smoke.jsonl"
DRIVER = PACKAGE_ROOT / "curated_openrouter_ablation.py"
SBATCH = PACKAGE_ROOT / "run_curated_10min_gemini25_flash_openrouter_smoke5.sbatch"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class CuratedOpenRouterSmokeTests(unittest.TestCase):
    def test_smoke_is_five_clean_rows_from_the_frozen_seventeen(self) -> None:
        full = {row["qa_id"]: row for row in read_jsonl(QA_17)}
        smoke = read_jsonl(QA_5)
        self.assertEqual(
            [row["qa_id"] for row in smoke],
            [
                "CURATED_TRACE_V3_Q01",
                "CURATED_TRACE_V3_Q05",
                "CURATED_TRACE_V3_Q10",
                "CURATED_TRACE_V3_Q15",
                "CURATED_TRACE_V3_Q18",
            ],
        )
        self.assertEqual(len({row["generation_group"] for row in smoke}), 5)
        for row in smoke:
            self.assertEqual(row, full[row["qa_id"]])
            self.assertEqual(row["review_status"], "pass")
            self.assertEqual(len(row["required_users"]), 2)
            self.assertNotIn("review_note", row)

    def test_prebilling_plan_resolves_exact_six_versus_pair_prefix(self) -> None:
        agent_dirs = {
            "Jake": "A1_JAKE",
            "Alice": "A2_ALICE",
            "Tasha": "A3_TASHA",
            "Lucia": "A4_LUCIA",
            "Katrina": "A5_KATRINA",
            "Shure": "A6_SHURE",
        }
        qas = [
            {
                **qa,
                "_qa_source_path": str(QA_5),
                "_qa_source_row": index,
                "_evaluation_id": f"smoke:{index}:{qa['qa_id']}",
            }
            for index, qa in enumerate(read_jsonl(QA_5), 1)
        ]
        evidence_rows = []
        six_view_rows = []
        for qa in qas:
            clips = {
                user: {
                    "agent_dir": agent_dir,
                    "agent_name": user,
                    "full_local_video": f"/fake/{qa['qa_id']}_{agent_dir}.mp4",
                }
                for user, agent_dir in agent_dirs.items()
            }
            required = qa["required_users"]
            evidence_rows.append(
                {
                    "evidence_id": qa["evidence_id"],
                    "clips": [clips[user] for user in required],
                }
            )
            six_view_rows.append(
                {
                    "evidence_id": qa["evidence_id"],
                    "remaining_full_clips": [
                        {
                            **clips[user],
                            "local_video": clips[user]["full_local_video"],
                            "alignment": "exact_synchronized",
                            "synchronized_with_selected_pair": True,
                        }
                        for user in agent_dirs
                        if user not in required
                    ],
                }
            )
        with (
            patch.object(curated_openrouter_ablation, "load_qa_rows", return_value=qas),
            patch.object(
                curated_openrouter_ablation,
                "iter_jsonl",
                side_effect=[iter(evidence_rows), iter(six_view_rows)],
            ),
            patch.object(six_video_qa_tester, "_nonempty_file", return_value=True),
        ):
            plan = curated_openrouter_ablation.validate_evaluation_plan(
                qa_path=QA_5,
                evidence_path="evidence.jsonl",
                six_view_manifest_path="six_view.jsonl",
                expected_count=5,
            )

        self.assertEqual(plan["logical_model_call_count"], 10)
        self.assertTrue(all(case["required_pair_is_six_prefix"] for case in plan["cases"]))

    def test_driver_and_embedded_python_are_syntactically_valid(self) -> None:
        ast.parse(DRIVER.read_text(encoding="utf-8"), filename=str(DRIVER))
        text = SBATCH.read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
        self.assertEqual(len(blocks), 2)
        for index, block in enumerate(blocks, 1):
            compile(block, f"embedded_python_{index}.py", "exec")
        self.assertNotIn(b"\r", SBATCH.read_bytes())

    def test_nitro_model_preflight_checks_the_underlying_catalog_model(self) -> None:
        response = {
            "data": [
                {
                    "id": "google/gemini-2.5-flash",
                    "architecture": {
                        "input_modalities": ["text", "video"],
                        "output_modalities": ["text"],
                    },
                }
            ]
        }
        with patch.object(
            curated_openrouter_ablation.urllib.request,
            "urlopen",
            return_value=_JsonResponse(response),
        ):
            snapshot = curated_openrouter_ablation.fetch_openrouter_model_snapshot(
                model_id="google/gemini-2.5-flash:nitro",
                base_url="https://openrouter.ai/api/v1",
            )

        self.assertEqual(snapshot["requested_model_id"], "google/gemini-2.5-flash:nitro")
        self.assertEqual(snapshot["catalog_model_id"], "google/gemini-2.5-flash")
        self.assertEqual(snapshot["routing_variant"], "nitro")

    def test_sbatch_has_billing_and_hpc_safety_gates(self) -> None:
        text = SBATCH.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres=gpu", text)
        self.assertNotIn("#SBATCH --gpus", text)
        self.assertIn('MODEL_ID="${MODEL_ID:-google/gemini-2.5-flash:nitro}"', text)
        self.assertIn('OPENROUTER_MAX_RETRIES=0', text)
        self.assertIn('OPENROUTER_VIDEO_MAX_EDGE="${OPENROUTER_VIDEO_MAX_EDGE:-512}"', text)
        self.assertIn('OPENROUTER_VIDEO_FPS="${OPENROUTER_VIDEO_FPS:-0.5}"', text)
        self.assertIn('OPENROUTER_VIDEO_CRF="${OPENROUTER_VIDEO_CRF:-30}"', text)
        self.assertIn('condition_order": ["six_users", "required_users"]', text)
        self.assertIn('--expected-count 5', text)
        self.assertIn('logical_model_call_count": 10', text)
        self.assertIn('"cpu_only": True', text)
        self.assertIn('"openrouter_routing_variant": "nitro"', text)
        self.assertIn('-m "${STORAGE_PREFLIGHT_MODULE}"', text)
        self.assertIn("from torchcodec.decoders import VideoDecoder", text)
        self.assertIn('test -s "${DRIVER}"', text)


if __name__ == "__main__":
    unittest.main()
