from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[5]
RUNBOOK = ROOT / "training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md"


class TorchRunbookTest(unittest.TestCase):
    def test_runbook_locks_pareto_dpo_gate_contract_and_safe_login_shell(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        required = (
            "feature/annotated-pareto-dpo",
            "32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7",
            "60/10/0",
            "/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo",
            "/scratch/xl6775/models/Qwen3-VL-8B-Instruct",
            "export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo",
            "export TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024",
            "export MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct",
            "export DATA_DIR=${PROJECT_ROOT}/data_RLHF/annotated_preference",
            "export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/annotated_preference",
            "export CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv",
            "export SPLIT_PATH=${DATA_DIR}/split_60_10.json",
            "export MEDIA_MAP=${DATA_DIR}/media_map.json",
            "export DPO_DATA_DIR=${DATA_DIR}/dpo",
            "gate0_data.sbatch",
            "structure_probe.sbatch",
            "smoke1.sbatch",
            "overfit_probe.sbatch",
            "train.sbatch",
            "evaluate.sbatch",
            "sbatch --parsable",
            "squeue -j",
            "sacct -j",
            "${JOBID}",
            "${OUTPUT_ROOT}/gate0_${JOBID}",
            "${OUTPUT_ROOT}/structure_${JOBID}",
            "${OUTPUT_ROOT}/smoke_${JOBID}",
            "${OUTPUT_ROOT}/overfit_${JOBID}",
            "${OUTPUT_ROOT}/train_${JOBID}",
            "${OUTPUT_ROOT}/validation_${JOBID}",
            "OVERFIT_RESULT=${OUTPUT_ROOT}/overfit_${OVERFIT_JOBID}/dpo_gate_result.json",
            "--export=ALL,OVERFIT_RESULT=\"${OVERFIT_RESULT}\"",
            "ADAPTER_DIR=${OUTPUT_ROOT}/train_${TRAIN_JOBID}/adapter",
            "--export=ALL,TRAIN_JOB_ID=${TRAIN_JOBID},ADAPTER_DIR=${ADAPTER_DIR}",
            "compact_qa_v1",
            "expanded schema",
            "Gate 6",
            "自由生成未验证",
        )
        for item in required:
            self.assertIn(item, text)
        for forbidden in ("latest_", "exit", "logout", "exec", "|| exit 1", "set -e", "set -euo pipefail"):
            self.assertNotIn(forbidden, text)

    def test_runbook_contains_copyable_git_sftp_and_data_bootstrap(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        required = (
            "git push -u origin $Branch",
            "git -C \"${SOURCE_ROOT}\" fetch origin \"${BRANCH}:refs/remotes/origin/${BRANCH}\"",
            "worktree add -b \"${BRANCH}\" \"${PROJECT_ROOT}\" \"origin/${BRANCH}\"",
            "sftp xl6775@greene.hpc.nyu.edu",
            "lcd C:/Users/20661/Documents/xwechat_files/wxid_i096w25uhusk22_e748/msg/file/2026-08",
            "cd /scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/data_RLHF/annotated_preference",
            'put "rlhf_candidate_scores_merged_70_packets.csv" rlhf_candidate_scores_merged_70_packets.csv',
            "git status --short --branch",
            "bash -n hpc/grpo_v3/annotated_preference/gate0_data.sbatch",
            "--split-output \"${SPLIT_PATH}\"",
            "--train-evidence-count 60",
            "--validation-evidence-count 10",
            "--locked-test-evidence-count 0",
            "prepare_media.sbatch",
            'assert len(mapping) == 140',
            "ANNOTATION_GATE_PASSED rows=420 evidence=70 split=60/10/0",
            "MEDIA_MAP_PASSED count=140",
            "DPO_DATA_PASSED",
        )
        for item in required:
            self.assertIn(item, text)
