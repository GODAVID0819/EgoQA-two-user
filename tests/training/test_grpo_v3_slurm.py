from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class V3SlurmContractTests(unittest.TestCase):
    def _script(self, gate: int) -> str:
        return (ROOT / "hpc" / f"grpo_v3_ms_swift_gate{gate}.sbatch").read_text(encoding="utf-8").lower()

    def test_all_gate_scripts_pin_native_video_bf16_lora(self) -> None:
        for gate in (0, 1, 2, 3, 4):
            text = self._script(gate)
            self.assertIn("ms-swift==4.2.2", text)
            self.assertIn("qwen3-vl-2b-instruct", text)
            self.assertIn("native_video", text)
            self.assertNotIn("sampled_frames", text)
            self.assertNotIn("qlora", text)
            self.assertNotIn("load_in_4bit", text)
            self.assertIn('export home="${job_home}"', text)
            self.assertNotIn('export home="${home:-', text)
            self.assertIn("pip freeze", text)
            self.assertIn("dependencies.txt", text)
            self.assertIn("rlhf --help", text)
            self.assertNotIn("grep -q", text)
            for cache in (
                "xdg_cache_home",
                "triton_cache_dir",
                "torchinductor_cache_dir",
                "vllm_cache_root",
                "cuda_cache_path",
                "flashinfer_workspace_dir",
            ):
                self.assertIn(cache, text)
            if gate in (1, 2, 3, 4):
                self.assertIn("--tuner_type lora", text)
                self.assertNotIn("--train_type", text)
                self.assertIn("--freeze_vit true", text)
                self.assertIn("--freeze_aligner true", text)

    def test_gate_order_and_reviewer_placement_are_explicit(self) -> None:
        gate0 = self._script(0)
        gate1 = self._script(1)
        gate2 = self._script(2)
        gate3 = self._script(3)
        gate4 = self._script(4)
        self.assertNotIn("vllm serve", gate0)
        self.assertNotIn("vllm serve", gate1)
        self.assertIn("gate0_result.json", gate1)
        self.assertIn("gate1_result.json", gate2)
        self.assertIn("parent_run", gate2)
        self.assertIn("dataset_sha256", gate2)
        self.assertIn("qwen3-vl-8b-instruct", gate2)
        self.assertIn("cuda_visible_devices=1", gate2)
        self.assertIn("vllm serve", gate2)
        self.assertIn('review_max_num_seqs="${review_max_num_seqs:-4}"', gate2)
        self.assertIn('--max-num-seqs "${review_max_num_seqs}"', gate2)
        self.assertIn('egoqa_format_reward_revision="json_three_tier_v1"', gate2)
        self.assertIn('export egoqa_format_reward_revision', gate2)
        repo_preflight = '--check-repo-reward'
        reviewer_start = 'vllm}" serve'
        self.assertIn(repo_preflight, gate2)
        self.assertLess(gate2.index(repo_preflight), gate2.index(reviewer_start))
        self.assertIn("gate2_result.json", gate3)
        self.assertIn("--adapters", gate3)
        self.assertIn("--max_steps 20", gate3)
        self.assertIn("--seed 42", gate3)
        self.assertIn("--data_seed 42", gate3)
        self.assertIn("--dataset_shuffle false", gate3)
        self.assertIn("--lr_scheduler_type constant", gate3)
        self.assertIn("gate3_result.json", gate4)
        self.assertIn("split_manifest.json", gate4)
        self.assertIn("--adapters", gate4)
        self.assertIn("--max_steps 40", gate4)
        self.assertIn("--val_dataset", gate4)
        self.assertIn("--eval_on_start true", gate4)
        self.assertIn("--eval_steps 40", gate4)
        self.assertIn("egoqa_eval_evidence_ids", gate4)

    def test_all_gate_scripts_default_to_validated_training_environment(self) -> None:
        expected = '/scratch/${user}/envs/egoqa-ms-swift-v4.2.2-vllm024'
        for gate in (0, 1, 2, 3, 4):
            self.assertIn(f'train_env="${{train_env:-{expected}}}"', self._script(gate))

    def test_default_resources_use_low_tier_l40s_ladder(self) -> None:
        expected = {
            0: ("#sbatch --gres=gpu:1", "#sbatch --cpus-per-task=8", "#sbatch --mem=32g", "#sbatch --time=01:00:00"),
            1: ("#sbatch --gres=gpu:1", "#sbatch --cpus-per-task=8", "#sbatch --mem=64g", "#sbatch --time=02:00:00"),
            2: ("#sbatch --gres=gpu:2", "#sbatch --cpus-per-task=16", "#sbatch --mem=96g", "#sbatch --time=03:00:00"),
            3: ("#sbatch --gres=gpu:2", "#sbatch --cpus-per-task=8", "#sbatch --mem=64g", "#sbatch --time=06:00:00"),
            4: ("#sbatch --gres=gpu:2", "#sbatch --cpus-per-task=8", "#sbatch --mem=64g", "#sbatch --time=18:00:00"),
        }
        for gate, resource_lines in expected.items():
            text = self._script(gate)
            self.assertIn("#sbatch --constraint=l40s", text)
            self.assertNotIn("--gres=gpu:h100", text)
            for line in resource_lines:
                self.assertIn(line, text)

    def test_all_gpu_jobs_record_peak_evidence(self) -> None:
        for gate in (0, 1, 2, 3, 4):
            text = self._script(gate)
            self.assertIn("nvidia-smi", text)
            self.assertIn("memory.total", text)
            self.assertIn("memory.used", text)
            self.assertIn("utilization.gpu", text)
            self.assertIn("gpu_metrics.csv", text)
            self.assertIn("gpu_monitor_pid", text)

    def test_gate3_and_gate4_keep_failure_manifests_before_exiting(self) -> None:
        for gate in (3, 4):
            text = self._script(gate)
            validate = f"training.grpo_v3_gate_validate --output-dir \"${{output_dir}}\" --gate {gate}"
            summary = f"training.grpo_v3_summary --output-dir \"${{output_dir}}\" --gate {gate}"
            failure_exit = 'exit "${gate_validate_status}"'
            latest = f'latest_gate{gate}_output.txt'
            self.assertIn("set +e", text)
            self.assertIn("gate_validate_status=$?", text)
            self.assertIn(failure_exit, text)
            self.assertLess(text.index(validate), text.index(summary))
            self.assertLess(text.index(summary), text.index(failure_exit))
            self.assertLess(text.index(failure_exit), text.rindex(latest))

    def test_all_v3_artifacts_live_under_one_canonical_root(self) -> None:
        for gate in (0, 1, 2, 3, 4):
            text = self._script(gate)
            self.assertIn('grpo_v3_root="${grpo_v3_root:-${project_root}/outputs/grpo_v3}"', text)
            self.assertIn(f'output_dir="${{output_dir:-${{grpo_v3_root}}/gate{gate}_${{slurm_job_id}}}}"', text)
            self.assertIn(f'${{grpo_v3_root}}/latest_gate{gate}_output.txt', text)
        gate0 = self._script(0)
        self.assertIn('evidence_jsonl="${evidence_jsonl:-${grpo_v3_root}/selected_packets_pruned.jsonl}"', gate0)
        self.assertIn('[[ -s "${evidence_jsonl}" ]]', gate0)
        self.assertNotIn("latest_grpo_evidence.txt", gate0)

    def test_runbook_contains_no_directory_placeholders(self) -> None:
        runbook = (ROOT / "docs" / "GRPO" / "v3" / "MS_SWIFT_NATIVE_VIDEO_TORCH_RUNBOOK_CN.md").read_text(
            encoding="utf-8"
        )
        for placeholder in ("你的实验目录", "替换为本次输出目录", "对应作业日志", "xxx目录", "XXX目录"):
            self.assertNotIn(placeholder, runbook)
        self.assertIn("outputs/grpo_v3/selected_packets_pruned.jsonl", runbook)
        self.assertIn('test -s "${EVIDENCE_JSONL}"', runbook)

    def test_runbook_has_interactive_sftp_torch_diagnostics_and_resource_escalation(self) -> None:
        runbook = (ROOT / "docs" / "GRPO" / "v3" / "MS_SWIFT_NATIVE_VIDEO_TORCH_RUNBOOK_CN.md").read_text(
            encoding="utf-8"
        ).lower()
        self.assertNotIn("## 9. 结果边界", runbook)
        gate34 = runbook[runbook.index("## 9.") :]
        for forbidden in (
            "```powershell",
            "sftp -b",
            "new-item",
            "remove-item",
            "set-content",
            "$localrepo",
            "$torchuser",
            "$remote",
        ):
            self.assertNotIn(forbidden, gate34)
        for required in (
            "sftp xl6775@torch-login-b-2",
            "lcd c:/users/20661/desktop/research/ar/multiuser/egoqa-two-user",
            "cd /scratch/xl6775/projects/egoqa-two-user",
            "put training/grpo_v3_*.py training/",
            "lcd c:/users/20661/desktop/research/ar/multiuser/egoqa-two-user/outputs/grpo_v3",
            "cd /scratch/xl6775/projects/egoqa-two-user/outputs/grpo_v3",
            "put selected_packets_pruned_gate4.jsonl",
            "selected_packets_pruned_gate4.jsonl",
            "scontrol show job",
            "timelimit",
            "traceback (most recent call last)",
            "gpu_metrics.csv",
            "failed_checks",
            "review_max_num_seqs=1",
            "--constraint=h100",
            "--time=12:00:00",
            "--time=36:00:00",
        ):
            self.assertIn(required, runbook)
        self.assertGreaterEqual(runbook.count("sacct -j"), 5)
        self.assertGreaterEqual(runbook.count("out_of_memory"), 3)

    def test_gate3_v2_is_audit_gated_multi_evidence_temperature_point_three(self) -> None:
        text = (ROOT / "hpc" / "grpo_v3_ms_swift_gate3_v2.sbatch").read_text(encoding="utf-8").lower()
        for required in (
            "ms-swift==4.2.2",
            "qwen3-vl-2b-instruct",
            "qwen3-vl-8b-instruct",
            "gate3_v2_train_native_video.jsonl",
            "gate3_v2_split_manifest.json",
            "groundedness_audit_summary.json",
            "approved_for_weight_change",
            'egoqa_content_reward_revision="ground_answer_gap_v1"',
            "--adapters",
            "--max_steps 20",
            "--temperature 0.3",
            "--top_p 1.0",
            "--lr_scheduler_type constant",
            "--dataset_shuffle false",
            'video_max_pixels="${video_max_pixels:-50176}"',
        ):
            self.assertIn(required, text)
        self.assertNotIn("sampled_frames", text)
        self.assertNotIn("qlora", text)

    def test_lora_greedy_eval_is_fixed_one_shot_native_video_job(self) -> None:
        text = (ROOT / "hpc" / "grpo_v3_lora_greedy_eval.sbatch").read_text(encoding="utf-8").lower()
        for required in (
            "qwen3-vl-2b-instruct",
            "qwen3-vl-8b-instruct",
            "gate3_v2_eval_native_video.jsonl",
            "training.grpo_v3_greedy_eval",
            "--adapter-label",
            "--max-image-pixels \"${video_max_pixels}\"",
            "cuda_visible_devices=1",
            '"${vllm}" serve',
            "gpu_metrics.csv",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
