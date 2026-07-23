from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


class V3SlurmContractTests(unittest.TestCase):
    def _script(self, gate: int) -> str:
        return (ROOT / "hpc" / "grpo_v3" / "baseline" / f"gate{gate}.sbatch").read_text(encoding="utf-8").lower()

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
            validate = f"training.grpo_v3.baseline.gate_validate --output-dir \"${{output_dir}}\" --gate {gate}"
            summary = f"training.grpo_v3.baseline.summary --output-dir \"${{output_dir}}\" --gate {gate}"
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

    def test_gate3_v2_is_audit_gated_multi_evidence_temperature_point_three(self) -> None:
        text = (ROOT / "hpc" / "grpo_v3" / "baseline" / "gate3_v2.sbatch").read_text(encoding="utf-8").lower()
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
        text = (ROOT / "hpc" / "grpo_v3" / "baseline" / "greedy_eval.sbatch").read_text(encoding="utf-8").lower()
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
