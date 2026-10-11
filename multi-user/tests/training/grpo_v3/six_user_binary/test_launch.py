"""训练接线与资源边界，不提交作业或加载模型。"""
import unittest
import json
from pathlib import Path
import tempfile
from test_pipeline import require_module
from unittest.mock import patch


def run_config():
    return {"project_root": "/scratch/test/project", "train_python": "/scratch/test/train/bin/python",
        "judge_python": "/scratch/test/judge/bin/python", "policy_model": "/scratch/test/model",
        "judge_config": "/scratch/test/judge.json", "train_dataset": "/scratch/test/train.jsonl",
        "val_dataset": "/scratch/test/val.jsonl", "output_root": "/scratch/test/outputs",
        "scratch_root": "/scratch/test/jobs", "reward_mode": "continuous", "policy_gpu_indices": [0],
        "judge_gpu_indices": [1, 2], "walltime_seconds": 10800, "keeper_reserve_gib": 15,
        "keeper_max_prealloc_gib": .25, "max_steps": 10, "num_generations": 4,
        "num_generations_eval": 4, "max_length": 32768, "max_completion_length": 1024,
        "max_pixels": 262144, "judge_port": 8766, "learning_rate": 1e-5, "beta": .04,
        "temperature": .85, "top_p": .95, "top_k": 40}


class LaunchTests(unittest.TestCase):
    def test_policy_allocator_is_explicit_and_does_not_change_judge_or_parent(self):
        launch = require_module(self, "launch")
        base = {'PATH': '/usr/bin', 'PYTORCH_ALLOC_CONF': 'expandable_segments:False'}
        policy = launch.role_environment('/scratch/train/bin/python', base,
                                        allocator_config='expandable_segments:True')
        self.assertEqual(policy['PYTORCH_CUDA_ALLOC_CONF'], 'expandable_segments:True')
        self.assertNotIn('PYTORCH_ALLOC_CONF', policy)
        self.assertEqual(base['PYTORCH_ALLOC_CONF'], 'expandable_segments:False')
        judge = launch.role_environment('/scratch/judge/bin/python', base)
        self.assertEqual(judge['PYTORCH_ALLOC_CONF'], 'expandable_segments:False')
        self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF', judge)
        with self.assertRaises(ValueError):
            launch.role_environment('/scratch/train/bin/python', base, allocator_config='backend:cudaMallocAsync')

    def test_synthetic_gpu_load_is_not_an_execution_option(self):
        launch = require_module(self, "launch")
        c = run_config()
        c["early_keeper_until_seconds"] = 7200
        with self.assertRaisesRegex(ValueError, '空转'):
            launch.validate_config(c)

    def test_external_baseline_must_be_complete_and_same_frozen_judge(self):
        launch = require_module(self, "launch")
        judge = {"status": "ready", "frozen": True, "model_id": "/scratch/Qwen3.8-27B",
                 "judge_mode": "baseline", "instance_id": "new"}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "validation_baseline.json"
            baseline = {"status": "completed", "adapter": None,
                        "input_bindings": [{"evidence_id": "E"}],
                        "rows": [{"evidence_id": "E", "slot": i} for i in range(4)],
                        "judge": {**judge, "instance_id": "old"}}
            p.write_text(json.dumps(baseline))
            self.assertEqual(launch.checked_baseline_source(p, judge, 4), p)
            baseline["status"] = "running"
            p.write_text(json.dumps(baseline))
            with self.assertRaises(ValueError):
                launch.checked_baseline_source(p, judge, 4)

    def test_swift_resumes_optimizer_state_from_checkpoint(self):
        launch = require_module(self, "launch")
        c = run_config()
        c["resume_from_checkpoint"] = "/scratch/test/old/checkpoint-8"
        cmd = launch.swift_command(c, "/scratch/test/outputs/train_1000")
        self.assertEqual(cmd[cmd.index("--resume_from_checkpoint") + 1], c["resume_from_checkpoint"])

    def test_policy_master_port_avoids_shared_node_default(self):
        launch = require_module(self, "launch")
        class Socket:
            def __init__(self, port): self.port = port
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def bind(self, address): self.address = address
            def getsockname(self): return ("0.0.0.0", self.port)
        with patch.object(launch.socket, "socket", side_effect=[Socket(29500), Socket(48621)]) as create:
            self.assertEqual(launch.select_master_port(), 48621)
        self.assertEqual(create.call_count, 2)
    def test_role_environment_exposes_own_executables_and_preserves_ffmpeg(self):
        launch = require_module(self, "launch")
        import os
        original = {"PATH": "/scratch/ffmpeg/bin" + os.pathsep + "/usr/bin", "LD_LIBRARY_PATH": "/scratch/ffmpeg/lib"}
        env = launch.role_environment("/scratch/judge/bin/python", original)
        self.assertEqual(env["PATH"].split(os.pathsep)[0], "/scratch/judge/bin")
        self.assertIn("/scratch/ffmpeg/bin", env["PATH"].split(os.pathsep))
        self.assertEqual(env["LD_LIBRARY_PATH"], original["LD_LIBRARY_PATH"])
        self.assertNotEqual(env["PATH"], original["PATH"])
        self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")

    def test_cleanup_continues_when_one_process_exits_between_poll_and_kill(self):
        launch = require_module(self, "launch")
        self.assertTrue(hasattr(launch, "cleanup_processes"))
        calls = []
        class Process:
            def __init__(self, pid):
                self.pid = pid
            def poll(self):
                return None
            def wait(self, timeout=None):
                calls.append(("wait", self.pid))
        def kill(pid, sig):
            calls.append(("kill", pid))
            if pid == 2:
                raise ProcessLookupError("already exited")
        launch.cleanup_processes([Process(1), Process(2)], kill_group=kill)
        self.assertIn(("kill", 1), calls)
        self.assertIn(("wait", 1), calls)
    def test_policy_and_judge_gpus_must_be_disjoint(self):
        launch = require_module(self, "launch")
        config = run_config()
        launch.validate_config(config)
        config["judge_gpu_indices"] = [0]
        with self.assertRaisesRegex(ValueError, "GPU"):
            launch.validate_config(config)

    def test_swift_retains_metadata_and_does_not_truncate_long_inputs(self):
        launch = require_module(self, "launch")
        cmd = launch.swift_command(run_config(), "/scratch/test/outputs/train_123")
        def value(flag):
            return cmd[cmd.index(flag) + 1]
        self.assertEqual(value("--reward_funcs"), "egoqa_six_user_binary_v1")
        self.assertEqual(value("--advantage_estimator"), "grpo")
        self.assertEqual(value("--scale_rewards"), "group")
        self.assertEqual(value("--enable_thinking"), "false")
        self.assertEqual(value("--remove_unused_columns"), "false")
        self.assertEqual(value("--truncation_strategy"), "delete")
        self.assertEqual(value("--strict"), "true")
        self.assertEqual(value("--val_dataset"), "/scratch/test/val.jsonl")
        self.assertNotIn("--test_dataset", cmd)

    def test_submission_parses_cluster_suffix_and_records_resources(self):
        submit = require_module(self, "submit")
        self.assertEqual(submit.parse_job_id("12345;torch\n"), "12345")
        with self.assertRaises(ValueError):
            submit.parse_job_id("warning: missing account")
        c = run_config()
        c["slurm"] = {"account": "a", "partition": "p", "qos": "q", "gres": "gpu:h200:3", "cpus": 16, "mem": "200G"}
        c["walltime_basis"] = "人工指定测量任务时限"
        cmd = submit.sbatch_command(c, "/scratch/test/run.json", "/scratch/test/task")
        self.assertIn("--parsable", cmd)
        self.assertFalse(any("nodelist" in x for x in cmd))
        self.assertIn("--account=a", cmd)


if __name__ == "__main__":
    unittest.main()
