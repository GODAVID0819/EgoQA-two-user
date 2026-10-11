"""27B 配置、窗口去重和资源接入不能退回旧 8B 流程。"""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from test_pipeline import require_module


class WorkflowTests(unittest.TestCase):
    def test_cold_judge_startup_timeout_reaches_runtime_configuration(self):
        workflow = require_module(self, 'workflow')
        c = {'project_root': '/scratch/a/project', 'model': '/scratch/a/models/Qwen3.8-27B',
             'train_python': '/scratch/a/train/bin/python', 'judge_python': '/scratch/a/judge/bin/python',
             'judge_startup_timeout_seconds': 1800}
        run = workflow.training_config(c, job_id='1002', phase='formal', max_steps=200)
        self.assertEqual(run.get('judge_startup_timeout_seconds'), 1800)

    def test_resume_selects_only_complete_formal_checkpoint(self):
        resume = require_module(self, "resume")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "formal_a/swift/v0/checkpoint-8"
            incomplete = root / "formal_b/swift/v0/checkpoint-10"
            for p, step in ((good, 8), (incomplete, 10)):
                p.mkdir(parents=True)
                (p / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 60}))
                for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                             "scheduler.pt", "rng_state.pth", "args.json", "training_args.bin"):
                    (p / name).write_bytes(b"valid")
            (incomplete / "optimizer.pt").unlink()
            self.assertEqual(resume.latest_complete_checkpoint([root / "formal_a", root / "formal_b"], 60), good)

    def test_resumed_job_runs_formal_only_and_retains_sixty_step_target(self):
        resume = require_module(self, "resume")
        self.assertEqual(resume.phase_sequence({"skip_smoke": True, "formal_max_steps": 60}), [("formal", 60)])
        self.assertEqual(resume.phase_sequence({"formal_max_steps": 60}), [("smoke", 1), ("formal", 60)])

    def test_formal_config_carries_resume_checkpoint_and_fixed_baseline(self):
        workflow = require_module(self, "workflow")
        c = {"project_root": "/scratch/a/project", "model": "/scratch/a/models/Qwen3.8-27B",
             "train_python": "/scratch/a/train/bin/python", "judge_python": "/scratch/a/judge/bin/python",
             "resume_from_checkpoint": "/scratch/a/old/checkpoint-8",
             "baseline_validation_source": "/scratch/a/old/validation_baseline.json",
             "paired_validation": True, "save_steps": 2, "eval_steps": 20}
        run = workflow.training_config(c, job_id="1000", phase="formal", max_steps=60)
        self.assertEqual(run["resume_from_checkpoint"], c["resume_from_checkpoint"])
        self.assertEqual(run["baseline_validation_source"], c["baseline_validation_source"])
        self.assertEqual(run["save_steps"], 2)
        self.assertEqual(run["eval_steps"], 20)
        smoke = workflow.training_config(c, job_id="1000", phase="smoke", max_steps=1)
        self.assertNotIn("resume_from_checkpoint", smoke)
        self.assertNotIn("baseline_validation_source", smoke)

    def test_policy_allocator_option_survives_workflow_without_changing_training(self):
        workflow = require_module(self, "workflow")
        c = {"project_root": "/scratch/a/project", "model": "/scratch/a/models/Qwen3.8-27B",
             "train_python": "/scratch/a/train/bin/python", "judge_python": "/scratch/a/judge/bin/python",
             "policy_allocator_config": "expandable_segments:True"}
        run = workflow.training_config(c, job_id="1000", phase="formal", max_steps=200)
        self.assertEqual(run['policy_allocator_config'], 'expandable_segments:True')
        self.assertEqual(run['num_generations'], 4)
        self.assertEqual(run['per_device_train_batch_size'], 1)
        self.assertEqual(run['gradient_accumulation_steps'], 4)

    def test_training_config_uses_actual_allocation_walltime(self):
        workflow = require_module(self, "workflow")
        c = {"project_root": "/scratch/a/project", "model": "/scratch/a/models/Qwen3.8-27B",
             "train_python": "/scratch/a/train/bin/python", "judge_python": "/scratch/a/judge/bin/python",
             "walltime_seconds": 46800}
        run = workflow.training_config(c, job_id="1001", phase="formal", max_steps=60)
        self.assertEqual(run["walltime_seconds"], 46800)

    def test_formal_config_does_not_enable_synthetic_load(self):
        workflow = require_module(self, "workflow")
        c = {"project_root": "/scratch/a/project", "model": "/scratch/a/models/Qwen3.8-27B",
             "train_python": "/scratch/a/train/bin/python", "judge_python": "/scratch/a/judge/bin/python",
             "early_keeper_until_seconds": 7200}
        run = workflow.training_config(c, job_id="1001", phase="formal", max_steps=60)
        self.assertNotIn("early_keeper_until_seconds", run)

    def test_retry_uses_completed_data_but_new_job_outputs_and_keeper(self):
        workflow = require_module(self, "workflow")
        attach = require_module(self, "attach")
        c = {"project_root": "/scratch/a/project", "model": "/scratch/a/models/Qwen3.8-27B",
             "train_python": "/scratch/a/env/bin/python", "judge_python": "/scratch/a/judge/bin/python",
             "prepared_data_root": "/scratch/a/project/data/grpo_18719659"}
        run = workflow.training_config(c, job_id="999", phase="smoke", max_steps=1)
        self.assertEqual(run["train_dataset"], "/scratch/a/project/data/grpo_18719659/smoke_train.jsonl")
        self.assertIn("allocation_999", run["allocation_manifest"])
        command = attach.step_command(c, "999", "/scratch/a/task")
        self.assertIn("training.grpo_v3.six_user_binary.resume", command)
        self.assertIn("--jobid=999", command)
        c["run_attempt"] = "portfix1"
        attempt = workflow.training_config(c, job_id="999", phase="smoke", max_steps=1)
        self.assertEqual(attempt["output_root"], "/scratch/a/project/outputs/smoke_portfix1")
        self.assertIn("allocation_999", attempt["allocation_manifest"])
        c["policy_cuda_home"] = "/scratch/a/project/preflight/nvcc130/packages/nvidia/cu13"
        attempt = workflow.training_config(c, job_id="999", phase="smoke", max_steps=1)
        self.assertEqual(attempt["policy_cuda_home"], c["policy_cuda_home"])
    def test_exited_keeper_returns_recovery_status_instead_of_aborting_training(self):
        launch = require_module(self, "launch")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "allocation.json"
            path.write_text(json.dumps({"job_id": "123", "hostname": "node", "keeper_pid": 10,
                "keeper_script": "/project/hpc/shared/cuda.py", "start_after_seconds": 7200,
                "job_start_epoch": 100.}))
            with patch.object(launch.socket, "gethostname", return_value="node"), patch.object(launch.os, "kill", side_effect=ProcessLookupError):
                result = launch.allocation_keeper({"allocation_manifest": str(path)}, "123")
            self.assertEqual(result["status"], "not_running")
            self.assertEqual(result["job_start_epoch"], 100.)
    def test_duplicate_speaker_records_are_one_source_window(self):
        workflow = require_module(self, "workflow")
        clips = [{"agent_dir": f"A{i}", "full_local_video": f"/data/{i}.mp4"} for i in range(6)]
        rows = [{"day": "DAY1", "time_token": "12000000", "clips": clips},
                {"day": "DAY1", "time_token": "12000000", "clips": list(reversed(clips))}]
        unique = workflow.unique_windows(rows)
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["duration_seconds"], 600.)

    def test_both_models_are_27b_and_training_uses_group_accumulation(self):
        workflow = require_module(self, "workflow")
        config = workflow.training_config({"project_root": "/scratch/a/project", "model": "/scratch/a/models/Qwen3.8-27B",
            "train_python": "/scratch/a/train/bin/python", "judge_python": "/scratch/a/judge/bin/python"},
            job_id="123", phase="formal", max_steps=12)
        self.assertEqual(config["policy_model"], "/scratch/a/models/Qwen3.8-27B")
        self.assertEqual(config["num_generations"], 4)
        self.assertEqual(config["per_device_train_batch_size"], 1)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertEqual(config["allocation_manifest"], "/scratch/a/project/outputs/allocation_123/allocation.json")

    def test_shared_keeper_must_belong_to_this_allocation(self):
        launch = require_module(self, "launch")
        self.assertTrue(hasattr(launch, "validate_allocation_keeper"))
        row = {"job_id": "123", "hostname": "node", "keeper_pid": 10,
               "keeper_script": "/project/hpc/shared/cuda.py", "start_after_seconds": 7200}
        launch.validate_allocation_keeper(row, job_id="123", hostname="node", command=["python", row["keeper_script"]])
        with self.assertRaises(ValueError):
            launch.validate_allocation_keeper(row, job_id="456", hostname="node", command=["python", row["keeper_script"]])


if __name__ == "__main__":
    unittest.main()
