"""等待已提交的 allocation 并自动接入；不申请第二份 GPU，不取消作业。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time


def step_command(config, job_id, task):
    root = config["project_root"]
    return ["srun", f"--jobid={job_id}", "--overlap", "--exact", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=16", "--gres=gpu:2", "--mem=500G", f"--chdir={root}/multi-user",
            f"--export=ALL,PYTHONPATH={root}/multi-user,PYTHONDONTWRITEBYTECODE=1",
            f"--output={task}/step-%J.out", f"--error={task}/step-%J.err",
            config["train_python"], "-m", "training.grpo_v3.six_user_binary." + ("resume" if config.get("prepared_data_root") else "workflow"),
            "--config", str(Path(task) / "workflow.json")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.job_id.isdigit():
        raise ValueError("JobID 必须为数字")
    c = json.loads(args.config.read_text(encoding="utf-8"))
    args.task_dir.mkdir(parents=True, exist_ok=True)
    claim = args.task_dir / "dispatcher.claim"
    with claim.open("x") as stream:
        stream.write(str(__import__("os").getpid()))
    (args.task_dir / "workflow.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    command = step_command(c, args.job_id, str(args.task_dir))
    record = {"job_id": args.job_id, "status": "waiting_for_allocation", "models": {"policy": c["model"], "judge": c["model"]},
              "command": command, "submitted_epoch": time.time()}
    def save():
        (args.task_dir / "attachment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    save()
    try:
        while True:
            result = subprocess.run(["squeue", "-j", args.job_id, "-h", "-o", "%T"], text=True, capture_output=True)
            result.check_returncode()
            state = result.stdout.strip().splitlines()
            if state and state[0] == "RUNNING":
                allocation = Path(c["project_root"]) / "outputs" / ("allocation_" + args.job_id) / "allocation.json"
                if allocation.is_file():
                    break
            elif not state or state[0] not in {"PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "REQUEUE_HOLD"}:
                raise RuntimeError(f"allocation 不再可接入：{state}")
            time.sleep(30)
        record.update(status="step_submitted", attached_epoch=time.time())
        save()
        result = subprocess.run(command)
        record.update(status="completed" if result.returncode == 0 else "failed", step_exit_code=result.returncode, finished_epoch=time.time())
        save()
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
