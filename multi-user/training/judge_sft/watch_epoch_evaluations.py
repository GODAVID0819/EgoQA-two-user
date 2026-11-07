"""checkpoint完整保存后安排独立评测，训练作业继续执行。"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def checkpoint_ready(checkpoint, step):
    names = ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors")
    if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in names):
        return False
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
    except json.JSONDecodeError:
        return False
    if state["global_step"] != step:
        raise ValueError("checkpoint step mismatch")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--train-job", type=int, required=True)
    parser.add_argument("--account", required=True)
    args = parser.parse_args()
    registry_path = args.project / f"epoch_evaluations_{args.train_job}.json"
    trainer = args.project / "jobs" / str(args.train_job) / "trainer"
    records = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    while True:
        for step in (40, 80, 120, 160):
            if str(step) in records:
                continue
            checkpoint = trainer / f"checkpoint-{step}"
            if not checkpoint_ready(checkpoint, step):
                continue
            exports = (f"ALL,JUDGE_PROJECT={args.project},JUDGE_REPO={args.repo},"
                       f"JUDGE_EVAL_ADAPTER={checkpoint},JUDGE_EVAL_LABEL=job{args.train_job}_epoch{step//40}")
            result = subprocess.check_output([
                "sbatch", "--parsable", f"--account={args.account}",
                f"--output={args.project}/logs/eval_%j.out", f"--export={exports}",
                str(args.repo / "hpc/judge_sft/evaluate_aria12.sbatch"),
            ], text=True).strip()
            job = result.split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"invalid evaluation JobID: {result}")
            records[str(step)] = {"job_id": int(job), "epoch": step//40,
                                  "checkpoint": str(checkpoint), "submitted_at_epoch": time.time()}
            temporary = registry_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry_path)
            print(json.dumps(records[str(step)]), flush=True)
        if (trainer / "test_metrics.json").is_file():
            print("Final epoch evaluation is available from the training job.", flush=True)
            return
        state = subprocess.check_output([
            "sacct", "-X", "-n", "-P", "-j", str(args.train_job), "-o", "JobID,State"
        ], text=True)
        terminal = [line.split("|")[1] for line in state.splitlines()
                    if line.startswith(f"{args.train_job}|")]
        if terminal and terminal[0] not in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"):
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
