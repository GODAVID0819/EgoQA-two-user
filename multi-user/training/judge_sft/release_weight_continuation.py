"""首轮完整checkpoint就绪后释放四卡正式接续任务。"""
import argparse
import json
import subprocess
import time
from pathlib import Path
from .watch_epoch_evaluations import checkpoint_ready
from .train import RESUME_CHECKPOINT_FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-job", type=int, required=True)
    parser.add_argument("--target-job", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    while True:
        if checkpoint_ready(args.checkpoint, 40) and all(
                (args.checkpoint / f).is_file() for f in RESUME_CHECKPOINT_FILES):
            subprocess.run(["scontrol", "release", str(args.target_job)], check=True)
            args.receipt.write_text(json.dumps({
                "source_job": args.source_job, "target_job": args.target_job,
                "checkpoint": str(args.checkpoint), "released_at_epoch": time.time(),
                "mode": "weights_only_new_optimizer", "completed_epochs": 1, "remaining_epochs": 4,
            }, indent=2))
            return
        status = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", str(args.source_job),
                                          "-o", "JobID,State"], text=True)
        states = [s.split("|")[1] for s in status.splitlines() if s.startswith(f"{args.source_job}|")]
        if states and states[0] not in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED"):
            raise RuntimeError("source job ended before a complete first-epoch checkpoint became available")
        time.sleep(60)


if __name__ == "__main__":
    main()
