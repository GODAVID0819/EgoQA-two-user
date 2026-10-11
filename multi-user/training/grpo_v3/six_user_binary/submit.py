"""提交单个新作业并立即记录 JobID；不取消或覆盖历史任务。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import uuid

from .launch import validate_config


def parse_job_id(text):
    if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", text.strip()):
        raise ValueError(f"sbatch 没有返回有效 JobID：{text!r}")
    return text.strip().split(";", 1)[0]


def sbatch_command(c, config_path, task_dir):
    validate_config(c)
    resources = c["slurm"]
    for key in ("account", "gres", "cpus", "mem"):
        if not resources.get(key):
            raise ValueError(f"必须提供已核验的 Slurm {key}")
    if not c.get("walltime_basis"):
        raise ValueError("必须记录 walltime 估时依据")
    seconds = c["walltime_seconds"]
    if any("," in str(c[k]) for k in ("project_root", "train_python")) or "," in config_path:
        raise ValueError("Slurm export 路径不得包含逗号")
    command = ["sbatch", "--parsable", "--job-name=egoqa-six-user-grpo", "--nodes=1", "--ntasks=1",
        f"--account={resources['account']}",
        f"--gres={resources['gres']}", f"--cpus-per-task={resources['cpus']}", f"--mem={resources['mem']}",
        f"--time={seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}",
        f"--output={task_dir}/slurm-%j.out", f"--error={task_dir}/slurm-%j.err",
        f"--chdir={c['project_root']}",
        f"--export=ALL,PROJECT_ROOT={c['project_root']},TRAIN_PYTHON={c['train_python']},RUN_CONFIG={config_path}",
        str(PurePosixPath(c["project_root"]) / "multi-user/hpc/grpo_v3/six_user_binary" /
            ("train_direct.sbatch" if c.get("execution_mode") == "direct" else "train.sbatch"))]
    # 当前Torch普通作业由站点自动路由分区；旧明确配置仍兼容历史路径。
    for key in ('partition', 'qos', 'constraint'):
        if resources.get(key):
            command.insert(-1, '--' + key + '=' + resources[key])
    return command


def main():
    parser = argparse.ArgumentParser(description="使用已核验配置提交六用户 GRPO")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    task = Path(c["output_root"]) / "submissions" / stamp
    # 先完成配置校验，再创建提交记录。
    command = sbatch_command(c, str(task / "run_config.json"), str(task))
    task.mkdir(parents=True, exist_ok=False)
    (task / "run_config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    result = subprocess.run(command, text=True, capture_output=True)
    (task / "sbatch.stdout").write_text(result.stdout, encoding="utf-8")
    (task / "sbatch.stderr").write_text(result.stderr, encoding="utf-8")
    result.check_returncode()
    job_id = parse_job_id(result.stdout)
    manifest = {"job_id": job_id, "submitted_at_utc": stamp, "config": c,
                "output_dir": str(Path(c["output_root"]) / ("train_" + job_id)), "command": command}
    (task / "submission.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"job_id": job_id, "manifest": str(task / "submission.json")}))


if __name__ == "__main__":
    main()
