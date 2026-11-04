"""不加载模型的 v3 数据与正式配置预检。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from training.grpo_v3.shared.contract import DEFAULTS, validate_formal_config
from training.grpo_v3.shared.data import read_jsonl, validate_swift_row


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_REWARD_SOURCE_PATHS = (
    PROJECT_ROOT / "video_qa_loop.py",
    PROJECT_ROOT / "prompts.py",
    PROJECT_ROOT / "schema.py",
    PROJECT_ROOT / "qwen3vl_runner.py",
    PROJECT_ROOT / "training" / "grpo_v3" / "shared" / "json_format.py",
    PROJECT_ROOT / "training" / "grpo_v3" / "baseline" / "judge_reward.py",
    PROJECT_ROOT / "training" / "grpo_v3" / "baseline" / "repo_reward.py",
    PROJECT_ROOT / "training" / "grpo_v3" / "runtime" / "reward_plugin.py",
)
CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def _python_sources(source_paths: Iterable[Path]) -> list[Path]:
    sources: list[Path] = []
    for path in source_paths:
        if path.is_dir():
            sources.extend(sorted(path.rglob("*.py")))
        else:
            sources.append(path)
    return sources


def validate_repo_reward_sources(
    project_root: Path = PROJECT_ROOT,
    *,
    source_paths: Iterable[Path] | None = None,
    import_modules: bool = True,
) -> dict[str, object]:
    paths = _python_sources(source_paths or REPO_REWARD_SOURCE_PATHS)
    if not paths:
        raise ValueError("repo-native reward 预检未找到 Python 源码")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Gate 2 缺少 repo-native reward 源码: {path}")
        source = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line.startswith(CONFLICT_MARKERS):
                relative = path.relative_to(project_root) if path.is_relative_to(project_root) else path
                raise ValueError(f"{relative}:{line_number} 存在未解决的 Git 冲突标记: {line}")
        compile(source, str(path), "exec")
    if import_modules:
        from training.grpo_v3.baseline.repo_reward import _repo_modules

        modules = _repo_modules()
        required = {
            "run_parallel_review_judges",
            "OpenAICompatibleLocalRunner",
            "compute_judge_reward",
        }
        missing = sorted(required.difference(modules))
        if missing:
            raise RuntimeError(f"repo-native reward 模块边界不完整: {missing}")
    return {"status": "passed", "python_sources": len(paths), "module_import": import_modules}


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO v3 CPU schema preflight")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-repo-reward",
        action="store_true",
        help="Gate 2：检查冲突标记、语法和 repo-native reward 导入边界",
    )
    args = parser.parse_args()
    validate_formal_config(DEFAULTS.to_dict())
    rows = read_jsonl(args.dataset)
    if not rows:
        raise ValueError("dataset 为空")
    for row in rows:
        validate_swift_row(row)
    result = {
        "status": "passed",
        "rows": len(rows),
        "framework": DEFAULTS.framework,
        "framework_version": DEFAULTS.framework_version,
        "policy_input": DEFAULTS.policy_input,
        "train_type": DEFAULTS.train_type,
    }
    if args.check_repo_reward:
        result["repo_reward_preflight"] = validate_repo_reward_sources()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
