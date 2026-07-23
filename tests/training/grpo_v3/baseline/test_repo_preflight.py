from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.baseline.preflight import REPO_REWARD_SOURCE_PATHS, validate_repo_reward_sources
from training.grpo_v3.baseline.repo_reward import _repo_modules


class RepoRewardSourcePreflightTests(unittest.TestCase):
    def test_rejects_unresolved_merge_conflict_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "video_qa_loop.py"
            source.write_text("<<<<<<< Updated upstream\nvalue = 1\n=======\nvalue = 2\n>>>>>>> Stashed changes\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"video_qa_loop\.py:1.*Git 冲突标记"):
                validate_repo_reward_sources(root, source_paths=(source,))

    def test_rejects_python_syntax_error_before_model_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "schema.py"
            source.write_text("def broken(:\n", encoding="utf-8")

            with self.assertRaisesRegex(SyntaxError, "schema.py"):
                validate_repo_reward_sources(root, source_paths=(source,))

    def test_repo_reward_modules_load_from_source_root_package(self) -> None:
        modules = _repo_modules()

        self.assertIn("run_parallel_review_judges", modules)
        self.assertIn("OpenAICompatibleLocalRunner", modules)
        self.assertIn("compute_judge_reward", modules)

    def test_repo_reward_preflight_includes_three_tier_format_sources(self) -> None:
        names = {path.name for path in REPO_REWARD_SOURCE_PATHS}

        self.assertIn("json_format.py", names)
        self.assertIn("repo_reward.py", names)
        self.assertIn("reward_plugin.py", names)


if __name__ == "__main__":
    unittest.main()
