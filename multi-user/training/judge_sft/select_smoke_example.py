"""Select one deterministic real all-six example for cluster smoke training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import JudgeTask, Verdict
from .data import load_normalized_manifest


def select_example_id(path: Path) -> str:
    examples = load_normalized_manifest(path)
    candidates = sorted(
        (
            example
            for example in examples
            if example.task is JudgeTask.ANSWERABILITY
            and example.condition_type == "combined_all_six_users"
            and example.verdict is Verdict.PASS
            and len(example.frame_sets) == 6
            and example.frame_count == 1_800
        ),
        key=lambda example: example.example_id,
    )
    if not candidates:
        raise ValueError("manifest has no PASS all-six 1,800-frame answerability example")
    return candidates[0].example_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    example_id = select_example_id(args.manifest)
    result = {"status": "passed", "example_id": example_id}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(example_id)


if __name__ == "__main__":
    main()
