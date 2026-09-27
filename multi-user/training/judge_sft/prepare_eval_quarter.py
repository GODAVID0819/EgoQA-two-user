#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


GROUP_ORDER = (
    "groundedness",
    "answerability_asker",
    "answerability_all",
    "formality",
)

TASK_TOKENS = {
    "groundedness",
    "answerability",
    "formality",
}


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            if "example_id" not in row:
                raise RuntimeError(
                    f"{path}:{line_no}: missing example_id"
                )

            rows.append(row)

    return rows


def task_text(row):
    for key in ("task", "judge_task", "dimension"):
        if key in row:
            value = row[key]

            if isinstance(value, dict) and "value" in value:
                value = value["value"]

            return str(value).lower()

    parts = str(row["example_id"]).lower().split("::")

    for part in parts:
        if part in TASK_TOKENS:
            return part

    raise RuntimeError(
        f"Cannot determine task: {row['example_id']}"
    )


def verdict_text(row):
    for key in ("verdict", "label", "gold_verdict"):
        if key not in row:
            continue

        value = row[key]

        if isinstance(value, dict) and "value" in value:
            value = value["value"]

        value = str(value).lower()

        if value in ("pass", "fail"):
            return value

        if value in ("1", "true"):
            return "pass"

        if value in ("0", "false"):
            return "fail"

    raise RuntimeError(
        f"Cannot determine verdict: {row['example_id']}"
    )


def invocation_group(row):
    eid = str(row["example_id"]).lower()
    task = task_text(row)

    if task == "groundedness":
        return "groundedness"

    if task == "formality":
        return "formality"

    if task == "answerability":
        if "::answerability::all-six" in eid:
            return "answerability_all"

        if "::answerability::speaker-only" in eid:
            return "answerability_asker"

        if "::answerability::asker-only" in eid:
            return "answerability_asker"

        raise RuntimeError(
            f"Unknown answerability scope: {row['example_id']}"
        )

    raise RuntimeError(
        f"Unknown task={task}: {row['example_id']}"
    )


def attempt_key(row):
    """
    Strip the judge task suffix.

    Example:

    X::attempt_03::answerability::all-six

    becomes:

    X::attempt_03
    """

    parts = str(row["example_id"]).split("::")
    lower = [x.lower() for x in parts]

    positions = [
        i
        for i, token in enumerate(lower)
        if token in TASK_TOKENS
    ]

    if len(positions) != 1:
        raise RuntimeError(
            f"Cannot uniquely identify task suffix: "
            f"{row['example_id']}"
        )

    i = positions[0]

    return "::".join(parts[:i])


def make_bundles(rows, name):
    bundles = defaultdict(list)

    for row in rows:
        bundles[attempt_key(row)].append(row)

    expected_groups = set(GROUP_ORDER)

    for key, bundle in bundles.items():
        groups = [
            invocation_group(row)
            for row in bundle
        ]

        if (
            len(bundle) != 4
            or set(groups) != expected_groups
            or len(set(groups)) != 4
        ):
            raise RuntimeError(
                f"{name}: incomplete QA bundle: "
                f"{key} rows={len(bundle)} groups={groups}"
            )

    return dict(bundles)


def summary(rows):
    by_group = Counter(
        invocation_group(row)
        for row in rows
    )

    by_group_label = Counter(
        (
            invocation_group(row),
            verdict_text(row),
        )
        for row in rows
    )

    return {
        "rows": len(rows),

        "by_group": dict(
            sorted(by_group.items())
        ),

        "by_group_label": {
            f"{group}::{label}": count
            for (group, label), count
            in sorted(by_group_label.items())
        },
    }


def bundle_signature(bundle):
    by_group = {
        invocation_group(row): verdict_text(row)
        for row in bundle
    }

    return tuple(
        by_group[group]
        for group in GROUP_ORDER
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--audit-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2718,
    )

    args = parser.parse_args()

    full_rows = read_jsonl(
        args.full_manifest
    )

    train_rows = read_jsonl(
        args.train_manifest
    )

    full_bundles = make_bundles(
        full_rows,
        "full",
    )

    train_bundles = make_bundles(
        train_rows,
        "train",
    )

    full_keys = set(
        full_bundles
    )

    train_keys = set(
        train_bundles
    )

    if not train_keys <= full_keys:
        raise RuntimeError(
            "Training contains QA attempts "
            "not present in full manifest"
        )

    # Verify training uses entire 4-row QA bundles.
    for key in train_keys:
        full_ids = {
            row["example_id"]
            for row in full_bundles[key]
        }

        train_ids = {
            row["example_id"]
            for row in train_bundles[key]
        }

        if full_ids != train_ids:
            raise RuntimeError(
                f"Partial train bundle: {key}"
            )

    unused_keys = sorted(
        full_keys - train_keys
    )

    print(
        "full_attempts=",
        len(full_keys),
    )

    print(
        "train_attempts=",
        len(train_keys),
    )

    print(
        "unused_attempts=",
        len(unused_keys),
    )

    # 110 unused attempts -> 27 attempts.
    #
    # 27 * 4 = 108 rows
    # 108 / 884 ~= 12.2%, approximately 1/8.
    target_attempts = (
        len(unused_keys) // 4
    )

    # Preserve joint PASS/FAIL pattern across
    # all four judge invocations as much as possible.
    strata = defaultdict(list)

    for key in unused_keys:
        signature = bundle_signature(
            full_bundles[key]
        )

        strata[signature].append(
            key
        )

    rng = random.Random(
        args.seed
    )

    selected_keys = []

    # Shuffle within each signature.
    for keys in strata.values():
        rng.shuffle(keys)

    # Proportional allocation.
    raw_allocations = []

    for signature, keys in strata.items():
        raw = (
            len(keys)
            * target_attempts
            / len(unused_keys)
        )

        base = int(raw)

        raw_allocations.append(
            (
                raw - base,
                signature,
                base,
            )
        )

    allocations = {
        signature: base
        for _, signature, base
        in raw_allocations
    }

    allocated = sum(
        allocations.values()
    )

    # Give remaining slots to strata
    # with largest fractional remainder.
    remaining = (
        target_attempts
        - allocated
    )

    raw_allocations.sort(
        reverse=True
    )

    for _, signature, _ in raw_allocations:
        if remaining <= 0:
            break

        if (
            allocations[signature]
            < len(strata[signature])
        ):
            allocations[signature] += 1
            remaining -= 1

    for signature in sorted(strata):
        k = allocations[signature]

        selected_keys.extend(
            strata[signature][:k]
        )

    if len(selected_keys) != target_attempts:
        raise RuntimeError(
            f"Expected {target_attempts} attempts, "
            f"got {len(selected_keys)}"
        )

    selected_set = set(
        selected_keys
    )

    overlap = (
        train_keys
        & selected_set
    )

    if overlap:
        raise RuntimeError(
            f"Train/eval attempt overlap: "
            f"{len(overlap)}"
        )

    # Emit exactly four rows per QA.
    selected_rows = []

    for key in sorted(selected_keys):
        bundle = full_bundles[key]

        by_group = {
            invocation_group(row): row
            for row in bundle
        }

        for group in GROUP_ORDER:
            selected_rows.append(
                by_group[group]
            )

    expected_rows = (
        target_attempts * 4
    )

    if len(selected_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"got {len(selected_rows)}"
        )

    args.output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        args.output_manifest,
        "w",
        encoding="utf-8",
    ) as f:
        for row in selected_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    unused_rows = []

    for key in unused_keys:
        unused_rows.extend(
            full_bundles[key]
        )

    audit = {
        "seed": args.seed,

        "design": (
            "atomic QA-attempt evaluation; "
            "select 25% of unused attempts "
            "(approximately 1/8 of full data)"
        ),

        "full_attempts": len(
            full_keys
        ),

        "train_attempts": len(
            train_keys
        ),

        "unused_attempts": len(
            unused_keys
        ),

        "eval_attempts": len(
            selected_keys
        ),

        "eval_rows_per_attempt": 4,

        "full": summary(
            full_rows
        ),

        "train": summary(
            train_rows
        ),

        "unused": summary(
            unused_rows
        ),

        "eval": summary(
            selected_rows
        ),

        "train_eval_attempt_overlap": len(
            overlap
        ),

        "unused_attempts_not_evaluated": (
            len(unused_keys)
            - len(selected_keys)
        ),
    }

    args.audit_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.audit_output.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
