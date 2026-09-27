#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from training.judge_sft.prepare_real_data import (
    load_labels,
    load_generation,
    _repo_module,
)


def read_packet_participants(dataset_root: Path, packet_id: str, asker_index: int):
    p = dataset_root / "packets" / packet_id / "packet.json"
    obj = json.loads(p.read_text())

    users = list(obj.get("users") or [])
    if len(users) != 6:
        raise ValueError(f"{packet_id}: expected 6 users, got {len(users)}")

    if not 0 <= asker_index < 6:
        raise ValueError(f"{packet_id}: bad asker_index={asker_index}")

    order = [asker_index] + [i for i in range(6) if i != asker_index]

    names = []
    for i in order:
        name = str(users[i].get("agent_name") or "").strip()
        if not name:
            raise ValueError(f"{packet_id}: missing agent_name for user {i}")
        names.append(name)

    if len(set(names)) != 6:
        raise ValueError(f"{packet_id}: participant names not unique: {names}")

    # This contains exactly what the Formality prompt needs.
    return {
        "required_users": names,
        "participant_names": names,
        "clips": [{"agent_name": x} for x in names],
        "packet_id": packet_id,
        "asker_user": names[0],
    }


def normalize_array_field(value, *, field, candidate_id):
    """Normalize legacy array fields stored as list/tuple/JSON/Python literals."""
    import ast

    if isinstance(value, list):
        return list(value)

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        text = value.strip()

        # First try proper JSON.
        try:
            decoded = json.loads(text)
        except Exception:
            decoded = None

        # Some legacy trajectories stored Python repr strings:
        # "['a', 'b', 'c']"
        if not isinstance(decoded, list):
            try:
                decoded = ast.literal_eval(text)
            except Exception as e:
                raise ValueError(
                    f"{candidate_id}: {field} string cannot be decoded "
                    f"as JSON or Python literal: {text[:200]!r}"
                ) from e

        if not isinstance(decoded, (list, tuple)):
            raise ValueError(
                f"{candidate_id}: decoded {field} is "
                f"{type(decoded).__name__}, expected list"
            )

        return list(decoded)

    raise ValueError(
        f"{candidate_id}: unsupported {field} type "
        f"{type(value).__name__}"
    )


def canonical_trajectory_qa(attempt):
    """
    Return the generation QA in canonical form.

    Some legacy trajectory rows serialized array-valued fields such as
    options as JSON strings. Normalize them before validation AND before
    constructing the model-visible Formality prompt.
    """
    qa = dict(attempt.qa)

    qa["options"] = normalize_array_field(
        qa.get("options"),
        field="options",
        candidate_id=attempt.candidate_id,
    )

    qa["required_users"] = normalize_array_field(
        qa.get("required_users"),
        field="required_users",
        candidate_id=attempt.candidate_id,
    )

    return qa


def strict_text_join(label, attempt, qa):
    # These are exactly the user-facing fields Formality judges.
    checks = {
        "question": str(label.get("question") or ""),
        "options": list(label["options"]),
        "correct": str(label.get("correct") or ""),
        "answer": str(label.get("answer") or ""),
    }

    for field, expected in checks.items():
        actual = qa.get(field)
        if actual != expected:
            raise ValueError(
                f"{attempt.candidate_id}: label/trajectory disagree on {field}: "
                f"label={expected!r} trajectory={actual!r}"
            )

    source_qa_id = str(label.get("source_qa_id") or "").strip()
    if qa.get("qa_id") and str(qa["qa_id"]) != source_qa_id:
        raise ValueError(
            f"{attempt.candidate_id}: source_qa_id mismatch"
        )

    if str(label.get("source_evidence_id") or "") != attempt.evidence_id:
        raise ValueError(
            f"{attempt.candidate_id}: source_evidence_id mismatch"
        )

    if str(label.get("evidence_id") or "") != attempt.packet_id:
        raise ValueError(
            f"{attempt.candidate_id}: packet/evidence_id mismatch"
        )

    if int(label.get("generation_attempt") or 0) != attempt.attempt:
        raise ValueError(
            f"{attempt.candidate_id}: generation_attempt mismatch"
        )

    if str(label.get("asker_user") or "") != attempt.asker_user:
        raise ValueError(
            f"{attempt.candidate_id}: asker_user mismatch"
        )


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )


def row_stats(rows):
    rows = list(rows)
    c = Counter(x["verdict"] for x in rows)
    return {
        "examples": len(rows),
        "pass": c["pass"],
        "fail": c["fail"],
        "pass_rate": c["pass"] / len(rows) if rows else None,
        "packets": len({x["group_id"] for x in rows}),
        "reviewers": dict(Counter(
            x["provenance"]["reviewer_id"] for x in rows
        )),
        "assignments": dict(Counter(
            x["provenance"]["assignment_id"] for x in rows
        )),
    }


def choose_packet_subset(packet_rows, fraction, seed):
    packets = sorted(packet_rows)
    k = max(1, round(len(packets) * fraction))

    all_rows = [
        r for p in packets
        for r in packet_rows[p]
    ]

    target_n = len(all_rows) * fraction
    target_pass = (
        sum(r["verdict"] == "pass" for r in all_rows)
        / len(all_rows)
    )

    def evaluate(chosen):
        chosen = tuple(sorted(chosen))
        rows = [
            r for p in chosen
            for r in packet_rows[p]
        ]

        n = len(rows)
        pass_rate = (
            sum(r["verdict"] == "pass" for r in rows) / n
        )

        size_err = abs(n - target_n) / max(target_n, 1)
        pass_err = abs(pass_rate - target_pass)

        score = size_err + pass_err
        return score, chosen, n, pass_rate

    best = None

    # Our normal case is 20 choose 4 = only 4,845 possibilities,
    # so search the exact optimum instead of random guessing.
    total_combinations = 1
    try:
        import math
        total_combinations = math.comb(len(packets), k)
    except Exception:
        pass

    if total_combinations <= 100_000:
        candidates = itertools.combinations(packets, k)
    else:
        rng = random.Random(seed)
        candidates = (
            rng.sample(packets, k)
            for _ in range(100_000)
        )

    for chosen in candidates:
        item = evaluate(chosen)
        if best is None or item < best:
            best = item

    assert best is not None

    return set(best[1]), {
        "available_packets": len(packets),
        "selected_packet_count": k,
        "selected_packets": list(best[1]),
        "heldout_examples": best[2],
        "heldout_pass_rate": best[3],
        "target_examples": target_n,
        "target_pass_rate": target_pass,
        "optimization_score": best[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", type=Path, required=True)
    ap.add_argument("--generation-root", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expected-candidates", type=int, default=446)
    ap.add_argument("--expected-packets", type=int, default=80)
    ap.add_argument("--heldout-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    prompts = _repo_module("prompts")

    labels, label_sha = load_labels(args.labels)
    attempts, generation_packets, generation_info = load_generation(
        args.generation_root
    )

    if len(labels) != args.expected_candidates:
        raise ValueError(
            f"expected {args.expected_candidates} labels, got {len(labels)}"
        )

    label_packets = {
        str(x.get("evidence_id") or "")
        for x in labels
    }

    if len(label_packets) != args.expected_packets:
        raise ValueError(
            f"expected {args.expected_packets} packets, "
            f"got {len(label_packets)}"
        )

    records = []
    exclusions = []
    incomplete_label_required_users = []
    incomplete_trajectory_required_users = []

    # Two legacy generation-trajectory rows have corrupted/missing options.
    # Exclude them entirely from Formality SFT rather than attempting recovery.
    dropped_candidates = {
        "RLHF6U_DAY2_16200000_A1_A2_A3_A4_A5_A6__ASKER_A1::attempt_03",
        "RLHF6U_DAY7_14200000_A1_A2_A3_A4_A5_A6__ASKER_A5::attempt_02",
    }

    view_cache = {}

    for label in sorted(labels, key=lambda x: x["candidate_id"]):
        cid = str(label["candidate_id"])

        if cid in dropped_candidates:
            exclusions.append({
                "candidate_id": cid,
                "reviewer_id": label.get("reviewer_id"),
                "assignment_id": label.get("assignment_id"),
                "human_verdict": label.get("formality_verdict"),
                "reason": "excluded: corrupted legacy generation trajectory options",
                "deterministic_errors": [],
            })
            continue

        attempt = attempts.get(cid)

        if attempt is None:
            raise ValueError(
                f"{cid}: no matching generation attempt"
            )

        qa = canonical_trajectory_qa(attempt)
        strict_text_join(label, attempt, qa)

        if attempt.packet_id not in generation_packets:
            raise ValueError(
                f"{cid}: packet absent from generation outcomes"
            )

        key = (attempt.packet_id, attempt.asker_index)

        if key not in view_cache:
            view_cache[key] = read_packet_participants(
                args.dataset_root,
                attempt.packet_id,
                attempt.asker_index,
            )

        view = view_cache[key]

        if view["asker_user"] != attempt.asker_user:
            raise ValueError(
                f"{cid}: packet asker mismatch: "
                f"view={view['asker_user']!r} "
                f"trajectory={attempt.asker_user!r}"
            )

        # The trajectory itself should know the real six participants.
        trajectory_users = [
            str(x or "").strip()
            for x in (qa.get("required_users") or [])
            if str(x or "").strip()
        ]

        canonical_users = list(view["required_users"])

        # Legacy generation trajectories may contain only the asker and one
        # provider in required_users. Formality is text-only, so packet.json is
        # the canonical source for the complete participant list.
        #
        # We still sanity-check any names that *are* present in the trajectory:
        # they must all belong to the packet, and the first one must be the asker.
        unknown_users = [
            x for x in trajectory_users
            if x not in canonical_users
        ]

        if unknown_users:
            raise ValueError(
                f"{cid}: trajectory required_users contain names absent "
                f"from packet: unknown={unknown_users!r} "
                f"trajectory={trajectory_users!r} "
                f"packet={canonical_users!r}"
            )

        if trajectory_users and trajectory_users[0] != attempt.asker_user:
            raise ValueError(
                f"{cid}: trajectory required_users first user does not match "
                f"asker: first={trajectory_users[0]!r} "
                f"asker={attempt.asker_user!r}"
            )

        if len(trajectory_users) != 6:
            incomplete_trajectory_required_users.append({
                "candidate_id": cid,
                "reviewer_id": label.get("reviewer_id"),
                "assignment_id": label.get("assignment_id"),
                "trajectory_required_users": trajectory_users,
                "canonical_required_users": canonical_users,
            })
        elif Counter(trajectory_users) != Counter(canonical_users):
            raise ValueError(
                f"{cid}: complete trajectory participant names disagree "
                f"with packet: trajectory={trajectory_users!r} "
                f"packet={canonical_users!r}"
            )

        # Audit broken/incomplete annotation metadata, but do not use it
        # as Formality supervision or participant provenance.
        label_users = [
            str(x or "").strip()
            for x in (label.get("required_users") or [])
        ]

        if (
            len(label_users) != 6
            or any(not x for x in label_users)
            or Counter(x for x in label_users if x)
               != Counter(view["required_users"])
        ):
            incomplete_label_required_users.append({
                "candidate_id": cid,
                "reviewer_id": label.get("reviewer_id"),
                "assignment_id": label.get("assignment_id"),
                "label_required_users": label_users,
                "canonical_required_users": view["required_users"],
            })

        participant_names = prompts.formality_participant_names(
            view, qa
        )

        formality_errors = prompts.qa_formality_errors(
            qa,
            list(attempt.schema_errors),
            participant_names=participant_names,
        )

        verdict = label["formality_verdict"]

        if formality_errors and verdict == "pass":
            exclusions.append({
                "candidate_id": cid,
                "reviewer_id": label.get("reviewer_id"),
                "assignment_id": label.get("assignment_id"),
                "human_verdict": verdict,
                "reason": "deterministic formality branch is FAIL",
                "deterministic_errors": formality_errors,
            })
            continue

        prompt = prompts.build_qa_formality_judge_prompt(
            qa,
            view,
            schema_errors=list(attempt.schema_errors),
        )

        records.append({
            "example_id": f"{cid}::formality",
            "group_id": attempt.packet_id,
            "task": "formality",
            "prompt": prompt,
            "verdict": verdict,
            "provenance": {
                "candidate_id": cid,
                "source_qa_id": label.get("source_qa_id"),
                "source_evidence_id": attempt.evidence_id,
                "generation_attempt": attempt.attempt,
                "assignment_id": label.get("assignment_id"),
                "reviewer_id": label.get("reviewer_id"),
            },
        })

    ids = [x["example_id"] for x in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate Formality example_id")

    # ------------------------------------------------------------------
    # Packet-clean split, stratified by each 20-packet assignment.
    # ------------------------------------------------------------------

    assignment_packet_rows = defaultdict(
        lambda: defaultdict(list)
    )

    packet_assignments = defaultdict(set)

    for r in records:
        assignment = r["provenance"]["assignment_id"]
        packet = r["group_id"]

        assignment_packet_rows[assignment][packet].append(r)
        packet_assignments[packet].add(assignment)

    cross = {
        p: sorted(v)
        for p, v in packet_assignments.items()
        if len(v) != 1
    }

    if cross:
        raise RuntimeError(
            f"packets occur in multiple assignments: {cross}"
        )

    heldout_packets = set()
    split_search = {}

    for i, assignment in enumerate(
        sorted(assignment_packet_rows)
    ):
        chosen, info = choose_packet_subset(
            assignment_packet_rows[assignment],
            args.heldout_fraction,
            args.seed + i * 100003,
        )

        heldout_packets |= chosen
        split_search[assignment] = info

    heldout = sorted(
        [
            r for r in records
            if r["group_id"] in heldout_packets
        ],
        key=lambda x: x["example_id"],
    )

    train = sorted(
        [
            r for r in records
            if r["group_id"] not in heldout_packets
        ],
        key=lambda x: x["example_id"],
    )

    train_packets = {x["group_id"] for x in train}
    held_packets = {x["group_id"] for x in heldout}

    train_ids = {x["example_id"] for x in train}
    held_ids = {x["example_id"] for x in heldout}

    if train_packets & held_packets:
        raise RuntimeError("packet leakage")
    if train_ids & held_ids:
        raise RuntimeError("example leakage")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.output_dir / "all_formality.jsonl", records)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "heldout.jsonl", heldout)

    # Convenience alias for evaluators expecting test.jsonl.
    write_jsonl(args.output_dir / "test.jsonl", heldout)

    write_jsonl(
        args.output_dir / "excluded_formality_rows.jsonl",
        exclusions,
    )

    write_jsonl(
        args.output_dir / "incomplete_label_required_users.jsonl",
        incomplete_label_required_users,
    )

    write_jsonl(
        args.output_dir / "incomplete_trajectory_required_users.jsonl",
        incomplete_trajectory_required_users,
    )

    audit = {
        "schema_version": "egolife_formality_combined_v1",
        "seed": args.seed,
        "heldout_fraction": args.heldout_fraction,

        "input": {
            "human_labels": len(labels),
            "human_label_packets": len(label_packets),
            "label_files_sha256": label_sha,
            "incomplete_label_required_users":
                len(incomplete_label_required_users),
            "incomplete_trajectory_required_users":
                len(incomplete_trajectory_required_users),
        },

        "canonical_formality": row_stats(records),
        "excluded_formality_rows": len(exclusions),

        "train": row_stats(train),
        "heldout": row_stats(heldout),

        "train_packet_count": len(train_packets),
        "heldout_packet_count": len(held_packets),
        "packet_overlap": sorted(train_packets & held_packets),
        "example_overlap": sorted(train_ids & held_ids),

        "split_search": split_search,

        "generation": {
            "generation_root": generation_info["generation_root"],
            "sha256": generation_info["sha256"],
            "trajectory_parseable_attempts":
                generation_info["trajectory_parseable_attempts"],
        },
    }

    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
