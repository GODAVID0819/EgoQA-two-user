from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import training.grpo_v3.experiments.annotated_preference.build_dataset as build_dataset_module
from training.grpo_v3.experiments.annotated_preference.build_dataset import (
    OUTPUT_FILENAMES,
    build_dpo_row,
    build_outputs,
    pair_index_row,
    publish_dataset,
)
from training.grpo_v3.experiments.annotated_preference.pareto import (
    PreferencePair,
    compact_fingerprint,
)
from training.grpo_v3.experiments.annotated_preference.prompting import (
    COMPACT_QA_CONTRACT,
    PROMPT_REVISION,
    prompt_sha256,
    serialize_compact_completion,
)
from training.grpo_v3.experiments.human_preference_reviewer.v1.data import (
    CONTRACT_VERSION,
    CandidateRecord,
    EvidenceRecord,
    sha256_file,
)


FIELDS = (
    "schema_version", "dataset_fingerprint", "assignment_id", "reviewer_id",
    "annotation_status", "packet_order", "evidence_id", "display_order",
    "candidate_id", "formality_score", "evidence_grounding_score",
    "answerability_score", "fea_total_score", "aggregate_rank", "packet_skipped",
    "skip_reason", "notes", "question", "options", "correct", "answer",
    "video_1_user", "video_1_source", "video_2_user", "video_2_source",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        cls.csv_path = cls.root / "annotations.csv"
        cls.split_path = cls.root / "split.json"
        cls.media_map_path = cls.root / "media_map.json"
        cls.rows: list[dict[str, str]] = []
        cls.media_map: dict[str, str] = {}
        media_dir = cls.root / "media"
        media_dir.mkdir()

        for evidence_number in range(70):
            evidence_id = f"e{evidence_number:03d}"
            source_a = f"https://media.test/{evidence_id}-speaker.mp4"
            source_b = f"https://media.test/{evidence_id}-provider.mp4"
            path_a = media_dir / f"{evidence_id}-speaker.mp4"
            path_b = media_dir / f"{evidence_id}-provider.mp4"
            path_a.write_bytes(b"speaker")
            path_b.write_bytes(b"provider")
            cls.media_map[source_a] = str(path_a)
            cls.media_map[source_b] = str(path_b)
            for candidate_number, grade in enumerate((1, 1, 2, 2, 3, 3), start=1):
                options = [f"{evidence_id}-{candidate_number}-{letter}" for letter in "ABCDE"]
                cls.rows.append({
                    "schema_version": "egolife_rlhf_packet_ranking_v4",
                    "dataset_fingerprint": "fixture",
                    "assignment_id": "assignment",
                    "reviewer_id": "HM",
                    "annotation_status": "completed",
                    "packet_order": str(evidence_number + 1),
                    "evidence_id": evidence_id,
                    "display_order": str(candidate_number),
                    "candidate_id": f"{evidence_id}::candidate_{candidate_number:02d}",
                    "formality_score": str(grade),
                    "evidence_grounding_score": str(grade),
                    "answerability_score": str(grade),
                    "fea_total_score": str(grade * 3),
                    "aggregate_rank": str(candidate_number),
                    "packet_skipped": "false",
                    "skip_reason": "",
                    "notes": "",
                    "question": f"Question {evidence_id} candidate {candidate_number}?",
                    "options": json.dumps(options),
                    "correct": "A",
                    "answer": options[0],
                    "video_1_user": "Jake",
                    "video_1_source": source_a,
                    "video_2_user": "Tasha",
                    "video_2_source": source_b,
                })

        cls._write_csv(cls.csv_path, cls.rows)
        cls.split = {
            "contract_version": CONTRACT_VERSION,
            "split_unit": "evidence_id",
            "csv_sha256": sha256_file(cls.csv_path).lower(),
            "train_evidence_ids": [f"e{number:03d}" for number in range(59, -1, -1)],
            "validation_evidence_ids": [f"e{number:03d}" for number in range(69, 59, -1)],
            "locked_test_evidence_ids": [],
            "reserve_evidence_ids": [],
        }
        cls.split_path.write_text(
            json.dumps(cls.split, ensure_ascii=False), encoding="utf-8"
        )
        cls.media_map_path.write_text(
            json.dumps(cls.media_map, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.cleanup()

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_row_and_pair_index_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            speaker = root / "speaker.mp4"
            provider = root / "provider.mp4"
            speaker.write_bytes(b"a")
            provider.write_bytes(b"b")
            evidence = EvidenceRecord(
                evidence_id="evidence-1",
                annotation_status="completed",
                video_a_user="Jake",
                video_a_source="source-a",
                video_b_user="Tasha",
                video_b_source="source-b",
                candidates=(),
            )
            chosen = CandidateRecord(
                "evidence-1::chosen", "evidence-1", 1, "Chosen?",
                ("a", "b", "c", "d", "e"), "A", "a", 3, 2, 3, 1,
            )
            rejected = CandidateRecord(
                "evidence-1::rejected", "evidence-1", 2, "Rejected?",
                ("a", "b", "c", "d", "e"), "B", "b", 2, 2, 1, 2,
            )
            pair = PreferencePair(
                "evidence-1", chosen, rejected,
                compact_fingerprint("evidence-1", chosen),
                compact_fingerprint("evidence-1", rejected),
            )

            row = build_dpo_row(pair, evidence, {"source-a": str(speaker), "source-b": str(provider)})
            index = pair_index_row(pair)

            self.assertEqual({"messages", "rejected_response", "videos"}, set(row))
            self.assertEqual(2, row["messages"][0]["content"].count("<video>"))
            self.assertEqual(serialize_compact_completion(chosen), row["messages"][1]["content"])
            self.assertEqual(serialize_compact_completion(rejected), row["rejected_response"])
            self.assertEqual([str(speaker), str(provider)], row["videos"])
            legacy_evidence = replace(
                evidence,
                video_a_user="A / Speaker",
                video_b_user="B / Provider",
            )
            legacy_row = build_dpo_row(
                pair,
                legacy_evidence,
                {"source-a": str(speaker), "source-b": str(provider)},
            )
            self.assertEqual(row["videos"], legacy_row["videos"])
            self.assertEqual(
                {
                    "evidence_id", "chosen_candidate_id", "rejected_candidate_id",
                    "chosen_scores", "rejected_scores", "chosen_fingerprint",
                    "rejected_fingerprint",
                },
                set(index),
            )
            self.assertEqual([3, 3, 2], index["chosen_scores"])
            self.assertEqual([1, 2, 2], index["rejected_scores"])
            self.assertNotEqual(index["chosen_candidate_id"], index["rejected_candidate_id"])

    def test_build_outputs_happy_path(self) -> None:
        outputs = build_outputs(self.csv_path, self.split_path, self.media_map_path)

        self.assertIsInstance(outputs.train_rows, tuple)
        self.assertEqual(len(outputs.train_rows), len(outputs.train_index))
        self.assertEqual(len(outputs.validation_rows), len(outputs.validation_index))
        self.assertEqual("e059", outputs.train_index[0]["evidence_id"])
        self.assertEqual("e069", outputs.validation_index[0]["evidence_id"])
        train_ids = {row["evidence_id"] for row in outputs.train_index}
        validation_ids = {row["evidence_id"] for row in outputs.validation_index}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(set(self.split["train_evidence_ids"]), train_ids)
        self.assertEqual(set(self.split["validation_evidence_ids"]), validation_ids)

        for split_name in ("train", "validation"):
            summary = outputs.audit["splits"][split_name]
            self.assertEqual(summary["pair_count"], summary["dominance_pair_count"])
            self.assertEqual(
                summary["total_combinations"],
                summary["dominance_pair_count"]
                + summary["equal_vector_pair_count"]
                + summary["incomparable_pair_count"],
            )
            self.assertEqual(summary["evidence_count"], len(summary["per_evidence"]))
            self.assertEqual(summary["evidence_count"], summary["evidence_with_pairs_count"])
        self.assertEqual(["e000", "e001", "e002", "e003"], outputs.audit["overfit_evidence_ids"])
        self.assertEqual(60, outputs.audit["splits"]["train"]["evidence_count"])
        self.assertEqual(10, outputs.audit["splits"]["validation"]["evidence_count"])

    def test_rejects_invalid_contracts(self) -> None:
        cases = (
            "csv_sha", "reserve", "missing_media", "relative_path", "same_media",
            "same_user", "placeholder_user", "audit_cancel",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                csv_path = root / "annotations.csv"
                split_path = root / "split.json"
                media_map_path = root / "media_map.json"
                rows = [dict(row) for row in self.rows]
                manifest = dict(self.split)
                media_map = dict(self.media_map)
                if case == "csv_sha":
                    manifest["csv_sha256"] = "0" * 64
                elif case == "reserve":
                    manifest["reserve_evidence_ids"] = ["not-eligible"]
                elif case == "missing_media":
                    media_map.pop(next(iter(media_map)))
                elif case == "relative_path":
                    media_map[next(iter(media_map))] = "relative/video.mp4"
                elif case == "same_media":
                    evidence_id = "e000"
                    media_map[f"https://media.test/{evidence_id}-provider.mp4"] = media_map[
                        f"https://media.test/{evidence_id}-speaker.mp4"
                    ]
                elif case == "same_user":
                    for row in rows[:6]:
                        row["video_1_user"] = row["video_2_user"]
                elif case == "placeholder_user":
                    for row in rows[:6]:
                        row["video_1_user"] = "speaker"
                self._write_csv(csv_path, rows)
                if case in {"same_user", "placeholder_user"}:
                    manifest["csv_sha256"] = sha256_file(csv_path)
                split_path.write_text(json.dumps(manifest), encoding="utf-8")
                media_map_path.write_text(json.dumps(media_map), encoding="utf-8")
                if case == "audit_cancel":
                    original = build_dataset_module.build_pareto_pairs

                    def corrupt_pair_audit(evidence: EvidenceRecord):
                        pairs, audit = original(evidence)
                        if evidence.evidence_id == "e059":
                            audit = replace(audit, total_combinations=audit.total_combinations + 1)
                        elif evidence.evidence_id == "e058":
                            audit = replace(audit, total_combinations=audit.total_combinations - 1)
                        return pairs, audit

                    with patch.object(
                        build_dataset_module, "build_pareto_pairs", corrupt_pair_audit
                    ):
                        with self.assertRaisesRegex(ValueError, "e05[89]"):
                            build_outputs(csv_path, split_path, media_map_path)
                else:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        build_outputs(csv_path, split_path, media_map_path)

    def test_publish_writes_manifest_and_exact_training_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            publish_dataset(self.csv_path, self.split_path, self.media_map_path, output_dir)

            self.assertEqual(set(OUTPUT_FILENAMES), {path.name for path in output_dir.iterdir()})
            for jsonl_name in ("train_dpo.jsonl", "validation_dpo.jsonl", "overfit_4_dpo.jsonl"):
                lines = (output_dir / jsonl_name).read_text(encoding="utf-8").splitlines()
                self.assertTrue(lines)
                self.assertTrue(all(set(json.loads(line)) == {"messages", "rejected_response", "videos"} for line in lines))
            manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(COMPACT_QA_CONTRACT, manifest["compact_qa_contract"])
            self.assertEqual(PROMPT_REVISION, manifest["prompt_revision"])
            self.assertEqual(prompt_sha256(), manifest["prompt_sha256"])
            self.assertEqual("human_fea_pareto_v1", manifest["preference_source"])
            for name in OUTPUT_FILENAMES[:-1]:
                self.assertEqual(_sha256(output_dir / name), manifest["outputs"][name]["sha256"])
            for input_name, path in (
                ("csv", self.csv_path), ("split", self.split_path), ("media_map", self.media_map_path)
            ):
                self.assertEqual(str(path.resolve()), manifest["inputs"][input_name]["path"])
                self.assertEqual(_sha256(path), manifest["inputs"][input_name]["sha256"])
            self.assertEqual(4, manifest["counts"]["overfit_evidence_count"])
            self.assertTrue(all((output_dir / name).read_bytes().endswith(b"\n") for name in OUTPUT_FILENAMES))

    def test_publish_rolls_back_existing_targets_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            output_dir.mkdir()
            expected: dict[str, bytes] = {}
            for number, name in enumerate(OUTPUT_FILENAMES):
                content = f"sentinel-{number}".encode()
                (output_dir / name).write_bytes(content)
                expected[name] = content
            other = output_dir / "keep.me"
            other.write_bytes(b"untouched")
            original_replace = Path.replace
            published = 0
            failed = False

            def fail_once(source: Path, target: str | Path) -> Path:
                nonlocal published, failed
                destination = Path(target)
                if (
                    source.parent.name.startswith(".annotated_preference-staging-")
                    and destination.parent == output_dir
                    and destination.name in OUTPUT_FILENAMES
                ):
                    published += 1
                    if published == 3 and not failed:
                        failed = True
                        raise OSError("injected publish failure")
                return original_replace(source, target)

            with patch.object(Path, "replace", fail_once):
                with self.assertRaisesRegex(OSError, "injected publish failure"):
                    publish_dataset(self.csv_path, self.split_path, self.media_map_path, output_dir)

            self.assertTrue(failed)
            self.assertEqual(b"untouched", other.read_bytes())
            self.assertEqual(expected, {name: (output_dir / name).read_bytes() for name in OUTPUT_FILENAMES})
            self.assertFalse(any(path.name.startswith(".annotated_preference-") for path in output_dir.iterdir()))

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            output_dir.mkdir()
            failed_name = OUTPUT_FILENAMES[2]
            for number, name in enumerate(OUTPUT_FILENAMES):
                (output_dir / name).write_bytes(f"sentinel-{number}".encode())
            original_replace = Path.replace
            publish_failed = False
            restore_failed = False

            def fail_publish_and_restore(source: Path, target: str | Path) -> Path:
                nonlocal publish_failed, restore_failed
                destination = Path(target)
                if (
                    source.parent.name.startswith(".annotated_preference-staging-")
                    and destination.parent == output_dir
                    and destination.name == failed_name
                    and not publish_failed
                ):
                    publish_failed = True
                    raise OSError("injected publish failure")
                if (
                    source.parent.name.startswith(".annotated_preference-backup-")
                    and destination.parent == output_dir
                    and destination.name == failed_name
                    and not restore_failed
                ):
                    restore_failed = True
                    raise OSError("injected restore failure")
                return original_replace(source, target)

            with patch.object(Path, "replace", fail_publish_and_restore):
                with self.assertRaisesRegex(RuntimeError, "rollback was incomplete") as raised:
                    publish_dataset(self.csv_path, self.split_path, self.media_map_path, output_dir)

            self.assertTrue(publish_failed)
            self.assertTrue(restore_failed)
            backups = list(output_dir.glob(".annotated_preference-backup-*"))
            self.assertEqual(1, len(backups))
            backup = backups[0]
            self.assertIn(str(backup.resolve()), str(raised.exception))
            self.assertEqual(b"sentinel-2", (backup / failed_name).read_bytes())


if __name__ == "__main__":
    unittest.main()
