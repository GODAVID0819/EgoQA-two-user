"""新模式的标签、窗口与划分边界回归；不依赖GPU。"""
import unittest
import json
import tempfile
from pathlib import Path
from training.judge_sft.aria12 import grouped_split, visual_messages
from training.judge_sft.data import normalized_record_to_example
from training.judge_sft.data import JudgeDataset
from training.judge_sft.trainer import audit_tensor_parallel_sampler
from training.judge_sft.watch_epoch_evaluations import checkpoint_ready


def row():
    order=['W11']+[f'W{i:02}' for i in range(1,13) if i!=11]
    return {'example_id':'sample','group_id':'packet_01','split_group_id':'blue_shirt',
        'task':'answerability','condition_type':'cross_view_requirement','verdict':'fail',
        'speaker_window_id':'W11','duration_seconds':233.9,'fps':0.5,'prompt':'Question and five choices.',
        'frame_sets':[{'label':w,'frames':[f'{w}/{i:04}.jpg' for i in range(117)],
            'times_seconds':[i*2 for i in range(117)],'cuts_before':[3]} for w in order]}


class Aria12Tests(unittest.TestCase):
    def test_four_tp_ranks_receive_each_example_once(self):
        examples = []
        for index in range(5):
            record = row()
            record["example_id"] = f"sample-{index}"
            examples.append(normalized_record_to_example(record, location="test", require_frame_files=False))
        audit = audit_tensor_parallel_sampler(JudgeDataset(examples), tensor_parallel_size=4, seed=17)
        self.assertEqual(audit["real_examples_per_epoch"], 5)
        self.assertTrue(audit["identical_order_on_every_tp_rank"])

    def test_partly_written_checkpoint_is_not_submitted_for_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "adapter_config.json").write_text("{}")
            (checkpoint / "adapter_model.safetensors").write_bytes(b"example")
            (checkpoint / "trainer_state.json").write_text('{"global_step":')
            self.assertFalse(checkpoint_ready(checkpoint, 40))
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 40}))
            self.assertTrue(checkpoint_ready(checkpoint, 40))
    def test_joint_c_accepts_twelve_complete_windows_and_keeps_fail(self):
        ex=normalized_record_to_example(row(),location='test',require_frame_files=False)
        self.assertEqual(ex.target,0)
        self.assertEqual(ex.frame_count,1404)
        self.assertEqual(ex.frame_sets[0].label,'W11')
        self.assertEqual(ex.split_group_id,'blue_shirt')

    def test_sampling_times_and_cut_boundary_are_retained(self):
        ex=normalized_record_to_example(row(),location='test',require_frame_files=False)
        self.assertEqual(ex.frame_sets[0].times_seconds[3],6)
        self.assertEqual(ex.frame_sets[0].cuts_before,(3,))

    def test_speaker_cannot_silently_move_to_another_group(self):
        r=row();r['speaker_window_id']='W05'
        with self.assertRaisesRegex(ValueError,'speaker'):
            normalized_record_to_example(r,location='test',require_frame_files=False)

    def test_missing_frame_is_not_padded_or_truncated(self):
        r=row();r['frame_sets'][0]['frames'].pop()
        with self.assertRaises(ValueError):
            normalized_record_to_example(r,location='test',require_frame_files=False)

    def test_visual_labels_are_adjacent_to_their_frames(self):
        example = normalized_record_to_example(row(), location="test", require_frame_files=False)
        content = visual_messages(example, min_pixels=3136, max_pixels=119808)[0]["content"]
        self.assertEqual(sum(item["type"] == "image" for item in content), 1404)
        self.assertIn("W11 SPEAKER", content[0]["text"])
        self.assertIn("W11 F000 t=0s", content[1]["text"])
        self.assertEqual(content[2]["image"], "W11/0000.jpg")
        self.assertIn("EDIT CUT", content[7]["text"])
        self.assertEqual(content[-1]["text"], example.prompt)

    def test_same_fact_positive_and_negative_stay_together(self):
        rows = [
            {"example_id": "a", "split_group_id": "same_fact", "verdict": "pass"},
            {"example_id": "b", "split_group_id": "same_fact", "verdict": "fail"},
            {"example_id": "c", "split_group_id": "different_fact", "verdict": "fail"},
        ]
        train, test = grouped_split(rows, test_count=2, test_pass=1)
        self.assertEqual({r["example_id"] for r in test}, {"a", "b"})
        self.assertEqual([r["example_id"] for r in train], ["c"])
        with self.assertRaisesRegex(ValueError, "splitting a fact group"):
            grouped_split(rows, test_count=1, test_pass=1)


if __name__=='__main__':unittest.main()
