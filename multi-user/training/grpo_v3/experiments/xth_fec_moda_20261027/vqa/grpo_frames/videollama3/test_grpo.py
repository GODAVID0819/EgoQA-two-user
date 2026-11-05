"""核对完整GRPO帧输入、原提示词和独立图片模态。"""
import copy
import json
import unittest
from pathlib import Path

import grpo_inputs as core

ROOT=Path(__file__).resolve().parent


class GrpoInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items,cls.frames,cls.header=core.load_inputs(ROOT)

    def test_all_131_and_original_generator_frame_counts(self):
        self.assertEqual(len(self.items),131)
        paths=set()
        for item in self.items:
            fr=self.frames[item['id']]
            self.assertEqual(len(fr['frames'][item['speaker']]),300)
            self.assertEqual(sum(map(len,fr['frames'].values())),sum(fr['frame_counts']))
            paths.update(p for group in fr['frames'].values() for p in group)
        self.assertEqual(len(paths),89255)

    def test_three_conditions_preserve_frame_order_and_evidence_exception(self):
        single=0
        for item in self.items:
            fr=self.frames[item['id']]
            for setting in core.SETTINGS:
                users,paths,conversation=core.build_input(item,fr,setting,self.header)
                self.assertEqual(paths,[p for u in users for p in fr['frames'][u]])
                self.assertTrue(all(c['type']=='image' for c in conversation[0]['content'][:-1]))
                self.assertEqual(len(conversation[0]['content'])-1,len(paths))
                self.assertEqual(users,core.reference.setting_users(item,fr,setting))
                if setting=='speaker1':self.assertEqual(users,[item['speaker']])
                if setting=='evidence2' and len(users)==1:single+=1
        self.assertEqual(single,1)

    def test_exact_upstream_text_and_no_gold_or_judge_leak(self):
        item=copy.deepcopy(self.items[0]);fr=self.frames[item['id']]
        item.update(answer='SECRET_GOLD_TEXT',correct='SECRET_LETTER',generator_evidence='SECRET_REASON',
                    baseline_results='SECRET_BASELINE',selection={'value':'SECRET_REVIEW'})
        for setting in core.SETTINGS:
            users,paths,conversation=core.build_input(item,fr,setting,self.header)
            text=conversation[0]['content'][-1]['text']
            self.assertEqual(text,core.reference.build_text(item,fr,users,self.header))
            self.assertNotIn('SECRET_',text)
            self.assertIn('Reason:',text)
            self.assertNotIn('Before returning the question',text)

    def test_parsers_are_exact_upstream(self):
        for text in ['Answer: b\nReason: from frames','Answer: (E)\nReason: no view','A then D','option c']:
            self.assertEqual(core.parse_letter(text),core.reference.parse_letter(text))
            self.assertEqual(core.parse_reason(text),core.reference.parse_reason(text))

    def test_original_swift_rescale_is_156(self):
        from PIL import Image
        self.assertEqual(core.reference.rescale_image(Image.new('RGB',(1408,1408)),24576).size,(156,156))


if __name__=='__main__':unittest.main()
