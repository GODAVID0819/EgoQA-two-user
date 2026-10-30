"""真实官方图片处理器与全部393条GRPO输入的CPU容量检查。"""
import unittest
from pathlib import Path
from PIL import Image
import torch

import grpo_inputs as core
from model_adapter import ImageAdapter

ROOT=Path(__file__).resolve().parent


class ImageAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.adapter=ImageAdapter(ROOT/'upstream')
        cls.items,cls.frames,cls.header=core.load_inputs(ROOT)

    def test_exact_393_prompts_use_images_and_fit_native_context(self):
        records=[]
        for item in self.items:
            for setting in core.SETTINGS:
                users,paths,conversation=core.build_input(item,self.frames[item['id']],setting,self.header)
                inputs,audit=self.adapter.audit_prompt(conversation,len(paths))
                self.assertEqual(audit['n_images'],len(paths))
                self.assertEqual(int(inputs['input_ids'].eq(self.adapter.processor.image_token_id).sum()),audit['visual_tokens'])
                self.assertLessEqual(audit['total_with_generation'],32768)
                records.append({'id':item['id'],'setting':setting,**audit})
        core.write_json(ROOT/'LOCAL_PROMPT_AUDIT.json',{'media':'synthetic_156_images_geometry_real_131_questions',
            'n':len(records),'max_total_with_generation':max(r['total_with_generation'] for r in records),
            'max_prompt_tokens':max(r['prompt_tokens'] for r in records),'records':records})
        self.assertEqual(len(records),393)

    def test_actual_processor_matches_geometry_with_images_not_video(self):
        item=max(self.items,key=lambda x:sum(x['frame_counts'].values()))
        users,paths,conversation=core.build_input(item,self.frames[item['id']],'all6',self.header)
        image=Image.new('RGB',(156,156))
        inputs,audit=self.adapter.prepare(conversation,[image]*len(paths))
        self.assertEqual(inputs['modals'],['image']*len(paths))
        self.assertTrue(torch.all(inputs['grid_sizes'][:,0]==1))
        predicted=self.adapter.geometry(len(paths))
        self.assertEqual(inputs['grid_sizes'].tolist(),predicted['grid_sizes'].tolist())
        self.assertEqual(audit['n_images'],1633)
        self.assertEqual(audit['actual_size_histogram'],{'56x56':1633})


if __name__=='__main__':unittest.main()
