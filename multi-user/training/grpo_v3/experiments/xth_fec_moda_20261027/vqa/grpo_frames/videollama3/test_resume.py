"""中断后仅复用同配置、同题号、同原始帧的完整结果。"""
import copy,json,tempfile,unittest
from pathlib import Path
import grpo_inputs as core
from resume_results import load_resume


class ResumeTests(unittest.TestCase):
    def test_partial_resume_validates_gold_and_original_frame_paths(self):
        root=Path(__file__).resolve().parent
        items,frames,header=core.load_inputs(root)
        config=json.loads((root/'experiment_config.json').read_text(encoding='utf-8'))
        item=items[0];users,paths,_=core.build_input(item,frames[item['id']],'speaker1',header)
        row={'id':item['id'],'setting':'speaker1','correct':item['correct'],'pred':item['correct'],
             'is_correct':True,'raw':f'Answer: {item["correct"]}\nReason: visible evidence',
             'users':users,'frame_paths':paths,'n_images':len(paths),'model':config['model'],
             'model_revision':config['model_revision'],'origin_job_id':'example_job'}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);core.write_json(source/'run_config.json',config)
            path=source/'speaker1_predictions.jsonl'
            path.write_text(json.dumps(row)+'\n'+ '{"incomplete":',encoding='utf-8')
            groups,notes=load_resume(source,items,frames,header,config)
            self.assertEqual(groups['speaker1'][item['id']]['correct'],item['correct'])
            self.assertEqual(len(notes),1)
            broken=copy.deepcopy(row);broken['frame_paths']=['frames/wrong.jpg']
            path.write_text(json.dumps(broken)+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):load_resume(source,items,frames,header,config)


if __name__=='__main__':unittest.main()
