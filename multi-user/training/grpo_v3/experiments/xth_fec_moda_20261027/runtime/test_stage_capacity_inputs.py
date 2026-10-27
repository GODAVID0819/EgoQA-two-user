"""防止把其他pair或不同设置的计时当成本轮最长输入容量证据。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from capacity_acceptance import verify_capacity_inputs


class CapacityInputTests(unittest.TestCase):
    def test_actual_images_and_conditions_must_match_longest_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'preflight').mkdir();output=root/'output';output.mkdir()
            source=[{'source_packet_id':f'packet{i}','asker_index':i,'images':[f'/scratch/a{i}.jpg',f'/scratch/b{i}.jpg']} for i in range(2)]
            formal=root/'formal.jsonl';measured=root/'measured.jsonl'
            dump=lambda p,rows:p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            dump(formal,source);dump(measured,source)
            config={'train_dataset':str(formal),'max_pixels':50176,'max_steps':360,'stop_after_steps':180}
            actual={**config,'train_dataset':str(measured),'capacity_measurement':True,'stop_after_steps':1}
            (output/'run_config.json').write_text(json.dumps(actual))
            (root/'preflight/input_lengths.json').write_text(json.dumps({'longest_pair_indices':[0,1]}))
            for rank in range(2):
                p=output/f'rank{rank}';p.mkdir();dump(p/'reward_trace.jsonl',[source[rank]]*4)
            self.assertEqual(verify_capacity_inputs(root,config,output),[0,1])
            bad=copy.deepcopy(source);bad[0]['images'].reverse();dump(measured,bad)
            with self.assertRaises(ValueError):verify_capacity_inputs(root,config,output)
            dump(measured,source);actual['max_pixels']=24576
            (output/'run_config.json').write_text(json.dumps(actual))
            with self.assertRaises(ValueError):verify_capacity_inputs(root,config,output)


if __name__=='__main__':unittest.main()
