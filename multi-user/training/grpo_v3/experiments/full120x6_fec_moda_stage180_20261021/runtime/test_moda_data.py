"""两种packet ID表示合并为720唯一训练条件，不继承验证身份。"""
from pathlib import Path
from collections import Counter
import importlib.util
import unittest

HERE=Path(__file__).resolve().parent


def fixtures():
    train=[];validation=[];definition={'train':[],'validation':[]}
    for window in range(120):
        token=f'{11000000+window:08d}'
        for speaker in range(6):
            split='validation' if speaker==window%6 else 'train'
            row={'source_packet_id':f'RLHF6U_DAY1_{token}_A1_JAKE','asker_index':speaker,
                 'required_users':['a','b','c','d','e','f'],'messages':[{'role':'user','content':'input'}]}
            (train if split=='train' else validation).append(row)
            definition[split].append({'source_packet_id':f'RLHF6U_DAY1_{token}_A1_A2_A3_A4_A5_A6',
                 'speaker_index':str(speaker),'input_view_count':'6','duration_seconds':'600','split':split})
    return train,validation,definition


class DataTests(unittest.TestCase):
    def setUp(self):
        path=HERE/'prepare_720_data.py'
        self.assertTrue(path.is_file(),'720全训练构建尚未实现')
        spec=importlib.util.spec_from_file_location('prepare_720_target',path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    def test_600_plus_120_preserves_all_six_speakers_as_train(self):
        train,validation,definition=fixtures()
        rows,by_key=self.module.assemble_720(train,validation,definition)
        self.assertEqual(len(rows),720)
        self.assertEqual(Counter(int(r['asker_index']) for r in rows),Counter({i:120 for i in range(6)}))
        self.assertEqual(len(by_key),720)
        self.assertTrue(all(r['split']=='train' for r in by_key.values()))
        self.assertEqual(sum(r['original_split']=='validation' for r in by_key.values()),120)

    def test_duplicate_condition_under_alias_or_missing_speaker_is_rejected(self):
        train,validation,definition=fixtures()
        validation[0]={**train[0],'source_packet_id':train[0]['source_packet_id'].replace('_A1_JAKE','_ALIAS')}
        with self.assertRaises(ValueError):self.module.assemble_720(train,validation,definition)

    def test_reusing_data_audit_requires_all_source_payloads_unchanged(self):
        self.assertTrue(callable(getattr(self.module,'verify_reused_conditions',None)))
        train,validation,definition=fixtures()
        rows,_=self.module.assemble_720(train,validation,definition)
        self.module.verify_reused_conditions(list(reversed(rows)),rows)
        changed=[dict(r) for r in rows]
        changed[0]['messages']=[{'role':'user','content':'changed'}]
        with self.assertRaises(ValueError):self.module.verify_reused_conditions(changed,rows)


if __name__=='__main__':unittest.main()
