import importlib.util
from pathlib import Path
import unittest


class SplitTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('splitter',Path(__file__).with_name('speaker_holdout_split.py'))
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.rows=[]; users=['Jake','Alice','Tasha','Lucia','Katrina','Shure']
        for window in range(8):
            for asker,user in enumerate(users):
                self.rows.append({'source_packet_id':f'window{window}','asker_index':asker,
                    'required_users':[user]+[u for u in users if u!=user],
                    'evidence_id':f'window{window}__ASKER_{asker}','images':[f'original_{window}_{asker}'],
                    'generator_frame_counts':[300,120,120,120,120,120]})

    def test_balance_preserve_media_and_five_one_partition(self):
        train,val,mapping=self.module.partition_training(self.rows)
        self.assertEqual((len(train),len(val)),(40,8))
        self.assertEqual(sorted(mapping['holdout_user_counts'].values()),[1,1,1,1,2,2])
        self.module.validate_holdout({'train':train,'validation':val})
        self.assertEqual({id(x) for x in train+val},{id(x) for x in self.rows})

    def test_overlap_or_same_window_test_rejected(self):
        train,val,_=self.module.partition_training(self.rows)
        with self.assertRaises(ValueError):self.module.validate_holdout({'train':train,'validation':val+[train[0]]})
        with self.assertRaises(ValueError):self.module.validate_holdout({'train':train,'validation':val,'test':[val[0]]})

    def test_duplicate_real_speaker_rejected_even_with_distinct_indices(self):
        self.rows[1]['required_users']=self.rows[0]['required_users'][:]
        with self.assertRaises(ValueError):self.module.partition_training(self.rows)


if __name__=='__main__':unittest.main()
