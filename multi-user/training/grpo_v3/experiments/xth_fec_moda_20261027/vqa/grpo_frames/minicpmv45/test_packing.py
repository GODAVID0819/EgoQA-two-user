"""检查跨视角分组边界、真实时间和三条件共同帧的不变性。"""
import unittest
from packing import pack_users, generation_kwargs
from grpo_inputs import load_inputs,build_input,SETTINGS

class PackingTests(unittest.TestCase):
    def test_boundaries_and_real_time(self):
        frames={'a':[f'frames/p/a/frame_{i:04d}_{t:.2f}s.jpg' for i,t in enumerate([0,2,4,6,8,10,50])],
                'b':['frames/p/b/frame_0000_0.00s.jpg']}
        groups=pack_users(['a','b'],frames)
        self.assertEqual([len(g['paths']) for g in groups],[6,1,1])
        self.assertEqual(groups[1]['temporal_ids'],[500])
        self.assertEqual(groups[2]['user'],'b')
        self.assertEqual(groups[0]['temporal_ids'],[0,20,40,60,80,100])
    def test_all_items_shared_groups_unchanged(self):
        items,store,header=load_inputs()
        for item in items:
            fr=store[item['id']]
            all_groups=pack_users(fr['required_users'],fr['frames'])
            for setting in SETTINGS:
                users,paths,messages=build_input(item,fr,setting,header)
                groups=pack_users(users,fr['frames'])
                self.assertEqual(groups,[g for g in all_groups if g['user'] in users])
                self.assertEqual([p for g in groups for p in g['paths']],paths)
                self.assertTrue(all(1<=len(g['paths'])<=6 for g in groups))
                self.assertIn('Answer:',messages[0]['content'][-1]['text'])
                self.assertIn('Reason:',messages[0]['content'][-1]['text'])
                self.assertLess(len(groups)*66+6000+512,40960)
    def test_decoder_overrides_official_sampling(self):
        params=generation_kwargs()
        self.assertFalse(params['do_sample'])
        self.assertEqual(params['num_beams'],1)
        self.assertEqual(params['repetition_penalty'],1.0)
        self.assertEqual(params['max_new_tokens'],512)

if __name__=='__main__':unittest.main()
