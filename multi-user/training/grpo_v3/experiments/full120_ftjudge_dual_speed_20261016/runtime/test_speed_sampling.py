from collections import Counter
import unittest
import dual_runtime


class SpeedTests(unittest.TestCase):
    def test_framework_image_dict_uses_preserved_raw_generator_paths(self):
        rows=[{'images':[{'path':'a.jpg','bytes':None}], 'generator_image_paths':['a.jpg']} for _ in range(4)]
        self.assertEqual(dual_runtime.frozen_visual_key(rows),('a.jpg',))

    def test_length_pairs_preserve_conditions_and_avoid_same_window(self):
        rows=[{'source_packet_id':w,'asker_index':s,'cost':cost} for s in range(3) for w,cost in [('a',10),('c',100),('b',11),('d',101)]]
        result=dual_runtime.make_length_pairs(rows,lambda r:r['cost'],seed=42)
        self.assertEqual(Counter((r['source_packet_id'],r['asker_index']) for r in rows),Counter((r['source_packet_id'],r['asker_index']) for r in result))
        self.assertTrue(all(result[i]['source_packet_id']!=result[i+1]['source_packet_id'] for i in range(0,len(result),2)))
        self.assertEqual(result,dual_runtime.make_length_pairs(rows,lambda r:r['cost'],seed=42))
        self.assertLessEqual(sum(max(result[i]['cost'],result[i+1]['cost']) for i in range(0,len(result),2)),sum(max(rows[i]['cost'],rows[i+1]['cost']) for i in range(0,len(rows),2)))


if __name__=='__main__':unittest.main()
