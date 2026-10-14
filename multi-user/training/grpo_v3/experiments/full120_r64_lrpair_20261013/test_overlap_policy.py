"""验证新增重叠的记录、连通组留出与已有源审计复用。"""
import collections
import json
from pathlib import Path
import tempfile
import unittest

import repair_sources as repair
from pair_data import validate_conditions, make_order


def window(name, start, day='DAY1'):
    return {'source_packet_id':name,'day':day,'time_token':repair.token(start)}


class OverlapPolicyTests(unittest.TestCase):
    def test_additional_overlap_counts_only_increases_and_excludes_duplicate(self):
        self.assertTrue(hasattr(repair, 'replacement_score'), '缺少允许最小新增重叠的候选评分')
        old=window('old',12*3600+9*60)
        neighbor=window('neighbor',12*3600)
        self.assertEqual(repair.replacement_score(old,12*3600+8*60+30,[old,neighbor]),(30,30,43710))
        self.assertIsNone(repair.replacement_score(old,12*3600,[old,neighbor]))

    def test_overlap_chain_uses_one_holdout_and_remains_exactly_balanced(self):
        self.assertTrue(hasattr(repair, 'assign_component_holdouts'), '缺少重叠连通组固定留出')
        windows=[window('w'+str(i),i*900,day='DAY'+str(i//20+1)) for i in range(120)]
        windows[1]['time_token']=repair.token(570)
        windows[2]['time_token']=repair.token(1140)
        preferred={w['source_packet_id']:i%6 for i,w in enumerate(windows)}
        held,groups=repair.assign_component_holdouts(windows,preferred,seed=42)
        self.assertEqual(held['w0'],held['w1'])
        self.assertEqual(held['w1'],held['w2'])
        self.assertEqual(collections.Counter(held.values()),{i:20 for i in range(6)})
        self.assertEqual((held,groups),repair.assign_component_holdouts(windows,preferred,seed=42))

    def test_real_120_window_split_and_four_window_sampler(self):
        self.assertTrue(hasattr(repair, 'repartition_conditions'), '缺少按连通组重分条件')
        root=Path(__file__).parent
        source=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
        definition=json.loads((root/'condition_definition.json').read_text(encoding='utf-8'))
        result,summary=repair.repartition_conditions(source['windows'],definition)
        validate_conditions(result['train'],result['validation'])
        ordered=make_order(result['train'],seed=42)
        self.assertTrue(all(len({r['source_packet_id'] for r in ordered[i:i+4]})==4 for i in range(0,600,4)))
        held={r['source_packet_id']:int(r['speaker_index']) for r in result['validation']}
        for component in summary['components']:
            self.assertEqual(len({held[w] for w in component}),1)

    def test_oversized_component_fails_instead_of_changing_balance(self):
        self.assertTrue(hasattr(repair, 'assign_component_holdouts'), '缺少连通组容量检查')
        windows=[window('w'+str(i),i*30) for i in range(21)]
        windows += [window('z'+str(i),i*900,day='DAY2') for i in range(99)]
        with self.assertRaisesRegex(ValueError,'20'):
            repair.assign_component_holdouts(windows,{w['source_packet_id']:0 for w in windows},seed=42)

    def test_reuse_audit_retains_short_sources_and_rejects_missing_files(self):
        self.assertTrue(hasattr(repair, 'reuse_source_audit'), '缺少已审计源复用')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);media=root/'media.mp4';media.write_bytes(b'abcd')
            audit=root/'source_media_audit.jsonl'
            rows=[{'path':'short','local_path':str(media),'bytes':4,'problem':'short_video','video_duration':21.1},
                  {'path':'missing','local_path':str(root/'missing.mp4'),'bytes':4,'problem':None}]
            audit.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            reused=repair.reuse_source_audit(audit)
            self.assertEqual(set(reused),{'short'})
            self.assertEqual(reused['short']['problem'],'short_video')


if __name__=='__main__':unittest.main()
