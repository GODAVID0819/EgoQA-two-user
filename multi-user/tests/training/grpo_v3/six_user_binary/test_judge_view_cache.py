"""重复评分复用已核验视图，元数据改动会使缓存失效。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_pipeline import require_module


class JudgeViewCacheTests(unittest.TestCase):
    def test_cached_view_isolated_and_invalidated_on_metadata_change(self):
        judge=require_module(self,'judge')
        with tempfile.TemporaryDirectory() as tmp:
            packet=Path(tmp)/'packets/P';packet.mkdir(parents=True)
            metadata=packet/'packet.json';metadata.write_text('{}')
            with patch.object(judge,'_load_full_judge_view',return_value={'clips':[{'path':'a'}]}) as load:
                first=judge.full_judge_view(tmp,'P',0)
                first['clips'][0]['path']='modified'
                second=judge.full_judge_view(tmp,'P',0)
                self.assertEqual(second['clips'][0]['path'],'a')
                self.assertEqual(load.call_count,1)
                metadata.write_text('{"changed":true}')
                judge.full_judge_view(tmp,'P',0)
                self.assertEqual(load.call_count,2)


if __name__=='__main__':unittest.main()
