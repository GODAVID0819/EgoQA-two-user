"""使用训练环境的真实 Torch CPU 验证数值、语言梯度和既有预算回退。"""
import copy
import json
from pathlib import Path
import unittest
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"runtime"))
from frozen_vision_cache import VisionCache
from dual_runtime import frozen_visual_key
from microbatch_vision_binding import install
from test_microbatch_binding import Trainer,row

torch.set_num_threads(1)


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=torch.nn.Linear(4,4)
        self.calls=0
        self.requires_grad_(False)
    def forward(self,x):
        self.calls+=1
        return self.proj(x).relu()


class TensorTrainer(Trainer):
    def __init__(self,c,visual,language):
        super().__init__(c);self.visual=visual;self.language=language
    def _get_per_token_logps_and_entropies(self,model,inputs,*args,**kwargs):
        pixels=torch.tensor([inputs['pixels']],dtype=torch.float32)
        return self.language(self.visual(pixels)).square().mean()


class TensorTests(unittest.TestCase):
    def run_case(self,enabled,budget=2*1024**3,rows=None):
        torch.manual_seed(12)
        visual=Visual();language=torch.nn.Linear(4,1)
        c=VisionCache(visual,torch,budget)
        if not enabled:c.enabled=False
        t=TensorTrainer(c,visual,language);install(t,c)
        rows=rows or [row('A') for _ in range(4)]+[row('B') for _ in range(4)]
        for r in rows:r['pixels']=r.get('pixels',[1,2,3,4])
        with torch.no_grad():encoded=t._prepare_batch_inputs(rows)
        result=[]
        for e in encoded:
            loss=t._get_per_token_logps_and_entropies(None,e)
            result.append(loss.detach().clone());loss.backward()
        grads=[p.grad.detach().clone() for p in language.parameters()]
        stats=copy.deepcopy(c.stats);calls=visual.calls;c.close()
        return result,grads,stats,calls

    def test_outputs_and_language_gradients_equal_when_cache_toggled(self):
        off=self.run_case(False);on=self.run_case(True)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(off[0],on[0])))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(off[1],on[1])))
        self.assertEqual(off[3],24);self.assertEqual(on[3],4)
        self.assertEqual(on[2]['hits'],20)

    def test_existing_input_output_budget_falls_back_without_numerical_change(self):
        normal=self.run_case(False);limited=self.run_case(True,budget=1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(normal[0],limited[0])))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(normal[1],limited[1])))
        self.assertEqual(limited[2]['hits'],0)
        self.assertLessEqual(limited[2]['bytes'],1)
        self.assertGreater(limited[2]['bypasses'],0)

    def test_exact_input_comparison_prevents_path_only_reuse(self):
        rows=[dict(row('A'),pixels=[1,2,3,4]),dict(row('A'),pixels=[4,3,2,1])]
        off=self.run_case(False,rows=copy.deepcopy(rows))
        on=self.run_case(True,rows=copy.deepcopy(rows))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(off[0],on[0])))
        self.assertEqual(on[3],4)

    def test_original_single_window_g4_function_unchanged(self):
        a=[row('A') for _ in range(4)]
        self.assertEqual(frozen_visual_key(a),('A/0','A/1'))
        self.assertIsNone(frozen_visual_key(a+[row('B') for _ in range(4)]))
        self.assertEqual(self.run_case(True,rows=a)[3],1)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TensorTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    Path('tensor_test_result.json').write_text(json.dumps({
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'passed':result.wasSuccessful(),'device':'CPU','torch':torch.__version__,
        'scope':'数值与路由验证；没有测量 GPU 提速'},ensure_ascii=False,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
