"""基于已部署 Swift 微批调用顺序的路由测试，不加载 GPU 或模型。"""
import unittest
from types import SimpleNamespace

try:
    from microbatch_vision_binding import install
except ImportError:
    # 红灯阶段：未安装路由时复现原来八条混窗输入的空 key。
    def install(trainer, controller, record=None):
        return None


def row(window, images=None):
    return {'source_packet_id':window,'asker_index':4,
            'generator_image_paths':images or [window+'/0',window+'/1']}


class Controller:
    enabled=True
    def __init__(self):
        self.key=None
        self.entry=None
        self.computations=0
    def set_key(self,key):
        if key!=self.key:
            self.entry=None
        self.key=key
    def visual(self,pixels):
        if self.key is not None and self.entry==pixels:
            return sum(pixels)
        self.computations+=1
        if self.key is not None:self.entry=pixels
        return sum(pixels)


class Trainer:
    dynamic_num_samples=False
    template=SimpleNamespace(sequence_parallel_size=1)
    args=SimpleNamespace(per_device_train_batch_size=1)
    def __init__(self,c):self.c=c;self.observed=[]
    def split_by_mini_batches(self,inputs):return [[r] for r in inputs]
    def _prepare_batch_inputs(self,inputs):
        result=[]
        for chunk in self.split_by_mini_batches(inputs):
            encoded={'pixels':tuple(chunk[0].get('pixels',[1,2]))}
            self._get_per_token_logps_and_entropies(None,encoded)
            self._get_per_token_logps_and_entropies(None,encoded)
            result.append(encoded)
        return result
    def _get_per_token_logps_and_entropies(self,model,inputs,*args,**kwargs):
        self.observed.append(self.c.key)
        return self.c.visual(inputs['pixels'])


class BindingTests(unittest.TestCase):
    def setup_trainer(self,**kwargs):
        c=Controller();t=Trainer(c)
        for k,v in kwargs.items():setattr(t,k,v)
        router=install(t,c)
        return t,c,router

    def test_two_windows_share_only_inside_actual_microbatches(self):
        t,c,_=self.setup_trainer()
        encoded=t._prepare_batch_inputs([row('A') for _ in range(4)]+[row('B') for _ in range(4)])
        self.assertEqual(c.computations,2)
        self.assertEqual(t.observed[:8],[('A/0','A/1')]*8)
        self.assertEqual(t.observed[8:],[('B/0','B/1')]*8)
        for e in encoded:t._get_per_token_logps_and_entropies(None,e)
        self.assertEqual(c.computations,4)

    def test_window_change_with_same_pixels_and_paths_clears_entry(self):
        t,c,_=self.setup_trainer()
        t._prepare_batch_inputs([row('A',['x','y']),row('B',['x','y'])])
        self.assertEqual(c.computations,2)

    def test_frame_order_change_clears_entry(self):
        t,c,_=self.setup_trainer()
        t._prepare_batch_inputs([row('A',['x','y']),row('A',['y','x'])])
        self.assertEqual(c.computations,2)

    def test_unknown_encoded_identity_falls_back(self):
        t,c,_=self.setup_trainer();e=t._prepare_batch_inputs([row('A')])[0]
        t._get_per_token_logps_and_entropies(None,dict(e))
        self.assertIsNone(c.key)

    def test_single_window_and_disabled_controller(self):
        t,c,_=self.setup_trainer()
        t._prepare_batch_inputs([row('A') for _ in range(4)])
        self.assertEqual(c.computations,1)
        c.enabled=False
        t._prepare_batch_inputs([row('A')])
        self.assertIsNone(c.key)

    def test_missing_condition_falls_back(self):
        t,c,_=self.setup_trainer()
        t._prepare_batch_inputs([{'generator_image_paths':['a']}])
        self.assertIsNone(c.key)

    def test_next_generation_discards_old_encoded_mapping(self):
        t,c,_=self.setup_trainer();e=t._prepare_batch_inputs([row('A')])[0]
        t._prepare_batch_inputs([row('B')])
        t._get_per_token_logps_and_entropies(None,e)
        self.assertIsNone(c.key)

    def test_dynamic_or_sequence_parallel_falls_back(self):
        for kwargs in [{'dynamic_num_samples':True},{'template':SimpleNamespace(sequence_parallel_size=2)}]:
            t,c,_=self.setup_trainer(**kwargs)
            t._prepare_batch_inputs([row('A')])
            self.assertIsNone(c.key)

    def test_mixed_actual_microbatch_falls_back(self):
        t,c,_=self.setup_trainer()
        # 原接口在 prepare 之外仍保持原返回类型。
        self.assertIsInstance(t.split_by_mini_batches([row('A')]),list)
        t.args=SimpleNamespace(per_device_train_batch_size=2)
        t._prepare_batch_inputs([row('A'),row('B')])
        self.assertIsNone(c.key)


if __name__=='__main__':unittest.main(verbosity=2)
