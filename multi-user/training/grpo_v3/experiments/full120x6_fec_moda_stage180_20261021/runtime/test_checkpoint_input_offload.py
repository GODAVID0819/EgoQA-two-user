"""用原生CPU autograd/checkpoint验证中间张量转存不改变数值、梯度及RNG。"""
import copy
import unittest
try:
    import checkpoint_input_offload as offload
except ModuleNotFoundError:
    offload=None


class CheckpointInputOffloadTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(offload,'缺少梯度检查点输入转存实现')
        import torch
        self.torch=torch

    def fixture(self,reentrant):
        torch=self.torch
        from torch.utils.checkpoint import checkpoint
        from transformers.modeling_layers import GradientCheckpointingLayer
        from transformers import PreTrainedModel,PretrainedConfig
        from functools import partial
        class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(16,16,dtype=torch.float64)
                self.dropout=torch.nn.Dropout(.05)
                self.gradient_checkpointing=True
                self._gradient_checkpointing_func=partial(checkpoint,use_reentrant=reentrant,preserve_rng_state=True)
            def forward(self,x):return x+self.dropout(torch.tanh(self.linear(x)))
        class Policy(PreTrainedModel):
            supports_gradient_checkpointing=True
            def __init__(self):
                super().__init__(PretrainedConfig());self.layers=torch.nn.ModuleList([Qwen3_5DecoderLayer() for _ in range(8)])
            def forward(self,x):
                for layer in self.layers:x=layer(x)
                return x
        torch.manual_seed(314);model=Policy();return model

    def compare(self,reentrant,layer_stride=4):
        torch=self.torch;baseline=self.fixture(reentrant);changed=copy.deepcopy(baseline)
        controller=offload.install(changed,expected_layer_count=8,layer_stride=layer_stride)
        self.assertEqual(controller.selected_layers,list(range(layer_stride-1,8,layer_stride)))
        source=torch.arange(160,dtype=torch.float64).reshape(2,5,16)/160
        results=[]
        for model in [baseline,changed]:
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
            x=source.clone().requires_grad_(True);torch.manual_seed(42)
            y=model(x);loss=y.square().mean();loss.backward()
            grads=[p.grad.detach().clone() for p in model.parameters()]
            input_grad=x.grad.detach().clone();rng=torch.get_rng_state().clone()
            optimizer.step();params=[p.detach().clone() for p in model.parameters()]
            results.append((y.detach(),loss.detach(),input_grad,grads,rng,params))
        a,b=results
        for index in [0,1,2,4]:self.assertTrue(torch.equal(a[index],b[index]),str(index))
        for ga,gb in zip(a[3],b[3]):self.assertTrue(torch.equal(ga,gb))
        for pa,pb in zip(a[5],b[5]):self.assertTrue(torch.equal(pa,pb))
        self.assertGreater(controller.stats['offloaded_bytes'],0)
        self.assertEqual(controller.stats['packed_tensors'],controller.stats['restored_tensors'])
        self.assertEqual(controller.stats['packed_tensors'],8//layer_stride)
        controller.close()
        before=controller.stats['packed_tensors'];torch.manual_seed(42)
        changed(source.clone().requires_grad_(True)).sum().backward()
        self.assertEqual(controller.stats['packed_tensors'],before)

    def test_reentrant_exact_output_gradient_optimizer_rng(self):self.compare(True)
    def test_nonreentrant_exact_output_gradient_optimizer_rng(self):self.compare(False)
    def test_all_layer_inputs_reentrant_exact_output_gradient_optimizer_rng(self):self.compare(True,layer_stride=1)
    def test_all_layer_inputs_nonreentrant_exact_output_gradient_optimizer_rng(self):self.compare(False,layer_stride=1)

    def test_does_not_pack_weights_or_unrelated_tensors(self):
        torch=self.torch;model=self.fixture(True);controller=offload.install(model,expected_layer_count=8)
        x=torch.ones(2,5,16,dtype=torch.float64,requires_grad=True)
        model(x).sum().backward()
        self.assertEqual(controller.stats['offloaded_bytes'],2*x.numel()*x.element_size())

    def test_rejects_unexpected_model_layout(self):
        with self.assertRaises(ValueError):offload.install(self.fixture(True),expected_layer_count=64)

    def test_actual_swift_context_reenable_keeps_offload_active_and_gradients_equal(self):
        torch=self.torch
        from swift.trainers import disable_gradient_checkpointing
        baseline=self.fixture(False);changed=copy.deepcopy(baseline)
        controller=offload.install(changed,expected_layer_count=8)
        kwargs={'use_reentrant':False,'preserve_rng_state':True}
        source=torch.arange(160,dtype=torch.float64).reshape(2,5,16)/160
        for model in [baseline,changed]:
            with torch.no_grad(),disable_gradient_checkpointing(model,kwargs):model(source)
        self.assertTrue(changed.is_gradient_checkpointing)
        values=[]
        for model in [baseline,changed]:
            torch.manual_seed(42);x=source.clone().requires_grad_(True)
            result=model(x);result.square().mean().backward()
            values.append((result.detach(),x.grad.detach().clone(),[p.grad.detach().clone() for p in model.parameters()],torch.get_rng_state()))
        self.assertTrue(torch.equal(values[0][0],values[1][0]));self.assertTrue(torch.equal(values[0][1],values[1][1]))
        self.assertTrue(torch.equal(values[0][3],values[1][3]))
        for a,b in zip(values[0][2],values[1][2]):self.assertTrue(torch.equal(a,b))
        self.assertEqual(controller.stats['packed_tensors'],2)
        self.assertEqual(controller.stats['restored_tensors'],2)
        self.assertGreaterEqual(controller.stats.get('wrapper_reinstalls',0),2)
        with torch.no_grad(),disable_gradient_checkpointing(changed,{'use_reentrant':True,'preserve_rng_state':True}):
            changed(source)
        latest=changed.layers[3]._gradient_checkpointing_func
        controller.close()
        self.assertIs(changed.layers[3]._gradient_checkpointing_func,latest)


if __name__=='__main__':unittest.main(verbosity=2)
