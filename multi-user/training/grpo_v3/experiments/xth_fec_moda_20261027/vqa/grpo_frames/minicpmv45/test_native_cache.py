"""使用官方小型随机模型核对原生三维视觉编码缓存和变长批处理，不加载权重或GPU。"""
import unittest
import torch
from upstream.configuration_minicpm import MiniCPMVConfig
from upstream.modeling_minicpmv import MiniCPMV

class Tokenizer:
    def convert_tokens_to_ids(self,word):return 2 if word=='<|im_end|>' else 3

class NativeCacheTests(unittest.TestCase):
    def test_grouped_vision_and_cached_variable_batch(self):
        torch.set_num_threads(2);torch.manual_seed(0)
        config=MiniCPMVConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,
            num_attention_heads=4,num_key_value_heads=2,head_dim=32,vocab_size=256,
            bos_token_id=1,eos_token_id=2,pad_token_id=0,query_num=64,
            vision_config={'hidden_size':64,'image_size':28,'intermediate_size':128,
                'num_attention_heads':4,'num_hidden_layers':1,'patch_size':14})
        config._attn_implementation='sdpa'
        model=MiniCPMV(config).eval()
        ids=torch.randint(10,250,(1,134),dtype=torch.int32)
        bounds=torch.tensor([[2,66],[68,132]])
        raw={'input_ids':ids,'image_bound':[bounds],
            'pixel_values':[[torch.randn(3,14,56) for _ in range(3)]],
            'tgt_sizes':[torch.tensor([[2,2]]*3)],'temporal_ids':[[[0,20],[40]]]}
        with torch.inference_mode():
            embedding,features=model.get_vllm_embedding(raw)
            self.assertEqual(features[0].shape,(2,64,128))
            cached,_=model.get_vllm_embedding({'input_ids':ids,'image_bound':[bounds],
                'vision_hidden_states':features})
            torch.testing.assert_close(embedding,cached,rtol=0,atol=0)
            short=ids[:,:68].clone()
            batched=torch.cat([ids,torch.cat([torch.zeros(1,66,dtype=ids.dtype),short],dim=1)])
            mask=batched.ne(0)
            kwargs=dict(tokenizer=Tokenizer(),decode_text=False,max_new_tokens=3,
                do_sample=False,num_beams=1,repetition_penalty=1.0,logits_to_keep=1)
            batch=model.generate(input_ids=batched,pixel_values=[[],[]],
                image_bound=[bounds,torch.tensor([[68,132]])],
                vision_hidden_states=[features[0],features[0][:1]],attention_mask=mask,**kwargs)
            for index,(tokens,positions,visual) in enumerate([
                (ids,bounds,features[0]),(short,torch.tensor([[2,66]]),features[0][:1])]):
                single=model.generate(input_ids=tokens,pixel_values=[[]],image_bound=[positions],
                    vision_hidden_states=[visual],attention_mask=tokens.ne(0),**kwargs)
                self.assertTrue(torch.equal(batch[index],single[0]))

if __name__=='__main__':unittest.main()
