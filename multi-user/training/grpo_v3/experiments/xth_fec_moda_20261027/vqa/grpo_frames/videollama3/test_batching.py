"""图片模式下原压缩掩码全真；核对单题与批处理生成一致。"""
import types
import unittest
from pathlib import Path
import torch
from model_adapter import ImageAdapter
from image_batching import collate_images,enable_image_batching

ROOT=Path(__file__).resolve().parent


class ImageBatchTests(unittest.TestCase):
    def test_variable_length_image_batches_match_original_single_item_generation(self):
        torch.set_num_threads(2);torch.manual_seed(0)
        adapter=ImageAdapter(ROOT/'upstream')
        config_class=adapter.model_class.config_class
        config=config_class(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
             num_attention_heads=2,num_key_value_heads=2,bos_token_id=1,eos_token_id=None,pad_token_id=0,
             image_token_index=20,max_position_embeddings=64,
             vision_encoder_config={'hidden_size':16,'intermediate_size':32,'num_hidden_layers':1,
                                    'num_attention_heads':2,'patch_size':14})
        model=adapter.model_class(config).eval()
        # 只替换CPU无法执行的视觉FlashAttention；测试实际官方多模态拼接、压缩和语言模型。
        model.encode_images=types.MethodType(lambda self,pixel_values,grid_sizes,merge_sizes:pixel_values[:,:16],model)
        examples=[]
        for count,text_ids in [(1,[4,5]),(2,[4,5,6])]:
            ids=torch.tensor([text_ids+[20]*(4*count)+[7]])
            pixels=torch.arange(4*count*16,dtype=torch.float32).reshape(4*count,16)/100
            examples.append({'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':pixels,
                 'grid_sizes':torch.tensor([[1,2,2]]*count),'merge_sizes':torch.ones(count,dtype=torch.long),
                 'modals':['image']*count})
        with torch.no_grad():
            expected=[model.generate(**x,do_sample=False,max_new_tokens=3).tolist()[0] for x in examples]
            enable_image_batching(model)
            batched=collate_images(examples,0)
            got=model.generate(**batched,do_sample=False,max_new_tokens=3,num_logits_to_keep=1).tolist()
        self.assertEqual(got,expected)
        self.assertFalse(model.config.use_token_compression)

    def test_video_input_cannot_enter_identity_image_batching(self):
        with self.assertRaises(ValueError):collate_images([{'modals':['video']}],0)


if __name__=='__main__':unittest.main()
