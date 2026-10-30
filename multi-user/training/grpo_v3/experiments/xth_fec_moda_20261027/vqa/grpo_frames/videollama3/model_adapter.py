"""官方VideoLLaMA3独立图片接口；保留默认处理器并记录真实尺寸。"""
import math
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parent/'local_hf_cache'))
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class ImageAdapter:
    def __init__(self,model_path):
        self.model_path=str(Path(model_path).resolve())
        self.processor=AutoProcessor.from_pretrained(self.model_path,trust_remote_code=True,local_files_only=True)
        get_class_from_dynamic_module('modeling_videollama3_encoder.Videollama3VisionEncoderModel',
                                      self.model_path,local_files_only=True)
        self.model_class=get_class_from_dynamic_module('modeling_videollama3.Videollama3Qwen2ForCausalLM',
                                                        self.model_path,local_files_only=True)
        import json
        self.native_context=json.loads((Path(model_path)/'config.json').read_text(encoding='utf-8'))['max_position_embeddings']

    def geometry(self,n_images):
        if n_images<1:raise ValueError('本次三个条件都必须提供图片')
        proc=self.processor.image_processor
        module=sys.modules[type(proc).__module__]
        merge=self.processor.image_merge_size
        # 直接调用官方尺寸算法，无自定义视觉预算；全部缓存输入已经过原156预缩放。
        sizes=module.simple_batched_resize([Image.new('RGB',(156,156))]*n_images,
                 factor=proc.patch_size*merge,min_tokens=proc.min_tokens,max_tokens=proc.max_tokens)
        grids=torch.tensor([[1,h//proc.patch_size,w//proc.patch_size] for h,w in sizes])
        return {'grid_sizes':grids,'merge_sizes':torch.tensor([merge]*n_images),'modals':['image']*n_images}

    def tokenize(self,conversation,image_inputs):
        n_images=len(image_inputs['grid_sizes'])
        if len(conversation[0]['content'])-1!=n_images:
            raise ValueError('占位符数量与图片数不一致')
        prompt=self.processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True,
                                                   add_system_prompt=False)
        text=self.processor.process_text(prompt,image_inputs,return_tensors='pt',padding=False,
                                          padding_side='right',truncation=False)
        visual=int(text['input_ids'].eq(self.processor.image_token_id).sum())
        expected=sum(math.prod(g)//(m*m) for g,m in
                     zip(image_inputs['grid_sizes'].tolist(),image_inputs['merge_sizes'].tolist()))
        if visual!=expected or image_inputs['modals']!=['image']*n_images:
            raise ValueError('视觉token计数或独立图片模态错误')
        length=int(text['input_ids'].shape[-1])
        sizes=Counter(f'{g[1]*self.processor.image_processor.patch_size}x{g[2]*self.processor.image_processor.patch_size}'
                      for g in image_inputs['grid_sizes'].tolist())
        audit={'n_images':n_images,'preprocessed_size':[156,156],
               'actual_size_histogram':dict(sizes),'visual_tokens':visual,'text_tokens':length-visual,
               'prompt_tokens':length,'max_new_tokens':512,'total_with_generation':length+512,
               'native_context':self.native_context,'modality':'image',
               'processor_max_tokens':self.processor.image_processor.max_tokens,
               'processor_min_tokens':self.processor.image_processor.min_tokens,
               'image_merge_size':self.processor.image_merge_size,'video_compression_used':False}
        if not prompt.endswith(self.processor.get_generation_prompt()):
            raise ValueError('缺少assistant生成前缀')
        if audit['total_with_generation']>self.native_context:
            raise ValueError(f'原生上下文超限：{length}+512>{self.native_context}')
        return dict(text),audit

    def audit_prompt(self,conversation,n_images):
        return self.tokenize(conversation,self.geometry(n_images))

    def prepare(self,conversation,images):
        if any(image.mode!='RGB' or image.size!=(156,156) for image in images):
            raise ValueError('模型输入图片必须是原流程预缩放后的156×156 RGB')
        inputs=self.processor.process_images([('image',image) for image in images],return_tensors='pt')
        if not torch.all(inputs['grid_sizes'][:,0]==1):
            raise ValueError('独立图片不能打包为视频')
        text,audit=self.tokenize(conversation,inputs)
        return {**text,**inputs},audit
