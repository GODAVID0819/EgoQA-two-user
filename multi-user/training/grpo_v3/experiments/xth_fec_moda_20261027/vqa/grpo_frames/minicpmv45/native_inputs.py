"""调用官方文字/占位符转换；视觉编码可缓存，题目和答案不进入视觉缓存。"""
import torch

def make_prompt(tokenizer,paths,text):
    content='\n'.join(['(<image>./</image>)']*len(paths)+[text])
    return tokenizer.apply_chat_template([{'role':'user','content':content}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)

def text_inputs(processor,records):
    sizes=[];temporal=[];skips=[];prompts=[]
    for record in records:
        paths=record['paths'];groups=record['groups']
        if [p for g in groups for p in g['paths']]!=paths:
            raise ValueError('视觉组与原始帧顺序不一致')
        pos=0;skip=[]
        for group in groups:
            skip.extend(range(pos+1,pos+len(group['paths'])))
            pos+=len(group['paths'])
        prompts.append(make_prompt(processor.tokenizer,paths,record['text']))
        sizes.append([(156,156)]*len(paths))
        temporal.append([g['temporal_ids'] for g in groups]);skips.append(skip)
    image_inputs={'pixel_values':[[] for _ in records], 'image_sizes':sizes,
        'tgt_sizes':[[] for _ in records], 'temporal_ids':temporal,'skip_image_idx':skips}
    result=processor._convert_images_texts_to_inputs(image_inputs,prompts,
        max_slice_nums=1,use_image_id=False,max_length=None)
    for record,bounds,mask in zip(records,result['image_bound'],result['attention_mask']):
        if len(bounds)!=len(record['groups']) or not torch.all(bounds[:,1]-bounds[:,0]==64):
            raise ValueError('官方处理器生成的视觉占位符数量不符')
        if int(mask.sum())+512>40960:raise ValueError('超过模型原生上下文；禁止截断')
    # 缓存特征已包含时间编码，生成阶段不再执行视觉编码；避免官方.to递归裸整数。
    result.pop('temporal_ids')
    result.pop('tgt_sizes')
    return result
