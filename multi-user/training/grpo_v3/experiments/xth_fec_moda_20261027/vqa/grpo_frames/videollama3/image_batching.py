"""独立图片的压缩掩码恒为真，绕过恒等操作以沿用原评测batch=4。"""
import torch


def enable_image_batching(model):
    original=model.config.use_token_compression
    model.config.use_token_compression=False
    return {'original_use_token_compression':original,'effective_use_token_compression':False,
            'reason':'image_only_compression_masks_are_identity_no_visual_tokens_removed'}


def collate_images(examples,pad_token_id):
    if not examples or any(not x.get('modals') or any(m!='image' for m in x['modals']) for x in examples):
        raise ValueError('仅支持独立图片，不能把视频放入图片恒等压缩批处理')
    max_len=max(x['input_ids'].shape[-1] for x in examples)
    ids=torch.full((len(examples),max_len),pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,example in enumerate(examples):
        if example['input_ids'].shape[0]!=1:raise ValueError('每个待合并输入必须只有一道题')
        length=example['input_ids'].shape[-1]
        ids[i,-length:]=example['input_ids'][0]
        mask[i,-length:]=example['attention_mask'][0]
    result={'input_ids':ids,'attention_mask':mask,
            **{k:torch.cat([x[k] for x in examples],dim=0) for k in ['pixel_values','grid_sizes','merge_sizes']},
            'modals':[m for x in examples for m in x['modals']]}
    if not torch.all(result['grid_sizes'][:,0]==1):raise ValueError('图片网格时间维必须为1')
    return result
