"""同一验证输入与种子下，独立比较基座和最终 LoRA；不要求指标必须上升。"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import time

from .data import read_rows
from .reward import aggregate
from .service import JudgeClient


def configure_engine_process(c, environ):
    """评分引擎保留隔离进程，避免与父进程保护线程并发捕获。"""
    if 'in_proj_qkv' in c.get('lora_target_modules', []):
        environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '1'


def configure_engine_kwargs(c, options):
    options = dict(options)
    if 'in_proj_qkv' in c.get('lora_target_modules', []):
        options['worker_extension_cls'] = (
            'training.grpo_v3.six_user_binary.packed_worker_extension.PartialPackedLoRAWorkerExtension')
    return options


def same_frozen_judge(baseline, current):
    if baseline.get('frozen') is not True or current.get('frozen') is not True:
        return False
    return {k:v for k,v in baseline.items() if k!='instance_id'} == {k:v for k,v in current.items() if k!='instance_id'}


def summarize(rows):
    if not rows or any(not math.isfinite(r['reward']) for r in rows):
        raise ValueError('验证奖励必须非空且有限')
    scored=[r for r in rows if r['status']=='scored']
    keys=('formality','groundedness','speaker_only','all_six')
    return {'candidate_count':len(rows),'scored_count':len(scored),'valid_rate':len(scored)/len(rows),
            'mean_reward':statistics.mean(r['reward'] for r in rows),
            'mean_probabilities_scored_only':{k:statistics.mean(r['probabilities'][k] for r in scored if k in r['probabilities'])
                if any(k in r['probabilities'] for r in scored) else None for k in keys}}


def compare_rows(baseline, policy):
    def indexed(rows):
        result={(r['evidence_id'],r['slot']):r for r in rows}
        if len(result)!=len(rows):raise ValueError('验证候选重复')
        return result
    b,p=indexed(baseline),indexed(policy)
    if not b or set(b)!=set(p):raise ValueError('baseline与Policy的验证候选不能完整配对')
    if any(b[k]['seed']!=p[k]['seed'] for k in b):raise ValueError('验证生成种子不一致')
    delta=statistics.mean(p[k]['reward']-b[k]['reward'] for k in b)
    ids=sorted({k[0] for k in b})
    return {'baseline':summarize(baseline),'policy':summarize(policy),'reward_delta':delta,
            'improved':delta>0,'paired_candidate_count':len(b),'paired_input_count':len(ids),
            'per_input':[{'evidence_id':e,'reward_delta':statistics.mean(p[k]['reward']-b[k]['reward'] for k in b if k[0]==e)} for e in ids],
            'boundary':'固定验证上的冻结Judge代理指标，不是独立测试或人工质量结论；同一视频窗口的提问者输入并非独立视频样本'}


def policy_template(c):
    from .policy_image_cache import configured_cache_bytes, install_policy_image_cache
    cache_bytes = configured_cache_bytes(c.get('policy_image_cache_gb'))
    if cache_bytes:
        install_policy_image_cache(max_bytes=cache_bytes)
    import torch
    from swift.model import get_model_processor
    from swift.template import get_template
    _,processor=get_model_processor(c['policy_model'],load_model=False,torch_dtype=torch.bfloat16,attn_impl='sdpa')
    return get_template(processor,max_length=c['max_length'],max_pixels=c['max_pixels'],
                        enable_thinking=False,remove_unused_columns=False,truncation_strategy='raise')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--judge-url',required=True)
    parser.add_argument('--judge-instance',required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    c=json.loads(args.config.read_text(encoding='utf-8'))
    configure_engine_process(c, os.environ)
    if 'in_proj_qkv' in c.get('lora_target_modules', []):
        from .packed_lora_compat import install
        install()
    from .utilization_runtime import enable_training_guard
    enable_training_guard()
    rows=read_rows(c['val_dataset'])
    template=policy_template(c)
    from swift.infer_engine import VllmEngine, InferRequest, RequestConfig
    import torch
    if args.adapter and not (args.adapter/'adapter_model.safetensors').is_file():
        raise FileNotFoundError(args.adapter/'adapter_model.safetensors')
    settings={k:c[k] for k in ('policy_model','max_length','max_completion_length','max_pixels','temperature','top_p','top_k','num_generations_eval','reward_mode')}
    settings['validation_seed']=c.get('validation_seed',42)
    bindings=[{k:r[k] for k in ('evidence_id','source_packet_id','asker_index','messages','images')} for r in rows]
    record={'status':'running','settings':settings,'input_bindings':bindings,'adapter':str(args.adapter) if args.adapter else None,
            'started_epoch':time.time(),'rows':[]}
    record['runtime']={'partial_qkv_compat':'in_proj_qkv' in c.get('lora_target_modules', []),
                       'v1_multiprocessing_environment':os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    client=JudgeClient(args.judge_url,expected_instance=args.judge_instance)
    record['judge']=client.health()
    def save():args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    save()
    try:
        engine=VllmEngine(c['policy_model'],template=template,torch_dtype=torch.bfloat16,
            enable_sleep_mode=bool(c.get('shared_gpu')),
            adapters=[str(args.adapter)] if args.adapter else None,enable_lora=True,max_lora_rank=c.get('lora_rank',8),
            max_model_len=c['max_length'],max_num_seqs=c['num_generations_eval'],
            gpu_memory_utilization=c.get('vllm_gpu_memory_utilization',.55),tensor_parallel_size=1,
            enable_prefix_caching=True,mm_processor_cache_gb=4,limit_mm_per_prompt={'image':1800},
            seed=settings['validation_seed'],engine_kwargs=configure_engine_kwargs(c,
                {'enable_chunked_prefill':True,'max_num_batched_tokens':8192,
                 'additional_config':{'gdn_prefill_backend':'triton'}}))
        record['runtime']['engine_core_client_class']=type(engine.engine.engine_core).__name__
        save()
        for index,row in enumerate(rows):
            seed=settings['validation_seed']+index
            request=InferRequest(messages=deepcopy(row['messages']),images=list(row['images']))
            decoding=RequestConfig(n=c['num_generations_eval'],seed=seed,max_tokens=c['max_completion_length'],
                                   temperature=c['temperature'],top_p=c['top_p'],top_k=c['top_k'])
            responses=engine.infer([request],decoding,use_tqdm=False)
            if len(responses)!=1 or len(responses[0].choices)!=c['num_generations_eval']:
                raise RuntimeError('验证生成候选数量不一致')
            requests=[]
            for slot,choice in enumerate(responses[0].choices):
                requests.append({**row,'request_id':f'{args.output.stem}:{index}:{slot}','completion':choice.message.content})
            from .shared_gpu import evaluation_judge_window
            with evaluation_judge_window(client, engine) if c.get('shared_gpu') else nullcontext():
                judged=client.score_many(requests)
            for slot,(request,result) in enumerate(zip(requests,judged)):
                reward=aggregate(result['probabilities'],mode=c['reward_mode']) if result['status']=='scored' else 0.
                record['rows'].append({'evidence_id':row['evidence_id'],'slot':slot,'seed':seed,
                    'completion':request['completion'],'status':result['status'],'reward':reward,
                    'probabilities':result.get('probabilities',{}),'judge_result':result})
            save()
        record.update(status='completed',summary=summarize(record['rows']),finished_epoch=time.time())
        if args.baseline:
            baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
            if baseline['status']!='completed' or baseline['adapter'] is not None:
                raise ValueError('baseline必须为完整的未训练基座验证')
            if baseline['settings']!=settings or baseline['input_bindings']!=bindings or not same_frozen_judge(baseline['judge'],record['judge']):
                raise ValueError('baseline与Policy的验证设置、输入或Judge不一致')
            record['comparison']=compare_rows(baseline['rows'],record['rows'])
            (args.output.parent/'validation_comparison.json').write_text(json.dumps(record['comparison'],ensure_ascii=False,indent=2),encoding='utf-8')
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_epoch=time.time())
        raise
    finally:save()


if __name__=='__main__':main()
