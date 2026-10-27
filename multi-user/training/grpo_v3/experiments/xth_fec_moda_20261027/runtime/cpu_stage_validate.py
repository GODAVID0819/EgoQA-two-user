"""本轮CPU实际检查；不加载模型权重，不创建CUDA上下文。"""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent


def setup():
    sys.path[:0]=[str(ROOT/'runtime_deps_train'),str(ROOT/'multi-user'),str(ROOT)]
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('必须在隐藏GPU的Slurm CPU作业运行')
    from training.torch_storage_preflight import REQUIRED_STORAGE_VARIABLES,validate_storage_environment
    scratch=Path('/scratch/xl6775/job_scratch')/('b50176_cpu_'+os.environ['SLURM_JOB_ID'])
    for name in REQUIRED_STORAGE_VARIABLES+('PIP_CACHE_DIR',):os.environ[name]=str(scratch/name.lower())
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',VLLM_NO_USAGE_STATS='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='8',
        PYTHONPATH=os.pathsep.join(map(str,[ROOT/'runtime_deps_train',ROOT/'multi-user',ROOT])))
    tools=json.loads((ROOT/'preflight/media_tools.json').read_text())
    os.environ['PATH']=str(Path(tools['ffmpeg']).parent)+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['LD_LIBRARY_PATH']=tools['ffmpeg_lib']+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    storage=validate_storage_environment(allowed_root=scratch,environ=os.environ)
    (ROOT/'preflight/cpu_storage.json').write_text(json.dumps(storage,indent=2))
    assert storage['status']=='passed'
    return scratch


def scheduler_and_sampler(c,rows,scratch):
    import torch
    from transformers.optimization import get_scheduler
    from torch.utils.data import BatchSampler
    from accelerate.data_loader import BatchSamplerShard
    from trl.trainer.utils import RepeatSampler
    def make():
        parameter=torch.nn.Parameter(torch.zeros(1))
        optimizer=torch.optim.AdamW([parameter],lr=c['learning_rate'])
        scheduler=get_scheduler(c['lr_scheduler_type'],optimizer,num_warmup_steps=c['warmup_steps'],
            num_training_steps=c['max_steps'],scheduler_specific_kwargs=c['lr_scheduler_kwargs'])
        return parameter,optimizer,scheduler
    def steps(p,o,s,count):
        result=[]
        for _ in range(count):
            p.grad=torch.ones_like(p);o.step();s.step();o.zero_grad();result.append(s.get_last_lr()[0])
        return result
    p,o,s=make();continuous=steps(p,o,s,360)
    p1,o1,s1=make();first=steps(p1,o1,s1,180)
    fixture=scratch/'scheduler_fixture.pt'
    torch.save({'parameter':p1.detach(),'optimizer':o1.state_dict(),'scheduler':s1.state_dict()},fixture)
    restored=torch.load(fixture,map_location='cpu',weights_only=False)
    p2,o2,s2=make();p2.data.copy_(restored['parameter'])
    o2.load_state_dict(restored['optimizer']);s2.load_state_dict(restored['scheduler'])
    assert s2.last_epoch==180 and {float(v['step']) for v in o2.state.values()}=={180.}
    split=first+steps(p2,o2,s2,180)
    assert continuous==split and torch.equal(p,p2)
    assert abs(split[-1]-1e-6)<1e-12 and s2.last_epoch==360
    visits=[];remaining=[]
    for rank in range(2):
        sampler=RepeatSampler(rows,mini_repeat_count=4,batch_size=2,repeat_count=4,shuffle=False,seed=42)
        batches=list(BatchSamplerShard(BatchSampler(sampler,batch_size=4,drop_last=False),
            num_processes=2,process_index=rank,split_batches=False,even_batches=True))
        assert len(batches)==1440 and all(len(b)==4 and len(set(b))==1 for b in batches)
        indexes=[b[0] for b in batches[::4]];visits.append(indexes)
        assert indexes==list(range(rank,720,2))
        remaining.append([b[0] for b in batches[720:][::4]])
    assert [r[0] for r in remaining]==[360,361]
    assert sorted(visits[0]+visits[1])==list(range(720))
    return {'status':'passed','fixture_only':True,'continuous_and_split_LR_identical':True,
        'optimizer_and_scheduler_last_step':360,'step180_lr':split[179],'last_lr':split[-1],
        'native_skipped_microbatches_per_rank':720,'native_next_condition_indices':[360,361],
        'first_stage_conditions':360,'full_conditions_once':720}


def input_lengths(c,rows):
    from PIL import Image
    from training.grpo_v3.six_user_binary.evaluation import policy_template
    from dual_runtime import condition_binding
    template=policy_template({**c,'policy_image_cache_gb':0})
    # 仅复用已通过的媒体完整性扫描；新增检查是图像几何与真实tokenizer长度。
    directory_geometry={};grid_by_geometry={};details=[]
    for index,row in enumerate(rows):
        images=row['images']
        assert images==row['generator_image_paths'] and sum(row['generator_frame_counts'])==len(images)
        assert row['generator_frame_counts'][0]==300 and len(row['required_users'])==6
        assert sum(m['content'].count('<image>') for m in row['messages'])==len(images)
        visual_tokens=0
        for path in images:
            directory=str(Path(path).parent)
            if directory not in directory_geometry:
                with Image.open(path) as image:directory_geometry[directory]=list(image.size)
            geometry=tuple(directory_geometry[directory])
            if geometry not in grid_by_geometry:
                probe=template.encode({'messages':[{'role':'user','content':'<image>\ngeometry'}],'images':[path]})
                grid=probe['image_grid_thw'][0].tolist()
                merge=int(template.processor.image_processor.merge_size)
                grid_by_geometry[geometry]={'source_size':list(geometry),'grid_thw':grid,
                    'merge_size':merge,'visual_tokens':int(grid[0]*grid[1]*grid[2]//merge**2)}
            visual_tokens+=grid_by_geometry[geometry]['visual_tokens']
        text=copy.deepcopy(row['messages'])
        for message in text:
            message['content']=message['content'].replace('<image>','<|vision_start|><|image_pad|><|vision_end|>')
        encoded=template.encode({'messages':text})
        prefix=len(encoded['input_ids'])+visual_tokens-len(images)
        assert prefix+c['max_completion_length']<=c['max_length'],f'超长条件{index}: {prefix}'
        details.append({'index':index,'condition':list(condition_binding(row)),'image_count':len(images),
            'prefix_tokens':prefix,'completion_reserve':c['max_completion_length'],'total_tokens':prefix+c['max_completion_length']})
        if index%120==0:print('LENGTH_PROGRESS',index,prefix,flush=True)
    pair=max(range(0,720,2),key=lambda i:(max(details[i]['total_tokens'],details[i+1]['total_tokens']),
        details[i]['total_tokens']+details[i+1]['total_tokens']))
    assert condition_binding(rows[pair])[0]!=condition_binding(rows[pair+1])[0]
    # 对容量测量的真实最长两行完整执行Swift encode，校验缓存几何计数无偏差。
    actual=[]
    for index in [pair,pair+1]:
        print('FULL_REAL_TEMPLATE_ENCODE',index,flush=True)
        encoded=template.encode(copy.deepcopy(rows[index]))
        length=len(encoded['input_ids']);assert length==details[index]['prefix_tokens'],(index,length,details[index])
        actual.append({'index':index,'actual_prefix_tokens':length,
            'grid_shapes':sorted({tuple(x) for x in encoded['image_grid_thw'].tolist()})})
        del encoded
    result={'status':'passed','actual_rows':720,'unique_conditions':720,'deleted_rows':0,'truncated_rows':0,
        'same_order_as_source':True,'grid_by_source_geometry':list(grid_by_geometry.values()),
        'frame_directories':len(directory_geometry),'counting_method':'实际Swift文本模板与tokenizer、原生图像网格缓存；最长两行完整真实encode复核',
        'minimum_total_tokens':min(x['total_tokens'] for x in details),
        'maximum_total_tokens':max(x['total_tokens'] for x in details),
        'longest_pair_indices':[pair,pair+1],'longest_pair_real_encodes':actual,'rows':details}
    (ROOT/'preflight/input_lengths.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    measure={**c,'capacity_measurement':True,'train_conditions':2,'stop_after_steps':1,
        'train_dataset':str(ROOT/'data/longest_pair.jsonl'),'output_root':str(ROOT/'outputs/capacity'),
        'walltime_seconds':14400,'save_steps':1,'run_attempt':'isolated_capacity_once'}
    Path(measure['train_dataset']).write_text(''.join(json.dumps(rows[i],ensure_ascii=False)+'\n' for i in [pair,pair+1]))
    (ROOT/'configs/capacity.json').write_text(json.dumps(measure,ensure_ascii=False,indent=2))
    return {key:value for key,value in result.items() if key!='rows'}


def main():
    started=time.time();scratch=setup()
    import torch
    assert not torch.cuda.is_initialized()
    c=json.loads((ROOT/'configs/train.json').read_text())
    from dual_runtime import validate_config,condition_binding
    validate_config(c)
    source=json.loads((ROOT/'preflight/source_cpu_validation_complete.json').read_text())
    assert source['status']=='passed' and source['actual_data_conditions']==720
    rows=[json.loads(line) for line in Path(c['train_dataset']).read_text().splitlines() if line.strip()]
    definition=json.loads((ROOT/'condition_definition.json').read_text())['train']
    assert len(rows)==720 and len({condition_binding(r) for r in rows})==720
    assert [(r['source_packet_id'],r['asker_index']) for r in rows]==[(r['source_packet_id'],r['speaker_index']) for r in definition]
    from training.grpo_v3.six_user_binary.launch import role_environment,swift_command
    role_proof=ROOT/'preflight/startup_roles_passed.json'
    if role_proof.exists():
        proof=json.loads(role_proof.read_text());assert proof['status']=='passed' and set(proof['roles'])=={'train','judge'}
        roles=proof['roles'];print('REUSED_STARTUP_ROLES_FROM_CPU',proof['job_id'],flush=True)
    else:
        roles={}
        for role,module in [('train','training.grpo_v3.six_user_binary.utilization_runtime'),
                            ('judge','training.grpo_v3.six_user_binary.service')]:
            env=role_environment(c[role+'_python'],os.environ.copy())
            env['PYTHONPATH']=os.pathsep.join(map(str,([ROOT/'runtime_deps_train'] if role=='train' else [])+[ROOT/'multi-user',ROOT]))
            subprocess.run([c[role+'_python'],str(ROOT/'startup_dependency_probe.py'),'--root',str(ROOT),'--role',role],env=env,check=True)
            subprocess.run([c[role+'_python'],'-m',module,'--help'],env=env,stdout=subprocess.DEVNULL,check=True)
            roles[role]={'status':'passed','python':c[role+'_python'],'entry':module}
        role_proof.write_text(json.dumps({'status':'passed','job_id':os.environ['SLURM_JOB_ID'],'roles':roles},indent=2))
    command=swift_command({**c,'policy_gpu_indices':[0],'judge_gpu_indices':[0]},str(scratch/'cli'))
    for key,value in {'max_pixels':'50176','max_length':'131072','max_steps':'360',
        'truncation_strategy':'delete','warmup_steps':'10','lr_scheduler_type':'cosine_with_min_lr',
        'num_generations':'4','gradient_accumulation_steps':'4'}.items():
        assert command[command.index('--'+key)+1]==value
    assert '--resume_from_checkpoint' not in command
    (ROOT/'preflight/effective_train_command.json').write_text(json.dumps(command,indent=2))
    from swift.arguments import RLHFArguments
    from transformers import HfArgumentParser
    parsed=HfArgumentParser(RLHFArguments).parse_args_into_dataclasses(command[2:]+['--use_cpu','true'])[0]
    assert (parsed.max_pixels,parsed.max_length,parsed.max_steps)==(50176,131072,360)
    scheduler_kwargs=parsed.lr_scheduler_kwargs
    if isinstance(scheduler_kwargs,str):scheduler_kwargs=json.loads(scheduler_kwargs)
    assert scheduler_kwargs=={'min_lr_rate':.1}
    assert parsed.save_only_model is False and parsed.resume_only_model is False and parsed.ignore_data_skip is False
    # 此安装版本将CLI delete映射为模板raise；strict与720行保留核验阻止默默过滤/重采样。
    assert parsed.get_template_kwargs()['truncation_strategy']=='raise'
    import unittest
    suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test_stage*.py')
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful() and not tests.skipped
    native=scheduler_and_sampler(c,rows,scratch)
    (ROOT/'preflight/scheduler_sampler.json').write_text(json.dumps(native,indent=2))
    lengths=input_lengths(c,rows)
    assert not torch.cuda.is_initialized()
    result={'status':'passed','job_id':os.environ['SLURM_JOB_ID'],'GPU_operations':0,
        'elapsed_seconds':time.time()-started,'tests_passed':tests.testsRun,'startup_import_roles':roles,
        'source_media_scan_reused_job':source['job_id'],'media_scan_repeated':False,
        'actual_conditions':720,'deleted_rows':0,'truncated_rows':0,'phase_resume_fixture_passed':True,
        'actual_CP180_reloaded':False,'input_lengths':lengths,'historical_rescoring_performed':False}
    (ROOT/'preflight/cpu_validation_complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('CPU_STAGE_VALIDATION_PASSED',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        (ROOT/'preflight/cpu_validation_failure.json').write_text(json.dumps({'status':'failed',
            'job_id':os.environ.get('SLURM_JOB_ID'),'error':repr(error),'traceback':traceback.format_exc()},ensure_ascii=False,indent=2))
        raise
