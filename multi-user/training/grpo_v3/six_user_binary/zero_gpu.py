"""登录节点零GPU检查；禁止加载模型权重，保留真实输入与CLI检查证据。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
import json
import os
import shlex
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time
import math


def normalize_scheduler_kwargs(value):
    """跳过GPU初始化的CLI解析可能保留JSON字符串，按同一字典严格核验。"""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('学习率调度附加参数必须是JSON对象或字典')
    return value


def compiler_check(directory):
    """实际执行C/C++编译器，防止仅发现名字却不能编译或加载。"""
    result={}
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for variable,defaults,suffix in [('CC',('cc','gcc','clang'),'c'),('CXX',('c++','g++','clang++'),'cpp')]:
        configured=os.environ.get(variable)
        command=shlex.split(configured) if configured else []
        if not command:
            executable=next((shutil.which(name) for name in defaults if shutil.which(name)),None)
            if not executable:raise FileNotFoundError(f'{variable}: 缺少可执行的C/C++编译器')
            command=[executable]
        source=directory/('compiler_check.'+suffix)
        binary=directory/('compiler_check_'+suffix)
        source.write_text('int main(void) { return 0; }\n',encoding='utf-8')
        compiled=subprocess.run([*command,str(source),'-o',str(binary)],capture_output=True,text=True,timeout=90)
        if compiled.returncode:raise RuntimeError(f'{variable}编译失败：{compiled.stderr}')
        subprocess.run([str(binary)],check=True,timeout=10)
        include=sysconfig.get_path('include')
        if not (Path(include)/'Python.h').is_file():raise FileNotFoundError(Path(include)/'Python.h')
        shared_source=directory/('load_check.'+suffix)
        shared=directory/('load_check_'+suffix+'.so')
        body='#include <Python.h>\nint egoqa_answer(void) { return 42; }\n'
        if suffix=='cpp':body='#include <Python.h>\n#include <vector>\nextern "C" int egoqa_answer(void) { std::vector<int> x(2); return 40+x.size(); }\n'
        shared_source.write_text(body,encoding='utf-8')
        compiled=subprocess.run([*command,'-shared','-fPIC','-I'+include,str(shared_source),'-o',str(shared)],capture_output=True,text=True,timeout=90)
        if compiled.returncode:raise RuntimeError(f'{variable}共享库编译失败：{compiled.stderr}')
        import ctypes
        if ctypes.CDLL(str(shared)).egoqa_answer()!=42:raise RuntimeError('编译产物加载结果不正确')
        result[variable]={'command':command,'compiled_and_executed':True,'shared_library_loaded':True,'python_headers':include}
    return result


def decode_images(paths):
    from PIL import Image
    for path in paths:
        with Image.open(path) as image:
            image.load()
            image.convert('RGB').load()
    return len(paths)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['environment','data','processor','checkpoint'],required=True)
    parser.add_argument('--role',choices=['train','judge'],default='train')
    args=parser.parse_args()
    c=json.loads(args.config.read_text(encoding='utf-8'))
    report={'status':'running','role':args.role,'mode':args.mode,'python':sys.executable,'started_epoch':time.time(),'checks':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    save()
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('零GPU检查必须显式隐藏CUDA设备')
        from .workflow import training_config
        run=training_config(c,job_id='0',phase='formal',max_steps=c['formal_max_steps'])
        if args.mode=='environment':
            packages=['torch','transformers','vllm','qwen_vl_utils','PIL','safetensors','ninja']
            if args.role=='train':packages+=['swift','peft','trl','fla']
            for name in packages:
                module=importlib.import_module(name)
                report['checks']['import_'+name]=getattr(module,'__file__','built-in')
                save()
            import torch
            if torch.cuda.is_available():raise RuntimeError('零GPU检查意外看到CUDA设备')
            result=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
            report['checks']['pip_check']={'returncode':result.returncode,'output':result.stdout+result.stderr}
            result.check_returncode()
            for name,version_arg in [('ninja','--version'),('ffmpeg','-version'),('ffprobe','-version')]:
                executable=shutil.which(name)
                if not executable:raise FileNotFoundError(name)
                result=subprocess.run([executable,version_arg],text=True,capture_output=True,check=True)
                report['checks'][name]={'path':executable,'version':result.stdout.splitlines()[0]}
            report['checks']['compiler_paths']={k:shutil.which(k) for k in ('gcc','g++','nvcc')}
            report['checks']['host_compilers']=compiler_check(args.output.parent/args.role)
            from vllm import SamplingParams
            from vllm.engine.arg_utils import EngineArgs
            from .predictor import engine_options, validate_config
            judge=json.loads(Path(c['judge_config']).read_text())
            EngineArgs(**engine_options(validate_config(judge)))
            SamplingParams(temperature=0.,max_tokens=1,logprob_token_ids=[1,2])
            report['checks']['vllm_engine_arguments']='accepted_without_engine_or_weight_loading'
            report['checks']['flashinfer_sampler']=os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')
            if os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')!='0':raise RuntimeError('FlashInfer sampler开关不一致')
            if args.role=='train':
                from unittest.mock import patch
                from transformers import HfArgumentParser
                from swift.arguments import RLHFArguments
                from .launch import swift_command
                command=swift_command(run,str(args.output.parent/'planned_training_output'))
                # 只用正式dataclass解析全部参数；不执行会初始化分布式/GPU的post_init。
                with patch.object(RLHFArguments,'__post_init__',lambda self:None):
                    parsed=HfArgumentParser(RLHFArguments).parse_args_into_dataclasses(command[2:])[0]
                if parsed.max_steps!=run['max_steps'] or parsed.eval_steps!=run.get('eval_steps',run['max_steps']):raise ValueError('训练步数或验证频率错误')
                for key in ('learning_rate','beta','temperature','warmup_steps'):
                    if key in run and getattr(parsed,key)!=run[key]:raise ValueError(f'{key}未正确传入实际CLI')
                if str(getattr(parsed.lr_scheduler_type,'value',parsed.lr_scheduler_type))!=run.get('lr_scheduler_type','constant'):
                    raise ValueError('学习率调度类型未正确传入实际CLI')
                scheduler_kwargs=normalize_scheduler_kwargs(parsed.lr_scheduler_kwargs)
                if 'lr_scheduler_kwargs' in run and scheduler_kwargs!=run['lr_scheduler_kwargs']:
                    raise ValueError('学习率下限未正确传入实际CLI')
                if 'stop_after_steps' in run and 'egoqa_stage_stop' not in parsed.callbacks:
                    raise ValueError('分阶段正常结束回调未正确传入实际CLI')
                if parsed.max_completion_length!=run['max_completion_length']:raise ValueError('生成长度上限未正确传入CLI')
                if c.get('resume_from_checkpoint') and parsed.resume_from_checkpoint!=c['resume_from_checkpoint']:
                    raise ValueError('恢复checkpoint未正确传入CLI')
                report['checks']['typed_cli']={'max_steps':parsed.max_steps,'eval_steps':parsed.eval_steps,'use_vllm':parsed.use_vllm,
                    'max_completion_length':parsed.max_completion_length,'resume_from_checkpoint':parsed.resume_from_checkpoint,
                    'learning_rate':parsed.learning_rate,'lr_scheduler_type':str(getattr(parsed.lr_scheduler_type,'value',parsed.lr_scheduler_type)),
                    'warmup_steps':parsed.warmup_steps,'lr_scheduler_kwargs':scheduler_kwargs,'callbacks':parsed.callbacks}
        elif args.mode=='checkpoint':
            import torch
            from .validate_run import finite_updated_adapter
            checkpoint=Path(c['resume_from_checkpoint'])
            state=json.loads((checkpoint/'trainer_state.json').read_text())
            step=state.get('global_step')
            if not isinstance(step,int) or not 0<step<c['formal_max_steps']:
                raise ValueError(f'续训步数不符：{checkpoint} global_step={step}')
            optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True)
            schedule=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=True)
            values=optimizer.get('state',{})
            if not values:raise ValueError('checkpoint缺少优化器状态')
            for value in values.values():
                for key,tensor in value.items():
                    if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
                        raise ValueError('优化器状态非有限：'+key)
                if int(value.get('step',-1))!=step:raise ValueError('优化器步数与trainer_state不一致')
            if schedule.get('last_epoch')!=step:raise ValueError('调度器步数与trainer_state不一致')
            if not finite_updated_adapter(checkpoint/'adapter_model.safetensors'):
                raise ValueError('LoRA权重缺少有限非零更新')
            for name in ('rng_state.pth','training_args.bin','adapter_config.json','args.json'):
                if not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size:raise FileNotFoundError(checkpoint/name)
            report['checks'].update(global_step=step,optimizer_states_loaded=len(values),
                optimizer_tensors_finite=True,scheduler_step=schedule['last_epoch'],finite_updated_adapter=True)
        elif args.mode=='data':
            from PIL import Image
            from .data import read_rows,validate_row,validate_splits
            split={k:read_rows(Path(c['prepared_data_root'])/(k+'.jsonl')) for k in ('train','validation')}
            validate_splits(split)
            if {k:len(v) for k,v in split.items()}!={'train':18,'validation':6}:raise ValueError('训练/验证数量错误')
            for rows in split.values():
                for row in rows:validate_row(row)
            files=set()
            for rows in split.values():
                for row in rows:
                    packet=Path(row['dataset_root'])/'packets'/row['source_packet_id']
                    data=json.loads((packet/'packet.json').read_text())
                    for user in data['users']:
                        for frame in user['frames']:files.add(packet/frame['path'])
            count=decode_images(sorted(files))
            report['checks'].update(rows={k:len(v) for k,v in split.items()},decoded_image_files=count,validation_unchanged=True)
        else:
            import torch
            torch.set_num_threads(2)
            from .data import read_rows
            row=read_rows(run['val_dataset'])[0]
            if args.role=='train':
                from .evaluation import policy_template
                template=policy_template(run)
                template.set_mode('train')
                request=deepcopy(row)
                bounded=c.get('processor_check_mode')=='bounded_cpu'
                if bounded:
                    from training.judge_sft.collator import qwen_vision_geometry
                    geometry=qwen_vision_geometry(template.processor)
                    per_image=math.ceil(run['max_pixels']/geometry['merged_token_pixel_area'])+8
                    bounds=[]
                    for split in ('train','validation'):
                        for item in read_rows(Path(c['prepared_data_root'])/(split+'.jsonl')):
                            content=item['messages'][0]['content'].replace('<image>','')
                            text_tokens=len(template.processor.tokenizer.encode(content,add_special_tokens=False))
                            bound=text_tokens+len(item['images'])*per_image+256+run['max_completion_length']
                            if bound>run['max_length']:raise ValueError(f'保守输入长度上界超限：{item["evidence_id"]} {bound}')
                            bounds.append({'evidence_id':item['evidence_id'],'image_count':len(item['images']),'bound_including_completion':bound})
                    report['checks']['full_media_input_bounds']=bounds
                    request['images']=row['images'][:2]
                    request['messages'][0]['content']='<image>\n<image>\n'+row['messages'][0]['content'].replace('<image>','')
                request['messages'].append({'role':'assistant','content':'CPU validation fixture.'})
                from swift.dataset.preprocessor.core import RowPreprocessor
                RowPreprocessor._cast_mm_data(request)
                report['checks']['framework_image_representation']=type(request['images'][0]).__name__
                encoded=[template.encode(deepcopy(request)) for _ in range(run['num_generations'])]
                batch=template.data_collator(encoded)
                length=int(batch['input_ids'].shape[-1])
                if length+run['max_completion_length']>run['max_length']:
                    raise ValueError(f'实际验证输入加生成预算超过上限：{length}+{run["max_completion_length"]}>{run["max_length"]}')
                key='four_candidate_component_encoding' if bounded else 'actual_four_candidate_encoding'
                report['checks'][key]={k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')}
                report['checks']['images_per_candidate']=len(request['images'])
                if c.get('policy_image_cache_gb'):
                    from .policy_image_cache import policy_cache_metrics
                    cache_metrics=policy_cache_metrics()
                    if not cache_metrics or cache_metrics['hits'] < len(request['images']):
                        raise ValueError('Policy真实四候选编码未观察到图像缓存复用')
                    report['checks']['policy_image_cache']=cache_metrics
                if bounded:report['checks']['boundary']='当前实际四候选编码验证新文本与两张真实图片；完整媒体长度另作保守上界核对，完整GPU显存与算子仍须运行时核验'
            else:
                from transformers import AutoProcessor
                from qwen_vl_utils import process_vision_info
                from training.judge_sft.collator import _apply_chat_template,qwen_vision_geometry
                processor=AutoProcessor.from_pretrained(c['model'],local_files_only=True,trust_remote_code=True)
                geometry=qwen_vision_geometry(processor)
                messages=[{'role':'user','content':[{'type':'image','image':row['images'][0],'max_pixels':24576}, {'type':'text','text':'CPU image encoding check.'}]}]
                text=_apply_chat_template(processor,messages)
                images,_=process_vision_info(messages,image_patch_size=geometry['patch_size'])
                batch=processor(text=[text]*4,images=images*4,padding=True,return_tensors='pt')
                report['checks']['four_request_image_component_encoding']={k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')}
                report['checks']['boundary']='Judge检查四请求图片组件；1800帧完整服务调度仍由原最小GPU smoke验证'
        report.update(status='passed',finished_epoch=time.time())
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_epoch=time.time())
        raise
    finally:save()


if __name__=='__main__':main()
