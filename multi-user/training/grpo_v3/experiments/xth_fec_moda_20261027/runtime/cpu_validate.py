"""在Torch CPU计算节点验证受影响的实际包、媒体、采样器及CLI。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import importlib

ROOT=Path(__file__).resolve().parent


def main():
    if 'SLURM_JOB_ID' not in os.environ or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('必须在显式禁用GPU的Slurm CPU作业执行')
    sys.path[:0]=[str(ROOT/'runtime_deps_train'),str(ROOT/'multi-user'),str(ROOT)]
    from training.torch_storage_preflight import REQUIRED_STORAGE_VARIABLES,validate_storage_environment
    scratch=Path('/scratch/xl6775/job_scratch')/('grpo720_cpu_'+os.environ['SLURM_JOB_ID'])
    for name in REQUIRED_STORAGE_VARIABLES+('PIP_CACHE_DIR',):os.environ[name]=str(scratch/name.lower())
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',VLLM_NO_USAGE_STATS='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONPATH=os.pathsep.join(map(str,[ROOT/'runtime_deps_train',ROOT/'multi-user',ROOT])))
    tools=json.loads((ROOT/'preflight/media_tools.json').read_text())
    os.environ['PATH']=str(Path(tools['ffmpeg']).parent)+os.pathsep+str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['LD_LIBRARY_PATH']=tools['ffmpeg_lib']+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    report=validate_storage_environment(allowed_root=scratch,environ=os.environ)
    (ROOT/'preflight/cpu_validation_storage.json').write_text(json.dumps(report,indent=2))
    if report['status']!='passed':raise RuntimeError('CPU验证scratch封闭失败')
    dependencies=ROOT/'runtime_deps_train'
    if not (dependencies/'onnxruntime/__init__.py').is_file():
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(dependencies),
                        'onnxruntime==1.31.0','coloredlogs','humanfriendly','flatbuffers'],check=True,env=os.environ.copy())
    importlib.invalidate_caches()
    import onnxruntime
    (ROOT/'preflight/train_encoder_dependencies.json').write_text(json.dumps({
        'status':'passed','python':sys.version,'onnxruntime':onnxruntime.__version__,
        'module':onnxruntime.__file__,'dependency_directory':str(dependencies)},indent=2))
    started=time.time()
    subprocess.run([sys.executable,str(ROOT/'make_configs.py')],check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT),'-p','test_moda*.py','-v'],check=True)
    config_path=ROOT/'configs/train.json'
    c=json.loads(config_path.read_text())
    data_path=ROOT/'preflight/data_720_cpu.json';sampler_path=ROOT/'preflight/dual_sampler_720_audit.json'
    reused_data=False
    if data_path.exists() and sampler_path.exists() and json.loads(data_path.read_text()).get('status')=='passed' and json.loads(sampler_path.read_text()).get('status')=='passed':
        from prepare_720_data import assemble_720,verify_reused_conditions
        read=lambda path:[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        expected,_=assemble_720(read(c['source_train_dataset']),read(c['source_validation_dataset']),json.loads(Path(c['source_condition_definition']).read_text()))
        verify_reused_conditions(read(c['train_dataset']),expected)
        ordered=json.loads((ROOT/'condition_definition.json').read_text())['train']
        from dual_runtime import condition_binding
        assert [condition_binding(r) for r in ordered]==[condition_binding(r) for r in read(c['train_dataset'])]
        sampler=json.loads(sampler_path.read_text())
        assert sampler['cpu_parser']['max_steps']==360 and sampler['shard']['global_unique_conditions']==720 and sampler['shard']['global_training_candidates']==2880
        reused_data=True
        print('REUSED_PASSED_MEDIA_AND_SAMPLER_AUDIT_ALL_720_PAYLOADS_EQUAL',flush=True)
    else:
        subprocess.run([sys.executable,str(ROOT/'prepare_720_data.py'),'--config',str(config_path)],check=True)
        subprocess.run([sys.executable,str(ROOT/'audit_dual_sampler.py'),'--definition',str(ROOT/'condition_definition.json'),
                        '--output',str(sampler_path)],check=True)
        c=json.loads(config_path.read_text())
    from history_stage import build_fixed_history
    history_report=build_fixed_history(c)
    rows=[json.loads(line) for line in Path(c['train_dataset']).read_text().splitlines() if line.strip()]
    from dual_runtime import condition_binding
    first=condition_binding(rows[0])[0]
    sample=sorted([r for r in rows if condition_binding(r)[0]==first],key=lambda r:r['asker_index'])
    if len(sample)!=6:raise ValueError('CPU实际输入测试须覆盖同窗全部6个speaker')
    from training.grpo_v3.six_user_binary.judge import build_examples
    from training.judge_sft.collator import model_visible_prompt,_apply_chat_template,_assert_thinking_disabled
    from training.judge_sft.contracts import VERDICT_ASSISTANT_PREFIX
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(c['policy_model'],local_files_only=True,trust_remote_code=True)
    previews=[];decode_frames=[]
    qa={'question':'What was next to my cup?','options':['book','bag','plate','bowl','hat'],'correct':'A','answer':'book'}
    for row in sample:
        request={**row,'request_id':'CPU_ROUTING_FIXTURE_'+str(row['asker_index']),'completion':json.dumps(qa)}
        examples=build_examples(request)
        if list(examples)!=['formality','groundedness','combined_answerability']:raise ValueError('实际Judge仍走旧四路')
        for key,example in examples.items():
            visible=model_visible_prompt(example)
            content=[{'type':'image','image':p,'min_pixels':3136,'max_pixels':24576} for p in example.frames]
            content.append({'type':'text','text':visible})
            rendered=_apply_chat_template(processor,[{'role':'user','content':content}]);_assert_thinking_disabled(rendered)
            if key=='formality':assert example.frame_count==0
            else:
                assert len(example.frame_sets)==6 and all(len(s.frames)==300 for s in example.frame_sets)
                assert example.frame_sets[0].label=='speaker: '+row['required_users'][0]
                decode_frames=list(example.frames[:8])
            previews.append({'speaker_index':row['asker_index'],'evidence_id':row['evidence_id'],'task':key,
                'frame_count':example.frame_count,'frame_labels':[s.label for s in example.frame_sets],
                'model_visible_text':visible,'assistant_prefix':VERDICT_ASSISTANT_PREFIX,'test_only_not_model_score':True})
    (ROOT/'preflight/model_visible_six_speakers.json').write_text(json.dumps(previews,ensure_ascii=False,indent=2))
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    content=[{'type':'image','image':p,'min_pixels':3136,'max_pixels':24576} for p in decode_frames]
    decoded,_=process_vision_info([{'role':'user','content':content}],image_patch_size=16)
    assert len(decoded)==8 and all(image.width>0 and image.height>0 for image in decoded)
    from training.grpo_v3.six_user_binary.history_pool import MiniLMEncoder
    import numpy as np
    embeddings=MiniLMEncoder(ROOT/'encoder').encode(['What was next to my cup?','What was next to my cup?','Where did I leave my bag?'])
    assert embeddings.shape==(3,384) and np.isfinite(embeddings).all() and np.max(np.abs(embeddings[0]-embeddings[1]))<1e-5
    # Judge与Policy环境分别导入自己的真实配置和编码器依赖，不实例化GPU模型。
    probe="import sys; from training.grpo_v3.six_user_binary.predictor import validate_config,engine_options; import json; c=validate_config(json.load(open(sys.argv[1]))); o=engine_options(c); assert o['max_lora_rank']==128 and o['max_cpu_loras']==3 and o['max_loras']==1; print('ENVIRONMENT_OK',sys.version)"
    for python in dict.fromkeys([c['train_python'],c['judge_python']]):
        subprocess.run([python,'-c',probe,c['judge_config']],check=True,env=os.environ.copy())
    # CLI帮助由真实安装解析器生成，检查本轮显式新增的参数确实存在。
    help_result=subprocess.run([str(Path(c['train_python']).parent/'swift'),'rlhf','--help'],capture_output=True,text=True,env=os.environ.copy())
    (ROOT/'preflight/swift_rlhf_help.txt').write_text(help_result.stdout+help_result.stderr)
    help_result.check_returncode()
    for flag in ['--lora_dropout','--epsilon','--num_iterations','--loss_type','--eval_strategy']:
        if flag not in help_result.stdout:raise ValueError('真实Swift CLI缺少参数：'+flag)
    summary={'status':'passed','job_id':os.environ['SLURM_JOB_ID'],'GPU_operations':0,'elapsed_seconds':time.time()-started,
             'actual_data_conditions':720,'real_sampler_global_candidates':2880,'model_visible_examples':len(previews),
             'decoded_real_frames':8,'MiniLM_shape':list(embeddings.shape),'new_E_GPU_load_tested':False,
             'history_selected_QA':history_report['selected_QA'],'history_empty_windows':history_report['empty_windows'],
             'historical_rescoring_performed':False,'passed_media_sampler_audits_reused':reused_data,
             'boundary':'Torch CPU实际链路通过；历史池直接用旧分数，新的Judge GPU评分仅用于正式训练新生成QA'}
    (ROOT/'preflight/cpu_validation_complete.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('CPU_VALIDATION_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__': main()
