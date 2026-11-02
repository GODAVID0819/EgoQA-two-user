"""最终四窗方案的定向回归测试。"""
import sys
import unittest
import ast
import importlib.util
import importlib
import json
import tempfile
import types
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG='multi-user/training/grpo_v3/six_user_binary'
sys.path.insert(0, str(ROOT/'runtime'/PKG))
SOURCE=ROOT/'runtime'
PKG='multi-user/training/grpo_v3/six_user_binary'


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def test_package(name):
    package=types.ModuleType(name)
    package.__path__=[str(ROOT/'runtime'/PKG),str(SOURCE/PKG)]
    sys.modules[name]=package
    diversity=types.ModuleType(name+'.diversity')
    from v7_batch import condition_key
    diversity.window_id=lambda packet:condition_key({'source_packet_id':packet,'asker_index':0})[0]
    sys.modules[diversity.__name__]=diversity


class FinalDesignTests(unittest.TestCase):
    def test_history_uses_previous_completed_updates(self):
            from v7_history import SpeakerHistory
            history = SpeakerHistory()
            self.assertEqual(history.snapshot()['speaker_lambda'], 0)
            history.stage(1, [1, 0]+[None]*6, list('abcdefgh'))
            self.assertEqual(history.snapshot()['speaker_lambda'], 0)
            history.commit(1, lambda local: [local, dict(local,hits=1,valid=8,count=8)])
            self.assertAlmostEqual(history.snapshot()['speaker_lambda'], 0)
            self.assertEqual(history.snapshot()['history_valid'], 10)

    def test_chance_scaling_and_no_wrong_answer_bonus(self):
            from v7_history import history_snapshot
            from v7_core import reward_parts
            p = dict(formality=.1,groundedness=.2,combined_answerability=.3)
            for hits, penalty in [(2,0),(3,.1),(4,.2)]:
                snapshot = history_snapshot([(hits,10)])
                self.assertAlmostEqual(snapshot['speaker_lambda'],penalty)
                self.assertAlmostEqual(reward_parts(p,'A','A',history=snapshot)['reward'],.22-penalty)
                self.assertAlmostEqual(reward_parts(p,'B','A',history=snapshot)['reward'],.22)
                self.assertAlmostEqual(reward_parts(p,'bad','A',history=snapshot)['reward'],.22)

    def test_history_window_excludes_old_updates_and_rejects_replay(self):
            from v7_history import SpeakerHistory
            h=SpeakerHistory()
            for step in range(1,10):
                values=[1]*8 if step==1 else [0]*8
                h.stage(step,values,[str(i) for i in range(8)])
                with self.assertRaises(RuntimeError):
                    h.stage(step,values,[str(i) for i in range(8)])
                h.commit(step,lambda row:[row,dict(row)])
            self.assertEqual(h.snapshot()['history_hits'],0)
            self.assertEqual(h.snapshot()['history_valid'],128)
            with self.assertRaises(RuntimeError):
                h.commit(9,lambda row:[row,row])
            h.stage(10,[0]*8,list('abcdefgh'))
            with self.assertRaises(RuntimeError):
                h.commit(10,lambda row:[row,dict(row,before={})])

    def test_two_independent_g4_and_identity_validation(self):
            from v7_core import score_group
            from v7_history import history_snapshot
            req=[dict(request_id=str(i),evidence_id='e'+str(i//4),source_packet_id='p'+str(i//4),asker_index=0) for i in range(8)]
            results=[dict(request_id=r['request_id'],evidence_id=r['evidence_id'],status='scored',
                          probabilities=dict(formality=.5,groundedness=.5,combined_answerability=.5),
                          speaker=dict(raw_output='A' if i%4==0 else 'B',gold='A')) for i,r in enumerate(req)]
            parts=score_group(req,results,history=history_snapshot([(4,10)]))
            self.assertEqual([p['generation_group_index'] for p in parts],[0]*4+[1]*4)
            self.assertAlmostEqual(parts[0]['reward'],.3)
            self.assertAlmostEqual(parts[1]['reward'],.5)
            with self.assertRaises(RuntimeError):
                score_group(req,list(reversed(results)),history=history_snapshot([]))
            bad=deepcopy(req);bad[3]['source_packet_id']='other'
            with self.assertRaises(RuntimeError):
                score_group(bad,results,history=history_snapshot([]))

    def test_decode_reuse_and_blind_whitelist(self):
            from v7_speaker import FrozenSpeaker,blind_payload
            speaker=FrozenSpeaker({})
            speaker.processor=types.SimpleNamespace(apply_chat_template=lambda *a,**kw:'prompt')
            qa=dict(question='Q?',options=list('abcde'),correct='E',rationale='SECRET')
            payloads=[blind_payload(qa,[f'{i//4}.jpg']) for i in range(8)]
            self.assertNotIn('SECRET',json.dumps(payloads))
            calls=[]
            def decode(messages,**kwargs):
                calls.append(kwargs)
                return [object()],None
            requests=speaker.prepare_requests(payloads,decode)
            self.assertEqual(len(calls),2)
            self.assertIs(requests[0]['multi_modal_data']['image'],requests[3]['multi_modal_data']['image'])
            self.assertIsNot(requests[0]['multi_modal_data']['image'],requests[4]['multi_modal_data']['image'])
            self.assertEqual(speaker.last_metrics['decoded_timelines'],2)

    def test_real_service_batches_two_windows_and_skips_extra_judge_wake(self):
            test_package('v7service')
            judge=types.ModuleType('v7service.judge')
            judge.InvalidCompletion=type('InvalidCompletion',(ValueError,),{})
            judge.parse_completion=json.loads
            def view(root,packet,asker):
                return dict(required_users=[packet],clips=[dict(agent_name=packet,media_role='speaker',frames=[dict(path=packet+'.jpg')])])
            judge.full_judge_view=view
            def build(request):
                if request['completion']=='invalid':
                    raise judge.InvalidCompletion('invalid JSON')
                return {key:types.SimpleNamespace(key=key) for key in ('formality','groundedness','combined_answerability')}
            judge.build_examples=build
            sys.modules[judge.__name__]=judge
            service=importlib.import_module('v7service.service')
            events=[]
            class Predictor:
                identity={'model_id':'fixture'}
                config={'v7':{'speaker':{'model_id':'fixture-Qwen2.5'}}}
                def resource(self,action):
                    events.append('judge_'+action)
                    return {'sleeping':action=='sleep'}
                def predict_many(self,examples):
                    events.append(examples[0].key)
                    return [dict(pass_probability=.5) for _ in examples]
            class Speaker:
                def predict_many(self,payloads):
                    events.append('speaker_predict')
                    self.payloads=payloads
                    return ['A']*len(payloads)
                def sleep(self):
                    events.append('speaker_sleep')
            qa=dict(question='Q?',options=list('abcde'),correct='A',rationale='SECRET')
            requests=[dict(request_id=str(i),evidence_id='e',dataset_root='root',source_packet_id=f'window{i//4}',asker_index=0,
                completion='invalid' if i==1 else json.dumps(qa)) for i in range(8)]
            scorer=service.CandidateScorer(Predictor(),'instance')
            scorer.v7_speaker=Speaker()
            results=scorer.score_many(requests)
            self.assertEqual(results[1]['status'],'invalid_completion')
            self.assertEqual(results[4]['speaker']['input_users'],['window1'])
            self.assertEqual(len(scorer.v7_speaker.payloads),7)
            self.assertNotIn('SECRET',json.dumps(scorer.v7_speaker.payloads))
            self.assertEqual(events,['judge_wake','formality','groundedness','combined_answerability',
                                    'judge_sleep','speaker_predict','speaker_sleep'])
            events.clear()
            scorer.score_many(requests)
            self.assertEqual(events[0],'judge_wake')
            scorer.v7_speaker.sleep=lambda:(_ for _ in ()).throw(RuntimeError('release failed'))
            with self.assertRaises(RuntimeError):
                scorer.score_many(requests)
            events.clear()
            with self.assertRaises(RuntimeError):
                scorer.resource(dict(instance_id='instance',action='sleep'))
            self.assertEqual(events,[])

    def test_real_reward_plugin_stages_two_groups_and_waits_for_optimizer(self):
            prefix='training.grpo_v3.six_user_binary'
            replacements={}
            for name in ('training','training.grpo_v3'):
                module=types.ModuleType(name);module.__path__=[]
                replacements[name]=module
            package=types.ModuleType(prefix);package.__path__=[str(ROOT/'overlay'/PKG),str(SOURCE/PKG)]
            replacements[prefix]=package
            data=types.ModuleType(prefix+'.data');data.validate_row=lambda *a,**kw:None
            replacements[data.__name__]=data
            guard=types.ModuleType(prefix+'.utilization_runtime');guard.enable_training_guard=lambda:None
            replacements[guard.__name__]=guard
            cache=types.ModuleType(prefix+'.policy_image_cache');cache.enable_from_environment=lambda:None
            replacements[cache.__name__]=cache
            replacements[prefix+'.stage_stop']=types.ModuleType(prefix+'.stage_stop')
            # 服务响应校验使用部署源文件的真实函数，后端只替代网络和GPU。
            test_package('pluginservice')
            judge=types.ModuleType('pluginservice.judge');judge.InvalidCompletion=ValueError;judge.build_examples=lambda x:{}
            sys.modules[judge.__name__]=judge
            service=importlib.import_module('pluginservice.service')
            replacements[prefix+'.service']=service
            with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,replacements),patch.dict('os.environ',
                {'EGOQA_GRPO_V3_REWARD_TRACE':str(Path(tmp)/'reward_trace.jsonl'),'EGOQA_GRPO_START_STEP':'0',
                 'EGOQA_SHARED_GPU':'0','EGOQA_QWEN_PARTIAL_PACKED_LORA':'0'}):
                plugin=importlib.import_module(prefix+'.plugin')
                class Client:
                    def score_many(self,requests):
                        return [dict(request_id=r['request_id'],evidence_id=r['evidence_id'],judge={},status='scored',
                            probabilities=dict(formality=.5,groundedness=.5,combined_answerability=.5),
                            speaker=dict(raw_output='A' if i%4<2 else 'B',gold='A')) for i,r in enumerate(requests)]
                reward=plugin.SixUserBinaryReward(client=Client(),trace_path=Path(tmp)/'reward_trace.jsonl',
                    reward_mode='continuous',diversity_lambda=0)
                metadata=dict(dataset_root='root',source_packet_id=['p0']*4+['p1']*4,
                    asker_index=[0]*8,evidence_id=['e0']*4+['e1']*4,required_users=['speaker'],
                    generator_image_paths=[['s0.jpg']]*4+[['s1.jpg']]*4)
                self.assertEqual(reward(['completion']*8,**metadata),[.5]*8)
                history=importlib.import_module(prefix+'.v7_history').current_history()
                self.assertEqual(history.completed_step,0)
                history.commit(1,lambda row:[row,dict(row)])
                values=reward(['completion']*8,**metadata)
                self.assertAlmostEqual(values[0],.3)
                self.assertAlmostEqual(values[2],.5)
                records=[json.loads(line) for line in (Path(tmp)/'reward_trace.jsonl').read_text().splitlines()]
                self.assertEqual(len(records),16)
                self.assertEqual([r['global_step'] for r in records],[1]*8+[2]*8)

    def test_terminal_replays_history_and_validates_all_four_windows(self):
            test_package('v7terminal')
            validator=importlib.import_module('v7terminal.validate_run')
            from v7_core import reward_parts,invalid_parts
            from v7_history import history_snapshot
            counts=[]
            planned=[dict(source_packet_id=f'RLHF6U_DAY1_{index:08d}_A1',asker_index=0) for index in range(12)]
            ranks=[[],[]]
            for step in range(1,4):
                snap={**history_snapshot(counts[-8:]),'history_completed_step':step-1}
                for rank in (0,1):
                    for slot,raw in enumerate(['A','A','A','A','B','bad',None,'B']):
                        result=dict(status='invalid_completion') if raw is None else dict(status='scored',
                            probabilities=dict(formality=.01,groundedness=.01,combined_answerability=.01),
                            speaker=dict(raw_output=raw,gold='A'))
                        parts={**invalid_parts(),**snap} if raw is None else reward_parts(result['probabilities'],raw,'A',history=snap)
                        ranks[rank].append(dict(parts,**planned[(step-1)*4+rank*2+slot//4],
                            request_id=f'{rank}-{step}:{slot}',evidence_id=f'e{rank}-{step}-{slot//4}',
                            global_step=step,judge_result=result))
                counts.append((8,12))
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);cp=out/'swift/v0/checkpoint-3';cp.mkdir(parents=True)
                data=out/'train.jsonl';data.write_text(''.join(json.dumps(row)+'\n' for row in planned))
                config=dict(max_steps=3,stop_after_steps=3,num_generations=4,world_size=2,train_conditions=12,
                    validation_conditions=0,val_dataset=None,eval_strategy='no',train_dataset=str(data))
                (out/'run_config.json').write_text(json.dumps(config))
                (cp/'trainer_state.json').write_text(json.dumps(dict(global_step=3,max_steps=3,log_history=[dict(loss=.1,grad_norm=.2)])))
                for name in ['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                             'rng_state_0.pth','rng_state_1.pth','args.json','training_args.bin']:
                    (cp/name).write_bytes(b'synthetic-test-only')
                for rank in (0,1):
                    path=out/f'rank{rank}';path.mkdir()
                    (path/'reward_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in ranks[rank]))
                    (path/'ddp_sync.jsonl').write_text(json.dumps(dict(global_step=3,finite=True,
                        checked_tensors=160,all_trainable_parameters_equal=True))+'\n')
                    (path/'gradient_sync.jsonl').write_text(''.join(json.dumps(dict(global_step=step,finite=True,
                        world_size=2,tensors=160,order='mean_then_clip_then_optimizer'))+'\n' for step in range(3)))
                    (path/'phase_timing.jsonl').write_text(''.join(json.dumps(dict(global_step=step,
                        phase='optimizer_step_wall',seconds=100+rank))+'\n' for step in range(1,4)))
                rows=ranks[0]+ranks[1]
                trace=out/'reward_trace.jsonl';trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                report=validator.validate(out,adapter_check=lambda path:True)
                self.assertEqual(report['status'],'passed',report)
                self.assertEqual(report['candidate_count'],48)
                self.assertEqual(report['throughput']['completed_update_count'],3)
                self.assertAlmostEqual(report['throughput']['candidate_slots_per_hour'],48*3600/303)
                self.assertLess(rows[8]['reward'],0)
                # 同时伪造第一步原输出和自报命中，后面的历史仍须验出。
                rows[0]['judge_result']['speaker']['raw_output']='B';rows[0]['speaker_hit']=0
                trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                report=validator.validate(out,adapter_check=lambda path:True)
                self.assertIn('history_replayed_from_previous_updates',report['failed_checks'])

if __name__=='__main__':unittest.main(verbosity=2)
