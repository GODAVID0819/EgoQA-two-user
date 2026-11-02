"""同一v7实验完整续跑，覆盖四窗前缀与历史统计恢复。"""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import ast
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PKG=Path('multi-user/training/grpo_v3/six_user_binary')
sys.path.insert(0,str(ROOT/'runtime'/PKG))


class ResumeTests(unittest.TestCase):
    def test_resume_acceptance_reads_actual_callback_file(self):
        worker=(ROOT/'runtime/dual_worker.py').read_text()
        validation=(ROOT/'runtime'/PKG/'validate_run.py').read_text()
        self.assertIn("record('resume_optimizer',check_restored_optimizer",worker)
        self.assertIn("resume_path=rank_dir/'resume_optimizer.jsonl'",validation)

    def test_final_odd_update_is_audited(self):
        source=ast.parse((ROOT/'runtime/dual_worker.py').read_text())
        conditions=[node.test for node in ast.walk(source) if isinstance(node,ast.If)
            and any(isinstance(child,ast.Call) and isinstance(child.func,ast.Attribute) and child.func.attr=='verify' for statement in node.body for child in ast.walk(statement))]
        condition=next(test for test in conditions if 'state.global_step%2' in ast.unparse(test).replace(' ',''))
        evaluate=lambda step,stop:eval(compile(ast.Expression(body=condition),'predicate','eval'),{},
            {'state':types.SimpleNamespace(global_step=step),'config':{'stop_after_steps':stop}})
        self.assertTrue(evaluate(3,3))
        self.assertFalse(evaluate(3,180))
        self.assertTrue(evaluate(180,180))

    def test_guard_rebuild_waits_for_cuda_and_keeps_live_graph(self):
        path=ROOT/'runtime'/PKG/'utilization_runtime.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='rearm_training_guard')
        events=[]
        active=types.SimpleNamespace(graph_live=False,error=None,config={'reserve_gib':8,'max_prealloc_mib':16,'max_duty':.9},log='fixture',close=lambda:events.append('close'))
        class Guard:
            def __init__(self,config,log):self.config=config;self.log=log;self.graph_live=True
            def start(self):events.append('capture');return self
            def close(self):pass
        scope=dict(_active=active,Guard=Guard,atexit=types.SimpleNamespace(register=lambda f:events.append('register')))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
        torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(synchronize=lambda:events.append('sync'))
        with patch.dict(sys.modules,{'torch':torch}):scope['rearm_training_guard']()
        self.assertEqual(events,['sync','close','capture','register'])
        self.assertEqual(scope['_active'].config,active.config)
        events.clear();scope['rearm_training_guard']();self.assertEqual(events,[])

    def test_prefix_and_population_history_continue_without_replay(self):
        package=types.ModuleType('formalfixture')
        package.__path__=[str(ROOT/'runtime'/PKG),str(ROOT/'runtime'/PKG)]
        sys.modules['formalfixture']=package
        history=importlib.import_module('formalfixture.v7_history')
        core=importlib.import_module('formalfixture.v7_core')
        batch=importlib.import_module('formalfixture.v7_batch')
        ranks=[[],[]];counts=[]
        planned=[dict(source_packet_id=f'RLHF6U_DAY1_{i:08d}_A1',asker_index=0) for i in range(12)]
        for step in (1,2):
            snapshot={**history.history_snapshot(counts),'history_completed_step':step-1}
            for rank in (0,1):
                for slot in range(8):
                    raw='A' if slot%4==0 else 'B'
                    p=dict(formality=.5,groundedness=.5,combined_answerability=.5)
                    parts=core.reward_parts(p,raw,'A',history=snapshot)
                    ranks[rank].append(dict(parts,**planned[4*(step-1)+rank*2+slot//4],
                        request_id=f'{rank}-{step}:{slot}',global_step=step,evidence_id=f'e{rank}-{step}-{slot//4}',
                        judge_result=dict(status='scored',probabilities=p,speaker=dict(raw_output=raw,gold='A'))))
            counts.append((4,16))
        with tempfile.TemporaryDirectory() as temporary:
            tmp=Path(temporary);source=tmp/'train_1000';source.mkdir()
            for rank in (0,1):
                path=source/f'rank{rank}';path.mkdir()
                (path/'reward_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in ranks[rank]))
            restored=history.restore_history(source,2)
            self.assertEqual(restored.completed_step,2)
            self.assertEqual(restored.snapshot()['history_valid'],32)
            self.assertAlmostEqual(restored.snapshot()['speaker_lambda'],.05)
            restored.stage(3,[0]*8,list('abcdefgh'))
            restored.commit(3,lambda row:[row,dict(row)])
            self.assertEqual(restored.snapshot()['history_valid'],48)
            checkpoint=source/'swift/v0/checkpoint-2';checkpoint.mkdir(parents=True)
            for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state_0.pth','rng_state_1.pth','args.json','training_args.bin']:
                (checkpoint/name).write_bytes(b'synthetic-unit-test-only')
            (checkpoint/'trainer_state.json').write_text(json.dumps(dict(global_step=2,max_steps=180)))
            (checkpoint/'adapter_config.json').write_text(json.dumps(dict(r=64,lora_alpha=128)))
            audit=checkpoint/'reload_cpu_audit.json'
            audit.write_text(json.dumps(dict(status='passed',global_step=2,checkpoint=str(checkpoint),scheduler={'_last_lr':[4e-6]},optimizer_states=160)))
            data=tmp/'train.jsonl';data.write_text(''.join(json.dumps(row)+'\n' for row in planned))
            judge=tmp/'judge.json';judge.write_text(json.dumps({'fixture':'same C630'}))
            old=dict(max_steps=180,stop_after_steps=2,lora_rank=64,lora_alpha=128,train_dataset=str(data),judge_config=str(judge))
            (source/'run_config.json').write_text(json.dumps(old))
            config=dict(old,resume_from_checkpoint=str(checkpoint),resume_source_output=str(source),
                        expected_resume_step=2,recovery_from_job='1000',resume_cpu_audit=str(audit),stop_after_steps=180)
            spec=importlib.util.spec_from_file_location('formal_recovery',ROOT/'runtime/recovery.py')
            recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
            # 根恢复模块通过真实包名导入回放函数，保持相同实现。
            canonical='training.grpo_v3.six_user_binary.v7_batch'
            previous=sys.modules.get(canonical);sys.modules[canonical]=batch
            try:
                out=tmp/'train_new'
                for rank in (0,1):(out/f'rank{rank}').mkdir(parents=True)
                report=recovery.seed_reward_prefix(config,out)
                self.assertEqual(report['inherited_candidates'],32)
                self.assertEqual(len((out/'rank0/reward_trace.jsonl').read_text().splitlines()),16)
                self.assertEqual(recovery.expected_resume_bindings(config),[(planned[8]['source_packet_id'],0),(planned[10]['source_packet_id'],0)])
                with self.assertRaises(FileExistsError):recovery.seed_reward_prefix(config,out)
                bad=dict(config,num_generations=8)
                with self.assertRaises(ValueError):recovery.validated_resume(bad)
                bad=dict(config,max_steps=90)
                with self.assertRaises(ValueError):recovery.validated_resume(bad)
            finally:
                if previous is None:sys.modules.pop(canonical,None)
                else:sys.modules[canonical]=previous
            changed=json.loads((source/'rank0/reward_trace.jsonl').read_text().splitlines()[0])
            changed['judge_result']['speaker']['raw_output']='B'
            rows=[changed]+ranks[0][1:]
            (source/'rank0/reward_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError):history.restore_history(source,2)


if __name__=='__main__':unittest.main(verbosity=2)
