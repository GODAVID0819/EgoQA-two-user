"""验证合并流程的失败边界和真实进程隔离，不运行GPU或重做原CPU检查。"""
import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

try:
    import combined_allocation as combined
except ModuleNotFoundError:
    combined = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class CombinedTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(combined, '缺少同allocation执行器')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        formal = dict(train_python=sys.executable, project_root=str(self.root),
            output_root=str(self.root/'outputs/stage180'), train_dataset='formal.jsonl',
            train_conditions=720, validation_conditions=0, stop_after_steps=180,
            max_steps=360, max_pixels=50176, max_length=131072, num_generations=4,
            learning_rate=1e-5, lr_scheduler_type='cosine_with_min_lr',
            lr_scheduler_kwargs={'min_lr_rate':0.1}, warmup_steps=10,
            seed=42, data_seed=42, initial_fresh_base=True,
            resume_from_checkpoint=None, walltime_seconds=216000,
            slurm={'cpus':32,'mem':'750G','gres':'gpu:2','constraint':'h200'})
        capacity = {**formal, 'output_root':str(self.root/'outputs/capacity'),
            'train_dataset':'longest.jsonl','train_conditions':2,
            'stop_after_steps':1,'walltime_seconds':7200,'capacity_measurement':True}
        write(self.root/'configs/train.json',formal)
        write(self.root/'configs/capacity.json',capacity)
        write(self.root/'submissions/current_training_submission.json',
            {'status':'submitted','job_id':'77','phase':'combined_capacity_then_stage180','output_dir':str(self.root/'outputs/stage180/train_77')})
        (self.root/'dual_launch.py').write_text('''import argparse,json,os,random,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config');c=json.loads(Path(p.parse_args().config).read_text())
root=Path(c['project_root']);phase='capacity' if c.get('capacity_measurement') else 'stage180'
with (root/'child_calls.jsonl').open('a') as f:
 f.write(json.dumps({'phase':phase,'pid':os.getpid(),'resume':c.get('resume_from_checkpoint'),'seed':c['seed'],'random':random.Random(c['seed']).random(),'inherited_resume':os.environ.get('EGOQA_RESUME_METADATA')})+'\\n')
if (root/('fail_'+phase)).exists():raise SystemExit(9)
out=Path(c['output_root'])/('train_'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
(out/'run_manifest.json').write_text(json.dumps({'status':'completed','cleanup_errors':[]}))
(out/'training_result.json').write_text(json.dumps({'status':'passed','global_step':c['stop_after_steps'],'candidate_count':8*c['stop_after_steps'],'unique_conditions':2*c['stop_after_steps'],'schedule_max_steps':360}))
''',encoding='utf-8')
        self.env = patch.dict(os.environ, {'SLURM_JOB_ID':'77','EGOQA_RESUME_METADATA':'must_be_removed'})
        self.env.start();self.addCleanup(self.env.stop)
        self.silence=patch('builtins.print');self.silence.start();self.addCleanup(self.silence.stop)
        self.live=patch.object(combined,'live_allocation',return_value={'elapsed_seconds':10,'time_limit_seconds':216000})
        self.live.start();self.addCleanup(self.live.stop)
        self.accept=patch('capacity_acceptance.accept_capacity',return_value=({}, {'status':'passed'}))
        self.accept_mock=self.accept.start();self.addCleanup(self.accept.stop)

    def calls(self):
        p=self.root/'child_calls.jsonl'
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_success_uses_two_fresh_processes_and_separate_outputs(self):
        combined.run_combined(self.root,'77')
        calls=self.calls()
        self.assertEqual([x['phase'] for x in calls],['capacity','stage180'])
        self.assertNotEqual(calls[0]['pid'],calls[1]['pid'])
        self.assertEqual(calls[0]['random'],calls[1]['random'])
        self.assertTrue(all(x['resume'] is None and x['inherited_resume'] is None for x in calls))
        state=json.loads((self.root/'preflight/combined_77.json').read_text())
        self.assertEqual(state['status'],'completed')
        self.assertEqual(state['formal_global_step'],180)
        self.assertTrue((self.root/'outputs/capacity/train_77').exists())
        self.assertTrue((self.root/'outputs/stage180/train_77').exists())

    def test_capacity_failure_never_starts_formal(self):
        (self.root/'fail_capacity').touch()
        with self.assertRaises(subprocess.CalledProcessError):combined.run_combined(self.root,'77')
        self.assertEqual([x['phase'] for x in self.calls()],['capacity'])
        self.accept_mock.assert_not_called()

    def test_rejected_measurement_never_starts_formal(self):
        self.accept_mock.side_effect=ValueError('真实验收不通过')
        with self.assertRaises(ValueError):combined.run_combined(self.root,'77')
        self.assertEqual([x['phase'] for x in self.calls()],['capacity'])

    def test_repeated_execution_is_rejected(self):
        combined.run_combined(self.root,'77')
        with self.assertRaises((FileExistsError,ValueError)):combined.run_combined(self.root,'77')
        self.assertEqual(len(self.calls()),2)

    def test_resume_configuration_rejected_before_gpu(self):
        p=self.root/'configs/train.json';c=json.loads(p.read_text());c['resume_from_checkpoint']='measurement/checkpoint-1';write(p,c)
        with self.assertRaises(ValueError):combined.run_combined(self.root,'77')
        self.assertEqual(self.calls(),[])

    def test_budget_includes_consumed_allocation_time(self):
        from capacity_acceptance import budget_report
        c={'walltime_seconds':216000}
        report=budget_report(c,{i:800 for i in range(1,181)},{1:800},64,'77',elapsed_seconds=36000)
        self.assertEqual(report['status'],'needs_review')
        self.assertEqual(report['allocation_elapsed_seconds'],36000)
        self.assertAlmostEqual(report['total_allocation_with_margin_hours'],60.5)

    def test_submission_refuses_live_legacy_jobs(self):
        from submit_combined import verify_legacy_terminal
        with self.assertRaises(ValueError):
            verify_legacy_terminal('20443244|PENDING|0:0\n20443249|PENDING|0:0\n')
        verify_legacy_terminal('20443244|CANCELLED by 4914731|0:0\n20443249|CANCELLED|0:0\n')

    def test_immediate_allocation_waits_for_accurate_sbatch_receipt(self):
        marker=self.root/'submissions/current_training_submission.json'
        write(marker,{'status':'submitting','phase':'combined_capacity_then_stage180'})
        def publish():
            time.sleep(.05)
            combined.save(marker,{'status':'submitted','phase':'combined_capacity_then_stage180','job_id':'77'})
        worker=threading.Thread(target=publish);worker.start()
        try:self.assertEqual(combined.await_submission(self.root,'77',timeout=2)['job_id'],'77')
        finally:worker.join()

    def test_slurm_resource_parser_uses_actual_torch_field_units(self):
        self.live.stop()
        text='JobId=77 JobState=RUNNING UserId=xl6775(4914731) TimeLimit=2-12:00:00 Features=h200 RunTime=00:15:00 ReqTRES=cpu=32,mem=750G,node=1,billing=32,gres/gpu=2'
        with patch.object(combined.subprocess,'check_output',return_value=text):
            self.assertEqual(combined.live_allocation('77')['elapsed_seconds'],900)
        with patch.object(combined.subprocess,'check_output',return_value=text.replace('mem=750G','mem=64G')):
            with self.assertRaises(ValueError):combined.live_allocation('77')


class InlineAcceptanceTests(unittest.TestCase):
    def test_running_allocation_uses_actual_completed_phase_and_rejects_missing_reload(self):
        from capacity_acceptance import accept_capacity
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'capacity';source=root/'source';old=source/'old_output'
            config={'train_dataset':str(root/'formal.jsonl'),'max_pixels':50176,
                'max_steps':360,'stop_after_steps':180,'walltime_seconds':216000}
            rows=[{'source_packet_id':'packet'+str(i),'asker_index':i,'images':['image'+str(i)]} for i in range(128)]
            dump=lambda p,values:(p.parent.mkdir(parents=True,exist_ok=True),p.write_text(''.join(json.dumps(x)+'\n' for x in values)))
            dump(root/'formal.jsonl',rows);dump(root/'pair.jsonl',rows[126:128])
            actual={**config,'train_dataset':str(root/'pair.jsonl'),'capacity_measurement':True,'stop_after_steps':1}
            write(output/'run_config.json',actual)
            for rank in range(2):dump(output/f'rank{rank}/reward_trace.jsonl',[rows[126+rank]]*4)
            checkpoint=output/'checkpoint-1'
            write(output/'training_result.json',{'status':'passed','global_step':1,'candidate_count':8,
                'unique_conditions':2,'schedule_max_steps':360,'checkpoint':str(checkpoint)})
            write(output/'run_manifest.json',{'status':'completed','cleanup_errors':[]})
            write(checkpoint/'reload_cpu_audit.json',{'optimizer_and_scheduler_step':1,'optimizer_state_entries':160})
            write(root/'submissions/combined_capacity_execution.json',{'job_id':'77','output_dir':str(output)})
            write(root/'submissions/current_training_submission.json',{'job_id':'77','phase':'combined_capacity_then_stage180'})
            write(root/'preflight/combined_77.json',{'status':'capacity_completed','capacity_exit_code':0})
            write(root/'preflight/input_lengths.json',{'longest_pair_indices':[126,127]})
            write(root/'preflight/source_provenance.json',{'source_root':str(source)})
            write(source/'submissions/current_training_submission.json',{'job_id':'20191258','output_dir':str(old)})
            for rank in range(2):
                dump(old/f'rank{rank}/phase_timing.jsonl',[{'phase':'optimizer_step_wall','global_step':i,'seconds':100} for i in range(1,181)])
                dump(output/f'rank{rank}/phase_timing.jsonl',[{'phase':'optimizer_step_wall','global_step':1,'seconds':200}])
            with patch.dict(os.environ,{'SLURM_JOB_ID':'77'}),patch.object(combined,'live_allocation',return_value={'elapsed_seconds':900}),patch('capacity_acceptance.terminal_success',side_effect=AssertionError('不能把运行中allocation冒充COMPLETED')):
                _,report=accept_capacity(root,config,inline_job_id='77')
                self.assertEqual(report['status'],'passed')
                self.assertEqual(report['matched_source_step'],64)
                self.assertEqual(report['allocation_elapsed_seconds'],900)
                write(checkpoint/'reload_cpu_audit.json',{'optimizer_and_scheduler_step':0,'optimizer_state_entries':0})
                with self.assertRaises(ValueError):accept_capacity(root,config,inline_job_id='77')

    def test_old_controller_exits_without_submission_when_replacement_requested(self):
        import pipeline_controller as controller
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'preflight').mkdir()
            write(root/'submissions/combined_replacement_requested.json',
                {'legacy_capacity_job_id':'20443244','legacy_controller_job_id':'20443249'})
            with patch.object(controller,'ROOT',root),patch.dict(os.environ,{'SLURM_JOB_ID':'20443249','CUDA_VISIBLE_DEVICES':''}),patch.object(sys,'argv',['pipeline_controller.py','--stage','after_capacity','--previous','20443244']),patch('pipeline_controller.subprocess.run',side_effect=AssertionError('旧控制器不得sbatch')),contextlib.redirect_stdout(io.StringIO()):
                controller.main()
            self.assertEqual(json.loads((root/'preflight/pipeline_state.json').read_text())['status'],'superseded_no_submission')


if __name__=='__main__':unittest.main(verbosity=2)
