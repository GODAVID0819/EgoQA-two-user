"""同视频speaker留出检查与独立测试评分路径。"""
from pathlib import Path


def data_validation_command(c):
    if c.get('full120_speaker_holdout') or c.get('minimum_runtime_check'):
        command=[c['train_python'],str(Path(c['project_root'])/'validate_full120.py'),'--train',c['train_dataset'],'--val',c['val_dataset'],'--definition',str(Path(c['project_root'])/'condition_definition.json')]
        if c.get('minimum_runtime_check'): command.append('--minimum-runtime-check')
        return command
    command=[c['train_python'],'-m','training.grpo_v3.six_user_binary.data','--validate',c['train_dataset'],c['val_dataset']]
    if c.get('speaker_holdout_validation'):
        command.extend([c['test_dataset'],'--speaker-holdout'])
    return command


def evaluation_files(c,output,adapter=None,split='validation',validation_baseline=None):
    output=Path(output)
    if split not in ('validation','test'):raise ValueError('未知评价集合')
    label=('test_' if split=='test' else '')+('policy' if adapter else 'baseline')
    config=dict(c)
    if split=='test':config['val_dataset']=c['test_dataset']
    return {'label':label,'destination':(output/(label+'.json' if split=='test' else 'validation_'+label+'.json')).as_posix(),
        'config_path':(output/('test_run_config.json' if split=='test' else 'run_config.json')).as_posix(),
        'config':config,'baseline':(output/'test_baseline.json').as_posix() if split=='test' else str(validation_baseline)}
