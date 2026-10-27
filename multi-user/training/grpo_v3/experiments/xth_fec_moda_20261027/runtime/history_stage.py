"""CPU直接筛选旧训练QA；正式GPU作业只读取固定池，不重评分。"""
from pathlib import Path
import json

from training.grpo_v3.six_user_binary.diversity import window_id
from training.grpo_v3.six_user_binary.judge import parse_completion
from training.grpo_v3.six_user_binary.history_pool import MiniLMEncoder, save_history, load_history, reference_qualified


def historical_records(source_rows):
    records=[]
    for row in source_rows:
        if row['status']!='scored': continue
        qa=row.get('qa') or parse_completion(row['completion'])
        if not isinstance(qa.get('question'),str) or not qa['question'].strip():
            raise ValueError('历史QA问题为空')
        records.append({'history_qa_id':'history:'+row['request_id'],
            'source_packet_id':row['source_packet_id'],'speaker_index':int(row['asker_index']),
            'source_global_step':row['global_step'],'source_job':row['source_job'],
            'question':qa['question'],'quality_reward_old':row['reward'],
            'probabilities_old_F_E_A_S':row['probabilities'],'judge_old':row.get('judge')})
    return records


def build_fixed_history(config):
    root=Path(config['project_root']);source=Path(config['history_source'])
    directory=Path(config['history_pool_directory'])
    metadata_path=source/'snapshot_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    if metadata['completed_global_step']!=300 or metadata['candidate_count']!=2400:
        raise ValueError('本轮直接使用完整旧300步2400候选来源')
    if directory.exists():
        _,report=load_history(directory)
        if report['source_metadata']!=str(metadata_path):raise ValueError('已有固定历史池来源不同，不能覆盖')
        return report
    rows=[json.loads(line) for line in (source/'complete_training_qa.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=2400:raise ValueError('旧QA来源候选数量不完整')
    records=historical_records(rows)
    windows=sorted({window_id(json.loads(line)['source_packet_id']) for line in Path(config['train_dataset']).read_text().splitlines() if line.strip()})
    if len(windows)!=120:raise ValueError('训练窗口应为120')
    report=save_history(records,windows,MiniLMEncoder(root/'encoder'),directory,
        source_metadata=str(metadata_path),old_judge_config=metadata['judge_config'])
    report.update(source_candidates=len(rows),source_scored_QA=len(records),source_completed_updates=300,
                  eligible_before_capacity=sum(reference_qualified(row) for row in records))
    (directory/'pool_metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    load_history(directory)
    print('OLD_SCORE_HISTORY_POOL_FIXED',json.dumps({k:v for k,v in report.items() if k!='old_judge_config'}),flush=True)
    return report


def prepare_history(config,output):
    output=Path(output);root=Path(config['project_root'])
    directory=Path(config['history_pool_directory'])
    _,report=load_history(directory)
    runtime={'diversity_lambda':.5,'encoder_directory':str(root/'encoder'),'history_pool_directory':str(directory)}
    path=output/'moda_runtime.json';path.write_text(json.dumps(runtime,indent=2))
    (output/'progress_stage.json').write_text(json.dumps({'phase':'fixed_old_score_history_loaded',
        'selected_QA':report['selected_QA'],'empty_windows':report['empty_windows'],
        'historical_rescoring_performed':False},indent=2))
    return path
