"""只复用已有完整预测；保留失败作业原文件及来源JobID。"""
import json
from pathlib import Path
import grpo_inputs as core


def load_resume(source,items,frames,header,config):
    result={s:{} for s in core.SETTINGS};notes=[]
    if source is None:return result,notes
    source=Path(source)
    previous=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    for key in ['model','model_revision','dataset_revision','reference_commit','reference_pipeline',
                'preprocess_max_pixels','max_new_tokens','processor_policy','repetition_penalty',
                'do_sample','batch_size','settings','image_only_identity_compression_bypass']:
        if previous.get(key)!=config.get(key):raise ValueError(f'续跑配置不一致：{key}')
    by_id={x['id']:x for x in items}
    for setting in core.SETTINGS:
        path=source/f'{setting}_predictions.jsonl'
        if not path.exists():continue
        lines=path.read_text(encoding='utf-8').splitlines(keepends=True)
        for index,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1 and not line.endswith('\n'):
                    notes.append({'file':str(path),'action':'ignored_torn_final_line'});continue
                raise
            item=by_id.get(row.get('id'))
            if item is None or row['id'] in result[setting]:raise ValueError('续跑题号未知或重复')
            users,paths,_=core.build_input(item,frames[item['id']],setting,header)
            if (row.get('setting')!=setting or row.get('correct')!=item['correct'] or row.get('users')!=users
                    or row.get('frame_paths')!=paths or row.get('n_images')!=len(paths)
                    or row.get('pred')!=core.parse_letter(row['raw'])
                    or row.get('is_correct')!=(row.get('pred')==item['correct'])
                    or row.get('model_revision')!=config['model_revision']):
                raise ValueError(f'续跑预测与当前题目或帧不匹配：{item["id"]}')
            result[setting][item['id']]=row
    return result,notes
