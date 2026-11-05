"""直接复用指定仓库版本的GRPO帧VQA提示词和计分函数。"""
import importlib.util
import json
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference_grpo_vqa',ROOT.parent/'vqa_grpo_frames.py')
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
SETTINGS=('evidence2','speaker1','all6')
parse_letter=reference.parse_letter
parse_reason=reference.parse_reason


def safe_relative(value):
    path=PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or ':' in value or '\\' in value or not value.startswith('frames/'):
        raise ValueError(f'不合法的题集帧路径：{value}')
    return path


def load_inputs(root=ROOT):
    root=Path(root)
    vqa=json.loads((root/'vqa_shuffled.json').read_text(encoding='utf-8'))
    full=json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    items=vqa['items'];metadata={i['id']:i for i in full['items']}
    if len(items)!=131 or len({i['id'] for i in items})!=131 or set(metadata)!={i['id'] for i in items}:
        raise ValueError('必须保留全部131个唯一题号')
    if vqa['meta']['max_pixels']!=24576:
        raise ValueError('预缩放面积与仓库GRPO设置不一致')
    frame_store={}
    for item in items:
        full_item=metadata[item['id']]
        users=full_item['required_users']
        if len(users)!=6 or len(set(users))!=6 or users[0]!=item['speaker']:
            raise ValueError(f'用户身份或顺序不一致：{item["id"]}')
        if users!=[v['agent_name'] for v in item['videos']]:
            raise ValueError('GRPO用户顺序与题集六路顺序不一致')
        if (item['frames']!=full_item['frames'] or item['question']!=full_item['question']
                or item['answer']!=full_item['answer'] or item['options'].get(item['correct'])!=item['answer']):
            raise ValueError(f'题面、答案或原始帧清单不一致：{item["id"]}')
        if set(item['frames'])!=set(users) or any(len(item['frames'][u])!=item['frame_counts'][u] for u in users):
            raise ValueError('帧数与原生成器清单不一致')
        if len(item['frames'][item['speaker']])!=300:
            raise ValueError('speaker必须保留完整300帧')
        for paths in item['frames'].values():
            for path in paths:safe_relative(path)
        meta=full_item['generator_media_metadata']
        if meta['required_users']!=users:
            raise ValueError('metadata中的用户顺序错误')
        for clip in meta['clips']:
            if clip['generator_frame_input']['frame_count']!=len(item['frames'][clip['user']]):
                raise ValueError('metadata与实际帧数不符')
        frame_store[item['id']]={'required_users':users,'frames':item['frames'],
                                'frame_counts':[len(item['frames'][u]) for u in users],'metadata':meta}
    header=json.loads((root/'prompt_header.json').read_text(encoding='utf-8'))
    return items,frame_store,header


def build_input(item,fr,setting,header):
    if setting not in SETTINGS:raise ValueError(f'未知条件：{setting}')
    users=reference.setting_users(item,fr,setting)
    paths=[p for user in users for p in fr['frames'][user]]
    if not paths or len(paths)!=len(set(paths)):
        raise ValueError('缺帧或同题输入重复')
    text=reference.build_text(item,fr,users,header)
    content=[{'type':'image'} for _ in paths]+[{'type':'text','text':text}]
    return users,paths,[{'role':'user','content':content}]


def cache_path(cache_root,relative):
    return Path(cache_root)/safe_relative(relative).with_suffix('.png')


def paired_summary(groups):
    maps={s:{r['id']:r for r in groups[s]} for s in SETTINGS}
    ids=set(maps[SETTINGS[0]])
    if len(ids)!=131 or any(set(maps[s])!=ids or len(groups[s])!=131 for s in SETTINGS):
        raise ValueError('三组必须完整配对全部131题；不能缩小分母')
    if any(len({maps[s][i]['correct'] for s in SETTINGS})!=1 for i in ids):
        raise ValueError('三组gold不一致')
    result={'conditions':{s:reference.summarize(groups[s]) for s in SETTINGS},'comparisons':{}}
    for a,b in [('evidence2','all6'),('evidence2','speaker1'),('all6','speaker1')]:
        first=sum(maps[a][i]['is_correct'] and not maps[b][i]['is_correct'] for i in ids)
        second=sum(not maps[a][i]['is_correct'] and maps[b][i]['is_correct'] for i in ids)
        both=sum(maps[a][i]['is_correct'] and maps[b][i]['is_correct'] for i in ids)
        result['comparisons'][f'{a}_minus_{b}']={'n':131,'gap_pp':100*(first-second)/131,
            'first_only_correct':first,'second_only_correct':second,'both_correct':both,
            'both_wrong':131-first-second-both}
    return result


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
