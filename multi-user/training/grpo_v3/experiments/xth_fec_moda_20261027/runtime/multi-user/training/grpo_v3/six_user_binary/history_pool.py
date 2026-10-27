"""直接用旧F/E/A/S筛固定历史池；本轮不做历史重评分。"""
from collections import Counter
from pathlib import Path
import csv
import json
import random
import math

import numpy as np

from training.grpo_v3.six_user_binary.diversity import checked_vectors, window_id
HISTORY_SCORE_SOURCE = 'old_training_F_E_A_S'


def reference_qualified(record):
    scores = record.get('probabilities_old_F_E_A_S')
    if not isinstance(scores,dict) or set(scores)!={'formality','groundedness','all_six','speaker_only'}:
        raise ValueError('历史池必须直接使用旧F/E/A/S，不混入或推造新C')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
        raise ValueError('旧Judge分数必须为[0,1]有限数值')
    return (record['quality_reward_old'] >= .4 and scores['formality'] >= .5
            and scores['groundedness'] >= .4 and scores['all_six'] >= .5 and scores['speaker_only'] < .5)


def select_history(records, windows, maximum=8, seed=42):
    if maximum != 8:
        raise ValueError('已批准历史容量上限为8，不强行补满')
    if len(set(windows)) != len(windows): raise ValueError('窗口列表重复')
    identifiers = [r['history_qa_id'] for r in records]
    if len(set(identifiers)) != len(identifiers): raise ValueError('历史候选ID重复')
    by_window = {window: [] for window in windows}
    rng = random.Random(seed)
    for record in sorted(records, key=lambda r:r['history_qa_id']):
        window = window_id(record['source_packet_id'])
        if window not in by_window: raise ValueError('历史QA不属于本轮训练窗口')
        if reference_qualified(record): by_window[window].append((rng.random(),record))
    selected = []
    counts = {}
    for window in sorted(windows):
        chosen = [r for _,r in sorted(by_window[window],key=lambda pair:pair[0])[:maximum]]
        selected.extend(chosen)
        counts[window] = len(chosen)
    return selected, counts


def mean_normalize(hidden, attention_mask):
    hidden = np.asarray(hidden)
    mask = np.asarray(attention_mask)[...,None].astype(np.float32)
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape[:2] or not np.isfinite(hidden).all():
        raise ValueError('编码器输出或attention mask维度无效')
    pooled = (hidden*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1e-9)
    norms = np.linalg.norm(pooled,axis=1,keepdims=True)
    if not np.isfinite(norms).all() or (norms<=0).any(): raise ValueError('编码器产生零或非有限问题向量')
    result = pooled/norms
    checked_vectors(result)
    return result.astype(np.float32)


class MiniLMEncoder:
    def __init__(self,directory):
        from tokenizers import Tokenizer
        import onnxruntime as ort
        directory = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(directory/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        options = ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session = ort.InferenceSession(str(directory/'onnx/model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_names = [item.name for item in self.session.get_inputs()]

    def encode(self,questions):
        if not questions: return np.empty((0,384),dtype=np.float32)
        if any(not isinstance(q,str) or not q.strip() for q in questions): raise ValueError('只编码非空问题正文')
        output = []
        for offset in range(0,len(questions),16):
            encoded = self.tokenizer.encode_batch(questions[offset:offset+16])
            inputs = {'input_ids':np.array([r.ids for r in encoded],dtype=np.int64),
                      'attention_mask':np.array([r.attention_mask for r in encoded],dtype=np.int64),
                      'token_type_ids':np.array([r.type_ids for r in encoded],dtype=np.int64)}
            hidden = self.session.run(None,{key:inputs[key] for key in self.input_names})[0]
            output.append(mean_normalize(hidden,inputs['attention_mask']))
        vectors=np.concatenate(output)
        checked_vectors(vectors,expected_count=len(questions),dimension=384)
        return vectors


def save_history(records,windows,encoder,output,*,source_metadata,old_judge_config=None):
    selected,counts = select_history(records,windows)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    vectors=encoder.encode([r['question'] for r in selected])
    bank=[]
    for index,row in enumerate(selected):
        bank.append({**row,'window_id':window_id(row['source_packet_id']),'embedding_row':index})
    (output/'history_bank.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in bank),encoding='utf-8')
    np.save(output/'question_embeddings.npy',vectors)
    with (output/'window_counts.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['window_id','selected_QA'])
        writer.writerows(sorted(counts.items()))
    metadata={'status':'passed','source_metadata':source_metadata,'old_judge_config':old_judge_config,
              'score_source':HISTORY_SCORE_SOURCE,
              'selection':'直接旧质量reward>=0.4、旧F>=0.5/E>=0.4/A>=0.5且S<0.5；seed42，每窗最多8，本轮固定，不重评分',
              'old_F_min':.5,'old_E_min':.4,'old_A_min':.5,'old_S_max_exclusive':.5,'capacity':8,'fixed_for_this_run':True,
              'window_count':len(windows),'selected_QA':len(bank),'empty_windows':sum(v==0 for v in counts.values()),
              'selected_count_distribution':dict(Counter(counts.values())),
              'encoder_revision':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41','dimension':384,
              'historical_rescoring_performed':False,'history_preparation_device':'CPU',
              'thresholds_are_not_human_quality_calibration':True}
    (output/'pool_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


def load_history(directory):
    directory=Path(directory)
    metadata=json.loads((directory/'pool_metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('status')!='passed' or metadata.get('score_source')!=HISTORY_SCORE_SOURCE
            or metadata.get('historical_rescoring_performed') is not False
            or [metadata.get(k) for k in ['old_F_min','old_E_min','old_A_min','old_S_max_exclusive','capacity']] != [.5,.4,.5,.5,8]):
        raise ValueError('历史池必须来自已批准的旧分数门槛，不进行历史重评分')
    rows=[json.loads(line) for line in (directory/'history_bank.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    vectors=np.load(directory/'question_embeddings.npy',allow_pickle=False)
    checked_vectors(vectors,expected_count=len(rows),dimension=384)
    if len({r['history_qa_id'] for r in rows})!=len(rows) or metadata['selected_QA']!=len(rows):
        raise ValueError('历史池数量或ID重复')
    by_window={}
    for index,row in enumerate(rows):
        if not reference_qualified(row) or row['embedding_row']!=index or row['window_id']!=window_id(row['source_packet_id']):
            raise ValueError('历史池质量、向量行或窗口不匹配')
        by_window.setdefault(row['window_id'],[]).append({**row,'embedding':vectors[index]})
    if any(len(rows)>8 for rows in by_window.values()): raise ValueError('历史池超出每窗容量')
    return by_window,metadata
