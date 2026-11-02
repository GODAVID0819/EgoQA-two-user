"""方案A：保留完整QA字段，只追加简洁输出要求并生成独立输入副本。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

PROFILE='compact_full_qa_v1'
BREVITY_INSTRUCTIONS='''Output brevity requirements for this run:
- Keep every field in the exact JSON shape above. Do not remove fields, add a preamble, or wrap the object in Markdown.
- Return compact JSON without indentation. Complete the entire JSON object before stopping.
- Preserve a clear question, exactly five complete options, the correct letter, and the exact answer text. Do not shorten them in a way that makes the question ambiguous.
- Describe each required evidence fact once, using a short concrete sentence. Include the evidence actually needed, without duplicating the same claim across entries or inventing support to fill space.
- In single_user_answerability, retain one entry for each of the six users; use one short sentence per entry, preferably at most 20 words.
- Keep generator_rationale, why_two_users_needed, and review.generator_self_check to at most 40 words each. State the decisive point without repeating the question, options, or judging instructions.
- Keep combined_answerability to one short sentence and each per_user_evidence_claims claim concise. Keep supported temporal references intact.
- These brevity requirements never permit omitting material evidence, guessing visual facts, or changing the required answer structure.'''


def profile_prompt(prompt, profile=None):
    if profile is None:return prompt
    if profile != PROFILE:raise ValueError('未知生成设置：'+str(profile))
    return prompt+'\n\n'+BREVITY_INSTRUCTIONS


def apply_profile(row, profile):
    if profile != PROFILE:raise ValueError('未知生成设置：'+str(profile))
    result=deepcopy(row)
    if result.get('generation_profile')==profile:return result
    if result.get('generation_profile') is not None:raise ValueError('不能覆盖另一生成设置')
    if len(result.get('messages',[]))!=1 or result['messages'][0]['role']!='user':
        raise ValueError('预期一条完整生成请求')
    result['messages'][0]['content']=profile_prompt(result['messages'][0]['content'],profile)
    result['generation_profile']=profile
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from .data import read_rows,validate_row,validate_splits
    rows={split:[apply_profile(row,PROFILE) for row in read_rows(args.source/(split+'.jsonl'))]
          for split in ('train','validation')}
    if {k:len(v) for k,v in rows.items()}!={'train':18,'validation':6}:
        raise ValueError('来源训练/验证输入数量不是18/6')
    validate_splits(rows)
    for values in rows.values():
        for row in values:validate_row(row)
    args.output.mkdir(parents=True,exist_ok=False)
    for split,values in rows.items():
        with (args.output/(split+'.jsonl')).open('x',encoding='utf-8') as stream:
            for row in values:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    record={'source':str(args.source.resolve()),'generation_profile':PROFILE,
            'rows':{k:len(v) for k,v in rows.items()},'media_and_split_unchanged':True}
    (args.output/'generation_profile.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
