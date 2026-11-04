"""将私有题库、公开视频元信息与可选翻译合成为独立标注页面。"""
import argparse,json
from pathlib import Path
from urllib.parse import quote


def build(rows,metadata,dataset,revision,assignment,translations=None):
    windows={};questions=[]; translations=translations or {}
    for row in rows:
        qa=row['qa'];zh=translations.get(row['id'],{}); names=row['windows']
        if len(set(names))!=12 or names[0]!=row['speaker'] or len(qa['options'])!=5:raise ValueError(row['id'])
        questions.append(dict(id=row['id'],seq=row['seq'],speaker=row['speaker'],windows=names,
            question=qa['question'],options=qa['options'],question_zh=zh.get('question',''),options_zh=zh.get('options',['']*5),
            proposed_option=qa['correct'],proposed_answer=qa['answer']))
        for name in names:
            if name in windows:continue
            start=(int(name[0])-1)*300
            clips=sorted((x for x in metadata if x['user']=='user_'+name[1].lower() and start<=x['source_start_seconds']<start+300 and x['group'].startswith(tuple(f'group_{k:02}_' for k in range(1,5)))),key=lambda x:x['source_start_seconds'])
            if [x['source_start_seconds'] for x in clips]!=list(range(start,start+300,30)):raise ValueError(name)
            windows[name]=dict(person='参与者'+name[1],start_seconds=start,duration_seconds=300,segments=[dict(source_start_seconds=x['source_start_seconds'],source_end_seconds=x['source_end_seconds'],url=f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{quote(x['file_name'],safe='/')}") for x in clips])
    data=json.dumps(dict(assignment_id=assignment,questions=questions,windows=windows),ensure_ascii=False).replace('</',r'<\/')
    return Path(__file__).with_name('review_template.html').read_text(encoding='utf8').replace('__DATA__',data)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('selection','metadata','output'):ap.add_argument(k,type=Path)
    for k in ('dataset','revision','assignment'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--translations',type=Path)
    args=ap.parse_args();rows=json.loads(args.selection.read_text(encoding='utf8'))
    meta=[json.loads(l) for l in args.metadata.read_text(encoding='utf8').splitlines() if l.strip()]
    zh=json.loads(args.translations.read_text(encoding='utf8')) if args.translations else {}
    with args.output.open('x',encoding='utf8') as f:f.write(build(rows,meta,args.dataset,args.revision,args.assignment,zh))
