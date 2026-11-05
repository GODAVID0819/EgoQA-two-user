"""egolife_131_AE_revised_with_original_20261101(2).jsonl -> vqa JSON，选项只用 revised_options（不带括号里的 original）。
correct 用文件里的 correct；answer 换成 revised_answer。其他字段原样保留。
  python build_ae_revised_vqa.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path('/scratch/tx856/multi_user_qa/dual_full120/egolife_131_AE_revised_with_original_20261101(2).jsonl')
OUT = HERE / 'ae_revised131_vqa.json'


def main():
    rows = [json.loads(s) for s in SRC.read_text().splitlines() if s.strip()]
    ids = json.loads((HERE / 'subsets/textwrong_any131.json').read_text())
    assert len(rows) == 131 and sorted(r['id'] for r in rows) == sorted(ids)
    items = []
    for r in rows:
        opts = r['revised_options']
        assert list(opts) == list('ABCDE') and len(set(opts.values())) == 5, r['id']
        assert opts[r['correct']] == r['revised_answer'], r['id']
        assert all('（original:' not in v for v in opts.values()), r['id']
        items.append({**r, 'options': dict(opts), 'answer': r['revised_answer']})
    meta = {'count': len(items), 'source': SRC.name, 'options': 'revised_options (no original text)'}
    OUT.write_text(json.dumps({'meta': meta, 'items': items}, ensure_ascii=False, indent=2))
    print('items', len(items), 'correct dist', {k: sum(i['correct'] == k for i in items) for k in 'ABCDE'})


if __name__ == '__main__':
    main()
