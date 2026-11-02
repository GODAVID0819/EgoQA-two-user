"""v7 不使用历史池奖励；保留现有 launcher 参数接口。"""
import json
from pathlib import Path
def prepare_history(config, output):
    p=Path(output)/'moda_runtime.json'
    p.write_text(json.dumps({'diversity_lambda':0}),encoding='utf-8')
    return p
