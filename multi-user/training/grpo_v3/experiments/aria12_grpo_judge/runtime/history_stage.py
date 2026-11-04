"""纯质量实验不读取历史题池或加载问题编码器。"""
from pathlib import Path
import json


def prepare_history(config, output):
    if config.get('diversity_lambda')!=0 or config.get('history_pool_directory') is not None:
        raise ValueError('已确认取消多样性和历史题池')
    output=Path(output)
    runtime={'diversity_lambda':0,'encoder_directory':None,'history_pool_directory':None}
    path=output/'moda_runtime.json'
    path.write_text(json.dumps(runtime,indent=2))
    (output/'progress_stage.json').write_text(json.dumps({'phase':'quality_only_no_history','diversity_lambda':0}))
    return path
