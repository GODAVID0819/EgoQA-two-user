"""核对Judge实际读取的三个提示词文件；保持原成功实验文字与占位符。"""
import json
from pathlib import Path

PROMPT_FIELDS={
    'formality_prompt.txt':['<DETERMINISTIC_SCHEMA_RESULT_JSON>','<PARTICIPANT_NAMES_JSON>','<QUESTION_ANSWER_ITEM_JSON>'],
    'groundedness_model_prompt.txt':['{video_packet_brief(packet)}','{qa_item JSON}'],
    'combined_answerability_judge_prompt.txt':['{question}'],
}


def validate_prompt_files(current,source):
    current=Path(current);source=Path(source);templates=[]
    for name,fields in PROMPT_FIELDS.items():
        original=(source/name).read_bytes();deployed=(current/name).read_bytes()
        if not original or original!=deployed:raise ValueError('Judge模板必须与原成功实验原文字节一致：'+name)
        text=deployed.decode('utf-8-sig')
        if any(text.count(field)!=1 for field in fields):raise ValueError('Judge模板占位符数量不符：'+name)
        templates.append({'name':name,'bytes':len(deployed),'source':str(source/name),'deployed':str(current/name)})
    return {'status':'passed','templates':templates,'reward_prompt_changed':False}


def validate_runtime_prompts(root):
    root=Path(root)
    provenance=json.loads((root/'preflight/source_provenance.json').read_text())
    judge=json.loads((root/'configs/judge_fec.json').read_text())
    source=Path(provenance['source_root'])/'prompts'
    if Path(judge['prompt_directory'])!=source:raise ValueError('Judge提示词来源与原成功实验不一致')
    report=validate_prompt_files(root/'prompts',source)
    (root/'preflight/judge_prompt_assets.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
