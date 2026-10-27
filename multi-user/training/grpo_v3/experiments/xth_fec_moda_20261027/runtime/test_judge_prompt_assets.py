"""复现20511562遗漏运行时提示词，并验证只复用原成功实验的原文。"""
from pathlib import Path
import tempfile
import unittest

try:
    from judge_prompt_assets import validate_prompt_files,PROMPT_FIELDS
except ModuleNotFoundError:
    validate_prompt_files=None
    PROMPT_FIELDS={'formality_prompt.txt':['<DETERMINISTIC_SCHEMA_RESULT_JSON>','<PARTICIPANT_NAMES_JSON>','<QUESTION_ANSWER_ITEM_JSON>'],
        'groundedness_model_prompt.txt':['{video_packet_brief(packet)}','{qa_item JSON}'],
        'combined_answerability_judge_prompt.txt':['{question}']}


class PromptAssetsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validate_prompt_files,'缺少运行时Judge模板检查')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'source';self.current=Path(self.temp.name)/'current'
        self.source.mkdir();self.current.mkdir()
        for name,fields in PROMPT_FIELDS.items():
            (self.source/name).write_text('Original template\n'+'\n'.join(fields),encoding='utf-8')

    def test_missing_deployed_template_fails_before_gpu(self):
        with self.assertRaises(FileNotFoundError):validate_prompt_files(self.current,self.source)

    def test_identical_source_templates_pass_without_modification(self):
        for name in PROMPT_FIELDS:(self.current/name).write_bytes((self.source/name).read_bytes())
        result=validate_prompt_files(self.current,self.source)
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['templates']),3)

    def test_changed_prompt_is_rejected(self):
        for name in PROMPT_FIELDS:(self.current/name).write_bytes((self.source/name).read_bytes())
        with (self.current/'formality_prompt.txt').open('a',encoding='utf-8') as f:f.write('\nchanged reward instruction')
        with self.assertRaises(ValueError):validate_prompt_files(self.current,self.source)


if __name__=='__main__':unittest.main(verbosity=2)
