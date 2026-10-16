"""从真实Judge的包路径运行评分CPU逻辑，禁止依赖项目根工作目录。"""
import importlib
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import json


class PackageDecodeImportTest(unittest.TestCase):
    def test_judge_initialization_failure_is_recorded_before_reraise(self):
        service = importlib.import_module('training.grpo_v3.six_user_binary.service')
        class Broken:
            @classmethod
            def from_config(cls, config):
                raise ModuleNotFoundError('controlled CPU initialization failure')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'judge_config.json'
            module = 'training.grpo_v3.six_user_binary.predictor'
            with patch.dict('sys.modules', {module:N(VllmJudge=Broken)}):
                with self.assertRaises(ModuleNotFoundError):
                    service.load_startup_scorer(str(config), 'cpu-test')
            error = json.loads(config.with_suffix('.startup_error.json').read_text())
            self.assertEqual(error['error'], 'ModuleNotFoundError')
            with self.assertRaisesRegex(RuntimeError, 'controlled CPU initialization failure'):
                service.raise_startup_error(config)

    def test_no_startup_error_does_not_block_normal_health_polling(self):
        service = importlib.import_module('training.grpo_v3.six_user_binary.service')
        with tempfile.TemporaryDirectory() as tmp:
            service.raise_startup_error(Path(tmp)/'judge_config.json')

    def test_predict_many_works_without_top_level_decode_pipeline(self):
        p = importlib.import_module('training.grpo_v3.six_user_binary.predictor')
        d = importlib.import_module('training.grpo_v3.six_user_binary.decode_pipeline')
        obj = p.VllmJudge.__new__(p.VllmJudge)
        obj.config = {'cpu_prefetch': True}
        obj.sleeping = False
        obj.adapters = {}
        obj.sampling = None
        obj.pass_id, obj.fail_id = 4, 5
        obj._effective_batch_size = 8
        obj._prepared = {}
        obj._next_examples = []
        obj._last_preemptions = None
        obj.pipeline = d.DecodePipeline(2)
        self.addCleanup(obj.pipeline.close)
        obj.prepare = lambda e: ({'prompt': e.example_id}, {})
        class Engine:
            def get_metrics(self):
                return [N(name='vllm:num_preemptions', value=0)]
            def generate(self, values, *args, **kwargs):
                return [N(outputs=[N(logprobs=[{4:N(logprob=-1), 5:N(logprob=-2)}])],
                          prompt_token_ids=[1], num_cached_tokens=0) for _ in values]
        obj.llm = Engine()
        result = obj.predict_many([N(example_id='a', task=N(value='answerability'))])
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]['pass_probability'], .5)


if __name__ == '__main__':
    unittest.main()
