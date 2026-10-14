import importlib.util
from pathlib import Path
import unittest


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('runtime',Path(__file__).with_name('holdout_runtime.py'))
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.config={'train_python':'/env/python','train_dataset':'/data/train.jsonl','val_dataset':'/data/validation.jsonl',
            'test_dataset':'/data/test.jsonl','speaker_holdout_validation':True,'validation_seed':42}

    def test_data_command_validates_three_sets_and_explicit_mode(self):
        command=self.module.data_validation_command(self.config)
        self.assertIn('/data/test.jsonl',command)
        self.assertIn('--speaker-holdout',command)
        command=self.module.data_validation_command(dict(self.config,speaker_holdout_validation=False))
        self.assertNotIn('--speaker-holdout',command)
        self.assertNotIn('/data/test.jsonl',command)

    def test_test_configuration_changes_only_dataset_and_keeps_baseline_separate(self):
        result=self.module.evaluation_files(self.config,Path('/output'),adapter='/checkpoint',split='test',validation_baseline='/old/base.json')
        self.assertEqual(result['config']['val_dataset'],'/data/test.jsonl')
        self.assertEqual(result['baseline'],'/output/test_baseline.json')
        self.assertEqual(result['label'],'test_policy')
        self.assertEqual(self.config['val_dataset'],'/data/validation.jsonl')
        self.assertEqual(result['config']['validation_seed'],42)

    def test_validation_uses_original_baseline(self):
        result=self.module.evaluation_files(self.config,Path('/output'),split='validation',validation_baseline='/old/base.json')
        self.assertEqual(result['baseline'],'/old/base.json')
        self.assertEqual(result['label'],'baseline')


if __name__=='__main__':unittest.main()
