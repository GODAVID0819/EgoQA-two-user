"""三轮反馈环：反馈插入位置与原流水线一致、Judge反馈解析/拼接、接受与续跑逻辑、服务端反馈路由。"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

VAL = Path('/scratch/tx856/multi_user_qa/dual_full120/data/trainval/validation.jsonl')


class FeedbackInjectionTests(unittest.TestCase):
    def test_matches_pipeline_prompt_with_feedback(self):
        """对同一packet：无反馈prompt经插入 == 原流水线 build_video_generation_prompt(feedback=..., previous_generation=...)。"""
        root = Path(__file__).resolve().parent / 'multi-user'
        sys.path.insert(0, str(root / 'tests'))
        from test_six_user_video_qa_loop import six_user_packet
        from training.judge_sft.prepare_real_data import _repo_module
        from training.grpo_v3.six_user_binary.feedback_loop import inject_feedback
        prompts = _repo_module('prompts')
        packet = six_user_packet()
        previous = json.dumps({'question': 'What did we eat?', 'options': list('abcde'), 'correct': 'A', 'answer': 'a'})
        plain = prompts.build_video_generation_prompt(packet, 'neutral')
        expected = prompts.build_video_generation_prompt(packet, 'neutral', feedback='qa_formality: bad. fix it',
                                                         previous_generation=previous)
        self.assertEqual(inject_feedback(plain, 'qa_formality: bad. fix it', previous), expected)
        self.assertEqual(inject_feedback(plain + '\n\nBREVITY', 'x', previous),
                         prompts.build_video_generation_prompt(packet, 'neutral', feedback='x', previous_generation=previous) + '\n\nBREVITY')
        self.assertEqual(inject_feedback(plain, None, None), plain)

    def test_every_validation_prompt_has_unique_insertion_point(self):
        from training.grpo_v3.six_user_binary.feedback_loop import MARKER
        rows = [json.loads(s) for s in VAL.read_text().splitlines() if s.strip()]
        self.assertEqual(len(rows), 120)
        self.assertTrue(all(r['messages'][0]['content'].count(MARKER) == 1 for r in rows))


class FeedbackTextTests(unittest.TestCase):
    def test_parse_complete_and_truncated(self):
        from training.grpo_v3.six_user_binary.feedback_loop import parse_feedback
        self.assertEqual(parse_feedback('", "reason": "Third-person.", "fix": "Use I."}'),
                         {'verdict': 'fail', 'reason': 'Third-person.', 'fix': 'Use I.'})
        self.assertEqual(parse_feedback('", "reason": "The \\"desk\\" is not shown", "missing_evidence": ["desk", "monitor"'),
                         {'reason': 'The "desk" is not shown', 'missing_evidence': ['desk', 'monitor']})

    def test_pipeline_format(self):
        from training.grpo_v3.six_user_binary.feedback_loop import judge_feedback
        explained = {'formality': {'continuation': '", "reason": "Third-person.", "fix": "Use I."}'},
                     'all_six': {'continuation': '", "reason": "Not visible.", "available_evidence": [], "missing_evidence": ["desk"]}'}}
        self.assertEqual(judge_feedback(['formality', 'all_six'], explained),
                         'qa_formality: Third-person. Use I. | answerability: Not visible. Missing evidence: desk')
        self.assertEqual(judge_feedback(['groundedness'], {'groundedness': {'continuation': '"'}}),
                         'evidence_groundedness: judge verdict is fail')


class LoopStateTests(unittest.TestCase):
    def test_accept_feedback_and_resume(self):
        from training.grpo_v3.six_user_binary.feedback_loop import new_items, pending, record_attempt, next_prompt_inputs
        items = new_items([{'evidence_id': 'e1', 'global_validation_index': 0}, {'evidence_id': 'e2', 'global_validation_index': 1}], 2)
        self.assertEqual([(i['evidence_id'], i['sample']) for i in pending(items, 1)], [('e1', 0), ('e1', 1), ('e2', 0), ('e2', 1)])
        good = {'status': 'scored', 'probabilities': {'formality': .9, 'groundedness': .8, 'all_six': .7}}
        bad = {'status': 'scored', 'probabilities': {'formality': .9, 'groundedness': .5, 'all_six': .2}}
        invalid = {'status': 'invalid_completion', 'reason': '必须有五个非空选项'}
        fb = {'groundedness': {'continuation': '", "reason": "R", "fix": "F"}'},
              'all_six': {'continuation': '", "reason": "A", "available_evidence": [], "missing_evidence": []}'}}
        record_attempt(items[0], 1, 7, None, 'q0', good, {})
        record_attempt(items[1], 1, 8, None, 'q1', bad, fb)
        record_attempt(items[2], 1, 9, None, 'q2', invalid, {})
        self.assertEqual(items[0]['accepted_attempt'], 1)
        self.assertIsNone(items[0]['attempts'][0]['feedback_out'])
        self.assertEqual(items[1]['attempts'][0]['pass'], {'formality': True, 'groundedness': False, 'all_six': False})
        self.assertEqual(items[1]['attempts'][0]['feedback_out'], 'evidence_groundedness: R F | answerability: A')
        self.assertEqual(next_prompt_inputs(items[1]), ('evidence_groundedness: R F | answerability: A', 'q1'))
        self.assertTrue(items[2]['attempts'][0]['feedback_out'].startswith('Generator output was not valid'))
        self.assertEqual(items[2]['attempts'][0]['pass'], {'formality': False, 'groundedness': False, 'all_six': False})
        # 第4题没跑完第1轮：续跑时第1轮只剩它；第2轮是未接受的1、2题（以及补完后的3题）
        self.assertEqual([i['evidence_id'] + str(i['sample']) for i in pending(items, 1)], ['e21'])
        self.assertEqual([i['evidence_id'] + str(i['sample']) for i in pending(items, 2)], ['e11', 'e20'])


class SummaryTests(unittest.TestCase):
    def test_round_rates_carry_accepted_results(self):
        from eval_feedback_loop import summarize
        def att(r, f, g, a):
            return {'attempt': r, 'status': 'scored', 'pass': {'formality': f, 'groundedness': g, 'all_six': a}, 'all_pass': f and g and a}
        items = [{'accepted_attempt': 1, 'attempts': [att(1, True, True, True)]},
                 {'accepted_attempt': 2, 'attempts': [att(1, True, False, True), att(2, True, True, True)]},
                 {'accepted_attempt': None, 'attempts': [att(1, False, False, False), att(2, True, False, False), att(3, True, True, False)]}]
        s = summarize(items, 3)
        self.assertEqual([r['pass_count_after_round'] for r in s['rounds']], [
            {'formality': 2, 'groundedness': 1, 'all_six': 2, 'all_three': 1},
            {'formality': 3, 'groundedness': 2, 'all_six': 2, 'all_three': 2},
            {'formality': 3, 'groundedness': 3, 'all_six': 2, 'all_three': 2}])
        self.assertEqual([r['judged_this_round'] for r in s['rounds']], [3, 2, 1])
        self.assertEqual(s['accepted_at_attempt'], {'1': 1, '2': 1, '3': 0})
        self.assertEqual((s['not_accepted'], s['mean_attempts']), (1, 2.0))
        self.assertAlmostEqual(s['final']['all_three'], 2 / 3)


class ServiceExplainTests(unittest.TestCase):
    def test_routes_only_requested_failed_keys(self):
        from training.grpo_v3.six_user_binary import service

        class Task:
            def __init__(self, v): self.value = v

        class Example:
            def __init__(self, key): self.key, self.task = key, Task(key)

        class Predictor:
            identity = {'model_id': 'm'}
            calls = []
            def explain_many(self, examples, max_tokens, forced='fail'):
                self.calls.append(([e.key for e in examples], max_tokens, forced))
                # formality 自然写出 reason；其余在 fail 后直接结束，需预填 reason 重试
                return [{'continuation': f'", "reason": "{e.key}"}}' if e.key == 'formality' or forced != 'fail' else ','}
                        for e in examples]

        scorer = service.CandidateScorer(Predictor(), 'inst')
        examples = lambda r: {k: Example(k) for k in ('formality', 'groundedness', 'all_six')}
        with patch.object(service, 'build_examples', side_effect=examples):
            out = scorer.explain_many([{'request_id': 'a', 'evidence_id': 'e', 'keys': ['groundedness'], 'max_tokens': 99},
                                       {'request_id': 'b', 'evidence_id': 'e', 'keys': ['formality', 'all_six']}])
        self.assertEqual([set(o['feedback']) for o in out], [{'groundedness'}, {'formality', 'all_six'}])
        pre = service.REASON_PREFILL
        self.assertEqual(Predictor.calls, [(['formality'], 99, 'fail'), (['groundedness'], 99, 'fail'), (['groundedness'], 99, pre),
                                           (['all_six'], 99, 'fail'), (['all_six'], 99, pre)])
        from training.grpo_v3.six_user_binary.feedback_loop import parse_feedback
        self.assertNotIn('reason_prefilled', out[1]['feedback']['formality'])
        self.assertEqual(out[0]['feedback']['groundedness']['unprefilled_continuation'], ',')
        self.assertEqual(parse_feedback(out[0]['feedback']['groundedness']['continuation'])['reason'], 'groundedness')
        with self.assertRaises(ValueError):
            scorer.explain_many([{'request_id': 'a', 'evidence_id': 'e', 'keys': ['speaker_only']}])


if __name__ == '__main__':
    unittest.main()
