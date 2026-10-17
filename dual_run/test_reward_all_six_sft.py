"""all_six_sft奖励与Judge prompt：A只判断六人合并足以回答(由340 SFT联合prompt删去speaker条件)，F用formality SFT prompt。"""
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

FIX=Path(__file__).resolve().parent/'fixtures'
ANS_MANIFEST=Path('/scratch/hm2991/egolife_judge_sft/answerability_842_question_split/manifests_video_split/train/answerability_joint.jsonl')
NAMES=['Lucia','Jake','Alice','Tasha','Katrina','Shure']


class AllSixRewardTests(unittest.TestCase):
    def test_joint_formula(self):
        from training.grpo_v3.six_user_binary.reward import aggregate
        p={'formality':.5,'groundedness':.25,'all_six':.75}
        self.assertAlmostEqual(aggregate(p,mode='all_six_sft'),.2*.5+.4*.25+.4*.75)

    def test_speaker_only_key_rejected(self):
        from training.grpo_v3.six_user_binary.reward import aggregate
        with self.assertRaises(ValueError):
            aggregate({'formality':.5,'groundedness':.5,'all_six':.5,'speaker_only':.5},mode='all_six_sft')
        for old in ('continuous','binary','all_six','joint'):
            with self.assertRaises(ValueError):aggregate({'formality':.5,'groundedness':.5,'all_six':.5},mode=old)

    def test_plugin_modes(self):
        from training.grpo_v3.six_user_binary.plugin import MODES
        self.assertIn('all_six_sft',MODES)


class PromptTests(unittest.TestCase):
    def test_joint_prompt_matches_sft_text(self):
        from training.grpo_v3.six_user_binary.joint_answerability_prompt import PROMPT_TEMPLATE,build_joint_answerability_prompt
        self.assertEqual(PROMPT_TEMPLATE,(FIX/'joint_answerability_judge_sft_prompt.txt').read_text())
        self.assertIn('Generated question:\nQ?\n\nReturn',build_joint_answerability_prompt('Q?'))

    def test_all_six_prompt_drops_only_speaker_condition(self):
        from training.grpo_v3.six_user_binary.joint_answerability_prompt import ALL_SIX_TEMPLATE,PROMPT_TEMPLATE
        self.assertEqual(ALL_SIX_TEMPLATE,(FIX/'all_six_answerability_judge_prompt.txt').read_text())
        self.assertNotIn('speaker-only',ALL_SIX_TEMPLATE)
        self.assertNotIn('insufficient to answer',ALL_SIX_TEMPLATE)
        kept=[l for l in PROMPT_TEMPLATE.splitlines() if 'speaker-only' not in l and 'requirement' not in l
              and not l.startswith(('1. ','2. ','Judge the following'))]
        self.assertTrue(all(l in ALL_SIX_TEMPLATE.splitlines() for l in kept))

    @unittest.skipUnless(ANS_MANIFEST.exists(),'SFT manifest不可读')
    def test_joint_prompt_matches_every_sft_row(self):
        from training.grpo_v3.six_user_binary.joint_answerability_prompt import build_joint_answerability_prompt
        rows=[json.loads(l) for l in ANS_MANIFEST.open()]
        q=lambda p:p[p.index('Generated question:\n')+20:p.index('\n\nReturn exactly one JSON object')]
        self.assertEqual(sum(build_joint_answerability_prompt(q(r['prompt']))==r['prompt'] for r in rows),len(rows))

    def test_formality_prompt_matches_sft_text(self):
        from training.judge_sft.prepare_real_data import _repo_module
        qa={'question_type':'neutral','question':'Q?','options':list('abcde'),'correct':'A','answer':'a',
            'required_users':NAMES,'speaker_user':NAMES[0]}
        g=_repo_module('prompts').build_qa_formality_judge_prompt(qa,{'required_users':NAMES},schema_errors=[])
        for head,tail,ph in (('formality branch:\n','\n\nKnown','<DETERMINISTIC_SCHEMA_RESULT_JSON>'),
                             ('detection only:\n','\n\nUser-facing','<PARTICIPANT_NAMES_JSON>'),
                             ('answer item:\n','\n\nReturn exactly','<QUESTION_ANSWER_ITEM_JSON>')):
            g=re.sub('('+re.escape(head)+').*?('+re.escape(tail)+')',lambda m:m[1]+ph+m[2],g,flags=re.S)
        self.assertEqual(g,(FIX/'formality_judge_sft_prompt.txt').read_text())

    def test_build_examples_routes_three_keys(self):
        from training.grpo_v3.six_user_binary import judge
        view={'required_users':NAMES,'source_packet_id':'pk','evidence_id':'ev'}
        qa={'question':'Where did the cup go?','options':list('abcde'),'correct':'A','answer':'a'}
        frames=lambda v,speaker_only:{'frame_order':['g']*(1 if speaker_only else 6),'speaker_only_flag':speaker_only}
        with patch.object(judge,'full_judge_view',return_value=view),patch.object(judge,'parse_completion',return_value=qa),\
             patch.object(judge,'_frame_fields',side_effect=frames),patch.object(judge,'normalized_record_to_example',side_effect=lambda r:r):
            out=judge.build_examples({'dataset_root':'x','source_packet_id':'pk','asker_index':0,'evidence_id':'ev',
                                      'completion':'{}','request_id':'r0'})
        self.assertEqual(set(out),{'formality','groundedness','all_six'})
        a=out['all_six']
        self.assertEqual((a['task'],a['condition_type'],len(a['frame_order']),a['speaker_only_flag']),
                         ('answerability','combined_all_six_users',6,False))
        self.assertTrue(a['prompt'].startswith('You are the cross-user answerability judge'))
        self.assertNotIn('speaker-only',a['prompt'])
        self.assertIn('Where did the cup go?',a['prompt'])


if __name__=='__main__':unittest.main()
