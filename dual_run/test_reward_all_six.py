"""all_six奖励：R=0.2F+0.4G+0.4·P(all_six)，与speaker_only无关；旧continuous不变。"""
import unittest


class AllSixRewardTests(unittest.TestCase):
    def test_all_six_ignores_speaker_only(self):
        from training.grpo_v3.six_user_binary.reward import aggregate
        p={'formality':.5,'groundedness':.25,'all_six':.75,'speaker_only':0.}
        self.assertAlmostEqual(aggregate(p,mode='all_six'),.2*.5+.4*.25+.4*.75)
        self.assertAlmostEqual(aggregate({**p,'speaker_only':1.},mode='all_six'),aggregate(p,mode='all_six'))

    def test_continuous_unchanged(self):
        from training.grpo_v3.six_user_binary.reward import aggregate
        p={'formality':.5,'groundedness':.25,'all_six':.75,'speaker_only':.5}
        self.assertAlmostEqual(aggregate(p,mode='continuous'),.2*.5+.4*.25+.4*.75*.5)

    def test_launch_accepts_mode(self):
        from training.grpo_v3.six_user_binary.plugin import MODES
        self.assertIn('all_six',MODES)


if __name__=='__main__':unittest.main()
