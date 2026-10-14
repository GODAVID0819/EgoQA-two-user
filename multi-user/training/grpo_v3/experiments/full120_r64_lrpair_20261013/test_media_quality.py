"""覆盖实测21.13秒源片段及音轨掩盖缺帧的失败场景。"""
import unittest
from media_quality import assess_probe,window_problem,overlaps


def metadata(video=30.,audio=30.033):
    return {'streams':[{'codec_type':'video','width':1408,'height':1408,'start_time':'0','duration':str(video)}],
            'format':{'duration':str(audio)}}


class MediaQualityTests(unittest.TestCase):
    def test_short_source_is_not_complete(self):
        self.assertEqual(assess_probe(metadata(21.1,21.13))['problem'],'short_video')

    def test_audio_must_not_hide_missing_video(self):
        self.assertEqual(assess_probe(metadata(21.1,30.03))['problem'],'short_video')

    def test_normal_jitter_and_full_window(self):
        result=assess_probe(metadata())
        self.assertIsNone(result['problem'])
        window={'clips':[{'paths':[str(i) for i in range(20)]} for _ in range(6)]}
        self.assertIsNone(window_problem(window,{str(i):result for i in range(20)}))

    def test_accumulated_duration_cannot_be_called_ten_minutes(self):
        result=assess_probe(metadata(30.,30.2))
        window={'clips':[{'paths':[str(i) for i in range(20)]} for _ in range(6)]}
        self.assertIsNotNone(window_problem(window,{str(i):result for i in range(20)}))

    def test_touching_window_boundaries_are_disjoint(self):
        self.assertFalse(overlaps(1000,1600))
        self.assertTrue(overlaps(1000,1599))


if __name__=='__main__':unittest.main()
