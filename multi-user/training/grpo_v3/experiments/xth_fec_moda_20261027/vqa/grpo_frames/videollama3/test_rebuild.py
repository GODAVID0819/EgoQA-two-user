"""核对原帧索引到30秒源片段的映射，以及逐片段抽帧与原concat流程相同。"""
import json,tempfile,unittest
from pathlib import Path
import cv2
import shutil
import numpy as np
from rebuild_from_videos import build_plan,window_command
from reference_decode import decode_source_segments_rgb_frames,_decode_ffmpeg_ppm_stream


class RebuildTests(unittest.TestCase):
    def test_full_original_frame_mapping_keeps_all_89255_targets(self):
        root=Path(__file__).resolve().parent
        items=json.loads((root/'dataset.json').read_text(encoding='utf-8'))['items']
        groups=build_plan(items)
        self.assertEqual(len(groups),414)
        self.assertEqual(sum(len(t['targets']) for t in groups.values()),89255)
        self.assertTrue(all(0<=i<300 for t in groups.values() for i in t['targets'].values()))
        found=[i for t in groups.values() for p,i in t['targets'].items() if p.endswith('frame_0288_576.00s.jpg')]
        self.assertTrue(found);self.assertEqual(set(found),{288})

    def test_thread_limited_sampling_matches_original_padded_concat_with_short_video(self):
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for j,count in enumerate([593,602]):
                path=Path(directory)/f'source_{j}.mp4';paths.append(path)
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(64,64))
                self.assertTrue(writer.isOpened())
                for i in range(count):writer.write(np.full((64,64,3),(i+j*79)%255,dtype=np.uint8))
                writer.release()
            combined=decode_source_segments_rgb_frames(paths,duration_seconds=60,sample_interval_seconds=2,
                                                       segment_durations_seconds=[30,30],ffmpeg_binary=ffmpeg)
            limited=_decode_ffmpeg_ppm_stream(window_command(paths,ffmpeg),
                    expected_frame_count=30,source_description='synthetic_unit_test')
            self.assertEqual(len(combined),30);self.assertEqual(len(limited),30)
            for a,b in zip(combined,limited):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


if __name__=='__main__':unittest.main()
