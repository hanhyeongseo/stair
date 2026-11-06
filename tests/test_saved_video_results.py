import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import saved_video_results as saved


class SavedResultTest(unittest.TestCase):
    def test_reuse_requires_matching_content_model_and_input_not_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'outputs/dark_demo_orange';folder.mkdir(parents=True)
            models=root/'models';models.mkdir()
            head=models/'mask2former_hazard_head.pt';head.write_bytes(b'head-v1')
            upload=root/'renamed.mp4';upload.write_bytes(b'original-video')
            (folder/'source_comparison_10fps.mp4').write_bytes(b'previously-computed-output')
            report={'checkpoint_sha256':saved.sha256(head),'source_sha256':saved.sha256(upload),'start_seconds':9}
            (folder/'report.json').write_text(json.dumps(report))
            (folder/'source_comparison_10fps.json').write_text(json.dumps({'duration_seconds':34.1,'frames':341,'fps':10}))
            with patch.object(saved,'ROOT',root):
                self.assertTrue(saved.matching_saved_result(upload,'mask2former','enhanced')['reused_analysis'])
                self.assertIsNone(saved.matching_saved_result(upload,'mask2former','original'))
                self.assertIsNone(saved.matching_saved_result(upload,'yolo','enhanced'))
                changed=root/'dark.mp4';changed.write_bytes(b'different-video-with-the-same-name')
                self.assertIsNone(saved.matching_saved_result(changed,'mask2former','enhanced'))
                head.write_bytes(b'head-v2')
                self.assertIsNone(saved.matching_saved_result(upload,'mask2former','enhanced'))


if __name__=='__main__':unittest.main()
