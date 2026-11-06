import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as web
from video_encoding import browser_mp4


class BrowserVideoTest(unittest.TestCase):
    def test_real_h264_faststart_and_seekable_inline_response_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);raw=folder/'raw.mp4'
            writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),2,(160,120))
            self.assertTrue(writer.isOpened())
            for level in (30,90):writer.write(np.full((120,160,3),level,np.uint8))
            writer.release()
            key='f'*32;result=folder/(key+'_result.mp4')
            browser_mp4(raw,result)
            data=result.read_bytes()
            self.assertLess(data.index(b'moov'),data.index(b'mdat'))
            capture=cv2.VideoCapture(str(result));codec=int(capture.get(cv2.CAP_PROP_FOURCC))
            self.assertIn(''.join(chr((codec>>(8*i))&255) for i in range(4)),('h264','avc1'))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),2)
            self.assertTrue(capture.read()[0]);capture.release()
            with patch.object(web,'OUTPUT',folder):
                client=web.app.test_client()
                self.assertEqual(client.get('/api/video/'+key).json['status'],'done')
                response=client.get('/api/video/'+key+'/play',headers={'Range':'bytes=0-99'})
                self.assertEqual(response.status_code,206)
                self.assertEqual(len(response.data),100)
                self.assertTrue(response.headers['Content-Disposition'].startswith('inline'))
                response.close()
                self.assertEqual(client.get('/videos').status_code,200)


if __name__=='__main__':unittest.main()
