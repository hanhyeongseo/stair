import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app
from mask2former_detector import StairsDetector


class Mask2FormerTests(unittest.TestCase):
    def test_reject_bad_requests_without_loading_large_model(self):
        client = app.test_client()
        with patch('mask2former_routes.get_detector') as load:
            self.assertEqual(client.get('/mask2former').status_code,200)
            self.assertEqual(client.post('/api/mask2former/analyze',json={'id':'../../bad'}).status_code,400)
            self.assertEqual(client.post('/api/mask2former/analyze',data={'image':(io.BytesIO(b'bad'),'a.jpg')}).status_code,400)
            ok, encoded = cv2.imencode('.png',np.zeros((20,30,3),np.uint8))
            self.assertTrue(ok)
            self.assertEqual(client.post('/api/mask2former/analyze',data={'mode':'bad','image':(io.BytesIO(encoded.tobytes()),'a.png')}).status_code,400)
            load.assert_not_called()

    def test_black_frame_abstains_without_model_forward(self):
        detector = StairsDetector.__new__(StairsDetector)
        detector.device = 'cpu'
        detector.labels = {53:'stairs',59:'stairway',121:'step'}
        detector.stair_ids = [53,59,121]
        with patch.object(detector,'segment') as forward:
            views, result = detector.infer(np.zeros((100,200,3),np.uint8))
        forward.assert_not_called()
        self.assertFalse(result['hazard_step_selected'])
        for name in ('original','enhanced'):
            self.assertEqual(result['views'][name]['status'],'insufficient_signal')
            self.assertFalse(np.any(views[name+'_mask']))
            self.assertFalse(np.any(views[name+'_transparent'][:,:,3]))


if __name__ == '__main__':
    unittest.main()
