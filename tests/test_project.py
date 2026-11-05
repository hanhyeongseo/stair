import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import app
import app as web
from vision import color_edges,enhance


class DatasetTest(unittest.TestCase):
    def test_heldout_source_and_exposure_groups_are_separate(self):
        data=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
        groups={};hashes={}
        for row in data['records']:
            groups.setdefault(row['group'],set()).add(row['split'])
            hashes.setdefault(row['sha256'],set()).add(row['split'])
            self.assertEqual(row['split']=='test',row['source']=='park')
        self.assertTrue(all(len(v)==1 for v in groups.values()))
        self.assertTrue(all(len(v)==1 for v in hashes.values()))
        self.assertEqual(len(data['records']),124)
        self.assertTrue(all(r['reason']=='unlabeled_or_empty_not_a_confirmed_negative' for r in data['excluded']))

    def test_labels_are_polygons_in_single_hazard_class(self):
        for path in (ROOT/'data'/'yolo'/'labels').rglob('*.txt'):
            for line in path.read_text().splitlines():
                values=line.split();self.assertEqual(values[0],'0')
                self.assertGreaterEqual(len(values),7)
                self.assertEqual((len(values)-1)%2,0)
                self.assertTrue(all(0<=float(x)<=1 for x in values[1:]))


class VisionTest(unittest.TestCase):
    def test_enhancement_leaves_input_unchanged_and_edges_stay_in_mask(self):
        image=np.full((240,320,3),20,np.uint8);copy=image.copy()
        cv2.rectangle(image,(40,40),(280,200),(40,90,140),-1);copy=image.copy()
        out=enhance(image)
        np.testing.assert_array_equal(image,copy)
        self.assertEqual(out.shape,image.shape)
        mask=np.zeros((240,320),np.uint8);mask[80:160,80:240]=255
        edge,_=color_edges(out,mask)
        self.assertFalse(np.any(edge[mask==0]))
        empty,lines=color_edges(out,np.zeros_like(mask))
        self.assertEqual(lines,[]);self.assertFalse(np.any(empty))


class WebTest(unittest.TestCase):
    def setUp(self):self.client=app.test_client()

    def test_pages_and_invalid_inputs(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(len(self.client.get('/api/catalog').json['records']),124)
        self.assertEqual(self.client.post('/api/analyze',json={'id':'../app.py'}).status_code,400)
        response=self.client.post('/api/upload',data={'image':(io.BytesIO(b'invalid'),'x.jpg')})
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.client.get('/api/video/unknown/download').status_code,404)

    @unittest.skipUnless((ROOT/'models'/'hazard_best.pt').exists(),'Training has not completed')
    def test_blank_is_abstained_and_transparent_overlay_is_available(self):
        ok,jpg=cv2.imencode('.jpg',np.zeros((320,240,3),np.uint8));self.assertTrue(ok)
        response=self.client.post('/api/upload',data={'image':(io.BytesIO(jpg.tobytes()),'black.jpg')})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['result']['status'],'insufficient_signal')
        self.assertIsNone(response.json['result']['hazard'])
        self.assertTrue(response.json['images']['transparent'].startswith('data:image/png;base64,'))
        self.assertFalse(response.json['model_input_used_annotation'])

    @unittest.skipUnless((ROOT/'models'/'hazard_best.pt').exists(),'Training has not completed')
    def test_video_worker_outputs_two_real_model_frames(self):
        manifest=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
        image=cv2.imread(manifest['records'][0]['image']);image=cv2.resize(image,(240,320))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'source.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),5,(240,320))
            self.assertTrue(writer.isOpened())
            writer.write(image);writer.write(image);writer.release()
            with patch.object(web,'OUTPUT',folder):
                with source.open('rb') as stream:
                    response=self.client.post('/api/video',data={'video':(io.BytesIO(stream.read()),'sample.mp4')})
                self.assertEqual(response.status_code,202)
                key=response.json['id'];deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    job=web.jobs[key]
                    if job['status']=='error':self.fail(job['error'])
                    if job['status']=='done' and not (folder/(key+'_input.mp4')).exists():break
                    time.sleep(.05)
                self.assertEqual(job['status'],'done',job)
                self.assertEqual(job['frames'],2)
                capture=cv2.VideoCapture(job['result'])
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),2)
                capture.release();web.jobs.pop(key)


if __name__=='__main__':unittest.main()
