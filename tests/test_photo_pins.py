import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app
import photo_pins as pins


class PhotoPinsTest(unittest.TestCase):
    def test_persistence_original_bytes_and_zip_with_split_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';raw.mkdir()
            original=raw/'han01_0.jpg';original.write_bytes(b'original-full-resolution-bytes')
            resized=root/'resized.jpg';resized.write_bytes(b'wrong-resized-copy')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'records':[{
                'id':'han_test','source':'han','split':'test','original':str(original),'image':str(resized)}]}))
            with patch.multiple(pins,MANIFEST=manifest,RAW=raw,SELECTION=root/'selection'/'pins.json'):
                client=app.test_client()
                for _ in range(2):self.assertEqual(client.post('/api/pins',json={'id':'han_test','pinned':True}).status_code,200)
                self.assertEqual(app.test_client().get('/api/pins').json['records'][0]['filename'],original.name)
                self.assertEqual(len(pins.read_pins()),1)
                response=client.get('/api/photos/han_test/original')
                self.assertEqual(response.data,original.read_bytes());response.close()
                response=client.get('/api/pins/originals.zip')
                self.assertEqual(response.status_code,200)
                with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                    self.assertEqual(archive.read('han/han01_0.jpg'),original.read_bytes())
                    self.assertEqual(json.loads(archive.read('review_selection.json'))['photos'][0]['original_split'],'test')
                response.close()
                original.unlink()
                self.assertEqual(client.get('/api/photos/han_test/original').status_code,404)
                self.assertEqual(client.get('/api/pins/originals.zip').status_code,404)
                self.assertEqual(client.post('/api/pins',json={'id':'han_test','pinned':False}).status_code,200)
                self.assertEqual(client.get('/api/pins').json['records'],[])
                self.assertEqual(client.get('/api/pins/originals.zip').status_code,400)
                self.assertEqual(client.post('/api/pins',json={'id':'bad','pinned':True}).status_code,404)
                self.assertEqual(client.post('/api/pins',json={'id':'han_test','pinned':'true'}).status_code,400)


if __name__=='__main__':unittest.main()
