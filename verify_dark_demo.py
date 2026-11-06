"""Verify every-frame offline demo files and HTTP playback responses."""
import hashlib
import json

import cv2
import requests

from mask2former_detector import ROOT


def main():
    folder=ROOT/'outputs/dark_demo'
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    expected=list(range(report['first_source_frame'],report['last_source_frame']+1))
    assert [r['source_frame'] for r in report['frames']]==expected
    assert report['output_frames']==len(expected)
    assert [r['output_frame'] for r in report['frames']]==list(range(len(expected)))
    assert len(list((folder/'frames').glob('frame_*.jpg')))==len(expected)
    videos=[]
    for name in ('overlay','comparison'):
        path=folder/(name+'.mp4');capture=cv2.VideoCapture(str(path))
        fps=capture.get(cv2.CAP_PROP_FPS);fourcc=int(capture.get(cv2.CAP_PROP_FOURCC));count=0
        codec=''.join(chr((fourcc>>(8*i))&255) for i in range(4))
        while True:
            ok,frame=capture.read()
            if not ok:break
            count+=1
            width,height=report['output_size']
            assert frame.shape[:2]==(height,width*(2 if name=='comparison' else 1))
        capture.release()
        assert count==len(expected),(name,count,len(expected))
        assert abs(fps-report['source_fps'])<.002
        assert codec in ('h264','avc1')
        data=path.read_bytes();assert data.index(b'moov')<data.index(b'mdat')
        response=requests.get('http://127.0.0.1:5001/dark-demo/'+name+'.mp4',headers={'Range':'bytes=0-99'},timeout=15)
        assert response.status_code==206 and len(response.content)==100
        videos.append({'file':path.name,'frames_decoded':count,'fps':fps,'codec':codec,'http_range_status':206,'bytes':len(data)})
    result={'all_frames_present':True,'source_original_sha256':report['source_sha256'],
            'original_fps_preserved':True,'offline_demo':True,'videos':videos}
    (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
