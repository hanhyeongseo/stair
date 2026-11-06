import json

import cv2
import numpy as np
import requests

from mask2former_detector import ROOT,StairsDetector


def main():
    folder=ROOT/'outputs/dark_demo_orange'
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    original=json.loads((ROOT/'outputs/dark_demo/report.json').read_text(encoding='utf-8'))
    assert report['source_sha256']==original['source_sha256']
    assert report['checkpoint_sha256']==original['checkpoint_sha256']
    assert [row['source_frame'] for row in report['rows']]==list(range(original['first_source_frame'],original['last_source_frame']+1))
    assert report['frames']==original['output_frames']==1025
    summaries=[]
    for name in ('overlay','comparison'):
        path=folder/(name+'.mp4');capture=cv2.VideoCapture(str(path));fps=capture.get(cv2.CAP_PROP_FPS)
        codec=int(capture.get(cv2.CAP_PROP_FOURCC));count=0
        while True:
            ok,image=capture.read()
            if not ok:break
            count+=1
        capture.release()
        assert count==1025
        assert abs(fps-report['fps'])<.002
        assert ''.join(chr((codec>>(8*i))&255) for i in range(4)) in ('h264','avc1')
        response=requests.get('http://127.0.0.1:5001/orange-demo/'+name+'.mp4',headers={'Range':'bytes=0-99'},timeout=15)
        assert response.status_code==206 and len(response.content)==100
        summaries.append({'file':path.name,'frames':count,'fps':fps,'http_range':206})
    # Confirm cached batch inference matches the ordinary production predictor.
    capture=cv2.VideoCapture('C:/Users/KDT76/OneDrive/Desktop/dark.mp4')
    frame=None
    for _ in range(401):ok,frame=capture.read()
    capture.release();assert ok
    h,w=frame.shape[:2];scale=min(1,640/max(h,w))
    frame=cv2.resize(frame,(max(2,round(w*scale)),max(2,round(h*scale))))
    net=StairsDetector();views,_=net.infer(frame,mode='enhanced')
    cached=cv2.imdecode(np.fromfile(str(ROOT/'outputs/dark_demo/mask_cache'/net.hazard_checkpoint_sha256[:16]/'000400_mask.png'),np.uint8),cv2.IMREAD_GRAYSCALE)>0
    actual=views['enhanced_hazard_mask']>0
    agreement=float(np.mean(cached==actual));assert agreement>.995
    result={'videos':summaries,'checkpoint_unchanged':True,'source_unchanged':True,
            'sample_cached_vs_single_pixel_agreement':agreement,'classification_of_danger':False}
    (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
