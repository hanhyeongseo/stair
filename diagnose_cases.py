"""Read-only diagnosis: distinguish score rejection from wrong localization."""
import json
from pathlib import Path

import cv2
import numpy as np

from detector import ROOT,Detector
from prepare import load_image
from vision import enhance


def main():
    manifest=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
    detector=Detector();rows=[]
    for filename in ('park_stair (7).jpg','park_stair (79).jpg'):
        row=next(r for r in manifest['records'] if Path(r['original']).name==filename)
        frame=load_image(Path(row['image']));h,w=frame.shape[:2]
        truth=np.zeros((256,256),np.uint8)
        for poly in row['polygons']:
            cv2.fillPoly(truth,[np.round(np.array(poly['points'])*256).astype(np.int32)],1)
        for view,image in [('original',frame),('enhanced',enhance(frame))]:
            result=detector.model.predict(image,conf=.01,imgsz=640,retina_masks=True,
                                         device=detector.device,verbose=False)[0]
            candidates=[]
            if result.masks is not None:
                for polygon,confidence in zip(result.masks.xy,result.boxes.conf.cpu().numpy()):
                    mask=np.zeros((256,256),np.uint8)
                    pts=np.round(polygon/[w,h]*256).astype(np.int32)
                    cv2.fillPoly(mask,[pts],1)
                    iou=float(np.logical_and(mask,truth).sum()/max(1,np.logical_or(mask,truth).sum()))
                    candidates.append({'score':round(float(confidence),4),'mask_iou':round(iou,4)})
            candidates.sort(key=lambda c:-c['score'])
            rows.append({'image':filename,'view':view,'top_candidates':candidates[:6],
                         'best_iou_any_candidate':max((c['mask_iou'] for c in candidates),default=0),
                         'candidates_at_deployed_threshold':sum(c['score']>=.4 for c in candidates)})
    (ROOT/'outputs'/'case_diagnosis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
