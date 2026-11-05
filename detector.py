"""Human-labeled hazard-step segmentation, with explicit abstention states."""
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from vision import enhance,color_edges

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache'/'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))


class Detector:
    def __init__(self,weights=None):
        import torch
        from ultralytics import YOLO
        self.weights=Path(weights or ROOT/'models'/'hazard_best.pt')
        if not self.weights.exists():raise FileNotFoundError('위험 단차 학습 모델이 아직 없습니다.')
        self.model=YOLO(str(self.weights))
        self.device=0 if torch.cuda.is_available() else 'cpu'
        self.lock=threading.RLock()

    def infer(self,image,confidence=.4,color=True):
        start=time.perf_counter()
        if not np.isfinite(confidence) or not .1<=confidence<=.95:raise ValueError('점수 문턱값은 0.1~0.95여야 합니다.')
        h,w=image.shape[:2];scale=min(1,1280/max(h,w))
        frame=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
        enhanced=enhance(frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        low_signal=float(np.mean(gray<8))>.95 or float(gray.std())<1.5
        candidates=[]
        if not low_signal:
            views=[('original',frame)]
            if float(np.mean(cv2.absdiff(frame,enhanced)))>4:views.append(('enhanced',enhanced))
            with self.lock:
                for name,view in views:
                    prediction=self.model.predict(view,conf=confidence,iou=.6,imgsz=640,
                        device=self.device,retina_masks=True,verbose=False)[0]
                    if prediction.masks is None:continue
                    for polygon,score in zip(prediction.masks.xy,prediction.boxes.conf.cpu().numpy()):
                        pts=np.asarray(polygon,dtype=np.float32)
                        if len(pts)<3:continue
                        polygon_norm=np.clip(pts/[frame.shape[1],frame.shape[0]],0,1)
                        area=abs(cv2.contourArea(polygon_norm.astype(np.float32)))
                        if not .0005<area<.7:continue
                        candidates.append({'polygon':polygon_norm.tolist(),'score':float(score),
                                           'input_view':name})
        candidates.sort(key=lambda c:-c['score'])
        best=candidates[0] if candidates else None
        status='insufficient_signal' if low_signal else 'candidate' if best else 'not_detected'
        mask=np.zeros(frame.shape[:2],np.uint8)
        overlay=enhanced.copy();transparent=np.zeros((*mask.shape,4),np.uint8)
        edge=np.zeros_like(mask);lines=[]
        if best:
            pts=np.round(np.asarray(best['polygon'])*[frame.shape[1],frame.shape[0]]).astype(np.int32)
            cv2.fillPoly(mask,[pts],255)
            overlay[mask>0]=cv2.addWeighted(overlay,.65,np.full_like(overlay,(35,65,245)),.35,0)[mask>0]
            cv2.polylines(overlay,[pts],True,(40,90,255),3)
            transparent[mask>0]=(35,65,245,95)
            cv2.polylines(transparent,[pts],True,(40,90,255,255),3)
            if color:
                edge,lines=color_edges(enhanced,mask)
                for x,y,xx,yy in lines:cv2.line(overlay,(x,y),(xx,yy),(255,220,0),2)
        report={'status':status,'hazard':best,'confidence_threshold':confidence,
                'processing_ms':round((time.perf_counter()-start)*1000,1),
                'input_size':[w,h],'analysis_size':list(frame.shape[1::-1]),
                'low_signal':low_signal,'color_edge_lines':lines,
                'warning':'미검출은 안전 판정이 아닙니다. 점수는 보정된 안전 확률이 아닙니다.',
                'label_definition':'하행 두 번째 단 / 상행 첫 번째 단으로 사람이 표시한 위험 단차 영역'}
        return {'original':frame,'enhanced':enhanced,'overlay':overlay,
                'transparent':transparent,'mask':mask,'color_edges':edge},report
