"""Before/after review comparison, with training replay clearly distinguished."""
import hashlib
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from hazard_head import HazardHead, probability_mask
from mask2former_detector import ROOT, StairsDetector
from prepare import load_image
from vision import enhance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',default='mask2former_hazard_review_v1')
    parser.add_argument('--output',default='pinned_review_v1')
    args=parser.parse_args()
    if not args.candidate.replace('_','').isalnum() or not args.output.replace('_','').isalnum():raise ValueError('Invalid experiment name')
    records=json.loads((ROOT/'data/experiments/pinned_review_v1/manifest.json').read_text(encoding='utf-8'))['records']
    candidate=ROOT/'models'/(args.candidate+'.pt')
    baseline=ROOT/'models/mask2former_hazard_head.pt'
    net=StairsDetector(head_path=candidate)
    oldstate=torch.load(baseline,map_location=net.device,weights_only=True)
    old=HazardHead().to(net.device).eval();old.load_state_dict(oldstate['state_dict'])
    output=ROOT/'outputs'/args.output;output.mkdir(parents=True,exist_ok=True)
    rows=[];cards=[]
    for row in records:
        frame=load_image(Path(row['image']));h,w=frame.shape[:2]
        scale=min(1,1280/max(h,w));frame=cv2.resize(frame,(round(w*scale),round(h*scale)));h,w=frame.shape[:2]
        truth=np.zeros((h,w),np.uint8)
        for poly in row['polygons']:cv2.fillPoly(truth,[np.round(np.array(poly['points'])*[w,h]).astype(np.int32)],1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        insufficient=np.mean(gray<8)>.95 or gray.std()<1.5
        metrics={};panels=[]
        for view,image in [('original',frame),('enhanced',enhance(frame))]:
            if not insufficient:
                semantic,features=net.segment(image,return_features=True)
                region=np.isin(semantic,net.stair_ids).astype(np.uint8)*255
                tensor=torch.from_numpy((region>0).astype(np.float32))[None,None].to(net.device)
            for name,head,threshold in [('before',old,float(oldstate['threshold'])),('after',net.hazard_head,net.hazard_threshold)]:
                if insufficient:mask=np.zeros((h,w),np.uint8)
                else:
                    with torch.inference_mode():probability=head(features,tensor).sigmoid()[0,0].cpu().numpy()
                    mask,_=probability_mask(probability,region,threshold)
                iou=float(np.logical_and(mask>0,truth).sum()/max(1,np.logical_or(mask>0,truth).sum()))
                metrics[view+'_'+name]={'iou':iou,'detected':bool(mask.any()),'correct':iou>=.5}
                display=image.copy();tint=np.full_like(image,(40,50,245))
                display[mask>0]=cv2.addWeighted(display,.5,tint,.5,0)[mask>0]
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display,contours,-1,(40,50,245),2)
                for poly in row['polygons']:cv2.polylines(display,[np.round(np.array(poly['points'])*[w,h]).astype(np.int32)],True,(0,255,0),2)
                cv2.putText(display,f'{view} {name} IoU={iou:.3f}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                panels.append(display)
        category='pinned_train' if row['review_pinned'] else 'validation' if row['split']=='val' else 'same_source_dev' if row['split']=='test' else 'other_train'
        filename=row['id']+'.jpg'
        grid=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
        ok,encoded=cv2.imencode('.jpg',grid)
        if not ok:raise RuntimeError('JPEG encode failed')
        encoded.tofile(str(output/filename))
        rows.append({'id':row['id'],'filename':Path(row['original']).name,'category':category,'previous_split':row['previous_split'],'split':row['split'],'metrics':metrics})
        cards.append(f'<article><h2>{Path(row["original"]).name} · {category}</h2><p>윗줄 원본 / 아랫줄 보정. 왼쪽 재학습 전 / 오른쪽 재학습 후. 빨강 예측 / 초록 정답.</p><img loading="lazy" src="{filename}"></article>')
        print(f'{len(rows)}/{len(records)} {row["id"]}',flush=True)
    summaries={}
    for category in ('pinned_train','validation','same_source_dev','other_train'):
        group=[r for r in rows if r['category']==category]
        summaries[category]={'images':len(group)}
        for key in ('original_before','original_after','enhanced_before','enhanced_after'):
            values=[r['metrics'][key] for r in group]
            summaries[category][key]={'mean_iou':float(np.mean([v['iou'] for v in values])),
                                     'correct':sum(v['correct'] for v in values),'wrong_candidates':sum(v['detected'] and not v['correct'] for v in values)}
    def average(category,stage):return np.mean([summaries[category][view+'_'+stage]['mean_iou'] for view in ('original','enhanced')])
    # Acceptance uses validation and training replay, never same-source dev.
    acceptable=bool(average('validation','after')>=average('validation','before')-.10 and
                    average('pinned_train','after')>=average('pinned_train','before')+.05)
    report={'summary':summaries,'records':rows,'can_activate':acceptable,
            'acceptance_rule':'Pinned mean IoU gain >= 0.05 and validation mean IoU drop <= 0.10; both views averaged. Dev results do not select checkpoint or activation.',
            'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
            'warning':'All 44 pins are now training examples. Old test score is retired. Validation groups are excluded from this retraining, but some were previously learned by the initialization model; this monitors retention, not blind generalization. Actual site independence is unknown and park remaining images share a source with training. Fresh labeled sites are required.'}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>핀 사진 재학습 비교</title><style>body{font:17px sans-serif;background:#14212e;color:white;margin:30px}img{max-width:100%}pre{white-space:pre-wrap}article{margin:40px 0}</style><a href="/">대시보드</a><h1>핀 사진 재학습 전후 비교</h1><p>핀 44장은 현재 학습 자료입니다. 핀 사진에서의 개선은 학습 자료 재현이며 새로운 계단 성능이 아닙니다. 시험 자료의 일부를 학습으로 옮겨 이전 시험 점수와 직접 비교할 수 없습니다.</p><pre>'+json.dumps(summaries,ensure_ascii=False,indent=2)+'</pre>'+''.join(cards),encoding='utf-8')
    print(json.dumps({'can_activate':acceptable,'summary':summaries},indent=2))


if __name__=='__main__':main()
