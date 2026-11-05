"""Evaluate fixed production inference on held-out source without tuning it."""
import json
from pathlib import Path

import cv2
import numpy as np

from detector import ROOT,Detector
from prepare import load_image,write_image


def main():
    manifest=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
    detector=Detector();output=ROOT/'outputs'/'heldout_pipeline';output.mkdir(parents=True,exist_ok=True)
    rows=[];true_positive=0;false_positive=0;false_negative=0
    for row in manifest['records']:
        if row['split']!='test':continue
        image=load_image(Path(row['image']))
        views,result=detector.infer(image,confidence=.4,color=True)
        h,w=views['mask'].shape
        truth=[]
        for shape in row['polygons']:
            mask=np.zeros((h,w),np.uint8)
            pts=np.round(np.array(shape['points'])*[w,h]).astype(np.int32)
            cv2.fillPoly(mask,[pts],255);truth.append(mask>0)
        pred=views['mask']>0
        ious=[float(np.logical_and(pred,t).sum()/max(1,np.logical_or(pred,t).sum())) for t in truth]
        best_iou=max(ious,default=0)
        matched=best_iou>=.5 and result['hazard'] is not None
        true_positive+=int(matched)
        false_positive+=int(result['hazard'] is not None and not matched)
        false_negative+=len(truth)-int(matched)
        rows.append({'id':row['id'],'original':row['original'],'source':row['source'],
                     'iou':best_iou,'matched_iou50':matched,**result})
        # Reference labels are drawn only in comparison images, never model input.
        comparison=views['overlay'].copy()
        for shape in row['polygons']:
            pts=np.round(np.array(shape['points'])*[w,h]).astype(np.int32)
            cv2.polylines(comparison,[pts],True,(0,255,0),2)
        write_image(output/(row['id']+'.jpg'),comparison)
    report={'split':'test','threshold':.4,'iou_threshold':.5,'images':len(rows),
            'true_positive':true_positive,'false_positive':false_positive,'false_negative':false_negative,
            'precision':true_positive/max(1,true_positive+false_positive),
            'recall':true_positive/max(1,true_positive+false_negative),
            'mean_mask_iou':float(np.mean([r['iou'] for r in rows])) if rows else None,
            'warning':'Positive annotated scenes only. False alerts on no-stair scenes and true unseen-location generalization remain unverified.',
            'records':rows}
    (ROOT/'models'/'pipeline_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
