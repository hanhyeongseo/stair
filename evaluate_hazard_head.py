"""Fixed evaluation of hazard masks; annotations enter metrics only."""
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from mask2former_detector import ROOT, StairsDetector
from prepare import load_image


def main():
    records = json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))['records']
    net = StairsDetector()
    if net.hazard_head is None:raise RuntimeError('Train hazard head first')
    output = ROOT/'outputs'/'hazard_head_evaluation';output.mkdir(parents=True,exist_ok=True)
    rows = [];cards = []
    for row in records:
        views,report = net.infer(load_image(Path(row['image'])),mode='both')
        h,w = views['original'].shape[:2]
        truth = np.zeros((h,w),np.uint8)
        for poly in row['polygons']:
            cv2.fillPoly(truth,[np.round(np.array(poly['points'])*[w,h]).astype(np.int32)],1)
        metrics = {};panels = []
        for name in ('original','enhanced'):
            predicted = views[name+'_hazard_mask']>0
            intersection = np.logical_and(predicted,truth).sum()
            union = np.logical_or(predicted,truth).sum()
            iou = float(intersection/max(1,union))
            found = bool(np.any(predicted))
            metrics[name] = {'iou':iou,'detected':found,'correct_iou_05':iou>=.5,
                             'score':report['views'][name]['hazard']['score']}
            panel = views[name+'_hazard_overlay'].copy()
            for poly in row['polygons']:
                cv2.polylines(panel,[np.round(np.array(poly['points'])*[w,h]).astype(np.int32)],True,(0,255,0),2)
            cv2.putText(panel,f'{name} IoU={iou:.3f}',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            panels.append(panel)
        filename = row['id']+'.jpg'
        ok,encoded = cv2.imencode('.jpg',np.hstack(panels))
        if not ok:raise RuntimeError('JPEG encode failed')
        encoded.tofile(str(output/filename))
        rows.append({'id':row['id'],'filename':Path(row['original']).name,'split':row['split'],
                     'metrics':metrics,'processing_ms':report['processing_ms']})
        cards.append(f'<article><h2>{Path(row["original"]).name} / {row["split"]}</h2><p>좌: 원본, 우: 보정. 빨강: 위험 단차 예측, 초록: 정답, 주황: 모서리 후보.</p><img loading="lazy" src="{filename}"></article>')
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    summaries = []
    for split in ('train','val','test'):
        selected = [r for r in rows if r['split']==split]
        for name in ('original','enhanced'):
            values = [r['metrics'][name] for r in selected]
            tp = sum(v['correct_iou_05'] for v in values)
            fp = sum(v['detected'] and not v['correct_iou_05'] for v in values)
            summaries.append({'split':split,'input':name,'images':len(values),'true_positive':tp,
                              'false_positive':fp,'false_negative':len(values)-tp,
                              'precision':tp/max(1,tp+fp),'recall':tp/max(1,len(values)),
                              'mean_mask_iou':float(np.mean([v['iou'] for v in values]))})
    report = {'model':'Frozen Mask2Former + trained hazard head','iou_threshold':.5,
              'checkpoint_sha256':hashlib.sha256((ROOT/'models'/'mask2former_hazard_head.pt').read_bytes()).hexdigest(),
              'threshold':net.hazard_threshold,'summary':summaries,'records':rows,
              'warning':'Park was already inspected during earlier development. These are development comparisons, not a new blind unseen-location safety evaluation. No negative-scene dataset available.'}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>위험 단차 학습 결과</title><style>body{background:#14212e;color:white;font:17px sans-serif;margin:30px}img{max-width:100%}article{margin:40px 0}pre{white-space:pre-wrap}</style><h1>위험 단차 모델 결과</h1><p>학습 자료 57장 / 선택용 검증 11장. test 56장은 이전 개발에서 이미 확인한 자료로 완전히 새로운 장소 검증을 의미하지 않습니다.</p><pre>'+json.dumps(summaries,ensure_ascii=False,indent=2)+'</pre>'+''.join(cards),encoding='utf-8')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
