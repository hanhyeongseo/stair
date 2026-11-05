"""Save reproducible, label-free predictions, with separate GT visualization.

Coverage measures whether whole-stairs predictions cover the selected step;
it cannot establish whole-stairs precision without whole-stairs annotations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from mask2former_detector import ROOT, MODEL_DIR, StairsDetector
from prepare import load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='All 124 photos instead of diagnostic samples')
    args = parser.parse_args()
    records = json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))['records']
    if not args.all:
        wanted = {'park_stair (7).jpg', 'park_stair (79).jpg', 'han01_0.jpg', 'han01_1.jpg', 'seo01_0.jpg', 'seo01_1.jpg'}
        records = [r for r in records if Path(r['original']).name in wanted]
    output = ROOT/'outputs'/'mask2former_evaluation'
    output.mkdir(parents=True, exist_ok=True)
    net = StairsDetector()
    rows = []
    cards = []
    for record in records:
        views, report = net.infer(load_image(Path(record['image'])), mode='both')
        h,w = views['original'].shape[:2]
        truth = np.zeros((h,w), np.uint8)
        for poly in record['polygons']:
            cv2.fillPoly(truth, [np.round(np.array(poly['points'])*[w,h]).astype(np.int32)],1)
        coverage = {}
        panels = []
        for name in ('original','enhanced'):
            mask = views[name+'_mask'] > 0
            coverage[name] = float(np.logical_and(mask, truth).sum()/max(1, truth.sum()))
            panel = views[name+'_overlay'].copy()
            for poly in record['polygons']:
                cv2.polylines(panel,[np.round(np.array(poly['points'])*[w,h]).astype(np.int32)],True,(0,255,0),2)
            cv2.putText(panel,name,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            panels.append(panel)
        filename = record['id']+'.jpg'
        ok, data = cv2.imencode('.jpg',np.hstack(panels))
        if not ok: raise RuntimeError('JPEG encode failed')
        data.tofile(str(output/filename))
        row = {'id':record['id'],'filename':Path(record['original']).name,'source':record['source'],
               'existing_yolo_split':record['split'],'report':report,'hazard_label_coverage_diagnostic':coverage}
        rows.append(row)
        # Save partial progress for long evaluations.
        (output/'records.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        cards.append(f'<article><h2>{row["filename"]}</h2><p>원본 / 밝기 보정. 파랑: 계단 전체, 초록: 기존 위험 단차 정답.</p><img loading="lazy" src="{filename}"></article>')
        print(json.dumps({'image':row['filename'],'views':report['views'],'coverage':coverage,
                          'ms':report['processing_ms']},ensure_ascii=False), flush=True)
    summary = {'model':'facebook/mask2former-swin-large-ade-semantic','device':net.device,
               'model_sha256':hashlib.sha256((MODEL_DIR/'model.safetensors').read_bytes()).hexdigest(),
               'input_processor_size':{'shortest_edge':512,'longest_edge':768},
               'images':len(rows),'team_training_performed':False,'annotation_used_as_input':False,
               'caveat':'Selected hazard step is not whole-stairs ground truth; coverage is diagnostic, not precision or overall accuracy.',
               'records':rows}
    (output/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mask2Former 실제 사진 결과</title><style>body{background:#14212e;color:white;font:17px sans-serif;margin:30px}img{max-width:100%}article{margin:40px 0}</style><h1>Mask2Former Swin-Large ADE20K 결과</h1><p>팀 사진으로 재학습하지 않은 모델입니다. 계단 전체 인식이며 특정 위험 단차 선택은 수행하지 않습니다.</p>'+''.join(cards),encoding='utf-8')


if __name__ == '__main__':
    main()
