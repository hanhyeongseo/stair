"""Compare an architectural stairs model without fitting to evaluation photos.

Stairs extent and the annotated hazard step are different targets: overlap here
is diagnostic coverage, not a valid stairs segmentation accuracy measurement.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from detector import ROOT, Detector
from prepare import load_image
from vision import enhance


def save(path, image):
    ok, data = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    data.tofile(str(path))


def main():
    from ultralytics import YOLO
    weights = ROOT / 'models' / 'archivision.pt'
    model = YOLO(str(weights))
    output = ROOT / 'outputs' / 'pretrained_comparison'
    output.mkdir(parents=True, exist_ok=True)
    stairs = [int(k) for k, v in model.names.items() if str(v).lower() in ('stairs', 'stair', 'staircase')]
    if not stairs:
        failure = {'status':'rejected_missing_stairs_class', 'actual_classes':model.names,
                   'source':'https://huggingface.co/TanTan2025/archivision-yolo',
                   'sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                   'comparison_performed':False,'training_performed':False,
                   'stairnet_status':'Official Baidu extraction page reached; checkpoint not acquired. No weight files in official GitHub tree.'}
        (output/'report.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>모델 검증 결과</title><style>body{font:18px sans-serif;max-width:900px;margin:50px auto;line-height:1.8}</style><h1>사전 학습 모델 검증 결과</h1><p>ArchiVision은 제외했습니다. 소개와 달리 실제 가중치에 계단 클래스가 없습니다.</p><p>실제 클래스: '+str(model.names)+' 수치 성능 비교와 재학습은 수행하지 않았습니다.</p><p>StairNet은 공식 배포 페이지에 접근했으나 가중치 파일은 확보하지 못했습니다. 가중치를 확보한 후 비교가 필요합니다.</p><p><a href="report.json">검증 보고서 JSON</a></p>',encoding='utf-8')
        print(json.dumps(failure,ensure_ascii=False,indent=2))
        return
    records = json.loads((ROOT / 'data' / 'manifest.json').read_text(encoding='utf-8'))['records']
    baseline = Detector()
    rows = []
    cards = []
    for row in records:
        frame = load_image(Path(row['image']))
        h, w = frame.shape[:2]
        scale = min(1, 960 / max(h, w))
        frame = cv2.resize(frame, (round(w*scale), round(h*scale)))
        h, w = frame.shape[:2]
        truth = np.zeros((h, w), np.uint8)
        for poly in row['polygons']:
            cv2.fillPoly(truth, [np.round(np.asarray(poly['points'])*[w, h]).astype(np.int32)], 1)
        views = [('original', frame), ('enhanced', enhance(frame))]
        panels = []
        results = []
        for view, image in views:
            prediction = model.predict(image, classes=stairs, conf=.25, imgsz=640,
                                       retina_masks=True, device='cpu', verbose=False)[0]
            union = np.zeros_like(truth)
            display = image.copy()
            detections = []
            if prediction.masks is not None:
                for polygon, score in zip(prediction.masks.xy, prediction.boxes.conf.cpu().numpy()):
                    pts = np.round(polygon).astype(np.int32)
                    cv2.fillPoly(union, [pts], 1)
                    cv2.polylines(display, [pts], True, (255, 160, 0), 3)
                    detections.append({'score': float(score), 'polygon': (polygon/[w, h]).tolist()})
            intersection = int(np.logical_and(union, truth).sum())
            results.append({'view': view, 'detections': detections,
                            'hazard_coverage': intersection/max(1, int(truth.sum())),
                            'hazard_iou_diagnostic': intersection/max(1, int(np.logical_or(union, truth).sum()))})
            cv2.putText(display, 'ArchiVision '+view, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            panels.append(display)
        old, report = baseline.infer(frame, confidence=.4, color=False)
        panels.append(old['overlay'])
        cv2.putText(panels[-1], 'Existing hazard model', (10,25), cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        for panel in panels:
            for poly in row['polygons']:
                cv2.polylines(panel,[np.round(np.asarray(poly['points'])*[w,h]).astype(np.int32)],True,(0,255,0),2)
        filename = row['id']+'.jpg'
        save(output / filename, np.hstack(panels))
        rows.append({'id':row['id'], 'filename':Path(row['original']).name, 'split':row['split'],
                     'mean_gray':row['mean_gray'], 'archivision':results,
                     'existing_hazard':report['hazard']})
        cards.append(f'<article><h3>{row["id"]} ({row["split"]})</h3><p>{Path(row["original"]).name}</p><img loading="lazy" src="{filename}"></article>')
        print(f'{len(rows)}/{len(records)} {Path(row["original"]).name}: '+str([len(r['detections']) for r in results]), flush=True)
    summary = []
    for split in ('train','val','test'):
        for view in ('original','enhanced'):
            for brightness in ('all','dark_mean_under_60','bright_mean_at_least_60'):
                group = [r for r in rows if r['split']==split and (brightness=='all' or
                         (r['mean_gray']<60 if brightness=='dark_mean_under_60' else r['mean_gray']>=60))]
                values = [next(v for v in r['archivision'] if v['view']==view) for r in group]
                summary.append({'split':split,'view':view,'brightness':brightness,'images':len(values),
                                'images_with_stairs':sum(bool(v['detections']) for v in values),
                                'mean_hazard_coverage':float(np.mean([v['hazard_coverage'] for v in values])) if values else None})
    result = {'model_source':'https://huggingface.co/TanTan2025/archivision-yolo',
              'sha256':hashlib.sha256(weights.read_bytes()).hexdigest(), 'actual_classes':model.names,
              'threshold':.25,'training_performed':False,
              'caveat':'Whole stairs vs selected hazard: coverage is diagnostic, NOT detection accuracy. No confirmed negative images. Brightness groups are measured intensity, not exposure filenames.',
              'summary':summary,'records':rows}
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Pretrained stairs comparison</title><style>body{font-family:sans-serif;background:#17212b;color:white;padding:20px}img{width:100%;max-width:1600px}article{margin:35px 0}</style><h1>계단 모델 비교</h1><p>왼쪽: ArchiVision 원본 / 가운데: 밝기 보정 / 오른쪽: 기존 위험 단차 모델. 초록: 정답, 파랑: 계단 예측, 빨강: 기존 위험 단차 예측. 계단 전체와 위험 단차는 다른 학습 대상입니다.</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
