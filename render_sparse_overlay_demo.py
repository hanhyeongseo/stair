"""Original FPS source/background, detector updates 5Hz, tracking in between."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from mask2former_detector import ROOT
from overlay_style import candidate_overlay
from sparse_overlay import SparseMaskTracker
from video_encoding import browser_mp4
from vision import enhance


def main():
    source=Path('C:/Users/KDT76/OneDrive/Desktop/dark.mp4')
    reference=json.loads((ROOT/'outputs/dark_demo/report.json').read_text(encoding='utf-8'))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=reference['source_sha256']:raise ValueError('Source changed')
    checkpoint=hashlib.sha256((ROOT/'models/mask2former_hazard_head.pt').read_bytes()).hexdigest()
    if checkpoint!=reference['checkpoint_sha256']:raise ValueError('Cached predictions belong to a different model')
    cache=ROOT/'outputs/dark_demo/mask_cache'/checkpoint[:16]
    output=ROOT/'outputs/dark_sparse_overlay';output.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(source));fps=capture.get(cv2.CAP_PROP_FPS)
    start=reference['first_source_frame'];last_bucket=-1;tracker=SparseMaskTracker()
    index=0;rows=[];writers=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            source_index=index;index+=1
            if source_index<start:continue
            n=len(rows);bucket=int(np.floor(n/fps*5+1e-9));update=bucket!=last_bucket
            h,w=frame.shape[:2];scale=min(1,640/max(h,w))
            analysis=cv2.resize(frame,(round(w*scale),round(h*scale)))
            detection=None
            if update:
                path=cache/f'{source_index:06d}_mask.png'
                detection=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE)
                if detection is None:raise ValueError('Missing detector mask')
                last_bucket=bucket;last_detection_frame=source_index
            mask=tracker.update(enhance(analysis),detection)
            scale=min(1,1280/max(h,w));size=(round(w*scale)//2*2,round(h*scale)//2*2)
            original=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
            # Crucially the rendered background is original, not enhance(original).
            overlay=candidate_overlay(original,cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST))
            canvas=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(canvas)
            draw.rectangle((8,size[1]-38,265,size[1]-8),fill=(12,20,30));draw.text((16,size[1]-36),'사전 분석 · 표시 추적',font=font,fill=(240,245,255))
            overlay=cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR)
            if not writers:
                for name,dimensions in [('overlay_working.mp4',size),('comparison_working.mp4',(size[0]*2,size[1]))]:
                    writer=cv2.VideoWriter(str(output/name),cv2.VideoWriter_fourcc(*'mp4v'),fps,dimensions)
                    if not writer.isOpened():raise RuntimeError('Video encoder failed')
                    writers.append(writer)
            writers[0].write(overlay);writers[1].write(np.hstack([original,overlay]))
            if n in (120,330):
                ok,encoded=cv2.imencode('.jpg',np.hstack([original,overlay]))
                if ok:encoded.tofile(str(output/f'preview_{n}.jpg'))
            rows.append({'source_frame':source_index,'detector_update':update,
                         'latest_detector_frame':last_detection_frame,
                         'detection_age_seconds':(source_index-last_detection_frame)/fps})
            if len(rows)%120==0:print(f'{len(rows)}/{reference["output_frames"]}',flush=True)
    finally:
        capture.release()
        for writer in writers:writer.release()
    if len(rows)!=reference['output_frames']:raise ValueError('Frame count mismatch')
    for name in ('overlay','comparison'):
        browser_mp4(output/(name+'_working.mp4'),output/(name+'.mp4'));(output/(name+'_working.mp4')).unlink(missing_ok=True)
    report={'source_sha256':reference['source_sha256'],'checkpoint_sha256':checkpoint,
            'source_start_seconds':9,'fps':fps,'output_frames':len(rows),'duration_seconds':len(rows)/fps,
            'detector_updates':sum(r['detector_update'] for r in rows),'detector_target_hz':5,
            'between_updates':'optical-flow tracking only; no fresh model predictions',
            'background':'original camera footage on both sides, no brightness correction',
            'inference_input':'brightness-corrected analysis image as before',
            'offline_saved_predictions':True,'real_time_speed_measured':False,'rows':rows}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>원본 영상 위 단차 표시</title><style>body{font:18px/1.8 sans-serif;max-width:1400px;margin:30px auto;padding:20px;background:#14212e;color:white}video{width:100%;max-height:80vh;background:black}a{color:#8ee6ff}.note{color:#aabccd;font-size:14px}</style><a href="/">대시보드</a><h1>원본 영상 위 단차 표시</h1><p>왼쪽 원본 · 오른쪽 같은 원본에 주황색 영역만 표시</p><p class="note">dark · 9초부터 끝까지 · 저장된 분석 결과와 표시 추적</p><video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="dark_original_sparse_overlay.mp4">비교 영상 다운로드</a></p><details><summary>표시 영상만 보기</summary><video controls preload="none" src="overlay.mp4"></video><a href="overlay.mp4" download="dark_sparse_overlay.mp4">표시 영상 다운로드</a></details><details><summary>처리 방식</summary><p>원본 속도로 모든 프레임을 재생하고, 초당 5번 선택한 탐지 결과 사이에서는 이동 추적만 수행합니다. 양쪽 배경에 밝기 보정을 적용하지 않았습니다. 기존 프레임별 분석 결과에서 필요한 시점만 사용한 사전 처리 영상이며 실시간 추론 성능을 측정한 결과는 아닙니다.</p><a href="report.json">프레임별 처리 정보</a></details></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
