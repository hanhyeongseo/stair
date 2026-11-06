"""Cache per-frame predictions and export orange raw/stabilized offline demos."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from hazard_head import probability_mask
from mask2former_detector import ROOT,StairsDetector
from overlay_style import candidate_overlay,MotionMaskSmoother
from video_encoding import browser_mp4
from vision import enhance


def masks_for_batch(net,frames):
    analyses=[];corrected=[]
    for image in frames:
        h,w=image.shape[:2];scale=min(1,640/max(h,w))
        small=cv2.resize(image,(max(2,round(w*scale)),max(2,round(h*scale))))
        analyses.append(small);corrected.append(enhance(small))
    inputs=net.processor(images=[cv2.cvtColor(im,cv2.COLOR_BGR2RGB) for im in corrected],
                         size={'shortest_edge':512,'longest_edge':768},return_tensors='pt')
    inputs={k:v.to(net.device) for k,v in inputs.items()}
    with torch.inference_mode():
        outputs=net.model(**inputs)
        semantic=net.processor.post_process_semantic_segmentation(outputs,target_sizes=[im.shape[:2] for im in analyses])
        result=[]
        for i,image in enumerate(analyses):
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if np.mean(gray<8)>.95 or gray.std()<1.5:
                mask=np.zeros(gray.shape,np.uint8);support=mask.copy()
            else:
                support=np.isin(semantic[i].cpu().numpy(),net.stair_ids).astype(np.uint8)*255
                region=torch.from_numpy((support>0).astype(np.float32))[None,None].to(net.device)
                probability=net.hazard_head(outputs.pixel_decoder_last_hidden_state[i:i+1],region).sigmoid()[0,0].cpu().numpy()
                mask,_=probability_mask(probability,support,net.hazard_threshold)
            result.append((mask,support))
    return result


def png(path,image):
    ok,data=cv2.imencode('.png',image)
    if not ok:raise RuntimeError('Mask encoding failed')
    data.tofile(str(path))


def main():
    source=Path('C:/Users/KDT76/OneDrive/Desktop/dark.mp4')
    original_report=json.loads((ROOT/'outputs/dark_demo/report.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=original_report['source_sha256']:raise ValueError('원본 영상이 변경됐습니다.')
    net=StairsDetector()
    output=ROOT/'outputs/dark_demo_orange';output.mkdir(parents=True,exist_ok=True)
    frame_folder=output/'frames';frame_folder.mkdir(exist_ok=True)
    cache=ROOT/'outputs/dark_demo/mask_cache'/net.hazard_checkpoint_sha256[:16];cache.mkdir(parents=True,exist_ok=True)
    capture=cv2.VideoCapture(str(source));fps=capture.get(cv2.CAP_PROP_FPS)
    start=original_report['first_source_frame'];expected=original_report['output_frames']
    smoother=MotionMaskSmoother();font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    writers=[];index=0;rows=[];batch=[];started=time.perf_counter()
    def process(items):
        nonlocal writers
        pending=[]
        for source_index,image in items:
            mp=cache/f'{source_index:06d}_mask.png';sp=cache/f'{source_index:06d}_stairs.png'
            pending.append(None if mp.exists() and sp.exists() else image)
        missing=[image for image in pending if image is not None]
        predicted=iter(masks_for_batch(net,missing)) if missing else iter(())
        for (source_index,image),missing_image in zip(items,pending):
            mp=cache/f'{source_index:06d}_mask.png';sp=cache/f'{source_index:06d}_stairs.png'
            if missing_image is not None:
                mask,support=next(predicted);png(mp,mask);png(sp,support)
            else:
                mask=cv2.imdecode(np.fromfile(str(mp),np.uint8),cv2.IMREAD_GRAYSCALE)
                support=cv2.imdecode(np.fromfile(str(sp),np.uint8),cv2.IMREAD_GRAYSCALE)
            # Optical flow runs in analysis coordinates and follows image motion.
            analysis=cv2.resize(image,(mask.shape[1],mask.shape[0]))
            stable=smoother.update(enhance(analysis),mask,support)
            h,w=image.shape[:2];scale=min(1,1280/max(h,w));size=(round(w*scale)//2*2,round(h*scale)//2*2)
            background=enhance(cv2.resize(image,size,interpolation=cv2.INTER_AREA))
            raw_view=candidate_overlay(background,cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST))
            stable_view=candidate_overlay(background,cv2.resize(stable,size,interpolation=cv2.INTER_NEAREST))
            for view,label in [(raw_view,'사전 분석 · 원래 예측'),(stable_view,'사전 분석 · 표시 안정화')]:
                canvas=Image.fromarray(cv2.cvtColor(view,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(canvas)
                draw.rectangle((8,size[1]-38,280,size[1]-8),fill=(12,20,30))
                draw.text((16,size[1]-36),label,font=font,fill=(240,245,255))
                view[:]=cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR)
            if not writers:
                for name,dimensions in [('overlay_working.mp4',size),('comparison_working.mp4',(size[0]*2,size[1]))]:
                    writer=cv2.VideoWriter(str(output/name),cv2.VideoWriter_fourcc(*'mp4v'),fps,dimensions)
                    if not writer.isOpened():raise RuntimeError('Video writer failed')
                    writers.append(writer)
            writers[0].write(stable_view);writers[1].write(np.hstack([raw_view,stable_view]))
            ok,encoded=cv2.imencode('.jpg',stable_view,[cv2.IMWRITE_JPEG_QUALITY,93])
            if not ok:raise RuntimeError('Frame encoding failed')
            encoded.tofile(str(frame_folder/f'frame_{len(rows):06d}.jpg'))
            rows.append({'source_frame':source_index,'raw_pixels':int(np.count_nonzero(mask)),
                         'stabilized_pixels':int(np.count_nonzero(stable))})
            if len(rows)%30==0 or len(rows)==1:
                print(json.dumps({'frames':len(rows),'expected':expected,'elapsed':round(time.perf_counter()-started,1)}),flush=True)
                (output/'progress.json').write_text(json.dumps({'status':'processing','frames':len(rows),'expected':expected}),encoding='utf-8')
    try:
        while True:
            ok,image=capture.read()
            if not ok:break
            source_index=index;index+=1
            if source_index<start:continue
            batch.append((source_index,image))
            if len(batch)==2:process(batch);batch=[]
        if batch:process(batch)
    finally:
        capture.release()
        for writer in writers:writer.release()
    if len(rows)!=expected:raise RuntimeError('프레임 수가 맞지 않습니다.')
    (output/'progress.json').write_text(json.dumps({'status':'encoding','frames':len(rows),'expected':expected}),encoding='utf-8')
    for name in ('overlay','comparison'):
        browser_mp4(output/(name+'_working.mp4'),output/(name+'.mp4'));(output/(name+'_working.mp4')).unlink(missing_ok=True)
    report={'source_sha256':digest,'checkpoint_sha256':net.hazard_checkpoint_sha256,'fps':fps,'frames':len(rows),
            'start_seconds':9,'offline':True,'color':'orange outline, 6% fill','raw_predictions_cached':str(cache),
            'postprocessing':'Backward optical flow, EMA current=.45/previous=.55, max missing 2 frames, candidate switch confirmation, current stairs gate',
            'danger_classification':False,'accuracy_measured':False,'rows':rows}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>주황색 발표 영상</title><style>body{font:18px/1.8 sans-serif;max-width:1400px;margin:30px auto;padding:20px;background:#14212e;color:white}video{width:100%;max-height:80vh;background:black}a{color:#8ee6ff}</style><a href="/">대시보드</a><h1>주황색 윤곽선 · 표시 안정화</h1><p>dark 영상 9초부터 끝까지 약 30fps 사전 분석 결과입니다. 주황은 단차 후보이며 위험도를 판정한 경고가 아닙니다. 색으로 위험 등급을 구분하지 않습니다.</p><video controls preload="metadata" src="overlay.mp4"></video><p><a href="overlay.mp4" download="dark_orange_stabilized.mp4">발표용 영상 다운로드</a></p><h2>표시 안정화 전후</h2><p>왼쪽: 원래 예측을 주황색으로 표시 / 오른쪽: 이동 추적·시간 보정 후. 추적은 표시를 다듬는 과정이며 실제 검출 정확도가 향상됐다는 뜻은 아닙니다. 최대 두 프레임 동안 표시가 유지될 수 있습니다.</p><video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="dark_orange_before_after.mp4">비교 영상 다운로드</a> · <a href="/dark-demo/">이전 빨간 원본 예측 영상</a> · <a href="report.json">처리 정보</a></p></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    (output/'progress.json').write_text(json.dumps({'status':'done','frames':len(rows),'expected':expected}),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
