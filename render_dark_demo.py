"""Precompute every frame after 9 seconds; export at original playback FPS.

This is an offline demonstration, not a measurement of real-time throughput.
No frames or detection masks are interpolated and no previous mask is reused.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from mask2former_detector import ROOT,StairsDetector
from video_encoding import browser_mp4
from vision import enhance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('video',type=Path)
    parser.add_argument('--start',type=float,default=9.)
    args=parser.parse_args()
    source=args.video.resolve();started=time.perf_counter()
    output=ROOT/'outputs/dark_demo';output.mkdir(parents=True,exist_ok=True)
    frames=output/'frames';frames.mkdir(exist_ok=True)
    net=StairsDetector()
    capture=cv2.VideoCapture(str(source));writers=[];rows=[]
    if not capture.isOpened():raise ValueError('영상을 읽을 수 없습니다.')
    fps=capture.get(cv2.CAP_PROP_FPS);total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps<1:raise ValueError('원본 FPS를 확인할 수 없습니다.')
    start_index=int(np.ceil(args.start*fps))
    if start_index>=total:raise ValueError('시작 시점이 영상 끝보다 늦습니다.')
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    index=0;output_size=None
    try:
        while True:
            ok,original=capture.read()
            if not ok:break
            source_index=index;index+=1
            if source_index<start_index:continue
            h,w=original.shape[:2];scale=min(1,640/max(h,w))
            analysis=cv2.resize(original,(max(2,round(w*scale)),max(2,round(h*scale))))
            views,result=net.infer(analysis,mode='enhanced')
            mask=views.get('enhanced_hazard_mask',np.zeros(analysis.shape[:2],np.uint8))
            display_scale=min(1,1280/max(h,w))
            width=max(2,round(w*display_scale)//2*2);height=max(2,round(h*display_scale)//2*2)
            raw=cv2.resize(original,(width,height),interpolation=cv2.INTER_AREA)
            corrected=enhance(raw)
            mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
            overlay=corrected.copy();tint=np.full_like(overlay,(40,50,245))
            overlay[mask>0]=cv2.addWeighted(overlay,.6,tint,.4,0)[mask>0]
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(40,50,255),2)
            # Label the offline nature in the artifact, not just its web page.
            canvas=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
            draw=ImageDraw.Draw(canvas)
            draw.rectangle((8,height-38,190,height-8),fill=(12,20,30))
            draw.text((16,height-36),'사전 분석 결과',font=font,fill=(240,245,255))
            overlay=cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR)
            if not writers:
                output_size=[width,height]
                for filename,size in [('overlay_working.mp4',(width,height)),('comparison_working.mp4',(width*2,height))]:
                    writer=cv2.VideoWriter(str(output/filename),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
                    if not writer.isOpened():raise RuntimeError('영상 기록을 시작하지 못했습니다.')
                    writers.append(writer)
            writers[0].write(overlay);writers[1].write(np.hstack([raw,overlay]))
            ok,encoded=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,93])
            if not ok:raise RuntimeError('프레임 저장 실패')
            encoded.tofile(str(frames/f'frame_{len(rows):06d}.jpg'))
            hazard=result['views']['enhanced'].get('hazard',{})
            rows.append({'source_frame':source_index,'source_seconds':source_index/fps,
                         'output_frame':len(rows),'detected':hazard.get('status')=='candidate',
                         'score':hazard.get('score')})
            if len(rows)%30==0 or len(rows)==1:
                elapsed=time.perf_counter()-started
                print(json.dumps({'done':len(rows),'expected':total-start_index,'source_seconds':round(source_index/fps,2),'elapsed_seconds':round(elapsed,1)}),flush=True)
                (output/'progress.json').write_text(json.dumps({'status':'processing','frames':len(rows),'expected':total-start_index}),encoding='utf-8')
    finally:
        capture.release()
        for writer in writers:writer.release()
    if not rows or len(rows)!=index-start_index:raise RuntimeError('원본과 출력의 프레임 수가 일치하지 않습니다.')
    (output/'progress.json').write_text(json.dumps({'status':'encoding','frames':len(rows),'expected':len(rows)}),encoding='utf-8')
    for stem in ('overlay','comparison'):
        browser_mp4(output/(stem+'_working.mp4'),output/(stem+'.mp4'))
        (output/(stem+'_working.mp4')).unlink(missing_ok=True)
    report={'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checkpoint_sha256':net.hazard_checkpoint_sha256,'start_requested_seconds':args.start,
            'first_source_frame':start_index,'last_source_frame':index-1,
            'first_source_seconds':start_index/fps,'source_frames':index,
            'output_frames':len(rows),'source_fps':fps,'output_fps':fps,'output_size':output_size,
            'output_duration_seconds':len(rows)/fps,'analysis_input_max_edge':640,
            'elapsed_seconds':round(time.perf_counter()-started,1),'offline_precomputed':True,
            'model_inference_on_every_output_frame':True,'interpolated_or_repeated_masks':False,
            'audio':'omitted','frames':rows}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>dark 발표용 영상</title><style>body{font:18px/1.8 sans-serif;max-width:1400px;margin:30px auto;padding:20px;background:#14212e;color:white}video{width:100%;max-height:80vh;background:black}a{color:#8ee6ff}</style><a href="/">← 대시보드</a><h1>dark · 9초부터 끝까지</h1><p>모든 프레임을 미리 분석한 사전 분석 영상입니다. 원본 FPS로 재생하며 실시간 분석 처리 속도를 나타내지 않습니다. 표시 영역의 흔들림이나 미검출은 그대로 남겼습니다.</p><h2>분석 결과만 보기</h2><video controls preload="metadata" src="overlay.mp4"></video><p><a href="overlay.mp4" download="dark_demo_9s_overlay.mp4">발표용 결과 MP4 다운로드</a></p><h2>원본과 결과 나란히 보기</h2><video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="dark_demo_9s_comparison.mp4">원본·결과 비교 MP4 다운로드</a> · <a href="report.json">분석 정보</a></p>'''
    page+=f'<p>프레임 {len(rows)}장 · {fps:.3f}fps · 약 {len(rows)/fps:.2f}초 · 오디오 없음.</p></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    (output/'progress.json').write_text(json.dumps({'status':'done','frames':len(rows),'expected':len(rows)}),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
