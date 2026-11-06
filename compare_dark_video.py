"""Compare baseline and active heads on exactly the same sampled video frames."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from hazard_head import HazardHead,probability_mask
from mask2former_detector import ROOT,StairsDetector
from video_encoding import browser_mp4
from vision import enhance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('video',type=Path)
    parser.add_argument('--input-view',choices=['original','enhanced'],default='enhanced')
    args=parser.parse_args()
    source=args.video.resolve()
    output=ROOT/'outputs/dark_video_comparison';output.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'models/backups/before_pin_review_v1/mask2former_hazard_head.pt'
    active=ROOT/'models/mask2former_hazard_head.pt'
    net=StairsDetector();state=torch.load(baseline,map_location=net.device,weights_only=True)
    old=HazardHead().to(net.device).eval();old.load_state_dict(state['state_dict'])
    old_threshold=float(state['threshold']);new_threshold=net.hazard_threshold
    capture=cv2.VideoCapture(str(source));writer=None;rows=[];index=0
    if not capture.isOpened():raise ValueError('영상을 읽지 못했습니다.')
    fps=capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<1:raise ValueError('원본 FPS를 읽지 못했습니다.')
    frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));step=max(1,int(np.ceil(fps)))
    working=output/'working.mp4';destination=output/'comparison.mp4'
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            selected=index%step==0;frame_index=index;index+=1
            if not selected:continue
            h,w=frame.shape[:2];scale=min(1,640/max(h,w))
            frame=cv2.resize(frame,(max(2,round(w*scale)//2*2),max(2,round(h*scale)//2*2)))
            view=enhance(frame) if args.input_view=='enhanced' else frame.copy()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            low=np.mean(gray<8)>.95 or gray.std()<1.5
            if not low:
                semantic,features=net.segment(view,return_features=True)
                region=np.isin(semantic,net.stair_ids).astype(np.uint8)*255
                tensor=torch.from_numpy((region>0).astype(np.float32))[None,None].to(net.device)
            panels=[frame.copy()];measurements={}
            for label,head,threshold in [('before',old,old_threshold),('after',net.hazard_head,new_threshold)]:
                if low:mask=np.zeros(frame.shape[:2],np.uint8);score=None
                else:
                    with torch.inference_mode():p=head(features,tensor).sigmoid()[0,0].cpu().numpy()
                    mask,score=probability_mask(p,region,threshold)
                display=view.copy();tint=np.full_like(display,(40,50,245))
                display[mask>0]=cv2.addWeighted(display,.5,tint,.5,0)[mask>0]
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display,contours,-1,(40,50,255),2)
                panels.append(display)
                measurements[label]={'detected':score is not None,'pixels':int(np.count_nonzero(mask)),'score':score}
            stamp=frame_index/fps
            for label,panel in zip(('SOURCE','BEFORE','AFTER'),panels):
                cv2.rectangle(panel,(0,0),(panel.shape[1],36),(8,15,24),-1)
                cv2.putText(panel,f'{label}  {stamp:.2f}s',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            joined=np.hstack(panels)
            if writer is None:
                writer=cv2.VideoWriter(str(working),cv2.VideoWriter_fourcc(*'mp4v'),fps/step,(joined.shape[1],joined.shape[0]))
                if not writer.isOpened():raise RuntimeError('영상 인코더 오류')
            writer.write(joined)
            rows.append({'frame_index':frame_index,'seconds':stamp,**measurements})
            if len(rows) in (1,5,10):
                ok,encoded=cv2.imencode('.jpg',joined)
                if ok:encoded.tofile(str(output/f'preview_{len(rows)}.jpg'))
            print(f'{frame_index}/{frame_count} {stamp:.2f}s',flush=True)
        if not rows:raise ValueError('분석한 프레임이 없습니다.')
    finally:
        capture.release()
        if writer:writer.release()
    browser_mp4(working,destination);working.unlink(missing_ok=True)
    report={'source_file':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
            'active_sha256':hashlib.sha256(active.read_bytes()).hexdigest(),
            'source_fps':fps,'source_frames':index,'analysis_frames':len(rows),'output_fps':fps/step,
            'analysis_interval_frames':step,'input_view':args.input_view,
            'before_threshold':old_threshold,'after_threshold':new_threshold,
            'shared_backbone_and_frames':True,'accuracy_measured':False,
            'note':'No video ground truth. Compare production settings, including different validation-selected thresholds. Max 1fps analysis, no interpolation or tracking. Audio omitted.',
            'frames':rows}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>dark 영상 재학습 전후 비교</title><style>body{font:17px/1.7 sans-serif;max-width:1500px;margin:30px auto;padding:20px;background:#14212e;color:white}a{color:#8ee6ff}video,img{width:100%;background:black}video{max-height:80vh}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #456}details{margin-top:25px}</style><a href="/">← 대시보드</a><h1>dark.mp4 · 재학습 전후 비교</h1><p><b>왼쪽 원본 / 가운데 재학습 전 / 오른쪽 재학습 후</b><br>동일 프레임·밝기 보정 입력으로 비교했습니다. 빨강은 위험 단차 후보입니다.</p><video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="dark_before_after.mp4">비교 영상 다운로드</a> · <a href="report.json">프레임별 분석 정보</a></p>'''
    page+=f'<p>원본 {fps:.3f}fps · 비교 분석 {len(rows)}프레임 · 결과 {fps/step:.3f}fps. 기존 영상 분석과 같은 최대 1fps 방식이며, 발표용 전체 프레임 분석은 아닙니다. 오디오는 없습니다.</p>'
    page+=f'<p>이전 문턱값 {old_threshold} / 새 문턱값 {new_threshold}. 실제 적용 설정끼리의 비교이며 가중치 변화만 분리한 실험은 아닙니다. 정답 라벨이 없어 영상 정확도는 계산하지 않았습니다.</p>'
    page+='<details><summary>프레임별 검출 여부 보기</summary><table><tr><th>원본 시각</th><th>이전</th><th>이후</th></tr>'
    for row in rows:
        page+=f'<tr><td>{row["seconds"]:.2f}초</td><td>'+('후보 있음' if row['before']['detected'] else '미검출')+'</td><td>'+('후보 있음' if row['after']['detected'] else '미검출')+'</td></tr>'
    page+='</table></details></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
