"""Keep synchronized source footage at left, stable orange result at right."""
import json
import os
import subprocess

import cv2
import imageio_ffmpeg

from mask2former_detector import ROOT


def main():
    folder=ROOT/'outputs/dark_demo_orange'
    # The older comparison retains the untouched camera frames in its left half.
    original=ROOT/'outputs/dark_demo/comparison.mp4'
    overlay=folder/'overlay.mp4'
    output=folder/'source_comparison_10fps.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
        '-i',str(original),'-i',str(overlay),'-filter_complex',
        '[0:v]crop=iw/2:ih:0:0[left];[left][1:v]hstack=inputs=2:shortest=1,fps=10[v]',
        '-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)],
        check=True,capture_output=True,text=True,timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    capture=cv2.VideoCapture(str(output));fps=capture.get(cv2.CAP_PROP_FPS);frames=0
    right=cv2.VideoCapture(str(folder/'overlay_10fps.mp4'));errors=[]
    while True:
        ok,image=capture.read()
        if not ok:break
        ok,reference=right.read();assert ok
        assert image.shape[1]==reference.shape[1]*2
        errors.append(float(cv2.absdiff(image[:,reference.shape[1]:],reference).mean()))
        if frames==110:
            success,encoded=cv2.imencode('.jpg',image)
            if success:encoded.tofile(str(folder/'source_comparison_preview.jpg'))
        frames+=1
    capture.release();right.release()
    assert frames==341 and abs(fps-10)<.001
    # Re-encoding may introduce small differences, but frame order must match.
    assert max(errors)<8,max(errors)
    report={'frames':frames,'fps':fps,'duration_seconds':frames/fps,
            'left':'Source camera footage, no brightness correction or overlay',
            'right':'Stable orange overlay, same frames and timestamps',
            'right_mean_encoding_difference':sum(errors)/len(errors),'model_rerun':False}
    (folder/'source_comparison_10fps.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>주황색 발표 영상 · 10fps</title><style>body{font:18px/1.8 sans-serif;max-width:1400px;margin:30px auto;padding:20px;background:#14212e;color:white}video{width:100%;max-height:80vh;background:black}a{color:#8ee6ff}</style><a href="/orange-demo/">30fps 버전 보기</a><h1>주황색 발표 영상 · 10fps</h1><p>dark 영상의 9초 이후 약 34초 구간입니다. 기존 사전 분석 결과를 10fps로 저장한 영상입니다.</p><h2>원본과 분석 결과 비교</h2><p><b>왼쪽: 보정·표시가 없는 원본 영상 / 오른쪽: 밝기 보정과 주황색 표시 안정화 결과</b><br>같은 프레임을 같은 시점에 재생합니다. 실시간 추론 속도를 나타내는 영상은 아닙니다.</p><video controls preload="metadata" src="source_comparison_10fps.mp4"></video><p><a href="source_comparison_10fps.mp4" download="dark_source_result_10fps.mp4">원본·결과 비교 영상 다운로드</a></p><h2>분석 결과만 보기</h2><video controls preload="metadata" src="overlay_10fps.mp4"></video><p><a href="overlay_10fps.mp4" download="dark_orange_10fps.mp4">결과 영상 다운로드</a></p><details><summary>표시 안정화 자체의 전후 비교</summary><p>이 영상의 왼쪽은 원본이 아니라 시간 보정 전 예측입니다.</p><video controls preload="none" src="comparison_10fps.mp4"></video></details></html>'''
    (folder/'10fps.html').write_text(page,encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
