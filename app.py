"""Independent hazard-step dashboard. Run and open http://127.0.0.1:5001."""
import base64
from functools import lru_cache
import json
from pathlib import Path
import threading
import uuid
import re

import cv2
import numpy as np
from flask import Flask,jsonify,render_template,request,send_file,send_from_directory

from detector import ROOT,Detector
from prepare import load_image

app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=128*1024*1024
OUTPUT=ROOT/'outputs'/'web';OUTPUT.mkdir(parents=True,exist_ok=True)
jobs={};video_slot=threading.BoundedSemaphore(1)

from mask2former_routes import bp as mask2former_bp
app.register_blueprint(mask2former_bp)


def read_json(path,default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


@lru_cache(maxsize=1)
def detector(modified):return Detector()


def model():
    path=ROOT/'models'/'hazard_best.pt'
    if not path.exists():raise FileNotFoundError('YOLO 위험 단차 학습이 아직 완료되지 않았습니다.')
    return detector(path.stat().st_mtime_ns)


def encode(image):
    png=image.ndim==3 and image.shape[2]==4
    ok,data=cv2.imencode('.png' if png else '.jpg',image)
    if not ok:raise ValueError('영상 인코딩 실패')
    return ('data:image/png;base64,' if png else 'data:image/jpeg;base64,')+base64.b64encode(data).decode('ascii')


def analyze(frame,confidence,color,record=None):
    views,report=model().infer(frame,confidence,color)
    if record:
        comparison=views['overlay'].copy();h,w=comparison.shape[:2]
        for polygon in record['polygons']:
            pts=np.round(np.array(polygon['points'])*[w,h]).astype(np.int32)
            cv2.polylines(comparison,[pts],True,(0,255,0),2)
        views['annotation_compare']=comparison
    return {'images':{k:encode(v) for k,v in views.items()},'result':report,
            'record':record,'model_input_used_annotation':False}


@app.get('/')
def index():return render_template('mask2former.html')


@app.get('/yolo')
def yolo_page():return render_template('index.html')


@app.get('/comparison/')
def pretrained_comparison():
    path=ROOT/'outputs'/'pretrained_comparison'
    if not (path/'index.html').exists():return jsonify(error='비교 실행이 아직 완료되지 않았습니다.'),503
    return send_from_directory(path,'index.html')


@app.get('/comparison/<name>')
def pretrained_asset(name):
    return send_from_directory(ROOT/'outputs'/'pretrained_comparison',name)


@app.get('/api/catalog')
def catalog():
    manifest=read_json(ROOT/'data'/'manifest.json',{})
    return jsonify(records=manifest.get('records',[]),excluded=manifest.get('excluded',[]),
                   label_policy=manifest.get('label_policy'),split_policy=manifest.get('split_policy'))


@app.get('/api/status')
def status():
    report=read_json(ROOT/'models'/'report.json')
    evaluation=read_json(ROOT/'models'/'pipeline_evaluation.json')
    if evaluation:evaluation={k:v for k,v in evaluation.items() if k!='records'}
    return jsonify(ready=(ROOT/'models'/'hazard_best.pt').exists(),report=report,evaluation=evaluation)


@app.post('/api/analyze')
def analyze_photo():
    try:
        body=request.get_json()
        if not isinstance(body,dict):raise ValueError('사진을 선택하세요.')
        manifest=read_json(ROOT/'data'/'manifest.json',{})
        record=next((r for r in manifest.get('records',[]) if r['id']==body.get('id')),None)
        if not record:raise ValueError('등록된 사진이 아닙니다.')
        return jsonify(**analyze(load_image(Path(record['image'])),float(body.get('confidence',.4)),body.get('color',True) is True,record))
    except (ValueError,TypeError) as exc:return jsonify(error=str(exc)),400
    except FileNotFoundError as exc:return jsonify(error=str(exc)),503


@app.post('/api/upload')
def upload():
    try:
        file=request.files.get('image')
        if not file:raise ValueError('사진을 선택하세요.')
        content=file.read(20*1024*1024+1)
        if len(content)>20*1024*1024:raise ValueError('사진은 20MB 이하로 선택하세요.')
        frame=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0]*frame.shape[1]>50_000_000:raise ValueError('읽을 수 없거나 너무 큰 이미지입니다.')
        return jsonify(**analyze(frame,float(request.form.get('confidence',.4)),request.form.get('color','true')=='true'))
    except (ValueError,TypeError) as exc:return jsonify(error=str(exc)),400
    except FileNotFoundError as exc:return jsonify(error=str(exc)),503


def process_video(key,path,confidence,color,engine='yolo',input_view='enhanced'):
    capture=None;writer=None
    try:
        capture=cv2.VideoCapture(str(path))
        if not capture.isOpened():raise ValueError('영상을 읽지 못했습니다.')
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<1:fps=25
        count=capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if count>0 and count/fps>60:raise ValueError('60초 이하 영상만 지원합니다.')
        output_fps=1 if engine=='mask2former' else 5
        step=max(1,int(np.ceil(fps/output_fps)));decoded=0;processed=0
        destination=OUTPUT/(key+'_result.mp4')
        working=OUTPUT/(key+'_working.mp4')
        if engine=='mask2former':
            from mask2former_routes import get_detector
            net=get_detector()
        else:net=model()
        while True:
            ok,frame=capture.read()
            if not ok or decoded/fps>=60:break
            selected=decoded%step==0;decoded+=1
            if not selected:continue
            h,w=frame.shape[:2];scale=min(1,640/max(h,w))
            frame=cv2.resize(frame,(max(2,round(w*scale)),max(2,round(h*scale))))
            if engine=='mask2former':
                views,result=net.infer(frame,mode=input_view)
                overlay=views.get(input_view+'_hazard_overlay',views[input_view+'_overlay'])
            else:
                views,result=net.infer(frame,confidence,color)
                overlay=views['overlay']
            joined=np.hstack([views['original'],overlay]);hh,ww=joined.shape[:2]
            joined=cv2.resize(joined,(ww-ww%2,hh-hh%2))
            if writer is None:
                writer=cv2.VideoWriter(str(working),cv2.VideoWriter_fourcc(*'mp4v'),fps/step,(joined.shape[1],joined.shape[0]))
                if not writer.isOpened():raise ValueError('MP4 인코더를 열지 못했습니다.')
            writer.write(joined);processed+=1
            jobs[key].update(status='processing',frames=processed,seconds=round(decoded/fps,1),
                             progress=min(99,round(decoded/max(count,1)*100)) if count>0 else None)
        if not processed:raise ValueError('유효한 프레임이 없습니다.')
        writer.release();writer=None
        jobs[key].update(status='encoding',progress=99)
        from video_encoding import browser_mp4
        browser_mp4(working,destination)
        working.unlink(missing_ok=True)
        jobs[key].update(status='done',progress=100,result=str(destination),codec='h264',pixel_format='yuv420p')
    except Exception as exc:jobs[key].update(status='error',error=str(exc))
    finally:
        if capture:capture.release()
        if writer:writer.release()
        path.unlink(missing_ok=True);video_slot.release()


@app.post('/api/video')
def video():
    if not video_slot.acquire(blocking=False):return jsonify(error='다른 영상 처리가 완료된 후 시도하세요.'),409
    path=None
    try:
        file=request.files.get('video')
        if not file:raise ValueError('영상을 선택하세요.')
        confidence=float(request.form.get('confidence',.4))
        if not .1<=confidence<=.95:raise ValueError('잘못된 점수 문턱값입니다.')
        engine=request.form.get('engine','yolo')
        input_view=request.form.get('input_view','enhanced')
        if engine not in ('yolo','mask2former') or input_view not in ('original','enhanced'):
            raise ValueError('잘못된 모델 또는 입력 방식입니다.')
        key=uuid.uuid4().hex;path=OUTPUT/(key+'_input.mp4');file.save(path)
        if path.stat().st_size>120*1024*1024:raise ValueError('영상은 120MB 이하만 지원합니다.')
        jobs[key]={'id':key,'status':'queued','progress':0,'engine':engine,'input_view':input_view}
        threading.Thread(target=process_video,args=(key,path,confidence,request.form.get('color','true')=='true',engine,input_view),daemon=True).start()
        return jsonify(id=key),202
    except (ValueError,OSError) as exc:
        if path:path.unlink(missing_ok=True)
        video_slot.release();return jsonify(error=str(exc)),400


@app.get('/api/video/<key>')
def video_status(key):
    job=completed_video(key)
    if not job:return jsonify(error='작업이 없습니다. 서버 재시작 후에는 다시 업로드하세요.'),404
    return jsonify(**{k:v for k,v in job.items() if k!='result'},
                   download='/api/video/'+key+'/download' if job['status']=='done' else None,
                   playback='/api/video/'+key+'/play' if job['status']=='done' else None)


def completed_video(key):
    if key in jobs:return jobs[key]
    if not re.fullmatch(r'[a-f0-9]{32}',key):return None
    path=OUTPUT/(key+'_result.mp4')
    if path.is_file():return {'id':key,'status':'done','progress':100,'result':str(path)}
    return None


@app.get('/api/video/<key>/play')
def play_video(key):
    job=completed_video(key)
    if not job or job['status']!='done':return jsonify(error='완료된 영상이 없습니다.'),404
    return send_file(job['result'],mimetype='video/mp4',conditional=True)


@app.get('/videos')
def video_gallery():
    paths=sorted(OUTPUT.glob('*_result.mp4'),key=lambda p:p.stat().st_mtime,reverse=True)
    keys=[p.name.removesuffix('_result.mp4') for p in paths if re.fullmatch(r'[a-f0-9]{32}_result.mp4',p.name)]
    return render_template('videos.html',keys=keys)


@app.get('/api/video/<key>/download')
def download(key):
    job=completed_video(key)
    if not job or job['status']!='done':return jsonify(error='완료된 영상이 없습니다.'),404
    return send_file(job['result'],mimetype='video/mp4',as_attachment=True,download_name='hazard_step.mp4')


if __name__=='__main__':app.run(host='127.0.0.1',port=5001,debug=False)
