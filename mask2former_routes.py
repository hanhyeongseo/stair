"""Separate endpoints keep whole-stairs and hazard-step claims distinct."""
import json
import base64
import hashlib
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request, send_from_directory

from mask2former_detector import ROOT, MODEL_DIR, StairsDetector
from prepare import load_image

bp = Blueprint('mask2former', __name__)
_net = None
_load_lock = threading.Lock()


def get_detector():
    global _net
    with _load_lock:
        if _net is None:
            _net = StairsDetector()
    return _net


@bp.get('/mask2former')
def page():
    return render_template('mask2former.html')


@bp.get('/mask2former/experiments/')
def attachment_experiment():
    return send_from_directory(ROOT/'outputs'/'attachment_experiment','index.html')


@bp.get('/mask2former/experiments/<name>')
def attachment_asset(name):
    return send_from_directory(ROOT/'outputs'/'attachment_experiment',name)


@bp.get('/mask2former/pin-review/')
def pin_review_results():
    folder='pinned_review_v2' if (ROOT/'outputs/pinned_review_v2/index.html').exists() else 'pinned_review_v1'
    return send_from_directory(ROOT/'outputs'/folder,'index.html')


@bp.get('/mask2former/pin-review/<name>')
def pin_review_asset(name):
    folder='pinned_review_v2' if (ROOT/'outputs/pinned_review_v2/index.html').exists() else 'pinned_review_v1'
    return send_from_directory(ROOT/'outputs'/folder,name)


@bp.get('/mask2former/results/')
def saved_results():
    return send_from_directory(ROOT/'outputs'/'mask2former_evaluation','index.html')


@bp.get('/mask2former/results/<name>')
def saved_asset(name):
    return send_from_directory(ROOT/'outputs'/'mask2former_evaluation',name)


@bp.get('/mask2former/hazard-results/')
def hazard_results():
    return send_from_directory(ROOT/'outputs'/'hazard_head_evaluation','index.html')


@bp.get('/mask2former/hazard-results/<name>')
def hazard_result_asset(name):
    return send_from_directory(ROOT/'outputs'/'hazard_head_evaluation',name)


@bp.get('/api/mask2former/status')
def status():
    evaluation_path = ROOT/'outputs'/'hazard_head_evaluation'/'report.json'
    head_path = ROOT/'models'/'mask2former_hazard_head.pt'
    evaluation = None
    if evaluation_path.exists() and head_path.exists():
        candidate = json.loads(evaluation_path.read_text(encoding='utf-8'))
        if candidate.get('checkpoint_sha256')==hashlib.sha256(head_path.read_bytes()).hexdigest():
            evaluation = {k:v for k,v in candidate.items() if k!='records'}
    active_path=ROOT/'models'/'active_hazard_model.json'
    active_review=None
    if active_path.exists() and head_path.exists():
        active=json.loads(active_path.read_text(encoding='utf-8'))
        if active.get('checkpoint_sha256')==hashlib.sha256(head_path.read_bytes()).hexdigest():active_review=active
    return jsonify(ready=all((MODEL_DIR/n).exists() for n in ('model.safetensors','config.json','preprocessor_config.json')),
                   loaded=_net is not None, task='stairs_and_hazard_candidates',
                   hazard_head_ready=head_path.exists(),hazard_evaluation=evaluation,active_review=active_review)


@bp.post('/api/mask2former/analyze')
def analyze():
    def encode(image):
        ok, data = cv2.imencode('.png' if image.ndim == 3 and image.shape[2] == 4 else '.jpg', image)
        if not ok: raise ValueError('이미지 인코딩 실패')
        mime = 'image/png' if image.ndim == 3 and image.shape[2] == 4 else 'image/jpeg'
        return 'data:'+mime+';base64,'+base64.b64encode(data).decode('ascii')
    try:
        record = None
        if request.is_json:
            body = request.get_json()
            if not isinstance(body, dict):
                raise ValueError('사진을 선택하세요.')
            mode = body.get('mode', 'both')
            records = json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))['records']
            record = next((r for r in records if r['id'] == body.get('id')), None)
            if record is None:
                raise ValueError('등록된 사진이 아닙니다.')
            frame = load_image(Path(record['image']))
        else:
            mode = request.form.get('mode', 'both')
            upload = request.files.get('image')
            if upload is None:
                raise ValueError('사진을 선택하세요.')
            content = upload.read(20*1024*1024+1)
            if len(content) > 20*1024*1024:
                raise ValueError('사진은 20MB 이하로 선택하세요.')
            frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[0]*frame.shape[1] > 50_000_000:
                raise ValueError('읽을 수 없거나 너무 큰 이미지입니다.')
        if mode not in ('original','enhanced','both'):
            raise ValueError('입력 방식을 확인하세요.')
        views, report = get_detector().infer(frame, mode)
        if record:
            for name in report['views']:
                compare = views[name+'_overlay'].copy()
                h,w = compare.shape[:2]
                for polygon in record['polygons']:
                    pts = np.round(np.array(polygon['points'])*[w,h]).astype(np.int32)
                    cv2.polylines(compare,[pts],True,(0,255,0),2)
                views[name+'_annotation'] = compare
                if name+'_hazard_overlay' in views:
                    hazard_compare = views[name+'_hazard_overlay'].copy()
                    for polygon in record['polygons']:
                        pts = np.round(np.array(polygon['points'])*[w,h]).astype(np.int32)
                        cv2.polylines(hazard_compare,[pts],True,(0,255,0),2)
                    views[name+'_hazard_annotation'] = hazard_compare
        return jsonify(images={k:encode(v) for k,v in views.items()}, result=report,
                       record=record, model_input_used_annotation=False)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    except RuntimeError as exc:
        return jsonify(error='Mask2Former 실행 실패: '+str(exc)), 503
