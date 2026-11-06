"""Persistent review selection; original bytes only, no training side effects."""
import json
import tempfile
import threading
import zipfile
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'data'/'manifest.json'
SELECTION=ROOT/'outputs'/'review_selection'/'pins.json'
RAW=ROOT/'data'/'raw'
bp=Blueprint('photo_pins',__name__)
lock=threading.RLock()


def records():
    return {r['id']:r for r in json.loads(MANIFEST.read_text(encoding='utf-8'))['records']} if MANIFEST.exists() else {}


def original(row):
    path=Path(row['original']).resolve()
    if not path.is_relative_to(RAW.resolve()) or not path.is_file():
        raise FileNotFoundError('원본 파일을 찾을 수 없습니다. 축소본으로 대체하지 않습니다.')
    return path


def read_pins():
    return json.loads(SELECTION.read_text(encoding='utf-8')) if SELECTION.exists() else []


@bp.get('/api/pins')
def list_pins():
    catalog=records()
    with lock:ids=read_pins()
    return jsonify(records=[{'id':key,'filename':Path(catalog[key]['original']).name,
                              'source':catalog[key]['source'],'split':catalog[key]['split']} for key in ids if key in catalog])


@bp.post('/api/pins')
def update_pin():
    body=request.get_json(silent=True)
    if not isinstance(body,dict) or not isinstance(body.get('id'),str) or type(body.get('pinned')) is not bool:
        return jsonify(error='사진 ID와 고정 여부를 확인하세요.'),400
    catalog=records();key=body['id']
    if key not in catalog:return jsonify(error='등록된 사진이 아닙니다.'),404
    with lock:
        ids=read_pins()
        if body['pinned'] and key not in ids:ids.append(key)
        if not body['pinned']:ids=[item for item in ids if item!=key]
        SELECTION.parent.mkdir(parents=True,exist_ok=True)
        temporary=SELECTION.with_suffix('.tmp')
        temporary.write_text(json.dumps(ids,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(SELECTION)
    return jsonify(id=key,pinned=body['pinned'])


@bp.get('/api/photos/<key>/original')
def download_original(key):
    row=records().get(key)
    if row is None:return jsonify(error='등록된 사진이 아닙니다.'),404
    try:path=original(row)
    except FileNotFoundError as exc:return jsonify(error=str(exc)),404
    return send_file(path,as_attachment=True,download_name=path.name,conditional=True)


@bp.get('/api/pins/originals.zip')
def download_pins():
    catalog=records()
    with lock:ids=read_pins()
    if not ids:return jsonify(error='먼저 사진을 핀 고정하세요.'),400
    selected=[]
    for key in ids:
        if key not in catalog:return jsonify(error='고정한 사진이 목록에서 사라졌습니다.'),409
        try:selected.append((catalog[key],original(catalog[key])))
        except FileNotFoundError as exc:return jsonify(error=str(exc)),404
    buffer=tempfile.SpooledTemporaryFile(max_size=8*1024*1024,mode='w+b')
    try:
        names=set();review=[]
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
            for row,path in selected:
                # Source directories preserve filenames and avoid common collisions.
                source=str(row['source'])
                if source not in ('han','seo','park'):source='photos'
                name=source+'/'+path.name
                if name in names:name=row['id']+'/'+path.name
                names.add(name)
                archive.write(path,arcname=name)
                review.append({'filename':name,'id':row['id'],'original_split':row['split']})
            archive.writestr('review_selection.json',json.dumps({'photos':review,
                'notice':'Selection only. Preserve originals and edit labels separately. Any test/val photo used for retraining is no longer an independent test/val sample.'},ensure_ascii=False,indent=2))
        length=buffer.tell();buffer.seek(0)
        response=send_file(buffer,mimetype='application/zip',as_attachment=True,download_name='pinned_originals.zip')
        response.content_length=length
        response.call_on_close(buffer.close)
        return response
    except Exception:
        buffer.close();raise
