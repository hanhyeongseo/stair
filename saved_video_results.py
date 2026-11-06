"""Reuse genuine saved inference only for matching bytes/model/input settings."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def matching_saved_result(upload,engine,input_view):
    if engine!='mask2former' or input_view!='enhanced':return None
    folder=ROOT/'outputs/dark_demo_orange'
    report_path=folder/'report.json';video=folder/'source_comparison_10fps.mp4'
    head=ROOT/'models/mask2former_hazard_head.pt'
    details_path=folder/'source_comparison_10fps.json'
    if not all(p.is_file() for p in (report_path,video,head,details_path)):return None
    report=json.loads(report_path.read_text(encoding='utf-8'))
    details=json.loads(details_path.read_text(encoding='utf-8'))
    if sha256(head)!=report.get('checkpoint_sha256') or sha256(upload)!=report.get('source_sha256'):return None
    return {'path':video,'reused_analysis':True,'source_start_seconds':report['start_seconds'],
            'seconds':details['duration_seconds'],'frames':details['frames'],
            'render_fps':details['fps'],'model_sha256':report['checkpoint_sha256'],
            'result_notice':'저장된 동일 파일 분석 결과 · 9초부터 끝까지'}
