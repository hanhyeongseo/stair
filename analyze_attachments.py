"""Inference-only gallery for the four user supplied experiment photos."""
import hashlib
import json
import shutil
from pathlib import Path

import cv2

from mask2former_detector import ROOT, StairsDetector
from prepare import load_image

IDS = ['4443997199372608069','13936205132366213117','10553638713044734614','14406477466387511729']
PREFIX = 'ai-chat-custom-attachment-temp-file-173b3755-4958-4150-9ff6-5aae23591591-'


def main():
    output = ROOT/'outputs'/'attachment_experiment'
    output.mkdir(parents=True,exist_ok=True)
    net = StairsDetector()
    rows = [];cards = []
    for number,ident in enumerate(IDS,1):
        source = Path('C:/Users/KDT76/AppData/Local/Temp')/(PREFIX+ident+'.png')
        shutil.copy2(source,output/f'{number}_input.png')
        views,report = net.infer(load_image(source),mode='both')
        figures = []
        for key,title in [('original','원본'),('enhanced','밝기 보정'),
                          ('original_overlay','원본 · 계단 전체'),('enhanced_overlay','보정 · 계단 전체'),
                          ('original_hazard_overlay','원본 · 위험 단차 후보'),('enhanced_hazard_overlay','보정 · 위험 단차 후보')]:
            image = views.get(key)
            if image is None:continue
            filename = f'{number}_{key}.jpg'
            ok,encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise RuntimeError('JPEG encode failed')
            encoded.tofile(str(output/filename))
            figures.append(f'<figure><figcaption>{title}</figcaption><a href="{filename}" target="_blank"><img loading="lazy" src="{filename}" alt="{title}"></a></figure>')
        for view in ('original','enhanced'):
            ok,encoded = cv2.imencode('.png',views[view+'_hazard_transparent'])
            if not ok:raise RuntimeError('PNG encode failed')
            encoded.tofile(str(output/f'{number}_{view}_hazard.png'))
        row = {'image_number':number,'source_file':source.name,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
               'used_for_training':False,'ground_truth_available':False,'result':report}
        rows.append(row)
        findings = []
        for view,label in [('original','원본'),('enhanced','보정')]:
            value = report['views'][view]
            findings.append(label+': '+('계단 영역 검출' if value['status']=='stairs_region' else '계단 미검출')+
                            ' / '+('위험 단차 후보 있음' if value.get('hazard',{}).get('status')=='candidate' else '위험 단차 미검출'))
        cards.append(f'<article id="photo-{number}"><h2>사진 {number}</h2><p>'+ ' · '.join(findings)+
                     '</p><div class="grid">'+''.join(figures)+f'</div><p><a href="{number}_original_hazard.png" download>원본 입력 단차 마스크 PNG</a> · <a href="{number}_enhanced_hazard.png" download>보정 입력 단차 마스크 PNG</a></p></article>')
        print(json.dumps({'number':number,'views':report['views']},ensure_ascii=False),flush=True)
    report = {'experiment':'four_user_attachments','training_performed':False,'weights_changed':False,
              'hazard_checkpoint_sha256':hashlib.sha256((ROOT/'models'/'mask2former_hazard_head.pt').read_bytes()).hexdigest(),
              'accuracy_measured':False,'reason':'No ground-truth labels for these four images. Detection does not establish correctness.',
              'records':rows}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>첨부 사진 4장 · 실험 결과</title>
<style>body{font:17px/1.7 sans-serif;background:#12202c;color:#edf6fc;margin:0;padding:24px}main{max-width:1500px;margin:auto}a{color:#8bddff}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0;background:#1c2e3c;padding:12px}img{width:100%;max-height:650px;object-fit:contain;background:#0a1119}figcaption{margin-bottom:8px}article{border-top:1px solid #486072;margin-top:40px;padding-top:20px}.note{background:#2a3842;padding:16px}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style><main>
<a href="/">← 대시보드</a><h1>첨부 사진 4장 분석 결과</h1><p class="note">현재 학습된 모델로 추론만 수행했습니다. 이번 4장은 학습에 추가하지 않았습니다.<br>파랑: 계단 전체 · 빨강: 위험 단차 후보 · 주황: 모서리 후보.<br>정답 라벨이 없어 정확도는 계산하지 않았습니다. 검출된 영역이 맞는지는 사진과 비교해 확인하세요. 사진을 클릭하면 크게 볼 수 있습니다.</p><p><a href="#photo-1">사진 1</a> · <a href="#photo-2">사진 2</a> · <a href="#photo-3">사진 3</a> · <a href="#photo-4">사진 4</a> · <a href="report.json">분석 JSON</a></p>'''
    (output/'index.html').write_text(page+''.join(cards)+'</main></html>',encoding='utf-8')


if __name__=='__main__':main()
