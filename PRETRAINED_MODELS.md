# 사전 학습 모델 검증

`compare_pretrained.py`는 실제 체크포인트의 클래스부터 검증합니다.
계단 클래스가 없으면 비교와 학습을 중단하고 제외 보고서를 만듭니다.
정상적인 계단 세그멘테이션 가중치일 때만 원본·밝기 보정·기존 모델 비교를 수행합니다.
전체 계단과 위험 단차는 서로 다른 대상이므로 겹침 수치를 탐지 정확도로 해석하지 않습니다.

## 현재 결과

- ArchiVision: 다운로드 성공, 실제 클래스는 window/door/column/wall/beam. 계단 클래스 없음. 제외.
- StairNet-DepthOut: 공식 GitHub 파일 목록에 가중치 없음. Baidu 배포 페이지는 추출 코드 입력 화면까지 접근. 가중치 미확보, 실행 및 성능 검증 미완료.
- 기존 hazard_best.pt는 교체하거나 재학습하지 않음.

보고서: `outputs/pretrained_comparison/index.html`, `report.json`.
Flask를 다시 시작한 뒤 `/comparison/`에서도 확인할 수 있습니다.

## 실행

```powershell
& .\night_v\.venv\Scripts\python.exe .\night_v\compare_pretrained.py
```

기존 가상환경이 참조하던 Python311 원본이 없어 실행되지 않았습니다.
`night_v/.runtime`에 Python 3.11.9를 설치하고 **night_v 가상환경만** 복구했습니다.
IDE의 루트 인터프리터 설정은 변경하지 않았습니다.
