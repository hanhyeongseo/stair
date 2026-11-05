# 팀원용 실행 안내

이 저장소는 `night_v` 프로젝트의 코드만 담습니다. 다른 수업 코드, 개인 사진·라벨,
로컬 경로가 들어 있는 manifest, 가중치, 가상환경, 분석 결과는 Git에 포함하지 않습니다.
GitHub에서 코드를 보는 것과 웹 대시보드를 실행하는 것은 별개입니다.

## 설치

Windows PowerShell, Python 3.11 권장. 아래 명령은 저장소 루트에서 실행합니다.

```powershell
git clone https://github.com/hanhyeongseo/stair.git
cd stair
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

기본 설치 후 CPU로 실행할 수 있습니다. CUDA GPU를 사용할 팀원은 PyTorch의 공식 설치
안내에 따라 GPU용 torch/torchvision을 설치하세요. 개발 환경은 Python 3.11,
torch 2.11.0+cu128 / torchvision 0.26.0+cu128, RTX 5060 Laptop GPU였습니다.

## 가중치 준비

가중치는 **이 Git 저장소에 포함되어 있지 않습니다.** 팀에서 별도 공유받아 다음 위치에
넣어야 실제 분석이 됩니다. 개인 사진과 학습 가중치를 공유할 권한은 따로 확인하세요.

```text
models/
  mask2former-swin-large-ade-semantic/
    config.json
    preprocessor_config.json
    model.safetensors
  mask2former_hazard_head.pt
  hazard_best.pt                 # 기존 YOLO 화면을 사용할 때
```

Mask2Former 원본은 다음 공식 배포본입니다. 약 866MB입니다.
https://huggingface.co/facebook/mask2former-swin-large-ade-semantic

`mask2former_hazard_head.pt`는 우리 팀 라벨로 학습한 파일이므로 팀에서 따로 받아야 합니다.
이 파일이 없으면 파란 계단 전체 영역만 표시되고, 빨간 단차 후보는 표시되지 않습니다.
`hazard_best.pt`가 없으면 `/yolo`의 기존 모델 추론은 사용할 수 없습니다.

## 실행

```powershell
& .\.venv\Scripts\python.exe .\app.py
```

브라우저: http://127.0.0.1:5001/

사진 목록이 비어 있어도 새 사진·영상 업로드로 분석할 수 있습니다. 데이터셋과 정답 비교
목록은 별도 공유받은 `data/manifest.json` 및 이미지가 있어야 표시됩니다. manifest에는
생성한 PC의 절대 경로가 들어가므로 다른 PC에서 그대로 쓰지 말고 데이터를 다시 준비하세요.
`prepare.py`의 SOURCES는 개발 PC 경로입니다. 팀원 PC의 실제 자료 위치에 맞춰 수정 후 실행합니다.

분석 결과 모아보기 페이지는 로컬에서 평가를 실행해 `outputs/`를 만든 뒤 사용할 수 있습니다.
`analyze_attachments.py`도 개발 PC의 임시 첨부 파일 경로를 사용하므로, 팀원은 웹 업로드를 사용하세요.

## 현재 모델의 한계

파랑은 계단 전체, 빨강은 위험 단차 후보, 주황은 모서리 후보입니다. 단의 순서나 상하행을
명시적으로 판단하지 않습니다. 기존 시험 자료 56장 중 밝기 보정 입력의 위치 일치는 22장
(mask IoU ≥ 0.5)이며, 잘못된 후보도 33장입니다. 실제 보행 안전을 검증한 제품이 아닙니다.

README의 `C:\Project\AI_Source` 및 `night_v/` 접두사가 있는 명령은 개발 PC의 기존 작업
기록입니다. 이 저장소를 단독으로 받은 팀원은 위의 저장소 루트 기준 명령을 사용하세요.

## 테스트

데이터셋 없이도 실행 가능한 테스트:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_hazard_head.py
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_mask2former.py
```

`test_project.py`의 전체 데이터 검사와 일부 통합 테스트는 개발용 데이터·가중치가 필요합니다.
