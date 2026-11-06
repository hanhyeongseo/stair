# 팀원에게 전달할 파일

코드는 GitHub, 가중치는 팀 공유 드라이브로 나눠 전달합니다.
가상환경은 각자 설치하며 `.venv`, `.runtime`, `.cache`를 복사하지 않습니다.

## 1. 코드 업데이트

저장소를 이미 받은 팀원은 저장소 루트에서 실행합니다.

```powershell
git pull --ff-only
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

이번 업데이트에는 영상 재생을 위한 `imageio-ffmpeg`가 추가됐으므로 패키지 업데이트도 필요합니다.
처음 설치하는 팀원은 [TEAM_SETUP.md](TEAM_SETUP.md)를 참고하세요.

## 2. 모델 묶음 전달 — 분석 실행에 필요

개발 PC의 `C:\Project\AI_Source\night_v`에서 아래 스크립트를 실행하면 필요한 모델만
골라 `outputs/sharing/`에 ZIP을 만듭니다. 원본 가중치를 수정하지 않습니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_models.ps1
```

`stair-models-날짜시간.zip`을 Google Drive, OneDrive, 학교 공유 드라이브 등 팀이 사용하는
공간에 올리고 링크를 전달하세요. 공개 Git 커밋에는 넣지 않습니다. 묶음은 약 873MB이며
압축 크기는 달라집니다. ZIP 옆 `.sha256.txt`도 함께 전달하면 전송 확인에 쓸 수 있습니다.

받는 팀원은 ZIP의 **models 폴더를 app.py 옆에** 놓습니다.

```text
stair/
  app.py
  MODEL_FILES.json                         # 파일별 해시·용도·패키지 생성 당시 코드 커밋
  models/
    mask2former-swin-large-ade-semantic/
      config.json
      preprocessor_config.json
      model.safetensors                   # 계단 전체 인식, 약 866MB
    mask2former_hazard_head.pt             # 팀 학습 위험 단차 모델, 약 0.68MB
    hazard_best.pt                        # 이전 YOLO 비교용, 약 6MB
```

ZIP 안에 `stair/stair/models`처럼 폴더가 중복되지 않도록 압축을 풀 때 구조를 확인하세요.
Windows에서는 ZIP을 임시 폴더에 풀고 그 안의 `models`를 저장소 루트로 옮기면 됩니다.

전송한 ZIP 확인:

```powershell
Get-FileHash .\stair-models-날짜시간.zip -Algorithm SHA256
```

출력한 해시가 함께 받은 `.sha256.txt`와 같아야 합니다.
Mask2Former 원본은 [공식 배포 페이지](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic/tree/main)에서
세 파일을 직접 받아도 됩니다. 팀 학습 `mask2former_hazard_head.pt`는 팀에서 별도로 받아야 합니다.
실험에서 제외한 `archivision.pt`, 초기 `yolo11n-seg.pt`, 학습 중간 체크포인트는 실행용 묶음에 포함하지 않습니다.

## 3. 사진·라벨 — 새 파일 업로드만 사용할 때는 불필요

팀원은 가중치만 있으면 자기 사진·영상을 웹에 업로드해 분석할 수 있습니다.
기존 사진 목록과 정답 비교, 재학습을 하려는 팀원에게만 원본 사진과 해당 JSON/TXT 라벨을
같은 상대 폴더 구조로 전달하세요. `Down`, `Up`, `park` 자료를 구분하고 파일명을 유지합니다.

`data/manifest.json`과 `data/hazard.yaml`에는 개발 PC의 절대 경로가 들어 있습니다.
그 파일만 복사하면 다른 PC에서는 경로가 맞지 않습니다. `prepare.py`의 SOURCES를 팀원 PC의
원본 자료 위치로 바꾸고 다음 명령으로 생성합니다.

```powershell
& .\.venv\Scripts\python.exe .\prepare.py --import-sources
```

생성 후 train/val/test 분리와 라벨 겹침을 확인하세요. 영상에서 추출한 같은 계단의 프레임을
서로 다른 학습·시험 묶음에 나누지 않습니다.

## 4. 발표·결과 확인 — 선택

완성된 H.264 결과 MP4는 파일 자체를 공유하면 모델 설치 없이 재생할 수 있습니다.
갤러리도 똑같이 보려면 필요한 `outputs` 하위 폴더만 별도로 전달하세요.

| 폴더 | 웹 주소 | 내용 |
|---|---|---|
| `outputs/web` | `/videos` | 처리한 영상 |
| `outputs/hazard_head_evaluation` | `/mask2former/hazard-results/` | 위험 단차 정답 비교 |
| `outputs/mask2former_evaluation` | `/mask2former/results/` | 계단 전체 결과 |
| `outputs/attachment_experiment` | `/mask2former/experiments/` | 첨부 사진 4장 실험 |

발표용 전체 프레임 분석은 아직 구현하지 않았습니다. 현재 Mask2Former 영상은 최대 1fps 분석입니다.

## 팀 채팅에 보낼 문구

> 코드는 https://github.com/hanhyeongseo/stair 에서 받아주세요.
> 기존 설치자는 git pull 후 requirements.txt를 다시 설치해주세요.
> 별도 공유한 모델 ZIP을 풀어 models 폴더가 app.py 옆에 오도록 놓으면 됩니다.
> 그다음 `.venv\Scripts\python.exe app.py` 실행 후 http://127.0.0.1:5001 에 접속하세요.
> 기존 학습 사진 없이도 새 사진·영상을 업로드해 분석할 수 있습니다.
