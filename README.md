# TrafficVLM_LLaVA_FineTuned
2025학년도 경찰대학 김경종 교수님의 '경찰활동과 캡스톤디자인' 수업에서 진행하였던 캡스톤 프로젝트 코드입니다.

# LLaVA-NeXT 기반 오토바이 안전모 착용 여부 자동 판독 시스템

## 1. 프로젝트 개요

본 프로젝트는 최신 대규모 언어·시각 모델(Vision–Language Model, VLM)인 **LLaVA-NeXT** (`llava-hf/llava-v1.6-mistral-7b-hf`)를 활용하여, CCTV 등에서 확보된 **사진**에서 오토바이 주행자 이미지의 안전모 착용 여부를 자동으로 판독하는 시스템을 개발합니다.

- 사용자가 “이 오토바이 주행자가 안전모를 착용했습니까?”와 같은 자연어 질문을 입력하면,
시스템은 이미지의 시각 정보를 바탕으로 “예, 착용했다.” 또는 “아니요, 착용하지 않았다.” 형태의 답변을 생성합니다.
- QLoRA(Quantized Low-Rank Adaptation) 기법과 PyTorch Lightning 프레임워크를 사용해 모델을 효율적으로 파인튜닝했으며,
파인튜닝된 모델은 **95% 정확도**를 달성해, 순수 모델(약 88%) 대비 우수한 성능을 보였습니다.

## 2. 연구 동기

이륜차 및 개인형 이동장치(PM) 운전자의 안전모 착용은 교통사고 시 심각한 부상을 예방하는 핵심 요소입니다.

그러나 실제 착용률은 낮고, 기존의 수동 단속 방식은 인력·효율성 면에서 한계가 있습니다.

본 연구는 이러한 문제를 해결하기 위한 AI 기반 자동 판독 시스템을 제안합니다.

## 3. 주요 기능 및 사용 기술

- **안전모 착용 여부 판독**
크롭된 주행자 이미지와 자연어 질문을 입력받아 “예/아니요” 스타일로 판독 결과 반환
- **기반 모델**: LLaVA-NeXT (`llava-hf/llava-v1.6-mistral-7b-hf`)
- **파인튜닝 기법**: QLoRA (4비트 양자화 + LoRA)
- **학습 프레임워크**: PyTorch Lightning
- **데이터셋**
    - AI Hub ‘교통법규 위반 상황 데이터’
    - Kaggle ‘Helmet Detection’ 이미지 크롭 및 Q&A 쌍으로 구성한 커스텀 데이터

## 4. 시스템 아키텍처 및 파이프라인

1. **데이터 전처리** (`preprocess_helmet_data.py`, `make_test_meta.py`)
    - 원본 이미지 및 어노테이션(JSON/XML)에서 주행자 영역 크롭
    - `(크롭 이미지 경로, 고정 질문, 정답 텍스트)` 형식의
        - `train_helmet_mixed.json` (학습/검증용)
        - `test_meta.json` (테스트용, `"label": "yes"/"no"`) 메타데이터 생성
2. **모델 파인튜닝** (`train_helmet_vlm.py`)
    - `train_helmet_mixed.json` 및 크롭 이미지를 사용해 LLaVA-NeXT을 QLoRA 방식으로 파인튜닝
    - PyTorch Lightning으로 학습 관리
    - 학습된 LoRA 어댑터와 프로세서 설정 저장
3. **추론** (`infer_finetuned_vlm.py`, `infer_vanilla_vlm.py`)
    - 파인튜닝된 모델 또는 순수(Vanilla) 모델 로드
    - 입력 이미지와 질문에 대한 답변 생성
4. **성능 평가** (`evaluate_finetuned_vlm.py`, `evaluate_vanilla_vlm.py`)
    - `test_meta.json`을 사용해 정확도, 정밀도, 재현율 계산 및 오차 행렬 생성

## 5. 파일 구조

```bash
VLM_and_YOLO/
├── data/
│ ├── annotations/ # 원본 어노테이션 파일 (JSON, XML)
│ ├── images/ # 원본 이미지 파일
│ ├── crops/ # 크롭된 주행자 이미지
│ ├── train_helmet_mixed.json # 학습/검증용 메타데이터
│ └── test_meta.json # 테스트용 메타데이터
├── outputs/
│ └── llava_next_helmet_finetuned/
│ └── final_checkpoint/ # 모델·프로세서·어댑터 파일
├── preprocess_helmet_data.py # 데이터 전처리 스크립트
├── make_test_meta.py # 테스트 메타데이터 생성 스크립트
├── train_helmet_vlm.py # 파인튜닝 스크립트
├── infer_finetuned_vlm.py # 파인튜닝 모델 추론 스크립트
├── evaluate_finetuned_vlm.py # 파인튜닝 모델 평가 스크립트
├── infer_vanilla_vlm.py # 순수 모델 추론 스크립트
├── evaluate_vanilla_vlm.py # 순수 모델 평가 스크립트
└── [README.md](http://readme.md/) # 본 파일
```

## 6. 환경 설정

Python 3.9 기반 Conda 가상환경에서 개발·테스트되었습니다.

1. **Conda 환경 생성 및 활성화**
    
    ```bash
    conda create -n llava_final_env python=3.9 -y
    conda activate llava_final_env
    
    ```
    

PyTorch (CUDA 12.1) 설치

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install protobuf==3.20.3
pip install transformers==4.38.2       # 또는 호환되는 최신 버전
pip install datasets==2.20.0
pip install peft==0.11.1
pip install accelerate==0.31.0
pip install bitsandbytes==0.43.1
pip install tokenizers==0.15.2
pip install sentencepiece==0.2.0
pip install pytorch-lightning==2.3.0
pip install Pillow==10.3.0
pip install nltk
pip install scikit-learn
pip install numpy==2.0.0
```

# (선택) tensorboard 설치: pip install tensorboard

참고: transformers, tokenizers, sentencepiece 버전 호환성을 확인하세요.

1. 스크립트 실행 방법
사전 준비

data/annotations/ 및 data/images/ 디렉토리에 원본 어노테이션과 이미지 파일을 준비

각 스크립트 상단의 경로 상수(IMAGE_ROOT, ANN_ROOT, DATA_JSON_PATH, OUTPUT_MODEL_DIR 등)를 실제 환경에 맞게 설정

1단계: 데이터 전처리

```bash
python preprocess_helmet_data.py
python make_test_meta.py
```

2단계: 모델 파인튜닝

```bash
python train_helmet_vlm.py
```

3단계: 파인튜닝된 모델 추론

```bash
python infer_finetuned_vlm.py \
--image_path path/to/crop.jpg \
--question "이 주행자가 안전모를 착용했습니까?"
```

4단계: 파인튜닝 모델 성능 평가

```bash
python evaluate_finetuned_vlm.py
```

5단계: 순수(Vanilla) 모델 추론 (비교용)

```bash
python infer_vanilla_vlm.py \
--image_path path/to/crop.jpg \
--question "이 주행자가 안전모를 착용했습니까?"
```

6단계: 순수(Vanilla) 모델 성능 평가 (비교용)

```bash
python evaluate_vanilla_vlm.py
```

8. 주요 결과 요약
지표	파인튜닝 전 (Vanilla)	파인튜닝 후
정확도	88%	95%
미착용 F1-score	0.86	0.94
착용 F1-score	0.90	0.96

재현율(착용 사례): FN 25건 → 3건 (98% 재현율)


