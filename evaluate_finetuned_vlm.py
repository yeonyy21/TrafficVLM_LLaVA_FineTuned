# evaluate_finetuned_vlm.py
# train_helmet_vlm.py로 파인튜닝된 LLaVA-NeXT 모델의 성능을 평가합니다.

import json
import sys
import os
import logging

# 파인튜닝된 모델의 추론 스크립트에서 예측 함수 및 초기화 함수 가져오기
# 추론 스크립트 파일명을 'infer_finetuned_vlm.py'로 가정합니다.
INFERENCE_MODULE_FINETUNED = "infer_finetuned_vlm" 
try:
    inference_module = __import__(INFERENCE_MODULE_FINETUNED)
    # infer_finetuned_vlm.py에 정의된 함수 이름과 일치해야 합니다.
    predict_helmet_fn = inference_module.predict_helmet_with_finetuned_vlm 
    initialize_model_fn = inference_module._initialize_finetuned_model_and_processor
except ImportError as e:
    logging.error(f"'{INFERENCE_MODULE_FINETUNED}.py'에서 함수를 임포트할 수 없습니다: {e}")
    logging.error("스크립트가 올바른 위치에 있는지, 필요한 함수가 정의되어 있는지 확인하세요.")
    sys.exit(1)
except AttributeError as e_attr:
    logging.error(f"'{INFERENCE_MODULE_FINETUNED}.py'에 필요한 함수가 정의되어 있지 않습니다: {e_attr}")
    sys.exit(1)

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    logging.error("scikit-learn 라이브러리가 설치되지 않았습니다.")
    logging.error("설치 명령어: pip install scikit-learn (또는 conda install scikit-learn -c conda-forge)")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s (%(process)d): %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
# 테스트 메타데이터 파일 경로 (make_test_meta.py의 결과 파일)
TEST_META_FILE = "data/test_meta.json" 
# JSON 파일 내 image_path가 상대 경로일 경우의 기준 디렉토리
# image_path가 "data/crops/..." 형태이므로, 이 스크립트가 프로젝트 루트에서 실행된다면 "."
IMAGE_BASE_DIRECTORY_EVAL = "." 
# --- 설정 끝 ---

def evaluate_model_performance():
    logger.info(f"'{TEST_META_FILE}'에서 테스트 메타데이터를 로드합니다...")
    try:
        with open(TEST_META_FILE, "r", encoding="utf-8") as f:
            test_meta = json.load(f)
    except FileNotFoundError:
        logger.error(f"오류: 테스트 메타데이터 파일 '{TEST_META_FILE}'을 찾을 수 없습니다."); return
    except json.JSONDecodeError:
        logger.error(f"오류: 테스트 메타데이터 파일 '{TEST_META_FILE}'의 JSON 형식이 올바르지 않습니다."); return

    y_true_labels = []  # 실제 레이블 (0 또는 1)
    y_pred_labels = []  # 모델 예측 레이블 (0 또는 1)

    if not test_meta: 
        logger.error("오류: 테스트 메타데이터가 비어있습니다."); return

    # 추론을 시작하기 전에 모델과 프로세서를 한 번만 로드
    logger.info("평가를 위해 파인튜닝된 추론 모델 및 프로세서 초기화 시도...")
    try:
        initialize_model_fn() # 추론 스크립트의 초기화 함수 호출
    except Exception as e_init:
        logger.error(f"평가 전 모델/프로세서 초기화 실패: {e_init}. 평가를 중단합니다.", exc_info=True); return
    logger.info("파인튜닝된 추론 모델 및 프로세서 준비 완료.")


    logger.info(f"총 {len(test_meta)}개의 테스트 샘플에 대해 파인튜닝 모델 평가를 시작합니다...")
    for i, item in enumerate(test_meta):
        # 'data/test_meta.json'은 "image_path"와 "label" 키를 가짐
        if not all(k in item for k in ["image_path", "label"]):
            logger.warning(f"{i+1}번째 아이템에 'image_path' 또는 'label' 필드 누락. 건너<0xEB><0x81><0xB5니다.")
            continue

        relative_image_path = item["image_path"] 
        true_label_text = item["label"].lower()  # "yes" 또는 "no"
        
        full_image_path = os.path.join(IMAGE_BASE_DIRECTORY_EVAL, relative_image_path) if IMAGE_BASE_DIRECTORY_EVAL != "." else relative_image_path
        
        if not os.path.exists(full_image_path):
            logger.warning(f"이미지 파일 '{full_image_path}' 누락 (원본 경로: '{relative_image_path}'). {i+1}번째 아이템 건너<0xEB><0x81><0xB5니다.")
            continue
            
        # logger.info(f"[{i+1}/{len(test_meta)}] 이미지 '{full_image_path}' 예측 중...") # 로그가 너무 많을 수 있어 주석 처리
        try:
            # infer_finetuned_vlm.py의 예측 함수 사용 (기본 질문 사용)
            prediction_text = predict_helmet_fn(full_image_path) 
            # logger.info(f"  모델 답변: '{prediction_text}'") # 로그가 너무 많을 수 있어 주석 처리
        except Exception as e_pred:
            logger.error(f"'{full_image_path}' 예측 중 예외 발생: {e_pred}. 이 샘플 제외.", exc_info=True)
            continue # 오류 발생 시 해당 샘플은 평가에서 제외

        # 실제 레이블을 0 또는 1로 변환
        if true_label_text == "yes": 
            y_true_labels.append(1)
        elif true_label_text == "no": 
            y_true_labels.append(0)
        else: 
            logger.warning(f"알 수 없는 정답 레이블 '{item['label']}'. {i+1}번째 아이템 건너<0xEB><0x81><0xB5니다.")
            continue # 알 수 없는 레이블은 평가에서 제외
        
        # 파인튜닝된 모델은 "예, 착용했습니다." 또는 "아니요, 착용하지 않았습니다." 형태로 답변할 것으로 기대
        # 따라서 단순 "예" 포함 여부로 판단 가능
        y_pred_labels.append(1 if "예" in prediction_text else 0)

    if not y_true_labels or not y_pred_labels: 
        logger.error("평가를 위한 유효한 예측 결과가 없습니다. 입력 데이터나 예측 과정을 확인하세요."); return

    if len(y_true_labels) != len(y_pred_labels): 
        logger.error(f"오류: 실제 레이블({len(y_true_labels)}개)과 예측 레이블({len(y_pred_labels)}개)의 수가 일치하지 않습니다."); return

    logger.info(f"\n최종 평가에 사용된 샘플 수: {len(y_true_labels)}")
    logger.info("\n--- 파인튜닝된 LLaVA-NeXT 모델 평가 결과 ---")
    target_names = ["no (미착용)", "yes (착용)"] # 0: no (미착용), 1: yes (착용)
    try:
        report = classification_report(y_true_labels, y_pred_labels, target_names=target_names, zero_division=0)
        conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=[0,1]) # labels 명시
        
        print("\nClassification Report (Fine-tuned LLaVA-NeXT):")
        print(report)
        print("\nConfusion Matrix ( [[TN, FP], [FN, TP]] ) (Fine-tuned LLaVA-NeXT):")
        print(conf_matrix)
    except ValueError as ve: 
        logger.error(f"sklearn.metrics에서 오류 발생: {ve}.")
        logger.info(f"실제 레이블 (y_true, {len(y_true_labels)}개, 일부): {y_true_labels[:20]}...")
        logger.info(f"예측 레이블 (y_pred, {len(y_pred_labels)}개, 일부): {y_pred_labels[:20]}...")


if __name__ == "__main__":
    evaluate_model_performance()
