# evaluate_vanilla_vlm.py
# 파인튜닝되지 않은 순수 LLaVA-NeXT 모델의 성능을 평가합니다.
# infer_vanilla_vlm.py의 예측 함수와 개선된 답변 분류 로직을 사용합니다.

import json
import sys
import os
import logging

# 순수 LLaVA 모델 추론 스크립트에서 함수 가져오기
# 이전 답변(#54)에서 생성한 추론 스크립트 파일명을 "infer_vanilla_vlm.py"로 가정합니다.
INFERENCE_MODULE_VANILLA = "infer_vanilla_vlm" 
try:
    inference_script_module = __import__(INFERENCE_MODULE_VANILLA)
    # infer_vanilla_vlm.py에 정의된 함수 이름과 일치해야 합니다.
    predict_helmet_vanilla_fn = inference_script_module.predict_helmet_with_vanilla_llava
    initialize_vanilla_model_fn = inference_script_module._initialize_vanilla_model_and_processor
except ImportError as e:
    logging.error(f"'{INFERENCE_MODULE_VANILLA}.py'에서 함수를 임포트할 수 없습니다: {e}")
    logging.error("스크립트가 올바른 위치에 있는지, 필요한 함수가 정의되어 있는지 확인하세요.")
    sys.exit(1)
except AttributeError as e_attr:
    logging.error(f"'{INFERENCE_MODULE_VANILLA}.py'에 필요한 함수가 정의되어 있지 않습니다: {e_attr}")
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

def classify_vanilla_llava_answer(prediction_text: str) -> str:
    """
    순수 LLaVA 모델의 다양한 답변을 "착용", "미착용", "확인 불가"로 분류합니다.
    반환값: "착용", "미착용", "확인 불가" 중 하나
    """
    if not prediction_text or not isinstance(prediction_text, str):
        logger.debug(f"  분류 입력 답변 유효하지 않음: {prediction_text}")
        return "확인 불가"

    # 모든 종류의 공백(연속 공백, 탭, 줄 바꿈 등) 제거하고 붙임
    processed_text = "".join(prediction_text.split())
    # logger.debug(f"  분류용 전처리된 답변: '{processed_text}'") # 상세 로그 필요시 주석 해제

    # 키워드 리스트 (실제 모델 답변 패턴을 보고 지속적으로 개선 필요)
    negative_indicators = [
        "아니요", "미착용", "착용하지않았습니다", "착용하고있지않습니다", "착용하고있지않은것으로", 
        "쓰고있지않습니다", "헬멧이없습니다", "안전모가없습니다", "안전모를착용하지않", "헬멧을쓰지않"
    ]
    positive_indicators = [
        "예", "착용했습니다", "착용하고있습니다", "착용하고있는것으로", "착용한것으로", "쓰고있습니다",
        "안전모를착용하고", "안전모를쓰고", "헬멧을착용", "헬멧을쓰고"
    ]
    uncertain_indicators = [
        "확인하기어렵습니다", "확인하기어려운", "확인이어렵", "확인할수없습니다", "확인할수없는",
        "흐릿하여", "흐릿하고", "불분명", "알수없", "판단이어렵", "자세한내용을파악하기어렵",
        "정보를제공하지않", "부분만보여", "아무것도없으므로답변할수없"
    ]

    # 1. 명확한 부정 표현 확인 (우선순위 높게)
    for neg_kw in negative_indicators:
        if neg_kw in processed_text:
            # logger.info(f"  답변 분류: '미착용' (사유: 부정 키워드 '{neg_kw}' 발견)")
            return "미착용"

    # 2. 명확한 긍정 표현 확인
    for pos_kw in positive_indicators:
        if pos_kw in processed_text:
            # 긍정 키워드가 있더라도, 동시에 판단 유보/불가 표현이 있는지 확인
            if any(unc_kw in processed_text for unc_kw in uncertain_indicators):
                # logger.info(f"  답변 분류: '확인 불가' (사유: 긍정 키워드 '{pos_kw}'와 불확실 키워드 동시 발견)")
                return "확인 불가"
            # logger.info(f"  답변 분류: '착용' (사유: 긍정 키워드 '{pos_kw}' 발견)")
            return "착용"
            
    # 3. 판단 유보/불가 표현 확인
    for unc_kw in uncertain_indicators:
        if unc_kw in processed_text:
            # logger.info(f"  답변 분류: '확인 불가' (사유: 불확실 키워드 '{unc_kw}' 발견)")
            return "확인 불가"
            
    logger.info(f"  답변 분류: '확인 불가' (명확한 긍정/부정/불확실 지표 없음 - 원본: '{prediction_text}')")
    return "확인 불가"


def evaluate_vanilla_model_performance():
    logger.info(f"'{TEST_META_FILE}'에서 테스트 메타데이터 로드 (VANILLA 모델 평가용)...")
    try:
        with open(TEST_META_FILE, "r", encoding="utf-8") as f: test_meta = json.load(f)
    except FileNotFoundError: logger.error(f"오류: 테스트 메타데이터 파일 '{TEST_META_FILE}'을 찾을 수 없습니다."); return
    except json.JSONDecodeError: logger.error(f"오류: 테스트 메타데이터 파일 '{TEST_META_FILE}'의 JSON 형식이 올바르지 않습니다."); return

    y_true_labels = []  # 실제 레이블 (0 또는 1)
    y_pred_labels = []  # 모델 예측 레이블 (0 또는 1)
    uncertain_predictions_count = 0 # "확인 불가"로 예측된 경우 카운트

    if not test_meta: 
        logger.error("오류: 테스트 메타데이터가 비어있습니다."); return

    logger.info("평가를 위해 VANILLA 추론 모델 및 프로세서 초기화 시도...")
    try:
        initialize_vanilla_model_fn() # Vanilla 모델/프로세서 초기화
    except Exception as e_init:
        logger.error(f"VANILLA 모델/프로세서 초기화 실패: {e_init}. 평가를 중단합니다.", exc_info=True); return
    logger.info("VANILLA 추론 모델 및 프로세서 준비 완료.")

    logger.info(f"총 {len(test_meta)}개의 테스트 샘플에 대해 VANILLA 모델 평가 시작...")
    for i, item in enumerate(test_meta):
        # 'data/test_meta.json'은 "image_path"와 "label" 키를 가짐
        if not all(k in item for k in ["image_path", "label"]):
            logger.warning(f"{i+1}번째 아이템에 'image_path' 또는 'label' 필드 누락. 건너<0xEB><0x81><0xB5니다.")
            continue
        
        relative_image_path = item["image_path"] 
        true_label_text = item["label"].lower()  # "yes" 또는 "no"
        
        # image_path가 "data/crops/..." 와 같은 전체 상대 경로로 되어 있다고 가정하고,
        # IMAGE_BASE_DIRECTORY_EVAL이 "."이면 full_image_path는 relative_image_path와 동일
        full_image_path = os.path.join(IMAGE_BASE_DIRECTORY_EVAL, relative_image_path) if IMAGE_BASE_DIRECTORY_EVAL != "." else relative_image_path
        
        if not os.path.exists(full_image_path):
            logger.warning(f"이미지 파일 '{full_image_path}' 누락 (원본 경로: '{relative_image_path}'). {i+1}번째 아이템 건너<0xEB><0x81><0xB5니다.")
            continue
            
        # 상세 예측 로그는 너무 많으므로, 필요시 주석 해제
        # logger.info(f"[{i+1}/{len(test_meta)}] 이미지 '{full_image_path}' (VANILLA) 예측 중...")
        try:
            # infer_vanilla_vlm.py의 예측 함수 사용 (내부적으로 제한된 프롬프트 사용)
            prediction_text = predict_helmet_vanilla_fn(full_image_path) 
            # logger.info(f"  VANILLA 모델 답변: '{prediction_text}'") 
        except Exception as e_pred:
            logger.error(f"'{full_image_path}' (VANILLA) 예측 중 예외 발생: {e_pred}. 이 샘플 제외.", exc_info=True)
            continue # 오류 발생 시 해당 샘플은 평가에서 제외

        # 개선된 답변 분류 함수 호출
        classified_prediction = classify_vanilla_llava_answer(prediction_text)

        # 실제 레이블을 0 또는 1로 변환
        current_true_label = -1 # 유효하지 않은 레이블 초기화
        if true_label_text == "yes": 
            current_true_label = 1
        elif true_label_text == "no": 
            current_true_label = 0
        else: 
            logger.warning(f"알 수 없는 정답 레이블 '{item['label']}'. {i+1}번째 아이템 평가에서 제외.")
            continue # 알 수 없는 레이블은 평가에서 제외
        
        # 예측 레이블 변환 및 "확인 불가" 처리
        if classified_prediction == "착용":
            y_true_labels.append(current_true_label)
            y_pred_labels.append(1)
        elif classified_prediction == "미착용":
            y_true_labels.append(current_true_label)
            y_pred_labels.append(0)
        elif classified_prediction == "확인 불가":
            uncertain_predictions_count += 1
            logger.info(f"  모델이 '{full_image_path}'에 대해 '확인 불가'로 답변 (원본 답변: '{prediction_text}'). 평가지표 계산에서는 제외하거나 특정 값으로 편향시킬 수 있음. 현재는 제외.")
            # "확인 불가"를 평가 지표 계산에 포함시키지 않음.
            # 만약 특정 값으로 편향시키려면 (예: 미착용(0)으로 간주):
            # y_true_labels.append(current_true_label)
            # y_pred_labels.append(0) 
            pass # 현재는 아무것도 추가 안 함 (평가 샘플 수에서 빠짐)
        

    if not y_true_labels or not y_pred_labels: 
        logger.error(f"평가를 위한 유효한 (확정적) 예측 결과가 없습니다. '확인 불가'로 분류된 예측 수: {uncertain_predictions_count}")
        return

    # y_true_labels와 y_pred_labels의 길이가 다를 수 있으므로 (확인 불가 제외 시), 이 부분은 불필요하거나 수정 필요
    # if len(y_true_labels) != len(y_pred_labels): 
    #     logger.error(f"오류: 실제 레이블({len(y_true_labels)}개)과 예측 레이블({len(y_pred_labels)}개)의 수가 일치하지 않습니다."); return

    logger.info(f"\n'확인 불가'로 분류된 예측 수: {uncertain_predictions_count}")
    logger.info(f"최종 이진 분류 평가에 사용된 샘플 수: {len(y_true_labels)}") # y_true_labels와 y_pred_labels 길이는 같아야 함
    
    if len(y_true_labels) > 0 : # 평가할 샘플이 있을 경우에만 리포트 생성
        logger.info("\n--- VANILLA LLaVA-NeXT 평가 결과 (개선된 분류 로직 적용) ---")
        target_names = ["no (미착용)", "yes (착용)"] # 0: no (미착용), 1: yes (착용)
        try:
            report = classification_report(y_true_labels, y_pred_labels, target_names=target_names, zero_division=0)
            # labels=[0,1]을 추가하여 항상 두 클래스에 대한 매트릭스를 생성하도록 강제
            conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=[0,1]) 
            
            print("\nClassification Report (VANILLA LLaVA-NeXT - Enhanced Parsing):")
            print(report)
            print("\nConfusion Matrix ( [[TN, FP], [FN, TP]] ) (VANILLA LLaVA-NeXT - Enhanced Parsing):")
            print(conf_matrix)
        except ValueError as ve: 
            logger.error(f"sklearn.metrics에서 오류 발생: {ve}. (y_true 또는 y_pred에 유효한 클래스가 하나만 존재할 수 있음)")
            logger.info(f"실제 레이블 (y_true, {len(y_true_labels)}개, 일부): {y_true_labels[:20]}...")
            logger.info(f"예측 레이블 (y_pred, {len(y_pred_labels)}개, 일부): {y_pred_labels[:20]}...")
    else:
        logger.warning("최종 평가를 위한 유효 샘플이 없어 Classification Report를 생성할 수 없습니다.")


if __name__ == "__main__":
    evaluate_vanilla_model_performance()
