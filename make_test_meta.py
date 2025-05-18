# make_test_meta.py
import json
import random
import os
import logging

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
# 이 스크립트는 프로젝트 루트 디렉토리(예: /knpu/traffic/VLM_and_YOLO/)에 있다고 가정합니다.
# data 폴더는 프로젝트 루트 하위에 있다고 가정합니다.
DATA_DIR = "data"
# preprocess_helmet_data.py의 OUT_JSON 값과 일치해야 합니다.
INPUT_JSON_FILE = os.path.join(DATA_DIR, "train_helmet_mixed.json")
# 생성될 테스트 메타데이터 파일 경로
OUTPUT_TEST_META_JSON = os.path.join(DATA_DIR, "test_meta.json")
# 테스트셋으로 분리할 비율 (예: 0.15는 15%)
TEST_SPLIT_RATIO = 0.15
# 재현성을 위한 랜덤 시드
RANDOM_SEED = 42
# --- 설정 끝 ---

def create_test_metadata_file():
    """
    전체 데이터셋에서 일부를 무작위로 샘플링하여 테스트용 메타데이터 파일을 생성합니다.
    생성된 파일은 "image_path"와 "label" ("yes" 또는 "no") 키를 가집니다.
    """
    if not os.path.exists(INPUT_JSON_FILE):
        logger.error(f"오류: 입력 파일 '{INPUT_JSON_FILE}'을 찾을 수 없습니다. "
                     f"먼저 preprocess_helmet_data.py를 실행하여 해당 파일을 생성하세요.")
        return

    logger.info(f"'{INPUT_JSON_FILE}' 파일 로딩 중...")
    try:
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"오류: '{INPUT_JSON_FILE}' 파일이 올바른 JSON 형식이 아닙니다.")
        return
    except Exception as e:
        logger.error(f"'{INPUT_JSON_FILE}' 파일 로딩 중 오류 발생: {e}", exc_info=True)
        return

    if not full_dataset or not isinstance(full_dataset, list):
        logger.error(f"오류: '{INPUT_JSON_FILE}' 파일이 비어있거나 리스트 형식이 아닙니다.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(full_dataset) # 데이터셋을 무작위로 섞음
    
    num_total_samples = len(full_dataset)
    num_test_samples = int(num_total_samples * TEST_SPLIT_RATIO)

    # 데이터가 매우 적을 경우, 테스트 샘플이 0개가 되는 것을 방지 (최소 1개)
    if num_test_samples == 0 and num_total_samples > 0:
        num_test_samples = 1
    
    if num_test_samples == 0 :
        logger.error("오류: 테스트 샘플을 생성하기에 전체 데이터가 너무 적습니다 (0개).")
        return

    test_split_data = full_dataset[:num_test_samples]
    logger.info(f"총 {num_total_samples}개 원본 데이터 중 {len(test_split_data)}개의 테스트 샘플 생성 예정.")

    test_meta_list = []
    for i, entry in enumerate(test_split_data):
        # 입력 JSON 파일(`train_helmet_mixed.json`)은 "output" 키에
        # "예, 착용했다." 또는 "아니요, 착용하지 않았다." 형식의 답변을 가지고 있음.
        # 이를 "yes" 또는 "no" 레이블로 변환.
        original_output_text = entry.get("output")
        image_path_val = entry.get("image_path")

        if original_output_text is None or image_path_val is None:
            logger.warning(f"{i+1}번째 항목에 'output' 또는 'image_path' 키가 없습니다. 건너<0xEB><0x81><0xB5니다. 항목: {entry}")
            continue
        
        # "예"가 포함되어 있으면 "yes", 그렇지 않으면 "no"로 레이블링
        # 이 로직은 원본 답변의 다양성을 고려하여 더 정교하게 만들 수 있음
        # (예: "아니요"가 명확히 포함되면 "no" 등)
        # 현재는 "예" 포함 여부로만 단순하게 판단
        label = "yes" if "예" in original_output_text else "no"
        
        test_meta_list.append({
            "image_path": image_path_val,
            "label":      label  # 평가 스크립트가 사용할 "label" 키
        })

    if not test_meta_list:
        logger.error("유효한 테스트 메타데이터를 생성하지 못했습니다.")
        return

    try:
        with open(OUTPUT_TEST_META_JSON, "w", encoding="utf-8") as f:
            json.dump(test_meta_list, f, ensure_ascii=False, indent=4) # indent로 가독성 좋게 저장
        logger.info(f"✔ '{OUTPUT_TEST_META_JSON}' 생성 완료: 총 {len(test_meta_list)}개 테스트 샘플 저장됨.")
    except Exception as e:
        logger.error(f"테스트 메타데이터 파일 저장 중 오류 '{OUTPUT_TEST_META_JSON}': {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("테스트 메타데이터 생성 스크립트 시작...")
    create_test_metadata_file()
    logger.info("테스트 메타데이터 생성 스크립트 종료.")
