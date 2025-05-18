# preprocess_helmet_data.py
import os
import json
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # 매우 큰 원본 이미지 로드를 위한 제한 해제
import xml.etree.ElementTree as ET
import logging

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s (%(process)d): %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 (사용자 환경에 맞게 경로 수정 필요) ---
# 원본 이미지가 저장된 루트 디렉토리 (하위에 train/val/test 등의 폴더 구조를 가질 수 있음)
IMAGE_ROOT = "data/images"
# 원본 어노테이션 파일(JSON, XML)이 저장된 루트 디렉토리
ANN_ROOT   = "data/annotations"
# 크롭된 이미지를 저장할 디렉토리
CROP_DIR   = "data/crops"
# 최종적으로 생성될 학습/검증용 메타데이터 JSON 파일 경로
OUT_JSON   = "data/train_helmet_mixed.json"
# 모든 데이터 샘플에 사용될 고정 질문
QUESTION   = "이 오토바이 주행자가 안전모를 착용했나요?"
# --- 설정 끝 ---

def create_training_json_from_annotations():
    """
    JSON 및 XML 어노테이션 파일을 순회하며 이미지를 크롭하고,
    (크롭 이미지 경로, 질문, 답변) 형식의 학습용 메타데이터를 생성하여 JSON 파일로 저장합니다.
    """
    if not os.path.isdir(IMAGE_ROOT):
        logger.error(f"원본 이미지 디렉토리 '{IMAGE_ROOT}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    if not os.path.isdir(ANN_ROOT):
        logger.error(f"원본 어노테이션 디렉토리 '{ANN_ROOT}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    os.makedirs(CROP_DIR, exist_ok=True) # 크롭 이미지 저장 폴더 생성
    dataset_records = []
    processed_object_count = 0 # 처리된 객체(크롭 이미지) 수 카운트

    # 1) JSON 어노테이션 처리
    logger.info(f"JSON 어노테이션 처리 시작... (검색 경로: {os.path.join(ANN_ROOT, '**', '*.json')})")
    json_files_found = glob.glob(os.path.join(ANN_ROOT, "**", "*.json"), recursive=True)
    logger.info(f"총 {len(json_files_found)}개의 JSON 파일 감지.")

    for jf_path in json_files_found:
        logger.info(f"JSON 파일 처리 중: {jf_path}")
        try:
            # subset_folder는 'train', 'val', 'test' 등 이미지 상위 폴더명을 가져오기 위함
            # 예: /data/annotations/train/file.json -> subset_folder = 'train'
            # 만약 ANN_ROOT 바로 밑에 json 파일들이 있다면, subset_folder 로직 수정 필요
            path_parts = jf_path.split(os.sep)
            subset_folder = ""
            if len(path_parts) > 2 and path_parts[-2] != os.path.basename(ANN_ROOT): # ANN_ROOT/subset/file.json 구조일 경우
                 subset_folder = path_parts[-2]
            else: # ANN_ROOT/file.json 구조일 경우 등, subset 구분이 어려우면 빈 문자열 또는 기본값 사용
                 logger.debug(f"'{jf_path}'에서 subset 폴더를 특정할 수 없어 기본 경로 사용 예정.")


            with open(jf_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # 메타데이터 구조가 다를 수 있으므로, get() 메소드와 기본값으로 안전하게 접근
            original_fname = meta.get("Meta", {}).get("Basic File Name")
            if not original_fname:
                logger.warning(f"'{jf_path}'에 'Meta/Basic File Name' 정보가 없습니다. 건너<0xEB><0x81><0xB5니다.")
                continue
            
            # 원본 이미지 경로 구성
            img_path_original = os.path.join(IMAGE_ROOT, subset_folder, original_fname) if subset_folder else os.path.join(IMAGE_ROOT, original_fname)

            if not os.path.exists(img_path_original):
                logger.warning(f"이미지 파일을 찾을 수 없습니다: {img_path_original} (JSON: {jf_path}). 건너<0xEB><0x81><0xB5니다.")
                continue
            
            img_obj = Image.open(img_path_original).convert("RGB")

            annotations_list = meta.get("Annotation", {}).get("annotations", [])
            if not annotations_list:
                logger.warning(f"'{jf_path}'에 'Annotation/annotations' 정보가 없거나 비어있습니다. 건너<0xEB><0x81><0xB5니다.")
                continue

            for ann_idx, ann_data in enumerate(annotations_list):
                if ann_data.get("Annotation Type", "").lower() != "bbox":
                    continue
                
                object_name = ann_data.get("Object Name", "")
                bbox_coords = ann_data.get("Bbox Cordinate")

                if not bbox_coords or len(bbox_coords) != 4:
                    logger.warning(f"잘못된 Bbox 좌표 ({bbox_coords}) in {jf_path} for object '{object_name}'. 건너<0xEB><0x81><0xB5니다.")
                    continue
                
                x1, y1, x2, y2 = map(int, bbox_coords) # 정수형으로 변환
                if not (0 <= x1 < x2 <= img_obj.width and 0 <= y1 < y2 <= img_obj.height):
                    logger.warning(f"이미지 크기 벗어난 Bbox 좌표: {(x1,y1,x2,y2)} in {img_path_original}. 건너<0xEB><0x81><0xB5니다.")
                    continue

                cropped_img = img_obj.crop((x1, y1, x2, y2))
                base_fname_no_ext = os.path.splitext(original_fname)[0]
                annotation_id_str = ann_data.get('Annotation ID', f"idx{ann_idx}") # Annotation ID가 없을 경우 대비
                
                # 크롭 이미지 파일명에 subset_folder 정보 포함 (폴더 구조가 복잡할 경우 대비)
                crop_fname_prefix = f"{subset_folder}_" if subset_folder else ""
                crop_fname = f"{crop_fname_prefix}{base_fname_no_ext}_json_{annotation_id_str}.png"
                out_crop_filepath = os.path.join(CROP_DIR, crop_fname)
                
                try:
                    cropped_img.save(out_crop_filepath)
                except Exception as e_save:
                    logger.error(f"크롭 이미지 저장 실패 '{out_crop_filepath}': {e_save}")
                    continue

                answer_text = "예, 착용했다." if "착용" in object_name else "아니요, 착용하지 않았다."
                
                # 저장된 크롭 이미지 경로를 메타데이터에 포함 (data/crops/...)
                # 스크립트 실행 위치 기준 상대 경로가 되도록 os.path.relpath 사용 가능 (선택)
                # 여기서는 CROP_DIR 기준으로 저장된 경로를 그대로 사용 (예: "data/crops/이미지명.png")
                # 만약 CROP_DIR이 "data/crops" 라면, out_crop_filepath는 "data/crops/..." 형태
                # 학습 스크립트의 IMAGE_BASE_DIRECTORY 설정과 일관성 유지가 중요
                dataset_records.append({
                    "image_path":  out_crop_filepath, # CROP_DIR을 포함한 경로
                    "instruction": QUESTION,
                    "output":      answer_text
                })
                processed_object_count += 1
        except FileNotFoundError: # open(jf_path, ...) 에서 발생 가능
            logger.error(f"어노테이션 파일 '{jf_path}'를 찾을 수 없습니다.", exc_info=True)
        except json.JSONDecodeError:
            logger.error(f"JSON 파일 파싱 오류 '{jf_path}'. 형식을 확인하세요.", exc_info=True)
        except Exception as e:
            logger.error(f"JSON 파일 처리 중 예기치 않은 오류 발생 '{jf_path}': {e}", exc_info=True)

    logger.info("XML 어노테이션 처리 시작...")
    xml_files_found = glob.glob(os.path.join(ANN_ROOT, "**", "*.xml"), recursive=True)
    logger.info(f"총 {len(xml_files_found)}개의 XML 파일 감지.")

    for xf_path in xml_files_found:
        logger.info(f"XML 파일 처리 중: {xf_path}")
        try:
            path_parts = xf_path.split(os.sep)
            subset_folder = ""
            if len(path_parts) > 2 and path_parts[-2] != os.path.basename(ANN_ROOT):
                 subset_folder = path_parts[-2]
            else:
                 logger.debug(f"'{xf_path}'에서 subset 폴더를 특정할 수 없어 기본 경로 사용 예정.")

            tree = ET.parse(xf_path)
            root = tree.getroot()
            
            original_fname_node = root.find("filename")
            if original_fname_node is None or not original_fname_node.text:
                logger.warning(f"'{xf_path}'에 'filename' 정보가 없습니다. 건너<0xEB><0x81><0xB5니다.")
                continue
            original_fname = original_fname_node.text
            img_path_original = os.path.join(IMAGE_ROOT, subset_folder, original_fname) if subset_folder else os.path.join(IMAGE_ROOT, original_fname)


            if not os.path.exists(img_path_original):
                logger.warning(f"이미지 파일을 찾을 수 없습니다: {img_path_original} (XML: {xf_path}). 건너<0xEB><0x81><0xB5니다.")
                continue
            
            img_obj = Image.open(img_path_original).convert("RGB")

            for obj_idx, object_node in enumerate(root.findall("object")):
                class_name_node = object_node.find("name")
                bndbox_node = object_node.find("bndbox")

                if class_name_node is None or not class_name_node.text or bndbox_node is None:
                    logger.warning(f"XML 객체에 'name' 또는 'bndbox' 정보 부족 in {xf_path}. 건너<0xEB><0x81><0xB5니다.")
                    continue
                
                class_name = class_name_node.text.lower() # 일관성을 위해 소문자로
                
                try:
                    x1 = int(float(bndbox_node.findtext("xmin"))) # 가끔 float으로 되어 있는 경우 대비
                    y1 = int(float(bndbox_node.findtext("ymin")))
                    x2 = int(float(bndbox_node.findtext("xmax")))
                    y2 = int(float(bndbox_node.findtext("ymax")))
                except (ValueError, TypeError) as e_coord:
                    logger.warning(f"잘못된 Bbox 좌표 형식 in {xf_path} for object '{class_name}': {e_coord}. 건너<0xEB><0x81><0xB5니다.")
                    continue

                if not (0 <= x1 < x2 <= img_obj.width and 0 <= y1 < y2 <= img_obj.height):
                    logger.warning(f"이미지 크기 벗어난 Bbox 좌표: {(x1,y1,x2,y2)} in {img_path_original}. 건너<0xEB><0x81><0xB5니다.")
                    continue

                cropped_img = img_obj.crop((x1, y1, x2, y2))
                base_fname_no_ext = os.path.splitext(original_fname)[0]
                
                tag = "with_helmet" if "with" in class_name else "without_helmet" # "with helmet", "without helmet" 등 가정
                if "without" not in class_name and "with" not in class_name:
                    logger.warning(f"XML 클래스명에서 착용/미착용 구분 불가: '{class_name_node.text}' in {xf_path}. '미착용'으로 간주.")
                    tag = "unknown_helmet_status" # 또는 기본값

                crop_fname_prefix = f"{subset_folder}_" if subset_folder else ""
                crop_fname = f"{crop_fname_prefix}{base_fname_no_ext}_xml_{tag}_{obj_idx}.png"
                out_crop_filepath = os.path.join(CROP_DIR, crop_fname)
                
                try:
                    cropped_img.save(out_crop_filepath)
                except Exception as e_save:
                    logger.error(f"크롭 이미지 저장 실패 '{out_crop_filepath}': {e_save}")
                    continue

                answer_text = "예, 착용했다." if "with" in tag else "아니요, 착용하지 않았다."
                if tag == "unknown_helmet_status": # 위에서 구분 불가 시
                    answer_text = "판단 불가" # 또는 다른 기본값

                if answer_text != "판단 불가": # 유효한 답변만 추가
                    dataset_records.append({
                        "image_path":  out_crop_filepath,
                        "instruction": QUESTION,
                        "output":      answer_text
                    })
                    processed_object_count += 1
        except ET.ParseError:
            logger.error(f"XML 파일 파싱 오류 '{xf_path}'. 형식을 확인하세요.", exc_info=True)
        except Exception as e:
            logger.error(f"XML 파일 처리 중 예기치 않은 오류 발생 '{xf_path}': {e}", exc_info=True)
    
    logger.info(f"총 {processed_object_count}개의 유효한 크롭 이미지 및 데이터 생성 완료.")
    if not dataset_records:
        logger.error("생성된 데이터셋이 비어있습니다. 입력 파일, 경로, 어노테이션 형식을 확인하세요.")
        return

    try:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(dataset_records, f, ensure_ascii=False, indent=4) # indent=4로 가독성 향상
        logger.info(f"✔ 전처리 완료: {len(dataset_records)}개 데이터 포인트 저장 → {OUT_JSON}")
    except Exception as e:
        logger.error(f"결과 JSON 파일 저장 중 오류 '{OUT_JSON}': {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("데이터 전처리 스크립트 시작...")
    create_training_json_from_annotations()
    logger.info("데이터 전처리 스크립트 종료.")
