# infer_finetuned_vlm.py
# train_helmet_vlm.py로 파인튜닝된 LLaVA-NeXT 모델을 사용하여 추론을 수행합니다.

import sys
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # 이미지 크기 제한 해제

from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel # LoRA 가중치 로드를 위해 필요
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s (%(process)d): %(message)s')
logger = logging.getLogger(__name__)

# --- 전역 변수 (모델 및 프로세서 인스턴스 저장) ---
PROCESSOR_FT = None # 파인튜닝된 모델용 프로세서
MODEL_FT = None     # 파인튜닝된 모델

# --- 상수 정의 ---
# 1. 파인튜닝 시 사용했던 기본 LLaVA-NeXT 모델 ID
BASE_MODEL_ID_FOR_LORA = "llava-hf/llava-v1.6-mistral-7b-hf" 

# 2. train_helmet_vlm.py에서 최종 LoRA 어댑터와 프로세서를 저장한 *정확한* 디렉토리 경로
#    예시: train_helmet_vlm.py의 OUTPUT_MODEL_DIR이 "outputs/llava_next_helmet_finetuned"이고,
#    그 안에 "final_checkpoint" 폴더를 만들어 저장했다면 아래와 같이 설정합니다.
FINETUNED_ARTIFACTS_PATH = "outputs/llava_next_helmet_finetuned/final_checkpoint" 
#    만약 OUTPUT_MODEL_DIR에 바로 저장했다면:
#    FINETUNED_ARTIFACTS_PATH = "outputs/llava_next_helmet_finetuned"

# 3. 추론 시 생성할 답변의 최대 새 토큰 수
ANSWER_MAX_NEW_TOKENS = 32 # 파인튜닝된 모델은 간결한 답변을 생성하므로, 적절히 조절 가능

# 4. 추론 시 사용할 기본 질문
DEFAULT_QUESTION = "이 오토바이 주행자가 안전모를 착용했나요?"

# 5. 프롬프트 템플릿 (train_helmet_vlm.py의 helmet_collate_fn_eval과 일치해야 함)
#    image_token은 processor 로드 후 실제 값으로 대체됨
_IMAGE_TOKEN_PLACEHOLDER = "{IMAGE_TOKEN}" 
PROMPT_START_TEMPLATE = f"<s>[INST] {_IMAGE_TOKEN_PLACEHOLDER}\n" # LLaVA-1.6/Mistral 스타일
PROMPT_END_TEMPLATE  = lambda q:    f"{q} [/INST]" # 답변 생성을 위해 ASSISTANT: 없이 이 부분까지만
# 또는 LLaVA 1.5 스타일을 학습에 사용했다면:
# PROMPT_START_TEMPLATE = f"USER: {_IMAGE_TOKEN_PLACEHOLDER}\n"
# PROMPT_END_TEMPLATE = lambda q: f"{q}\nASSISTANT:"


def _initialize_finetuned_model_and_processor():
    """
    파인튜닝된 모델과 프로세서를 초기화하고 전역 변수에 할당합니다.
    이미 초기화된 경우 아무 작업도 수행하지 않습니다.
    """
    global PROCESSOR_FT, MODEL_FT, PROMPT_START_TEMPLATE # PROMPT_START_TEMPLATE도 업데이트 필요

    if PROCESSOR_FT is not None and MODEL_FT is not None:
        return

    logger.info(f"추론용 파인튜닝 모델/프로세서 로딩 시작: {FINETUNED_ARTIFACTS_PATH}")
    if not os.path.exists(FINETUNED_ARTIFACTS_PATH):
        logger.error(f"오류: 지정된 경로에 학습된 모델 아티팩트가 없습니다: {FINETUNED_ARTIFACTS_PATH}")
        logger.error("train_helmet_vlm.py를 먼저 실행하여 모델을 학습 및 저장했는지, 경로가 올바른지 확인하세요.")
        raise FileNotFoundError(f"Artifacts not found at {FINETUNED_ARTIFACTS_PATH}")
    try:
        # 1. 저장된 프로세서 로드
        # FINETUNED_ARTIFACTS_PATH는 processor.save_pretrained()로 저장된 모든 파일 포함해야 함
        PROCESSOR_FT = AutoProcessor.from_pretrained(FINETUNED_ARTIFACTS_PATH)
        logger.info(f"파인튜닝된 프로세서 로드 완료. Pad token ID: {PROCESSOR_FT.tokenizer.pad_token_id}")
        if PROCESSOR_FT.tokenizer.pad_token is None: # 만약을 위해 다시 한번 확인 및 설정
            PROCESSOR_FT.tokenizer.pad_token = PROCESSOR_FT.tokenizer.eos_token
            if PROCESSOR_FT.tokenizer.pad_token_id is None:
                 PROCESSOR_FT.tokenizer.pad_token_id = PROCESSOR_FT.tokenizer.eos_token_id
            logger.info(f"추론용 프로세서 PAD 토큰 재설정: ID {PROCESSOR_FT.tokenizer.pad_token_id}")
        
        # 프롬프트 템플릿의 실제 이미지 토큰 업데이트
        actual_image_token = getattr(PROCESSOR_FT, 'image_token', 
                                   getattr(PROCESSOR_FT.tokenizer, 'image_token', "<image>"))
        PROMPT_START_TEMPLATE = f"<s>[INST] {actual_image_token}\n" # LLaVA-1.6/Mistral 스타일
        # PROMPT_START_TEMPLATE = f"USER: {actual_image_token}\n" # LLaVA 1.5 스타일 사용 시
        logger.info(f"추론 시 사용될 실제 이미지 토큰: '{actual_image_token}'")


        # 2. 기본 LLaVA-NeXT 모델 로드 (QLoRA 학습 시와 동일한 설정으로)
        logger.info(f"추론용 기본 LLaVA-NeXT 모델 로딩: {BASE_MODEL_ID_FOR_LORA}")
        # QLoRA로 학습했으므로, 기본 모델 로드 시에도 양자화 옵션 고려 가능 (메모리 절약)
        # 또는 float16/bfloat16으로 로드 후 LoRA 적용
        quantization_config_load = None
        if USE_QLORA: # train_helmet_vlm.py의 USE_QLORA 설정과 일치시키는 것이 좋음
             quantization_config_load = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
             logger.info("추론 시 기본 모델 로드에 4비트 양자화 설정 적용됨.")
        
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID_FOR_LORA,
            torch_dtype=torch.float16, # 학습 시 사용한 dtype
            quantization_config=quantization_config_load,
            low_cpu_mem_usage=True if quantization_config_load else False,
            device_map="auto"
        )
        logger.info("기본 모델 로드 완료.")
        
        # 3. 저장된 LoRA 어댑터 가중치를 기본 모델에 적용
        logger.info(f"저장된 LoRA 어댑터 로딩 및 적용: {FINETUNED_ARTIFACTS_PATH}")
        MODEL_FT = PeftModel.from_pretrained(base_model, FINETUNED_ARTIFACTS_PATH)
        logger.info("LoRA 어댑터 적용 완료.")
        
        # 4. (선택 사항) LoRA 가중치를 기본 모델에 병합하여 추론 속도 향상
        # 병합 후에는 LoRA 어댑터를 분리하거나 추가 학습하기 어려워짐.
        logger.info("LoRA 가중치 병합 시도 (선택 사항)...")
        try:
            MODEL_FT = MODEL_FT.merge_and_unload()
            logger.info("LoRA 가중치 병합 완료. 이제 일반 모델처럼 작동합니다.")
        except Exception as e_merge:
            logger.warning(f"LoRA 어댑터 병합 실패 (어댑터가 적용된 상태로 계속 사용): {e_merge}")

        MODEL_FT.eval() # 추론 모드로 설정
        logger.info("추론용 파인튜닝된 LLaVA-NeXT 모델 및 프로세서 초기화 완료.")

    except Exception as e:
        logger.error(f"추론용 모델/프로세서 초기화 실패: {e}", exc_info=True)
        PROCESSOR_FT = None; MODEL_FT = None; raise


def predict_helmet_with_finetuned_vlm(image_path: str, question: str = DEFAULT_QUESTION) -> str:
    """
    파인튜닝된 LLaVA-NeXT 모델을 사용하여 이미지와 질문에 대한 답변을 생성합니다.
    """
    global PROCESSOR_FT, MODEL_FT # 전역 변수 사용

    if PROCESSOR_FT is None or MODEL_FT is None:
        try:
            _initialize_finetuned_model_and_processor()
        except Exception as e_init: # 초기화 실패 시 오류 메시지 반환
            return f"모델 초기화 중 오류 발생: {e_init}"

    if not os.path.exists(image_path):
        return "오류: 이미지 파일을 찾을 수 없습니다."

    try:
        image = Image.open(image_path).convert("RGB")
        
        # 프롬프트 구성 (학습 시 검증 프롬프트와 동일한 형식)
        prompt_text = PROMPT_START_TEMPLATE + PROMPT_END_TEMPLATE(question)
        
        # image_sizes 준비
        # 모델이 GPU에 있다면, 이 텐서도 같은 GPU로 보내야 함
        image_sizes_tensor = torch.tensor([[image.height, image.width]]).to(MODEL_FT.device)

        inputs = PROCESSOR_FT(text=prompt_text, images=image, return_tensors="pt").to(MODEL_FT.device)
        
        # logger.debug(f"추론 입력 (input_ids): {inputs['input_ids']}")
        # logger.debug(f"추론 입력 (pixel_values shape): {inputs['pixel_values'].shape}")
        # logger.debug(f"추론 입력 (image_sizes): {image_sizes_tensor}")

        with torch.no_grad(): # 추론 시에는 그래디언트 계산이 필요 없음
            outputs = MODEL_FT.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values'].to(MODEL_FT.dtype), # 모델의 dtype과 일치
                image_sizes=image_sizes_tensor, # image_sizes 전달
                max_new_tokens=ANSWER_MAX_NEW_TOKENS, # 답변 길이 조절
                eos_token_id=PROCESSOR_FT.tokenizer.eos_token_id,
                pad_token_id=PROCESSOR_FT.tokenizer.pad_token_id # generate는 pad_token_id를 내부적으로 사용
            )
        
        # 생성된 전체 토큰에서 입력 프롬프트 부분을 제외하고 디코딩
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids_only = outputs[0][input_token_len:]
        answer = PROCESSOR_FT.decode(generated_ids_only, skip_special_tokens=True)
        
    except Exception as e:
        logger.error(f"파인튜닝된 LLaVA-NeXT 추론 중 오류 (이미지: {image_path}): {e}", exc_info=True)
        return "오류: 예측 중 오류가 발생했습니다."
        
    return answer.strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"스크립트 직접 실행 사용법:\n  python {os.path.basename(__file__)} <image_path> [question]")
        print(f"  예시: python {os.path.basename(__file__)} data/crops/some_image.png")
        sys.exit(1)

    img_path_arg = sys.argv[1]
    question_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION

    logger.info(f"명령줄에서 직접 실행 (파인튜닝된 LLaVA-NeXT): '{img_path_arg}'에 대한 예측 요청...")
    # 모델/프로세서 로드를 위해 첫 호출 시 초기화 함수 실행
    try:
        _initialize_finetuned_model_and_processor()
    except Exception as e:
        print(f"초기화 실패: {e}")
        sys.exit(1)

    prediction = predict_helmet_with_finetuned_vlm(img_path_arg, question_arg)

    print("\n--- 파인튜닝된 LLaVA-NeXT 추론 결과 ---")
    print(f"이미지: {img_path_arg}")
    print(f"질문: {question_arg}")
    print(f"답변: {prediction}")
