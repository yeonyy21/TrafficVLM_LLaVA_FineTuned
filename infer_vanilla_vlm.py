# infer_vanilla_vlm.py
# 파인튜닝되지 않은 순수 LLaVA-NeXT 모델을 사용하여,
# "예" 또는 "아니요" 답변을 유도하는 제한된 프롬프트로 추론을 수행합니다.

import sys
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # 이미지 크기 제한 해제

from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s (%(process)d): %(message)s')
logger = logging.getLogger(__name__)

# --- 전역 변수 ---
PROCESSOR_VANILLA = None
MODEL_VANILLA = None

# --- 상수 정의 ---
# 파인튜닝의 기반이 되었던 원본 LLaVA-NeXT 모델 ID
BASE_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf" 

# 순수 모델 로드 시 QLoRA와 유사한 양자화 적용 여부 (메모리 및 비교 일관성 위함)
# False로 설정하면 fp16/bf16으로 로드 (더 많은 메모리 필요)
USE_QUANTIZATION_FOR_VANILLA = True 

# 답변 생성 시 최대 새 토큰 수. "예" 또는 "아니오" 만을 기대하므로 짧게 설정 가능.
# 다만, 모델이 지시를 완벽히 따르지 않고 추가 설명을 붙일 수 있으므로 약간의 여유를 둠.
ANSWER_MAX_NEW_TOKENS = 32 # 이전 10에서 약간 늘림 (안정성)

# !!!! 새로운 제한된 질문 !!!!
CONSTRAINED_QUESTION = "이 이미지에서 주행자가 안전모를 착용하고 있습니까? ‘예’ 또는 ‘아니오’로만 답해주세요."

# 프롬프트 템플릿 (학습 시 eval 프롬프트와 유사한 형식 사용)
# image_token은 processor 로드 후 실제 값으로 대체됨
_IMAGE_TOKEN_PLACEHOLDER_VANILLA = "{IMAGE_TOKEN}" 
PROMPT_START_TEMPLATE_VANILLA = f"<s>[INST] {_IMAGE_TOKEN_PLACEHOLDER_VANILLA}\n" # LLaVA-1.6/Mistral 스타일
PROMPT_END_TEMPLATE_VANILLA  = lambda q:    f"{q} [/INST]" # 답변 생성을 위해 ASSISTANT: 제외


def _initialize_vanilla_model_and_processor():
    """
    순수 LLaVA-NeXT 모델과 프로세서를 초기화하고 전역 변수에 할당합니다.
    이미 초기화된 경우 아무 작업도 수행하지 않습니다.
    """
    global PROCESSOR_VANILLA, MODEL_VANILLA, PROMPT_START_TEMPLATE_VANILLA # 프롬프트 템플릿도 업데이트

    if PROCESSOR_VANILLA is not None and MODEL_VANILLA is not None:
        return

    logger.info(f"추론용 VANILLA LLaVA-NeXT 프로세서 및 모델 로딩 시작: {BASE_MODEL_ID}")
    try:
        PROCESSOR_VANILLA = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        if PROCESSOR_VANILLA.tokenizer.pad_token is None:
            PROCESSOR_VANILLA.tokenizer.pad_token = PROCESSOR_VANILLA.tokenizer.eos_token
            if PROCESSOR_VANILLA.tokenizer.pad_token_id is None:
                 PROCESSOR_VANILLA.tokenizer.pad_token_id = PROCESSOR_VANILLA.tokenizer.eos_token_id
        logger.info(f"프로세서 로드 완료. Pad token ID: {PROCESSOR_VANILLA.tokenizer.pad_token_id}")
        
        # 실제 이미지 토큰을 가져와 프롬프트 템플릿 업데이트
        actual_image_token = getattr(PROCESSOR_VANILLA, 'image_token', 
                                   getattr(PROCESSOR_VANILLA.tokenizer, 'image_token', "<image>"))
        PROMPT_START_TEMPLATE_VANILLA = f"<s>[INST] {actual_image_token}\n"
        logger.info(f"추론 시 사용될 이미지 토큰 (바닐라): '{actual_image_token}'")

        quant_config_vanilla = None
        if USE_QUANTIZATION_FOR_VANILLA:
            quant_config_vanilla = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 
            )
            logger.info("바닐라 모델 로드 시 QLoRA와 유사한 4비트 양자화 설정 적용됨.")

        MODEL_VANILLA = LlavaNextForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID, 
            torch_dtype=torch.float16, # 학습 시 사용한 dtype과 일치 권장
            quantization_config=quant_config_vanilla,
            low_cpu_mem_usage=True if USE_QUANTIZATION_FOR_VANILLA else False,
            device_map="auto" # 사용 가능한 GPU에 자동 할당
        )
        MODEL_VANILLA.eval() # 추론 모드
        logger.info(f"추론용 VANILLA LLaVA-NeXT 모델 ({BASE_MODEL_ID}) 및 프로세서 초기화 완료.")

    except Exception as e:
        logger.error(f"추론용 VANILLA 모델/프로세서 초기화 실패: {e}", exc_info=True)
        PROCESSOR_VANILLA = None; MODEL_VANILLA = None; raise


def predict_helmet_with_vanilla_llava(image_path: str, question: str = CONSTRAINED_QUESTION) -> str:
    """
    파인튜닝되지 않은 순수 LLaVA-NeXT 모델과 제한된 프롬프트를 사용하여 예측합니다.
    """
    global PROCESSOR_VANILLA, MODEL_VANILLA # 전역 변수 사용

    if PROCESSOR_VANILLA is None or MODEL_VANILLA is None:
        try: 
            _initialize_vanilla_model_and_processor()
        except Exception as e_init: # 초기화 실패 시 오류 메시지 반환
            return f"모델 초기화 오류: {e_init}"

    if not os.path.exists(image_path): 
        return "오류: 이미지 파일을 찾을 수 없습니다."

    try:
        image = Image.open(image_path).convert("RGB")
        
        # 프롬프트 구성 (학습 시 eval 프롬프트와 동일한 형식, 제한된 질문 사용)
        prompt_text = PROMPT_START_TEMPLATE_VANILLA + PROMPT_END_TEMPLATE_VANILLA(question)
        
        # image_sizes 준비
        image_sizes_tensor = torch.tensor([[image.height, image.width]]).to(MODEL_VANILLA.device)

        inputs = PROCESSOR_VANILLA(text=prompt_text, images=image, return_tensors="pt").to(MODEL_VANILLA.device)
        
        # logger.debug(f"VANILLA 추론 입력 (input_ids): {inputs['input_ids']}")

        with torch.no_grad(): # 추론 시에는 그래디언트 계산 비활성화
            outputs = MODEL_VANILLA.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values'].to(MODEL_VANILLA.dtype), # 모델의 dtype과 일치
                image_sizes=image_sizes_tensor, # image_sizes 전달
                max_new_tokens=ANSWER_MAX_NEW_TOKENS, # 위에서 정의한 상수 사용
                eos_token_id=PROCESSOR_VANILLA.tokenizer.eos_token_id,
                pad_token_id=PROCESSOR_VANILLA.tokenizer.pad_token_id
                # 답변 형식을 더 제어하기 위한 추가적인 generate 파라미터 (선택 사항):
                # num_beams=1,
                # do_sample=False, 
                # temperature=0.1, # 매우 낮은 온도로 설정하여 결정성을 높이고 "예/아니요" 유도
                # top_p=None,
                # top_k=None,
            )
        
        # 생성된 전체 토큰에서 입력 프롬프트 부분을 제외하고 디코딩
        input_token_len = inputs['input_ids'].shape[1]
        generated_ids_only = outputs[0][input_token_len:]
        answer = PROCESSOR_VANILLA.decode(generated_ids_only, skip_special_tokens=True)
        
    except Exception as e:
        logger.error(f"VANILLA LLaVA-NeXT (제한된 프롬프트) 추론 중 오류 (이미지: {image_path}): {e}", exc_info=True)
        return "오류: 예측 중 오류가 발생했습니다."
        
    return answer.strip()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"스크립트 직접 실행 사용법:\n  python {os.path.basename(__file__)} <image_path> [question]")
        print(f"  [question] 생략 시 기본 질문: '{CONSTRAINED_QUESTION}'")
        sys.exit(1)
        
    img_path_arg = sys.argv[1]
    # 명령줄에서 질문을 주면 그것을 사용하고, 아니면 CONSTRAINED_QUESTION 사용
    question_arg = sys.argv[2] if len(sys.argv) > 2 else CONSTRAINED_QUESTION
    
    # 모델/프로세서 로드를 위해 첫 호출 시 초기화 함수 실행
    try:
        _initialize_vanilla_model_and_processor()
    except Exception as e:
        print(f"초기화 실패: {e}")
        sys.exit(1)

    logger.info(f"명령줄에서 직접 실행 (VANILLA LLaVA-NeXT, 제한된 프롬프트): '{img_path_arg}'에 대한 예측 요청...")
    prediction = predict_helmet_with_vanilla_llava(img_path_arg, question_arg)

    print(f"\n--- VANILLA LLaVA-NeXT (제한된 프롬프트) 추론 결과 ---")
    print(f"이미지: {img_path_arg}")
    print(f"질문: {question_arg}") # 실제 사용된 질문 출력
    print(f"답변: {prediction}")
