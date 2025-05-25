# train_helmet_vlm.py
# LLaVA 1.6 모델 (`llava-hf/llava-v1.6-mistral-7b-hf`)을 
# PyTorch Lightning과 QLoRA를 사용하여 오토바이 헬멧 착용 여부 감지 작업에 파인튜닝합니다.
# 실행 전 `preprocess_helmet_data.py`를 통해 `data/train_helmet_mixed.json` 파일이 생성되어 있어야 합니다.

import os
import json
import random
from typing import Any, Dict, List 

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L # PyTorch Lightning
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # 이미지 크기 제한 해제

from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, AutoProcessor
# from datasets import load_dataset as hf_load_dataset # 현재 직접 사용 안 함
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import numpy as np # validation_step에서 np.mean 사용 시 필요 (현재는 단순 정확도)
import logging

# 로깅 설정
logger = logging.getLogger(__name__) # PyTorch Lightning은 자체 로깅 시스템도 가짐
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s (%(process)d): %(message)s')

# --- 1. 설정 (Configurations) ---
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
DATA_JSON_PATH = "data/train_helmet_mixed.json" # preprocess_helmet_data.py의 결과물
OUTPUT_MODEL_DIR = "outputs/llava_next_helmet_finetuned" # README.md와 일치하는 학습 결과물 저장 기본 경로

MAX_TEXT_LENGTH = 512  # 입력 텍스트(프롬프트+답변)의 최대 토큰 길이
ANSWER_MAX_LENGTH = 32 # 답변 생성 시 최대 새 토큰 수 (검증용)
MAX_EPOCHS = 3         # 총 학습 에포크 수
BATCH_SIZE = 1         # GPU 메모리에 맞춰 조절 (A100 80GB에서는 2 또는 4 시도 가능)
ACCUMULATE_GRAD_BATCHES = 16 # 실질 배치 크기 = BATCH_SIZE * ACCUMULATE_GRAD_BATCHES
LEARNING_RATE = 1e-5   # LLaVA 계열 모델에 권장되는 낮은 학습률
GRADIENT_CLIP_VAL = 1.0 # 그래디언트 클리핑 값
USE_QLORA = True       # QLoRA 사용 여부 (True 권장)
FIXED_QUESTION = "이 오토바이 주행자가 안전모를 착용했나요?" # 데이터셋 생성 시 사용된 질문

# JSON 파일 내 image_path의 기준 디렉토리.
# data/train_helmet_mixed.json 내 image_path가 "data/crops/이미지명.png" 형태이므로,
# 이 스크립트를 프로젝트 루트에서 실행한다면 "."으로 설정하거나,
# image_path가 "crops/이미지명.png" 이고 IMAGE_BASE_DIRECTORY = "data" 로 설정.
# 여기서는 image_path가 이미 "data/crops/..." 전체 상대 경로라고 가정하고 "." 사용.
IMAGE_BASE_DIRECTORY = "." 

# --- 2. 전역 변수 (모델 및 프로세서) ---
# 이들은 main 함수 내에서 초기화되고 PyTorch Lightning 모듈에 전달됨
processor_main = None
model_main = None

# --- 3. 데이터셋 및 데이터로더 정의 ---
class HelmetDataset(Dataset):
    def __init__(self, json_file_path: str, image_base_dir: str, split_name: str = "all"):
        super().__init__()
        self.image_base_dir = image_base_dir
        self.dataset_list = []
        self.dataset_length = 0
        logger.info(f"'{split_name}' 데이터셋 원본 로딩 시도 (경로: {json_file_path})...")
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            # 여기서 `all_data`는 `preprocess_helmet_data.py`의 `dataset_records`임
            self.dataset_list = all_data 
            self.dataset_length = len(self.dataset_list)
            logger.info(f"'{split_name}' 데이터셋 원본 로드 완료. 총 샘플 수: {self.dataset_length}")
            if self.dataset_length == 0:
                logger.warning(f"'{split_name}' 데이터셋이 비어있습니다! 경로 및 파일 내용을 확인하세요: {json_file_path}")
        except FileNotFoundError:
            logger.error(f"'{split_name}' 데이터셋 파일({json_file_path})을 찾을 수 없습니다.", exc_info=True)
        except json.JSONDecodeError:
            logger.error(f"'{split_name}' 데이터셋 파일({json_file_path}) JSON 디코딩 오류.", exc_info=True)
        except Exception as e:
            logger.error(f"'{split_name}' 데이터셋 로딩 중 예기치 않은 오류 ({json_file_path}): {e}", exc_info=True)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset_list[idx]
        
        relative_image_path = item.get("image_path") # 예: "data/crops/이미지.png"
        # instruction 필드는 json 파일에 있지만, 여기서는 FIXED_QUESTION 사용
        question_text = FIXED_QUESTION 
        answer_text = item.get("output") # 예: "예, 착용했다."
        image = None

        if relative_image_path and answer_text:
            # image_base_dir이 "."이면, relative_image_path는 "data/crops/..."와 같이
            # 스크립트 실행 위치 기준의 전체 상대 경로여야 함.
            full_image_path = os.path.join(self.image_base_dir, relative_image_path) if self.image_base_dir != "." else relative_image_path
            
            if os.path.exists(full_image_path):
                try:
                    image = Image.open(full_image_path).convert("RGB")
                except Exception as e_img:
                    logger.error(f"이미지 로드 실패 (인덱스 {idx}, 경로 {full_image_path}): {e_img}")
            else:
                logger.warning(f"이미지 파일 경로 없음: {full_image_path} (데이터 인덱스 {idx})")
        else:
            logger.warning(f"잘못된 데이터 샘플 (인덱스 {idx}): image_path 또는 output 누락. item: {item}")

        if image is None: # 이미지 로드 실패 또는 경로/데이터 누락 시
            # LLaVA-NeXT는 보통 336x336 입력을 기본으로 하므로, 그 크기의 더미 이미지 생성
            image = Image.new('RGB', (336, 336), (255, 255, 255)) 
            answer_text = "오류: 이미지 정보 없음" # 답변도 변경하여 문제 인지
            logger.warning(f"더미 이미지 및 답변 사용됨 (인덱스 {idx})")


        return {"image": image, "question": question_text, "answer": answer_text, "original_image_path": relative_image_path}

# LLaVA-1.6 / Mistral-Instruct 스타일 프롬프트 또는 LLaVA 1.5 스타일
# processor 로드 후 image_token_str을 실제 값으로 설정
image_token_str_placeholder = "{IMAGE_TOKEN}" 
# 예시 프롬프트 템플릿 (학습 데이터의 질문/답변 형식과 일치해야 함)
PROMPT_FORMAT_TRAIN = f"USER: {image_token_str_placeholder}\n{{question}}\nASSISTANT: {{answer}}" # EOS는 processor가 붙이거나, 직접 추가
PROMPT_FORMAT_EVAL  = f"USER: {image_token_str_placeholder}\n{{question}}\nASSISTANT:"

def helmet_collate_fn_train(batch_samples: List[Dict]):
    global processor_main # 메인 프로세스에서 초기화된 processor 사용
    if processor_main is None: raise ValueError("Processor가 초기화되지 않았습니다.")

    images = [sample["image"] for sample in batch_samples]
    questions = [sample["question"] for sample in batch_samples]
    answers = [sample["answer"] for sample in batch_samples]
    
    # 실제 이미지 토큰 사용
    current_image_token = getattr(processor_main, 'image_token', getattr(processor_main.tokenizer, 'image_token', "<image>"))
    
    full_texts = [
        PROMPT_FORMAT_TRAIN.replace(image_token_str_placeholder, current_image_token).format(question=q, answer=a) + processor_main.tokenizer.eos_token
        for q, a in zip(questions, answers)
    ]
    
    image_sizes_tensor = torch.tensor([[img.height, img.width] for img in images])

    inputs = processor_main(
        text=full_texts, images=images,
        padding="longest", truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt"
    )
    labels = inputs.input_ids.clone()
    # 패딩 토큰에 해당하는 레이블은 -100으로 마스킹 (손실 계산에서 제외)
    labels[labels == processor_main.tokenizer.pad_token_id] = -100 
    inputs["labels"] = labels
    inputs["image_sizes"] = image_sizes_tensor
    return {key: value for key, value in inputs.items()}


def helmet_collate_fn_eval(batch_samples: List[Dict]):
    global processor_main
    if processor_main is None: raise ValueError("Processor가 초기화되지 않았습니다.")

    images = [sample["image"] for sample in batch_samples]
    questions = [sample["question"] for sample in batch_samples]
    ground_truth_answers = [sample["answer"] for sample in batch_samples] # 평가용 정답
    
    current_image_token = getattr(processor_main, 'image_token', getattr(processor_main.tokenizer, 'image_token', "<image>"))

    prompt_texts = [
        PROMPT_FORMAT_EVAL.replace(image_token_str_placeholder, current_image_token).format(question=q)
        for q in questions
    ]
    image_sizes_tensor = torch.tensor([[img.height, img.width] for img in images])
    inputs = processor_main(
        text=prompt_texts, images=images,
        padding="longest", truncation=True, max_length=MAX_TEXT_LENGTH, return_tensors="pt"
    )
    inputs["image_sizes"] = image_sizes_tensor
    inputs["ground_truth_answers"] = ground_truth_answers
    return {key: value for key, value in inputs.items()}

# --- 4. PyTorch Lightning 모듈 정의 ---
class LlavaHelmetPLModule(L.LightningModule):
    def __init__(self, model_instance, processor_instance, learning_rate_cfg, batch_size_cfg):
        super().__init__()
        self.model = model_instance
        self.processor = processor_instance # collate_fn 등에서 사용될 수 있도록 전달 (여기서는 전역 사용)
        self.learning_rate = learning_rate_cfg
        self.batch_size_for_logging = batch_size_cfg # 로깅용 배치 크기
        self.save_hyperparameters(ignore=['model_instance', 'processor_instance']) 

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        # 모델 forward pass
        # LlavaNextForConditionalGeneration은 image_sizes를 받을 수 있음
        outputs = self.model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            labels=batch.get("labels"),
            image_sizes=batch.get("image_sizes") 
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size_for_logging)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        pixel_values = batch.get("pixel_values")
        image_sizes = batch.get("image_sizes") 
        ground_truth_answers = batch.get("ground_truth_answers")

        # 답변 생성
        generated_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_sizes=image_sizes,
            max_new_tokens=ANSWER_MAX_LENGTH, # 답변 최대 길이
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id  # generate는 pad_token_id를 내부적으로 사용
        )
        
        predictions = []
        correct_predictions = 0
        
        for i in range(generated_ids.shape[0]):
            # 입력 프롬프트 부분 제외하고 디코딩
            actual_input_len = input_ids[i].ne(self.processor.tokenizer.pad_token_id).sum().item()
            decoded_prediction = self.processor.decode(generated_ids[i, actual_input_len:], skip_special_tokens=True)
            predictions.append(decoded_prediction)
            
            # 간단한 예/아니요 판단 (평가 로직은 evaluate_*.py에서 더 상세히)
            pred_binary = 1 if "예" in decoded_prediction else 0
            true_binary = 1 if "예" in ground_truth_answers[i] else 0 # 정답도 "예" 포함 여부로
            if pred_binary == true_binary: 
                correct_predictions += 1
            
            # 첫 번째 배치의 첫 번째 샘플만 자세히 로깅 (디버깅용)
            if batch_idx == 0 and self.global_rank == 0 and i == 0 : # self.global_rank는 DDP 사용 시
                 logger.info(f"\nVal Sample - Pred: '{decoded_prediction}'\nTrue: '{ground_truth_answers[i]}'")

        accuracy = correct_predictions / len(predictions) if len(predictions) > 0 else 0.0
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size_for_logging)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        global train_helmet_dataset # 전역 변수 사용
        if train_helmet_dataset:
            return DataLoader(train_helmet_dataset, collate_fn=helmet_collate_fn_train, 
                              batch_size=self.batch_size_from_config, shuffle=True, num_workers=2) # num_workers는 환경에 맞게 조절
        return None

    def val_dataloader(self):
        global val_helmet_dataset # 전역 변수 사용
        if val_helmet_dataset:
            return DataLoader(val_helmet_dataset, collate_fn=helmet_collate_fn_eval, 
                              batch_size=self.batch_size_from_config, shuffle=False, num_workers=2)
        return None

# --- 5. 학습 실행 ---
def main_train():
    global model_main, processor_main, train_helmet_dataset, val_helmet_dataset # 전역 변수 명시

    # 모델 및 프로세서 초기화 (스크립트 실행 시 한 번)
    try:
        logger.info(f"프로세서 로딩: {MODEL_ID}")
        processor_main = AutoProcessor.from_pretrained(MODEL_ID)
        processor_main.tokenizer.padding_side = "right"
        if processor_main.tokenizer.pad_token is None:
            processor_main.tokenizer.pad_token = processor_main.tokenizer.eos_token
            if processor_main.tokenizer.pad_token_id is None:
                 processor_main.tokenizer.pad_token_id = processor_main.tokenizer.eos_token_id
            logger.info(f"메인 프로세서 PAD 토큰 설정: ID {processor_main.tokenizer.pad_token_id}")
        logger.info(f"메인 프로세서 로드 완료. Pad token ID: {processor_main.tokenizer.pad_token_id}")

        logger.info(f"모델 로딩: {MODEL_ID}")
        quantization_config_to_pass = None
        if USE_QLORA:
            quantization_config_to_pass = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            logger.info("QLoRA 설정 활성화됨 (4비트 양자화).")
        
        model_main = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16,
            quantization_config=quantization_config_to_pass, low_cpu_mem_usage=True
        )
        logger.info("모델 로드 완료.")

        if USE_QLORA:
            logger.info("PEFT LoRA 설정 적용 중...")
            def find_all_linear_names_main(model_to_search): # 함수 이름 변경 (중복 방지)
                cls = torch.nn.Linear; lora_module_names = set()
                multimodal_keywords = ['multi_modal_projector', 'vision_model', 'vision_tower']
                for name, module_item in model_to_search.named_modules():
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords): continue
                    if isinstance(module_item, cls):
                        names = name.split('.'); lora_module_names.add(names[0] if len(names) == 1 else names[-1])
                if 'lm_head' in lora_module_names: lora_module_names.remove('lm_head')
                return list(lora_module_names)
            lora_target_modules_main = find_all_linear_names_main(model_main)
            logger.info(f"LoRA 타겟 모듈: {lora_target_modules_main}")
            lora_config_main = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=lora_target_modules_main, init_lora_weights="gaussian", bias="none"
            )
            model_main = prepare_model_for_kbit_training(model_main)
            model_main = get_peft_model(model_main, lora_config_main)
            logger.info("LoRA 적용된 모델 정보:"); model_main.print_trainable_parameters()
    except Exception as e:
        logger.error(f"메인 모델 또는 프로세서 초기화 중 치명적 오류: {e}", exc_info=True); return

    # 데이터셋 준비
    logger.info(f"'{DATA_JSON_PATH}' 전체 데이터셋 로딩 (기준경로: '{IMAGE_BASE_DIRECTORY}')...")
    full_custom_dataset_main = HelmetDataset(DATA_JSON_PATH, image_base_dir=IMAGE_BASE_DIRECTORY)
    
    if len(full_custom_dataset_main) > 0:
        train_val_ratio = 0.95; train_size = int(train_val_ratio * len(full_custom_dataset_main))
        val_size = len(full_custom_dataset_main) - train_size
        if val_size == 0 and train_size > 1: train_size -= 1; val_size += 1
        if train_size > 0:
            if val_size > 0:
                train_helmet_dataset, val_helmet_dataset = torch.utils.data.random_split(full_custom_dataset_main, [train_size, val_size])
                logger.info(f"학습 데이터셋: {len(train_helmet_dataset)}, 검증: {len(val_helmet_dataset)}")
            else: 
                train_helmet_dataset = full_custom_dataset_main; val_helmet_dataset = None # 전역 변수 설정
                logger.info(f"학습: {len(train_helmet_dataset)} (검증 없음)")
        else: logger.error("유효 학습 데이터셋 생성 불가."); return
    else: logger.error("데이터셋 로드 실패 또는 데이터 없음."); return


    L.seed_everything(42, workers=True) # PyTorch Lightning 시드 고정

    if not train_helmet_dataset or model_main is None or processor_main is None:
        logger.error("학습 데이터셋 또는 모델/프로세서 준비 안됨. 종료.")
        return
    
    logger.info(f"PyTorch Lightning 모듈 및 Trainer 초기화 시작...")
    pl_model_module = LlavaHelmetPLModule(model_main, processor_main, LEARNING_RATE, BATCH_SIZE)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=[0], 
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES, gradient_clip_val=GRADIENT_CLIP_VAL,
        precision="16-mixed", 
        val_check_interval=0.5 if val_helmet_dataset else 1.0, 
        limit_val_batches=10 if val_helmet_dataset else 0,   
        num_sanity_val_steps=0, # 이전 오류로 인해 0으로 설정. 안정화 후 2로 복원 가능.
    )
    
    if val_helmet_dataset is None: # 검증셋 없을 시 관련 설정 조정
        trainer.val_check_interval = 1.0 
        trainer.limit_val_batches = 0
        trainer.num_sanity_val_steps = 0

    logger.info("PyTorch Lightning Trainer 학습 시작!")
    try:
        trainer.fit(pl_model_module) # 모델과 데이터로더는 pl_model_module에서 가져옴
        logger.info("학습 정상 완료.")
        
        final_save_path = os.path.join(OUTPUT_MODEL_DIR, "final_checkpoint")
        os.makedirs(final_save_path, exist_ok=True)
        logger.info(f"모델/프로세서 저장 중: {final_save_path}")
        
        # PEFT 모델 저장 (LoRA 어댑터)
        pl_model_module.model.save_pretrained(final_save_path) 
        logger.info(f"PEFT 모델(어댑터) 저장 완료: {final_save_path}")
        
        # 프로세서 저장
        processor_main.save_pretrained(final_save_path)
        logger.info(f"프로세서 저장 완료: {final_save_path}")
        print(f"✅ LLaVA-NeXT (PL) 파인튜닝 및 저장 완료: {final_save_path}")

    except Exception as e_fit: 
        logger.error(f"Trainer.fit() 실행 중 오류: {e_fit}", exc_info=True)

if __name__ == "__main__":
    main_train()
