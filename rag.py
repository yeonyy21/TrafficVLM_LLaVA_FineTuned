import os
import base64
import openai
import chromadb
import random

# --------------------------------------------------------------------------------------
# 섹션 1: 설정 및 클라이언트 초기화
# --------------------------------------------------------------------------------------
OPENAI_API_KEY='your_key'
GPT_VISION_MODEL = "gpt-4o"
GPT_REASONING_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

KNOWLEDGE_BASE = [
    { "id": "law_50_3", "content": "도로교통법 제50조 제3항: 이륜자동차와 원동기장치자전거의 운전자는 인명보호 장구(안전모)를 착용하고 운행하여야 하며, 동승자에게도 이를 착용하도록 하여야 한다. 위반 시 범칙금은 2만원이다." },
    { "id": "law_5", "content": "도로교통법 제5조: 모든 차의 운전자는 신호기가 표시하는 신호 또는 교통정리를 하는 경찰공무원등의 신호를 따라야 한다. 신호위반 시 범칙금은 6만원, 벌점은 15점이다." },
    { "id": "law_13_1", "content": "도로교통법 제13조 제1항: 차마의 운전자는 보도와 차도가 구분된 도로에서는 차도로 통행하여야 한다. 보도 통행위반 시 범칙금은 4만원, 벌점은 10점이다." }
]

try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    client.models.list()
except openai.AuthenticationError:
    print(f"[오류] OpenAI API 키가 유효하지 않습니다.")
    exit()
except Exception as e:
    print(f"[오류] OpenAI 클라이언트 초기화 중 문제 발생: {e}")
    exit()

# --------------------------------------------------------------------------------------
# 섹션 2: 기존 로컬 모델 연동
# --------------------------------------------------------------------------------------
try:
    import infer_final
    print("✅ [연동] 로컬 헬멧 탐지 모델('infer_final.py')을 성공적으로 불러왔습니다.")
    infer_final._initialize_model_and_processor_for_inference()
    print("✅ [연동] 로컬 VLM이 추론 준비를 마쳤습니다.")
except Exception as e:
    print(f"[오류] 로컬 모델 연동 또는 초기화 중 문제 발생: {e}")
    exit()

# --------------------------------------------------------------------------------------
# 섹션 3: 파이프라인 함수 정의 (GPT-4o Vision 재도입)
# --------------------------------------------------------------------------------------

def setup_chroma_db(knowledge_base: list, chroma_client: chromadb.Client, openai_client: openai.OpenAI, collection_name="traffic_law"):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    if collection.count() == 0:
        documents = [item['content'] for item in knowledge_base]
        ids = [item['id'] for item in knowledge_base]
        response = openai_client.embeddings.create(input=documents, model=EMBEDDING_MODEL)
        embeddings = [embedding.embedding for embedding in response.data]
        collection.add(embeddings=embeddings, documents=documents, ids=ids)
    return collection

def get_rich_description_from_gpt4v(image_path: str) -> str:
    """GPT-4o를 사용하여 이미지에 대한 상세한 상황 묘사를 얻습니다."""
    print(f"🤖 [GPT-4o Vision] 이미지 심층 분석 요청: '{image_path}'")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        specific_prompt = "이것은 안전모 미착용이 의심되는 인물의 크롭 이미지입니다. 이미지 속 인물이 머리에 무엇을 쓰고 있는지, 또는 아무것도 쓰지 않았는지 사실 그대로 한국어로 묘사해주세요."
        response = client.chat.completions.create(
            model=GPT_VISION_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": specific_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[오류] GPT-4o Vision API 호출 중 오류 발생: {e}")
        return "이미지 분석 중 오류가 발생했습니다."

def query_rag_pipeline(query_text: str, collection: chromadb.Collection, openai_client: openai.OpenAI) -> str:
    """RAG 파이프라인을 실행하여 최종 답변을 생성합니다."""
    print(f"🔍 [RAG] 관련 법규 검색 실행. Query: \"{query_text[:30]}...\"")
    try:
        query_embedding = openai_client.embeddings.create(input=[query_text], model=EMBEDDING_MODEL).data[0].embedding
        retrieved_results = collection.query(query_embeddings=[query_embedding], n_results=1)
        retrieved_context = retrieved_results['documents'][0][0]

        system_prompt = "당신은 법률 전문가입니다. 주어진 [상황 묘사]와 관련 [법규]를 바탕으로, 어떤 법을 위반했는지 명시하고 처분 사항을 요약하여 리포트 형식으로 답변해주세요."
        human_prompt = f"[상황 묘사]\n{query_text}\n\n[관련 법규]\n{retrieved_context}"
        response = client.chat.completions.create(
            model=GPT_REASONING_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": human_prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[오류] RAG 파이프라인 실행 중 오류 발생: {e}")
        return "법률 분석 중 오류가 발생했습니다."

# --------------------------------------------------------------------------------------
# 섹션 4: 메인 파이프라인 실행
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    crops_dir = "data/crops"
    NUM_SAMPLES_TO_TEST = 13

    all_images = [f for f in os.listdir(crops_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.isdir(crops_dir) else []
    if not all_images:
        print(f"[실행 오류] '{crops_dir}' 폴더에 분석할 이미지가 없습니다.")
        exit()

    images_to_process = random.sample(all_images, min(len(all_images), NUM_SAMPLES_TO_TEST))

    print(f"\n🚀 VLM-RAG 교통 법규 분석 파이프라인을 시작합니다. 🚀\n{len(images_to_process)}개의 샘플을 테스트합니다.\n" + "="*60)

    chroma_client = chromadb.Client()
    law_collection = setup_chroma_db(KNOWLEDGE_BASE, chroma_client, client)

    for i, filename in enumerate(images_to_process):
        image_to_analyze = os.path.join(crops_dir, filename)

        print(f"\n[{i+1}/{len(images_to_process)}] >> 분석 시작: {image_to_analyze}")
        print("-" * 60)

        print("--- 1단계: 로컬 전문 모델로 빠른 스크리닝 ---")
        initial_check_result = infer_final.predict_helmet_with_llava_next(image_to_analyze)
        print(f"👤 [로컬 VLM 결과] 헬멧 착용 여부: '{initial_check_result}'")

        if "아니요" in initial_check_result:
            print("\n[알림] 헬멧 미착용 감지. 심층 분석을 시작합니다.\n")

            # GPT-4o Vision을 이용한 심층 분석 호출
            print("--- 2단계: 범용 VLM(GPT-4o)을 이용한 심층 상황 분석 ---")
            rich_description = get_rich_description_from_gpt4v(image_to_analyze)
            print(f"💬 [GPT-4o VLM 묘사]\n{rich_description}\n")

            if "오류" not in rich_description:
                print("--- 3단계: RAG + LLM 기반 법률 분석 ---")
                final_report = query_rag_pipeline(rich_description, law_collection, client)
                print("\n" + "="*50 + "\n✅ 최종 분석 리포트\n" + "="*50)
                print(final_report)
        else:
            print("\n[알림] 헬멧 착용으로 판단. 심층 분석을 진행하지 않습니다.")

        print("=" * 60)

    print("\n✅ 모든 이미지에 대한 파이프라인 실행이 종료되었습니다.")
