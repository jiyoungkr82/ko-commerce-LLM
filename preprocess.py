import json
import pandas as pd
from deep_translator import GoogleTranslator
from datasets import Dataset

def preprocess_pipeline():
    # 1. JSON 파일 로드'
    with open("Ecommerce_FAQ_Chatbot_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_questions = data['questions']
    translator = GoogleTranslator(source='en', target='ko')
    processed_data = []

    print(f"데이터 로컬라이징 시작...(총 {len(raw_questions)}건)")

    # 2. 데이터 변환 로직 (전략 B 반영)
    for i, item in enumerate(raw_questions):
        current_num =  i+1
        try:
            # 현재 진행 상황 출력
            print(f"[{current_num}/{len(raw_questions)}] 번역 중...")
            # 영문 -> 국문 번역
            q_ko = translator.translate(item['question'])
            a_ko = translator.translate(item['answer'])
            
            # 필수 체크리스트: 커머스 페르소나 주입
            formatted_text = f"### 문의: {q_ko}\n### 답변: 안녕하세요! 제미나이 커머스 상담원입니다. {a_ko}"
            
            processed_data.append({
                "question_ko": q_ko,
                "answer_ko": a_ko,
                "text": formatted_text
            })
        except Exception as e:
            print(f"Error during translation: {e}")

    # 3. Pandas를 거쳐 Hugging Face Dataset으로 변환
    df = pd.DataFrame(processed_data)
    hf_dataset = Dataset.from_pandas(df)
    
    # 4. 결과 저장 (학습 단계에서 불러올 수 있게)
    hf_dataset.save_to_disk("./processed_data")
    print("데이터 전처리 완료 및 저장 성공!")

if __name__ == "__main__":
    preprocess_pipeline()
