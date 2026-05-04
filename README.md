# ko-commerce-LLM
QLoRA를 이용한 이커머스 특화 한국어 상담 모델 로컬라이징 및 최적화
<br><br>
**🚀 Project Overview**
<br>
본 프로젝트는 글로벌 이커머스 FAQ 데이터를 기반으로 한국어 커머스 환경에 최적화된 상담 챗봇을 구축하는 프로젝트입니다. 일반적인 LLM이 답변하기 어려운 쇼핑몰 특화 정책(회원가입, 배송, 환불 등)에 대해 일관된 한국어 말투와 정확한 정보를 제공하는 것을 목표로 합니다.

**🛠 Tech Stack**<br>
• **Model:** `beomi/Llama-3-Open-Ko-8B` (Base Model)<br>
• **Library:** `Hugging Face (Transformers, PEFT, Datasets)`, `PyTorch`<br>
• **Technique:** `QLoRA (4-bit Quantization)`, `Instruction Tuning`<br>
• **Infrastructure:** `Google Colab (T4 GPU)` / `Local RTX 3060`<br><br>
**📊 Data Pipeline**<br>
• **Source:** Kaggle Ecommerce FAQ (English JSON)<br>
• **Preprocessing:**<br>
    ◦ 영문 데이터의 한국어 로컬라이징 (상담원 페르소나 주입).<br>
    ◦ Nested JSON 구조를 `Instruction-Input-Output` 포맷의 Hugging Face Dataset으로 변환.<br>
    ◦ 한국어 어휘 확장을 고려한 `Llama-3` 전용 토크나이징.<br>
<br>
**💡 Key Selling Points**<br>
1. **리소스 최적화 (Memory Efficiency):**<br>
    ◦ 4-bit 양자화(BitsAndBytes)를 적용하여 8B 파라미터 모델을 16GB 미만의 VRAM 환경에서 학습 성공.<br>
    ◦ Full Fine-tuning 대신 **LoRA** 기법을 사용하여 학습 파라미터를 90% 이상 절감.<br>
2. **도메인 특화 성능 (Domain Adaptation):**<br>
    ◦ 일반적인 대화형 AI가 아닌, 커머스 도메인 특유의 **'다나까/해요체'** 혼용 및 전문 용어(송장 번호, 입금 확인 등) 처리 능력 강화.<br>
3. **환각 현상 방지 (Hallucination Control):**<br>
    ◦ 학습 시 질문 범위를 벗어난 답변을 지양하도록 프롬프트 가드레일 설계.<br>**📈 Results (Evaluation)**<br>
• **Quantitative:** 학습 전/후 도메인 질문에 대한 Loss 추이 시각화 (W&B 사용).<br>
• **Qualitative:** "계정 생성 방법" 질문에 대해 일반 모델의 답변과 파인튜닝된 모델의 '제미나이 커머스 전용 답변' 비교.<br>
<br>
**📂 Directory Structure**

<aside>

├── data/<br>
│   ├── raw/ (Kaggle JSON)<br>
│   └── processed/ (Hugging Face Dataset)<br>
├── scripts/<br>
│   ├── [preprocess.py](http://preprocess.py/) (Data Pipeline)<br>
│   └── [train.py](http://train.py/) (Fine-tuning Logic)<br>
├── notebooks/<br>
│   └── inference_test.ipynb<br>
└── [README.md](http://readme.md/)<br>

</aside>
