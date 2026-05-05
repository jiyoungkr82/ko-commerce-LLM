import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 모델 로드 함수 (캐싱 처리)
def load_model():
    base_model_id = "beomi/Llama-3-Open-Ko-8B"
    adapter_path = "./final_adapter" # 업로드한 폴더 이름
    
    # CPU 환경을 위한 경량 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16, # float32 대신 float16 사용 (용량 절반)
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    return model, tokenizer

model, tokenizer = load_model()

# 챗봇 응답 생성 함수
def respond(message, history):
    prompt = f"### 문의: {message}\n### 답변:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            temperature=0.2, 
            top_p=0.9,
            repetition_penalty=1.2
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 질문 이후의 답변 내용만 추출
    ans = response.split("### 답변:")[-1].strip()
    return ans

# Gradio 챗봇 인터페이스 실행
# 최신 버전에서는 theme을 gr.ChatInterface에 직접 넣는 대신 
# Blocks 환경에서 설정하거나 기본값으로 사용하는 것을 권장합니다.

demo = gr.ChatInterface(
    fn=respond,
    title="🤖 제미나이 커머스 AI 상담원",
    description="제미나이 커머스 전용 상담봇입니다. 계정 생성, 환불 등 궁금한 점을 물어보세요!",
    examples=["계정은 어떻게 만드나요?", "비밀번호를 잊어버렸어요."]
)

if __name__ == "__main__":
    # theme 설정이 꼭 필요하다면 launch 함수에서 지정할 수 있습니다.
    demo.launch()