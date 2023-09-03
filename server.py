from flask import Flask, request, jsonify
from transformers import AutoTokenizer, pipeline

app = Flask(__name__)

# Hugging Face Transformers 모델 및 토크나이저 로드
model_name = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline("text-generation", model=model_name, device=0)  # CUDA 디바이스 0 사용


# 대화 생성 함수
def generate_response(input_text, max_length=500):
    response = generator(
        input_text,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    generated_text = response[0]["generated_text"]
    # 앞뒤 공백 제거
    generated_text = generated_text.strip()

    return generated_text


# POST 요청 핸들러
@app.route("/gen", methods=["POST"])
def generate_chat():
    try:
        # 클라이언트에서 JSON 데이터 가져오기
        request_data = request.get_json()
        user_message = request_data["text"]

        # 대화 생성
        ai_response = generate_response(user_message)

        # AI 응답을 JSON 형태로 응답
        return jsonify({"text": ai_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
