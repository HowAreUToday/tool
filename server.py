from flask import Flask, request, jsonify
from transformers import AutoTokenizer, pipeline
import torch, transformers
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# chatbot
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# llm
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chat_ai = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

chat_template = """
필수 : 한번의 답변에 한가지 질문만 한다. 여러가지 질문하지 않는다. 반말해야한다 반말로 친근하게 한다. 너는 상대방과 제일 친한 친구다! 친근한 말투로 대화해야 한다. 

0. 상대방의 상황에 있어서 상대방의 마음이 어땠는지 속마음을 묻는 질문을 종종한다.

1. 한국어를 사용해서 대화한다.

2. 첫 대답은 오늘 어땠는지 질문한다.

3. 오늘 하루에 대한 질문을 이어간다.

4. 대화할 때 여러가지 질문하지 않고 하나만 질문한다.

5. 공감할 때를 제외하고 스스로의 경험에 대해서 이야기하지 않는다.

6. 같은 주제에 대한 질문은 3번 정도만 하고 다른 질문으로 넘어간다.

7. 대답하기 편한 개방형 질문하기.
"""

# 저장된 모델 및 토크나이저 경로
model_path = "HowAreYouToday/KoT5-summarization"  # 모델 디렉토리 경로 (config.json 파일을 포함해야 함)
tokenizer_path = "psyche/KoT5-summarization"  # 토크나이저 경로 (Hugging Face 모델 허브 경로)

# 토크나이저 및 모델 로드
tokenizer2 = AutoTokenizer.from_pretrained(tokenizer_path)
model2 = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# 대화 생성 함수
def generate_response(message, history):
    history_langchain_format = []
    # 프롬프트 추가
    history_langchain_format.append(SystemMessage(content=chat_template))
    # 이전 대화 기억
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = chat_ai(history_langchain_format)
    return gpt_response.content


# 대화 생성 함수
def generate_daily_T5_response(history, max_length=100):
    input_text = ""

    for i, (human, ai) in enumerate(history):
        input_text += human
        input_text += ai

    try:
        # 입력 문장을 토큰화하여 모델에 주입
        input_ids = tokenizer2.encode(
            input_text, return_tensors="pt", max_length=5000, truncation=True
        )

        # 모델을 사용하여 요약 생성
        summary_ids = model2.generate(
            input_ids,
            max_length=400,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

        # 생성된 요약 텍스트 디코딩
        return tokenizer2.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        return str(e)


def generate_daily_OEPNAI_response(history):
    for i, (human, ai) in enumerate(history):
        history[i] = (f"사용자 : {human}", f"AI : {ai}")

    summarize_llm = OpenAI(
        temperature=0.3, model="gpt-3.5-turbo-instruct", max_tokens=512
    )

    summarize_template = """
    필수 : Ai는 대화의 보조이고 주요 관점은 사용자의 글이야. 모든 말은 반말로 한다.
    0. 친한 친구가 대신 일기를 작성해주는 느낌으로 대화내용을 기반으로 사용자의 일기 작성한다.
    1. "사용자 :", "AI : "는 태그일 뿐 일기에 포함하지 않는다.

    {texts} 대화내용을 기반으로 일기를 작성해라
    """

    summarize_prompt = PromptTemplate(
        template=summarize_template, input_variables=["texts"]
    )
    summarize_chain = LLMChain(prompt=summarize_prompt, llm=summarize_llm)

    return summarize_chain.run(history)


# POST 요청 핸들러
@app.route("/gen", methods=["POST"])
def generate_chat():
    try:
        # 클라이언트에서 JSON 데이터 가져오기
        request_data = request.get_json()
        user_message = request_data["message"]
        history = request_data["history"]
        print(history)

        # 대화 생성
        ai_response = generate_response(user_message, history)

        return jsonify({"text": ai_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

        # POST 요청 핸들러


@app.route("/makeDaily", methods=["POST"])
def generate_daily():
    try:
        # 클라이언트에서 JSON 데이터 가져오기
        request_data = request.get_json()
        # user_message 값을 로그로 출력
        user_message = request_data["history"]

        if not user_message:  # user_message가 빈 리스트 또는 빈 문자열일 때 작동
            return jsonify({"error": str(e)}), 500

        # 대화 생성
        ai_response_T5 = generate_daily_T5_response(user_message)
        ai_response_OPENAI = generate_daily_OEPNAI_response(user_message)

        response_data = {"T5_text": ai_response_T5, "OPENAI_text": ai_response_OPENAI}

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
