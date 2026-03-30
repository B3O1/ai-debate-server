# ai_debate.py
import google.generativeai as genai
import groq
import cohere
import asyncio
import time
import json
import os
from dotenv import load_dotenv

# 💡 DB 사용을 위한 임포트
from sqlalchemy.orm import Session
from database import DebateSession, Message

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.5-flash"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
COHERE_MODEL_NAME = "command-r-08-2024"

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(f'models/{GEMINI_MODEL_NAME}')
groq_client = groq.Groq(api_key=GROQ_API_KEY)
cohere_client = cohere.Client(api_key=COHERE_API_KEY)

# 글로벌 변수는 토큰 사용량만 남깁니다. (메모리는 DB가 담당)
model_token_usage = {"groq": 0, "gemini": 0, "cohere": 0}


def extract_json(text):
    try:
        bt = "`" * 3
        clean_text = text.replace(bt + "json", "").replace(bt, "").strip()
        start = clean_text.find('{')
        end = clean_text.rfind('}') + 1
        if start != -1 and end != 0:
            clean_text = clean_text[start:end]
        return json.loads(clean_text)
    except:
        return None


async def run_debate_pipeline(user_claim, model_type="groq", debate_style="logical", atmosphere="adversarial",
                              topic=None, background=None, goal=None, condition=None, db: Session = None,
                              session_string_id: str = "default"):
    global model_token_usage
    print(f"\n▶️ [AI Engine] 분석 시작... (모델: {model_type})")

    # 1. 세션 확인 및 생성
    current_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if not current_session:
        current_session = DebateSession(session_string_id=session_string_id, model_type=model_type,
                                        atmosphere=atmosphere)
        db.add(current_session)
        db.commit()
        db.refresh(current_session)

    # 2. 과거 5턴 기록 DB에서 불러오기
    past_msgs = db.query(Message).filter(Message.session_id == current_session.id).order_by(Message.created_at).all()
    user_summaries = [msg.summary for msg in past_msgs if msg.role == "user"]
    ai_summaries = [msg.summary for msg in past_msgs if msg.role == "ai"]

    recent_u = user_summaries[-5:]
    recent_a = ai_summaries[-5:]
    history_context = "".join([f"[{i + 1}턴] 사용자: {u} / AI: {a}\n" for i, (u, a) in enumerate(zip(recent_u, recent_a))])
    history_text = history_context if history_context else "이전 대화 없음"

    style_guide = {
        "aggressive": "차가운 팩트폭격기. 억지 주장을 펴지 않고, 오직 팩트와 논리적 모순만을 바탕으로 사용자의 빈틈을 날카롭게 찌릅니다.",
        "logical": "냉철한 분석가. 감정을 배제하고 객관적인 데이터와 논리적 근거만을 바탕으로 타당성을 검토합니다.",
        "kind": "따뜻한 상담가. 부드럽고 공감하는 어조로 상대방의 의견을 존중하되, 친절하게 허점을 짚어줍니다."
    }
    atmosphere_guide = {
        "cooperative": "우리는 현재 [같은 목표를 달성하기 위해 협력하는 회의] 중입니다. 사용자의 의견을 무조건 비난하지 말고, 더 완벽한 결론을 내기 위해 리스크와 보완점을 예리하게 짚어주세요.",
        "adversarial": "우리는 현재 [이해관계가 충돌하는 팽팽한 대립/협상] 중입니다. 사용자의 논리적 모순을 찾아내어 완벽히 논파하고, 당신의 반대 안건이 더 우월함을 증명하세요."
    }

    persona = style_guide.get(debate_style, style_guide['logical'])
    situation = atmosphere_guide.get(atmosphere, atmosphere_guide['adversarial'])

    custom_instruction = f"- [현재 토론 주제]: {topic if topic else '자유 토론'}\n"
    if background or goal or condition:
        custom_instruction += "\n[🚨 커스텀 시나리오 절대 규칙 🚨]\n당신은 아래의 시나리오와 역할을 완벽하게 연기해야 합니다.\n"
        if background: custom_instruction += f"- 배경 상황: {background}\n"
        if goal: custom_instruction += f"- 당신(AI)의 달성 목표: {goal}\n"
        if condition: custom_instruction += f"- 필수 조건 및 규칙: {condition}\n"
        custom_instruction += "위 규칙을 그 어떤 상황에서도 최우선으로 지키며 대답하세요.\n\n"

    full_prompt = (
        f"당신은 토론 전문가입니다.\n[당신의 성격]: {persona}\n[현재 상황]: {situation}\n\n{custom_instruction}"
        "[🔥 핵심 규칙]\n- 사용자가 토론의 전제나 맥락을 정정하면, 고집부리지 말고 수용하세요.\n"
        "- 아래 제공된 [요약본 히스토리]를 보고 대화의 맥락을 파악하세요.\n\n"
        f"--------------------------\n[요약본 히스토리]\n{history_text}\n--------------------------\n"
        f"[사용자의 새로운 주장]: {user_claim}\n\n"
        "반드시 아래의 순수 JSON 형식으로만 답하세요:\n"
        "{\n  \"ai_rebuttal\": \"반박 내용 (3문장 이내)\",\n"
        "  \"user_summary\": \"사용자의 핵심 주장을 명사 위주의 단답식/개조식(키워드 중심)으로 짧고 명확하게 요약하되, 수치는 절대 누락하지 말 것.\",\n"
        "  \"ai_summary\": \"당신의 핵심 반박을 명사 위주의 단답식/개조식(키워드 중심)으로 짧고 명확하게 요약하되, 수치는 절대 누락하지 말 것.\",\n"
        "  \"evaluation\": {\n    \"logic_score\": 0,\n    \"persuasion_score\": 0,\n    \"feedback\": \"개선 방향\",\n    \"is_emotional\": false\n  }\n}"
    )

    try:
        raw_response = ""
        used_tokens = 0

        if model_type == "groq":
            c = await asyncio.to_thread(groq_client.chat.completions.create, model=GROQ_MODEL_NAME,
                                        messages=[{"role": "user", "content": full_prompt}],
                                        response_format={"type": "json_object"})
            raw_response = c.choices[0].message.content
            if hasattr(c, 'usage') and c.usage: used_tokens = c.usage.total_tokens
        elif model_type == "cohere":
            r = await asyncio.to_thread(cohere_client.chat, message=full_prompt, model=COHERE_MODEL_NAME)
            raw_response = r.text
            if hasattr(r, 'meta') and r.meta and r.meta.billed_units: used_tokens = getattr(r.meta.billed_units,
                                                                                            'input_tokens',
                                                                                            0) + getattr(
                r.meta.billed_units, 'output_tokens', 0)
        else:
            r = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
            raw_response = r.text
            if hasattr(r, 'usage_metadata') and r.usage_metadata: used_tokens = r.usage_metadata.total_token_count

        result = extract_json(raw_response)

        if result:
            model_token_usage[model_type] += used_tokens
            result['total_tokens'] = model_token_usage[model_type]
            print(f"📊 [토큰 사용량] 이번 턴: {used_tokens} / 누적: {result['total_tokens']}")

            # 3. 새로운 메시지 DB에 저장
            new_user_msg = Message(session_id=current_session.id, role="user", content=user_claim,
                                   summary=result.get('user_summary', '요약 누락'))
            new_ai_msg = Message(session_id=current_session.id, role="ai", content=result.get('ai_rebuttal', ''),
                                 summary=result.get('ai_summary', '요약 누락'))
            db.add_all([new_user_msg, new_ai_msg])
            db.commit()

            user_summaries.append(new_user_msg.summary)
            ai_summaries.append(new_ai_msg.summary)
            result['user_history'] = user_summaries
            result['ai_history'] = ai_summaries

        return result
    except Exception as e:
        print(f"❌ 에러: {e}")
        return {"ai_rebuttal": "분석 오류 발생",
                "evaluation": {"logic_score": 0, "persuasion_score": 0, "feedback": "Error", "is_emotional": False}}


async def run_evaluation_pipeline(db: Session, session_string_id: str):
    global model_token_usage

    current_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if not current_session:
        return {"score": 0, "logic_score": 0, "persuasion_score": 0, "strengths": ["오류"], "weaknesses": ["오류"],
                "feedback": "세션을 찾을 수 없습니다."}

    # DB에서 원본 대화 가져오기
    past_msgs = db.query(Message).filter(Message.session_id == current_session.id).order_by(Message.created_at).all()
    if len(past_msgs) < 2:
        return {"score": 0, "logic_score": 0, "persuasion_score": 0, "strengths": ["기록 부족"], "weaknesses": ["기록 부족"],
                "feedback": "평가할 대화 내용이 충분하지 않습니다."}

    history_text = "\n".join([f"[{'나' if msg.role == 'user' else 'AI'}]: {msg.content}" for msg in past_msgs])

    prompt = f"""당신은 객관적이고 예리한 토론 심판입니다.
    아래 제공된 [대화 원본]을 모두 읽고, 전체적인 흐름을 파악하여 '사용자(나)'의 토론 실력을 엄격하게 평가해주세요.

    [대화 원본]
    {history_text}

    반드시 아래 순수 JSON 형식으로만 답하세요:
    {{
      "score": 86,
      "logic_score": 90,
      "persuasion_score": 82,
      "strengths": ["장점 요약 1 (10자 내외)", "장점 요약 2 (10자 내외)"],
      "weaknesses": ["단점(아쉬운점) 요약 1 (10자 내외)", "단점 요약 2 (10자 내외)"],
      "feedback": "앞으로의 발전을 위한 종합적인 상세 피드백 (3문장 이내)"
    }}
    """

    try:
        r = await asyncio.to_thread(cohere_client.chat, message=prompt, model=COHERE_MODEL_NAME)
        result = extract_json(r.text)

        if hasattr(r, 'meta') and r.meta and r.meta.billed_units:
            used_tokens = getattr(r.meta.billed_units, 'input_tokens', 0) + getattr(r.meta.billed_units,
                                                                                    'output_tokens', 0)
            model_token_usage["cohere"] += used_tokens

        if result:
            result['total_tokens'] = model_token_usage["cohere"]
            result['raw_chat'] = history_text
            return result
    except Exception as e:
        print(f"❌ 평가 에러: {e}")

    return {"score": 0, "logic_score": 0, "persuasion_score": 0, "strengths": ["오류"], "weaknesses": ["오류"],
            "feedback": "평가 중 서버 오류가 발생했습니다."}


def reset_memory(db: Session, session_string_id: str):
    # DB에서 세션을 날리면 연결된 메시지도 싹 다 날아갑니다 (Cascade)
    current_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if current_session:
        db.delete(current_session)
        db.commit()