import os
import json
import re
from dotenv import load_dotenv

# 각 LLM 라이브러리
import groq
import google.generativeai as genai
import cohere
from sqlalchemy.orm import Session

# 💡 동환 님의 실제 database.py 모델들을 완벽하게 불러옵니다!
from database import DebateSession, Message

load_dotenv()

# ==========================================
# 💡 [API 키 세팅] (하드코딩 방지 유지!)
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

model_token_usage = {"groq": 0, "gemini": 0, "cohere": 0}

# ==========================================
# 💡 [1. 통합된 AI 성격/분위기 가이드 (3종)]
# ==========================================
atmosphere_guide = {
    "aggressive": "매우 공격적이고 자비 없는 팩트폭격기입니다. 상대방의 주장에 있는 논리적 허점, 모순, 억지를 찾아내어 무자비하게 짓밟습니다. 감정적인 호소는 철저히 조롱하고 무시하며, 얼음장처럼 차갑고 날카로운 어조로 숨 막히게 압박하세요. 절대 타협하거나 동의하지 않으며, 상대를 완벽하게 논파하는 것만을 목표로 합니다. 단, 매 턴마다 똑같은 표현이나 비꼬기를 앵무새처럼 반복하지 말고, 상대의 발언에 맞춰 다채롭고 창의적인 수사의문문과 날카로운 어휘를 구사하세요.",
    "logical": "감정에 휩쓸리지 않고 오직 객관적인 데이터, 근거, 논리적 타당성만 깐깐하게 따지는 이성적인 토론자입니다.",
    "kind": "다정하고 인내심 많은 멘토입니다. 부드럽고 존중하는 어조로 대화를 이끌며, 상대방이 더 나은 논리를 펼칠 수 있도록 돕습니다."
}

# ==========================================
# 💡 [2. 코히어 및 유틸리티 함수]
# ==========================================
DYNAMIC_COHERE_MODEL = None


def get_best_cohere_model():
    global DYNAMIC_COHERE_MODEL
    if DYNAMIC_COHERE_MODEL: return DYNAMIC_COHERE_MODEL
    if not cohere_client: return "command-r-08-2024"
    try:
        models_data = cohere_client.models.list().models
        models = [m.name for m in models_data if 'chat' in m.endpoints]
        priority = ["command-r-08-2024", "c4ai-aya-expanse-8b", "c4ai-aya-expanse-32b"]
        for p in priority:
            if p in models:
                DYNAMIC_COHERE_MODEL = p
                return DYNAMIC_COHERE_MODEL
        DYNAMIC_COHERE_MODEL = models[0] if models else "command-r-08-2024"
        return DYNAMIC_COHERE_MODEL
    except:
        return "command-r-08-2024"


def extract_json(text):
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx + 1])
        return None
    except:
        return None


def remove_cjk(text: str) -> str:
    if not text: return ""
    return re.sub(r'[\u3040-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]+', '', text).strip()


def sanitize_rebuttal(text: str) -> str:
    if not text: return text
    text = remove_cjk(text)
    leak_keywords = ["당신은 최고 수준", "한국인 토론 전문가", "JSON 형식으로만"]
    for k in leak_keywords:
        if k in text:
            return "⚠️ 응답 생성 오류가 감지되었습니다. 다시 입력해주세요."
    return text


# ==========================================
# 💡 [3. 프롬프트 생성 (팀원 고도화 로직 적용)]
# ==========================================
def create_debate_prompt(user_claim, personality, attitude, atmosphere, topic, background, goal, condition,
                         history_text):
    p_desc = atmosphere_guide.get(atmosphere, atmosphere_guide["aggressive"])

    custom_scenario = ""
    if background or goal or condition:
        custom_scenario = (
            f"[상황극 배경]: {background}\n"
            f"[AI의 목표]: {goal}\n"
            f"[특수 조건]: {condition}\n"
        )

    if atmosphere == "aggressive":
        style_rules = (
            "4. [점수별 반응 및 예의 수준]: 당신이 평가한 점수에 따라 ai_rebuttal의 어투를 다르게 하세요. (점수 언급 금지, 3문장 이내 유지)\n"
            "   - 둘 다 50점 이상 (존대 100%): 정중한 어조로 다른 사각지대를 찌르며 공격하세요.\n"
            "   - 둘 중 하나만 50점 미만 (존대 80%): 가시 돋친 말투로 논리/설득력의 부족함을 쏘아붙이세요.\n"
            "   - 둘 다 50점 미만 (존대 50%): 반말을 섞으며 '여기 뭐 하러 오셨습니까?'라며 한심해하세요."
        )
    elif atmosphere == "logical":
        style_rules = (
            "4. [점수별 반응 및 문장 길이]: 당신이 평가한 점수에 따라 ai_rebuttal의 **문장 길이**를 철저히 다르게 하세요. (어투는 항상 차갑고 깐깐한 정중함 유지)\n"
            "   - 둘 다 50점 이상: '그럼 이 부분의 논리적 허점은 어쩔 겁니까?'라며 화제를 전환하세요.\n"
            "   - 둘 다 50점 미만: 주장의 모순점과 데이터 부재를 집요하게 따져 물으세요."
        )
    else:
        style_rules = (
            "4. [점수별 반응 및 멘토링]: 평가한 점수에 따라 ai_rebuttal의 어투를 다르게 하되, **끝까지 다정하고 친절한 존댓말**을 유지하세요.\n"
            "   - 둘 다 50점 미만: 멘토처럼 아주 친절하게 어떤 부분이 부족한지 피드백을 섞어 대답해 주세요."
        )

    full_prompt = (
        f"당신은 최고 수준의 한국인 토론 전문가입니다.\n"
        f"[토론 분위기 및 말투]: {p_desc}\n\n"
        f"[현재 주제]: {topic if topic else '자유 토론'}\n"
        f"{custom_scenario}"
        "[🔥 핵심 절대 규칙]\n"
        "1. [언어]: 오직 한글만 사용하세요. 한자, 중국어 절대 금지!\n"
        "2. [어투]: 자연스러운 '해요체'나 '하십시오체'를 사용하세요.\n"
        "3. [3인칭 금지]: '사용자의 주장은~'처럼 제3자 평가자처럼 말하지 말고, 1:1로 직접 반박하세요.\n"
        f"{style_rules}\n\n"
        f"[요약본 히스토리]\n{history_text}\n"
        f"[사용자의 새로운 주장]: {user_claim}\n\n"
        "🔥 반드시 아래 JSON 형식의 사고 흐름을 거쳐서 답하세요 (명시된 key값 절대 유지):\n"
        "{\n"
        "  \"step1_context\": \"생략된 주어와 진짜 의도 파악\",\n"
        "  \"step2_attitude\": \"내 점수와 규칙을 바탕으로 이번 턴의 내 말투 결정\",\n"
        "  \"evaluation\": { \"logic_score\": 0, \"persuasion_score\": 0, \"feedback\": \"...\" },\n"
        "  \"ai_rebuttal\": \"결정된 태도로 작성된 최종 반응\",\n"
        "  \"user_summary\": \"...\",\n"
        "  \"ai_summary\": \"...\"\n"
        "}"
    )
    return full_prompt


# ==========================================
# 💡 [4. 메인 토론 파이프라인 (DB 연동 유지)]
# ==========================================
async def run_debate_pipeline(user_claim, model_type, personality, attitude, atmosphere, topic, background, goal,
                              condition, db: Session, session_string_id: str):
    global model_token_usage

    db_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if not db_session:
        db_session = DebateSession(session_string_id=session_string_id, model_type=model_type, atmosphere=atmosphere)
        db.add(db_session)
        db.commit()
        db.refresh(db_session)

    past_messages = db.query(Message).filter(Message.session_id == db_session.id).order_by(Message.id.asc()).all()

    history_text = ""
    for msg in past_messages:
        role_name = "유저" if msg.role == "user" else "AI"
        history_text += f"[{role_name}]: {msg.content}\n"

    full_prompt = create_debate_prompt(user_claim, personality, attitude, atmosphere, topic, background, goal,
                                       condition, history_text)

    try:
        if model_type == "groq" and groq_client:
            res1 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"},
                temperature=0.4
            )
            raw1 = res1.choices[0].message.content

            res2 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"아래 JSON에서 한국어는 그대로 두고 한자만 한글로 바꾸세요:\n{raw1}"}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_response = res2.choices[0].message.content
            used_tokens = res1.usage.total_tokens + res2.usage.total_tokens

        elif model_type == "cohere" and cohere_client:
            live_model = get_best_cohere_model()
            res = cohere_client.chat(model=live_model, message=full_prompt, temperature=0.7)
            raw_response = res.text
            used_tokens = len(raw_response) * 2
        else:
            raw_response = "{}"
            used_tokens = 0

    except Exception as e:
        print(f"Error: {e}")
        raw_response = "{}"
        used_tokens = 0

    result = extract_json(raw_response)
    if result:
        result['ai_rebuttal'] = sanitize_rebuttal(result.get('ai_rebuttal', ''))
        model_token_usage[model_type] += used_tokens

        user_msg = Message(session_id=db_session.id, role="user", content=user_claim,
                           summary=result.get('user_summary', ''))
        ai_msg = Message(session_id=db_session.id, role="ai", content=result.get('ai_rebuttal', ''),
                         summary=result.get('ai_summary', ''))

        db.add(user_msg)
        db.add(ai_msg)
        db.commit()

        all_messages = db.query(Message).filter(Message.session_id == db_session.id).order_by(
            Message.id.asc()).all()

        user_history_list = [msg.summary for msg in all_messages if msg.role == "user" and msg.summary]
        ai_history_list = [msg.summary for msg in all_messages if msg.role == "ai" and msg.summary]

        return {
            **result,
            "user_history": user_history_list,
            "ai_history": ai_history_list,
            "total_tokens": model_token_usage[model_type]
        }

    return {"ai_rebuttal": "통신 에러", "total_tokens": 0}


# ==========================================
# 💡 [5. 심판 평가 파이프라인]
# ==========================================
async def run_evaluation_pipeline(db: Session, session_string_id: str):
    db_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if not db_session:
        return {"score": 0, "logic_score": 0, "persuasion_score": 0, "strengths": ["데이터 없음"],
                "weaknesses": ["대화 기록 없음"], "feedback": "토론 기록이 존재하지 않습니다."}

    past_messages = db.query(Message).filter(Message.session_id == db_session.id).order_by(Message.id.asc()).all()

    chat_history = ""
    for msg in past_messages:
        role_prefix = "[나]" if msg.role == "user" else "[AI]"
        chat_history += f"{role_prefix}: {msg.content}\n"

    if not chat_history.strip():
        return {"score": 0, "logic_score": 0, "persuasion_score": 0, "strengths": ["데이터 없음"],
                "weaknesses": ["대화 기록 없음"], "feedback": "토론 기록이 존재하지 않습니다."}

    live_model = get_best_cohere_model()

    prompt = (
        f"당신은 냉철한 토론 심판입니다. 아래 대화를 분석해 JSON으로만 답하세요.\n"
        f"반드시 다음 구조를 지키세요:\n"
        f'{{"score": 0~100점, "logic_score": 0~100점, "persuasion_score": 0~100점, '
        f'"strengths": ["장점1", "장점2"], "weaknesses": ["단점1", "단점2"], "feedback": "상세평"}}'
        f"\n\n[대화 기록]\n{chat_history}"
    )

    try:
        if cohere_client:
            res = cohere_client.chat(model=live_model, message=prompt, temperature=0.3)
            result = extract_json(res.text)
            if result:
                result['raw_chat'] = chat_history
                return result
        return {"score": 0, "feedback": "심판 호출 실패"}
    except Exception as e:
        print(f"Eval Error: {e}")
        return {"score": 0, "feedback": f"에러 발생: {str(e)}"}


# ==========================================
# 💡 [6. 대화 초기화]
# ==========================================
def reset_memory(db: Session, session_string_id: str):
    db_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if db_session:
        db.query(Message).filter(Message.session_id == db_session.id).delete()
        db.commit()