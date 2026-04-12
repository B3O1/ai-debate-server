import os
import json
import re
from dotenv import load_dotenv

# 각 LLM 라이브러리
import groq
import google.generativeai as genai
import cohere
from sqlalchemy.orm import Session

# DB 모델 불러오기
from database import DebateSession, Message

load_dotenv()

# ==========================================
# 💡 [API 키 세팅 및 스위칭 로직]
# ==========================================
# 여러 개의 Groq 키를 리스트로 모아둡니다. (.env에 GROQ_API_KEY_2, 3 등을 추가하세요)
GROQ_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4")
]
# 값이 있는 키만 필터링해서 클라이언트 리스트 생성
GROQ_KEYS = [k for k in GROQ_KEYS if k]
groq_clients = [groq.Groq(api_key=k) for k in GROQ_KEYS]

current_groq_index = 0  # 현재 사용 중인 Groq 키 인덱스

if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY")) if os.getenv("COHERE_API_KEY") else None

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
# 💡 [3. 프롬프트 생성 (동환 님의 깔끔한 버전 유지)]
# ==========================================
def create_debate_prompt(user_claim, personality, attitude, atmosphere, topic, background, goal, condition,
                         history_text):
    p_desc = atmosphere_guide.get(atmosphere, atmosphere_guide["aggressive"])

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
            "   - 둘 다 50점 이상 (3문장 이내): '그럼 이 부분의 논리적 허점은 어쩔 겁니까?'라며 화제를 전환하세요.\n"
            "   - 둘 다 50점 미만 (3~4문장 이내): 주장의 모순점과 데이터 부재를 집요하게 따져 물으세요."
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
# 💡 [4. 메인 토론 파이프라인 (Groq 스위칭 및 ERD 우회)]
# ==========================================
async def run_debate_pipeline(user_claim, model_type, personality, attitude, atmosphere, topic, background, goal,
                              condition, db: Session, session_string_id: str):
    global model_token_usage, current_groq_index

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
        # AI 요약본에 붙여둔 점수( ||80||90 )가 있으면 앞부분 요약만 잘라서 역사에 넣습니다.
        clean_summary = msg.summary.split("||")[0] if msg.summary else ""
        history_text += f"[{role_name}]: {clean_summary}\n"

    full_prompt = create_debate_prompt(user_claim, personality, attitude, atmosphere, topic, background, goal,
                                       condition, history_text)

    raw_response = "{}"
    used_tokens = 0

    try:
        if model_type == "groq" and groq_clients:
            success = False
            # 💡 [핵심] 등록된 키 개수만큼 반복하며 시도 (에러 나면 다음 키로 스위칭)
            for _ in range(len(groq_clients)):
                try:
                    client = groq_clients[current_groq_index]
                    res1 = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": full_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.4
                    )
                    raw1 = res1.choices[0].message.content

                    res2 = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": f"아래 JSON에서 한국어는 그대로 두고 한자만 한글로 바꾸세요:\n{raw1}"}],
                        response_format={"type": "json_object"},
                        temperature=0.0
                    )
                    raw_response = res2.choices[0].message.content
                    used_tokens = res1.usage.total_tokens + res2.usage.total_tokens
                    success = True
                    break  # 성공 시 반복문 탈출!
                except Exception as e:
                    print(f"[경고] Groq API 에러 (Key Index {current_groq_index}): {e}")
                    # 에러 시 다음 키로 이동 (예: 0 -> 1 -> 2 -> 3 -> 0)
                    current_groq_index = (current_groq_index + 1) % len(groq_clients)

            if not success:
                raise Exception("등록된 모든 Groq 키가 한도 초과 또는 에러 상태입니다.")

        elif model_type == "cohere" and cohere_client:
            live_model = get_best_cohere_model()
            res = cohere_client.chat(model=live_model, message=full_prompt, temperature=0.7)
            raw_response = res.text
            used_tokens = len(raw_response) * 2

    except Exception as e:
        print(f"Pipeline Error: {e}")
        # 💡 [500 에러 방지] 통신 에러 시 준성님이 요청한 예외 JSON 반환!
        return {
            "step1_context": "통신 지연",
            "step2_attitude": "통신 지연",
            "evaluation": {"logic_score": 0, "persuasion_score": 0, "feedback": "통신 지연"},
            "ai_rebuttal": "⚠️ AI 서버 연결이 지연되었습니다. 방금 하신 말씀을 한 번 더 전송해 주세요!",
            "user_summary": "",
            "ai_summary": "",
            "user_history": [],
            "ai_history": [],
            "total_tokens": 0
        }

    result = extract_json(raw_response)
    if result:
        result['ai_rebuttal'] = sanitize_rebuttal(result.get('ai_rebuttal', ''))
        model_token_usage[model_type] += used_tokens

        # 💡 [ERD 우회 핵심] DB 스키마를 건드리지 않고, summary 문자열 뒤에 점수를 숨겨서 저장합니다. ("요약||80||90")
        eval_data = result.get('evaluation', {})
        encoded_ai_summary = f"{result.get('ai_summary', '')}||{eval_data.get('logic_score', 0)}||{eval_data.get('persuasion_score', 0)}"

        user_msg = Message(session_id=db_session.id, role="user", content=user_claim,
                           summary=result.get('user_summary', ''))
        ai_msg = Message(session_id=db_session.id, role="ai", content=result.get('ai_rebuttal', ''),
                         summary=encoded_ai_summary)

        db.add(user_msg)
        db.add(ai_msg)
        db.commit()

        all_messages = db.query(Message).filter(Message.session_id == db_session.id).order_by(Message.id.asc()).all()

        # 프론트엔드로 보낼 때는 점수 부분(||80||90)을 잘라내고 순수 요약본만 보냅니다.
        user_history_list = [msg.summary for msg in all_messages if msg.role == "user" and msg.summary]
        ai_history_list = [msg.summary.split("||")[0] for msg in all_messages if msg.role == "ai" and msg.summary]

        return {
            **result,
            "user_history": user_history_list,
            "ai_history": ai_history_list,
            "total_tokens": model_token_usage[model_type]
        }

    return {"ai_rebuttal": "JSON 파싱 에러 발생", "total_tokens": 0}


# ==========================================
# 💡 [5. 심판 평가 파이프라인 (그록 스위칭 + 콤보 시스템 결합!)]
# ==========================================
async def run_evaluation_pipeline(db: Session, session_string_id: str):
    global current_groq_index

    db_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if not db_session:
        return {"score": 0, "logic_score": 0, "persuasion_score": 0, "strengths": ["데이터 없음"],
                "weaknesses": ["대화 기록 없음"], "feedback": "토론 기록이 존재하지 않습니다.", "raw_chat": ""}

    past_messages = db.query(Message).filter(Message.session_id == db_session.id).order_by(Message.id.asc()).all()

    chat_history = ""
    groq_logic_sum = 0
    groq_persuasion_sum = 0
    ai_turn_count = 0

    # 💡 준성 님 기획: 30점 시작 & 콤보 카운터 장전!
    final_score = 30
    combo_count = 0

    for msg in past_messages:
        role_prefix = "[나]" if msg.role == "user" else "[AI]"

        # summary에 점수가 숨겨져 있다면 내용에는 안 보이게 잘라냅니다. (DB 에러 방어)
        clean_content = msg.content
        chat_history += f"{role_prefix}: {clean_content}\n"

        if msg.role == "ai":
            ai_turn_count += 1
            turn_logic = 0
            turn_persuasion = 0

            # 숨겨둔 점수 해독 (||80||90)
            if msg.summary and "||" in msg.summary:
                parts = msg.summary.split("||")
                if len(parts) == 3:
                    try:
                        turn_logic = int(parts[1])
                        turn_persuasion = int(parts[2])
                    except:
                        pass

            groq_logic_sum += turn_logic
            groq_persuasion_sum += turn_persuasion

            turn_total = turn_logic + turn_persuasion

            # 💡 [콤보 시스템 결합] 매 턴 기본 가감산 & 감점 여부 확인
            is_minus_turn = False

            if turn_total < 50:
                final_score -= 5
                is_minus_turn = True
            elif turn_total < 75:
                final_score -= 3
                is_minus_turn = True
            elif turn_total < 100:
                final_score += 0
            elif turn_total < 125:
                final_score += 3
            elif turn_total < 150:
                final_score += 4
            else:
                final_score += 5

            # 야생의 콤보 시스템 발동!
            if is_minus_turn:
                combo_count = 0  # 감점받으면 콤보 와장창 초기화
            else:
                combo_count += 1  # 감점이 아니면(방어 성공) 콤보 스택 1 증가!
                final_score += combo_count  # 보너스 팍팍 추가!

    if not chat_history.strip() or ai_turn_count == 0:
        return {"score": 0, "logic_score": 0, "persuasion_score": 0, "strengths": ["데이터 부족"], "weaknesses": ["대화 부족"],
                "feedback": "평가할 대화 턴이 부족합니다.", "raw_chat": ""}

    # 💡 [초고속 유지] 코히어 버리고 그록으로 계속 갑니다!
    prompt = (
        f"당신은 냉철한 토론 심판입니다. 아래 대화를 분석해 JSON으로만 답하세요.\n"
        f"반드시 다음 구조를 지키세요:\n"
        f'{{"strengths": ["장점1", "장점2"], "weaknesses": ["단점1", "단점2"], "feedback": "상세평"}}'
        f"\n\n[대화 기록]\n{chat_history}"
    )

    try:
        if groq_clients:
            success = False
            for _ in range(len(groq_clients)):
                try:
                    client = groq_clients[current_groq_index]
                    res = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.3
                    )
                    raw_eval = res.choices[0].message.content
                    result = extract_json(raw_eval)
                    success = True
                    break
                except Exception as e:
                    print(f"[경고] Eval Groq API 에러 (Key Index {current_groq_index}): {e}")
                    current_groq_index = (current_groq_index + 1) % len(groq_clients)

            if not success:
                raise Exception("등록된 모든 Groq 키가 한도 초과 또는 에러 상태입니다.")

            if result:
                final_score = max(0, final_score)

                avg_logic = int(groq_logic_sum / ai_turn_count)
                avg_persuasion = int(groq_persuasion_sum / ai_turn_count)

                result['score'] = final_score
                result['logic_score'] = avg_logic
                result['persuasion_score'] = avg_persuasion
                result['raw_chat'] = chat_history

                return result

        return {
            "score": max(0, final_score),
            "logic_score": int(groq_logic_sum / ai_turn_count),
            "persuasion_score": int(groq_persuasion_sum / ai_turn_count),
            "strengths": ["평가 실패"], "weaknesses": ["평가 실패"],
            "feedback": "심판 호출 실패: 응답을 해석할 수 없습니다.", "raw_chat": chat_history
        }
    except Exception as e:
        print(f"Eval Error: {e}")
        return {
            "score": max(0, final_score),
            "logic_score": int(groq_logic_sum / ai_turn_count) if ai_turn_count > 0 else 0,
            "persuasion_score": int(groq_persuasion_sum / ai_turn_count) if ai_turn_count > 0 else 0,
            "strengths": ["통신 지연"], "weaknesses": ["통신 지연"],
            "feedback": "⚠️ 통신 에러 발생: 서버가 혼잡하여 응답이 지연되었습니다.", "raw_chat": chat_history
        }


# ==========================================
# 💡 [6. 대화 초기화]
# ==========================================
def reset_memory(db: Session, session_string_id: str):
    db_session = db.query(DebateSession).filter(DebateSession.session_string_id == session_string_id).first()
    if db_session:
        db.query(Message).filter(Message.session_id == db_session.id).delete()
        db.commit()