# ai_debate.py
# ai_debate.py
import google.generativeai as genai
import groq
import cohere
import asyncio
import time
import json
import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 불러오기
load_dotenv()

# ==========================================
# 1. API 키 및 모델 설정
# ==========================================
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

# 💡 대화 원본 메모리 (이걸 드디어 코히어 심판이 씁니다!)
debate_memory = []
user_claims_summary = []
ai_rebuttals_summary = []

model_token_usage = {
    "groq": 0,
    "gemini": 0,
    "cohere": 0
}

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

# ==========================================
# 2. 메인 스파링 엔진 (요약본만 사용)
# ==========================================
# 💡 파라미터에 atmosphere(회의 분위기) 추가
async def run_debate_pipeline(user_claim, model_type="groq", debate_style="logical", atmosphere="adversarial"):
    global debate_memory, user_claims_summary, ai_rebuttals_summary, model_token_usage
    print(f"\n▶️ [AI Engine] 분석 시작... (모델: {model_type} / 성격: {debate_style} / 상황: {atmosphere})")

    # 최근 5턴까지만 기억 (토큰 최적화 유지)
    recent_u = user_claims_summary[-5:]
    recent_a = ai_rebuttals_summary[-5:]
    history_context = "".join([f"[{i+1}턴] 사용자: {u} / AI: {a}\n" for i, (u, a) in enumerate(zip(recent_u, recent_a))])
    history_text = history_context if history_context else "이전 대화 없음"
    
    style_guide = {
        "aggressive": "차가운 팩트폭격기. 억지 주장을 펴지 않고, 오직 팩트와 논리적 모순만을 바탕으로 사용자의 빈틈을 날카롭게 찌릅니다.",
        "logical": "냉철한 분석가. 감정을 배제하고 객관적인 데이터와 논리적 근거만을 바탕으로 타당성을 검토합니다.",
        "kind": "따뜻한 상담가. 부드럽고 공감하는 어조로 상대방의 의견을 존중하되, 친절하게 허점을 짚어줍니다."
    }
    
    # 💡 상황(회의 분위기) 가이드 추가
    atmosphere_guide = {
        "cooperative": "우리는 현재 [같은 목표를 달성하기 위해 협력하는 회의] 중입니다. 사용자의 의견을 무조건 비난하지 말고, 더 완벽한 결론을 내기 위해 예상되는 리스크와 보완점을 예리하게 짚어주세요.",
        "adversarial": "우리는 현재 [이해관계가 충돌하는 팽팽한 대립/협상] 중입니다. 사용자의 논리적 모순을 찾아내어 완벽히 논파하고, 당신의 반대 안건이 더 우월함을 증명하세요."
    }

    persona = style_guide.get(debate_style, style_guide['logical'])
    situation = atmosphere_guide.get(atmosphere, atmosphere_guide['adversarial'])

    full_prompt = (
        f"당신은 토론 전문가입니다.\n"
        f"[당신의 성격]: {persona}\n"
        f"[현재 상황]: {situation}\n\n"
        "[🔥 핵심 규칙]\n"
        "- 사용자가 토론의 전제나 맥락을 정정하면, 고집부리지 말고 수용하세요.\n"
        "- 아래 제공된 [요약본 히스토리]를 보고 대화의 맥락을 파악하세요.\n\n"
        f"--------------------------\n[요약본 히스토리]\n{history_text}\n--------------------------\n"
        f"[사용자의 새로운 주장]: {user_claim}\n\n"
        "반드시 아래의 순수 JSON 형식으로만 답하세요:\n"
        "{\n"
        "  \"ai_rebuttal\": \"반박 내용 (3문장 이내)\",\n"
        "  \"user_summary\": \"사용자의 이번 주장을 15자 내외로 짧게 요약\",\n"
        "  \"ai_summary\": \"당신의 이번 반박을 15자 내외로 짧게 요약\",\n"
        "  \"evaluation\": {\n"
        "    \"logic_score\": 0,\n"
        "    \"persuasion_score\": 0,\n"
        "    \"feedback\": \"개선 방향\",\n"
        "    \"is_emotional\": false\n"
        "  }\n"
        "}"
    )

    try:
        raw_response = ""
        used_tokens = 0

        if model_type == "groq":
            c = await asyncio.to_thread(groq_client.chat.completions.create, model=GROQ_MODEL_NAME, messages=[{"role": "user", "content": full_prompt}], response_format={"type": "json_object"})
            raw_response = c.choices[0].message.content
            if hasattr(c, 'usage') and c.usage: used_tokens = c.usage.total_tokens
        elif model_type == "cohere":
            r = await asyncio.to_thread(cohere_client.chat, message=full_prompt, model=COHERE_MODEL_NAME)
            raw_response = r.text
            if hasattr(r, 'meta') and r.meta and r.meta.billed_units: used_tokens = getattr(r.meta.billed_units, 'input_tokens', 0) + getattr(r.meta.billed_units, 'output_tokens', 0)
        else:
            r = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
            raw_response = r.text
            if hasattr(r, 'usage_metadata') and r.usage_metadata: used_tokens = r.usage_metadata.total_token_count

        result = extract_json(raw_response)
        
        if result:
            model_token_usage[model_type] += used_tokens
            result['total_tokens'] = model_token_usage[model_type]
            
            # 💡 코히어 평가를 위해 원문을 차곡차곡 모아둠!
            debate_memory.append(f"사용자: {user_claim}")
            debate_memory.append(f"AI: {result.get('ai_rebuttal', '')}")

            user_claims_summary.append(result.get('user_summary', '요약 누락'))
            ai_rebuttals_summary.append(result.get('ai_summary', '요약 누락'))
            
            result['user_history'] = user_claims_summary
            result['ai_history'] = ai_rebuttals_summary
            
        return result
    except Exception as e:
        print(f"❌ 에러: {e}")
        return {"ai_rebuttal": "분석 오류 발생", "evaluation": {"logic_score":0, "persuasion_score":0, "feedback":"Error", "is_emotional":False}}

# ==========================================
# 3. 심판 엔진 (코히어 전용, 대화 원문 100% 읽기)
# ==========================================
async def run_evaluation_pipeline():
    global debate_memory, model_token_usage
    
    if len(debate_memory) < 2:
        return {"score": 0, "best": "기록 부족", "worst": "기록 부족", "feedback": "평가할 대화 내용이 충분하지 않습니다."}

    # 숨겨뒀던 대화 원본을 싹 다 합칩니다!
    history_text = "\n".join(debate_memory)
    
    prompt = f"""당신은 객관적이고 예리한 토론 심판입니다.
    아래 제공된 [대화 원본]을 모두 읽고, 전체적인 흐름을 파악하여 '사용자(User)'의 토론 실력을 엄격하게 평가해주세요.

    [대화 원본]
    {history_text}

    반드시 아래 순수 JSON 형식으로만 답하세요:
    {{
      "score": 85,
      "best": "사용자의 가장 좋았던 논리나 칭찬할 만한 발언 (2문장 이내)",
      "worst": "사용자의 가장 아쉬웠던 논리적 허점이나 감정적 대처 (2문장 이내)",
      "feedback": "앞으로의 발전을 위한 심판의 종합적인 조언 (3문장 이내)"
    }}
    """
    
    try:
        # 코히어는 횟수 차감이라 원문을 왕창 던져도 안전합니다.
        r = await asyncio.to_thread(cohere_client.chat, message=prompt, model=COHERE_MODEL_NAME)
        result = extract_json(r.text)
        
        # 코히어 호출 횟수(토큰) 누적
        if hasattr(r, 'meta') and r.meta and r.meta.billed_units: 
            used_tokens = getattr(r.meta.billed_units, 'input_tokens', 0) + getattr(r.meta.billed_units, 'output_tokens', 0)
            model_token_usage["cohere"] += used_tokens
            
        if result:
            result['total_tokens'] = model_token_usage["cohere"]
            return result
    except Exception as e:
        print(f"❌ 평가 에러: {e}")
    
    return {"score": 0, "best": "오류", "worst": "오류", "feedback": "평가 중 서버 오류가 발생했습니다."}

def reset_memory():
    global debate_memory, user_claims_summary, ai_rebuttals_summary
    debate_memory.clear()
    user_claims_summary.clear()
    ai_rebuttals_summary.clear()