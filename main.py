# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# 💡 평가 함수(run_evaluation_pipeline) 추가로 불러오기!
from ai_debate import run_debate_pipeline, reset_memory, run_evaluation_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DebateRequest(BaseModel):
    user_id: str = "guest"
    session_id: str = "default"
    message: str
    model_type: str = "groq"
    debate_style: str = "logical"
    atmosphere: str = "adversarial" # 💡 추가됨: 회의 분위기 (협력/대립)

@app.post("/api/v1/debate/chat")
async def start_debate(data: DebateRequest):
    print(f"\n[서버 알림] 요청 도착! (모델: {data.model_type} / 성격: {data.debate_style} / 상황: {data.atmosphere})")
    # 💡 상황 데이터도 같이 넘겨줌
    result = await run_debate_pipeline(data.message, data.model_type, data.debate_style, data.atmosphere)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return result

# 💡 추가됨: 코히어 심판에게 최종 성적표를 요구하는 통신구
@app.post("/api/v1/debate/evaluate")
async def evaluate_debate():
    print(f"\n[서버 알림] 🏁 코히어 심판 모드 가동! 전체 대화 분석 중...")
    result = await run_evaluation_pipeline()
    return result

@app.post("/api/v1/debate/reset")
async def reset_debate_memory():
    reset_memory()
    return {"status": "success", "message": "서버 메모리가 초기화되었습니다."}