from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime

# 💡 DB 및 평가/실행 함수 불러오기
from database import SessionLocal, engine, Base
from ai_debate import run_debate_pipeline, reset_memory, run_evaluation_pipeline
from typing import Optional, List # 💡 맨 위 import 부분에 List가 없다면 꼭 추가해주세요!

# ... (기존 DebateRequest, SessionRequest 코드) ...

# ==========================================
# ==========================================
# 💡 [새로 추가] 스웨거 명세용 응답(Response) 모델
# ==========================================
class ChatEvaluation(BaseModel):
    logic_score: int
    persuasion_score: int
    feedback: str


class ChatResponse(BaseModel):
    # 👇 새로 추가된 AI 속마음 데이터 (에러 안 나게 Optional 처리)
    step1_context: Optional[str] = None
    step2_attitude: Optional[str] = None

    # 기존 데이터들
    ai_rebuttal: str
    user_summary: str
    ai_summary: str
    evaluation: ChatEvaluation
    user_history: List[str]
    ai_history: List[str]
    total_tokens: int
    timestamp: str


class EvaluateResponse(BaseModel):
    # 👇 새로 추가된 심판의 속마음 데이터 (에러 안 나게 Optional 처리)
    step1_turn_by_turn: Optional[str] = None
    step2_ai_fault_check: Optional[str] = None

    # 기존 데이터들
    score: int
    logic_score: int
    persuasion_score: int
    strengths: List[str]
    weaknesses: List[str]
    feedback: str
    raw_chat: str
app = FastAPI()

# 시작 시 DB 테이블 자동 생성
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 💡 DB 세션 생성기
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 💡 프론트엔드 데이터 수신 규격 (팀원분이 만든 매트릭스 적용)
class DebateRequest(BaseModel):
    user_id: str = "guest"
    session_id: str = "default"
    message: str
    model_type: str = "groq"
    personality: str = "cynical"      # 성격 (말투/온도)
    attitude: str = "egoist"          # 태도 (논리/가치관)
    atmosphere: str = "adversarial"   # 상황 및 분위기
    topic: Optional[str] = None
    background: Optional[str] = None
    goal: Optional[str] = None
    condition: Optional[str] = None

@app.post("/api/v1/debate/chat", response_model=ChatResponse)
async def start_debate(data: DebateRequest, db: Session = Depends(get_db)):
    print(f"\n[서버 알림] 요청 도착! (주제: {data.topic} / 성격: {data.personality} / 태도: {data.attitude})")

    # 💡 db 객체와 함께 새로운 변수들을 넘겨줍니다.
    result = await run_debate_pipeline(
        user_claim=data.message,
        model_type=data.model_type,
        personality=data.personality,
        attitude=data.attitude,
        atmosphere=data.atmosphere,
        topic=data.topic,
        background=data.background,
        goal=data.goal,
        condition=data.condition,
        db=db,                              # DB 추가
        session_string_id=data.session_id   # 세션 ID 추가
    )
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return result
# (위쪽 코드는 그대로 둡니다)

# 💡 평가/리셋용 데이터 규격 추가 (방 번호만 받음)
class SessionRequest(BaseModel):
    session_id: str = "default"

@app.post("/api/v1/debate/evaluate", response_model=EvaluateResponse)
async def evaluate_debate(req: SessionRequest, db: Session = Depends(get_db)):
    print(f"\n[서버 알림] 🏁 {req.session_id} 방 코히어 심판 모드 가동!")
    result = await run_evaluation_pipeline(db, req.session_id)
    return result

@app.post("/api/v1/debate/reset")
async def reset_debate_memory(req: SessionRequest, db: Session = Depends(get_db)):
    reset_memory(db, req.session_id)
    return {"status": "success", "message": f"[{req.session_id}] 방 서버 메모리가 초기화되었습니다."}