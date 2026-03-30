# main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

# 💡 DB 및 평가/실행 함수 불러오기
from database import SessionLocal, engine, Base
from ai_debate import run_debate_pipeline, reset_memory, run_evaluation_pipeline

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


class DebateRequest(BaseModel):
    user_id: str = "guest"
    session_id: str = "default"
    message: str
    model_type: str = "groq"
    debate_style: str = "logical"
    atmosphere: str = "adversarial"
    topic: Optional[str] = None
    background: Optional[str] = None
    goal: Optional[str] = None
    condition: Optional[str] = None


@app.post("/api/v1/debate/chat")
async def start_debate(data: DebateRequest, db: Session = Depends(get_db)):
    print(f"\n[서버 알림] 요청 도착! (주제: {data.topic} / 커스텀 모드: {'O' if data.background else 'X'})")

    # 💡 db 객체와 session_id를 함께 넘겨줍니다.
    result = await run_debate_pipeline(
        user_claim=data.message,
        model_type=data.model_type,
        debate_style=data.debate_style,
        atmosphere=data.atmosphere,
        topic=data.topic,
        background=data.background,
        goal=data.goal,
        condition=data.condition,
        db=db,
        session_string_id=data.session_id
    )
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return result


@app.post("/api/v1/debate/evaluate")
async def evaluate_debate(db: Session = Depends(get_db)):
    print(f"\n[서버 알림] 🏁 코히어 심판 모드 가동! 전체 대화 분석 중...")
    result = await run_evaluation_pipeline(db, "default")
    return result


@app.post("/api/v1/debate/reset")
async def reset_debate_memory(db: Session = Depends(get_db)):
    reset_memory(db, "default")
    return {"status": "success", "message": "서버 메모리가 초기화되었습니다."}