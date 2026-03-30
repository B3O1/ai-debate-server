# database.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# 로컬 테스트용 SQLite DB 설정 (나중에 애저 PostgreSQL로 주소만 바꾸면 됩니다)
SQLALCHEMY_DATABASE_URL = "sqlite:///./debate_app.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DebateSession(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_string_id = Column(String, unique=True, index=True)  # "default" 등 방 이름
    model_type = Column(String)
    atmosphere = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 세션 삭제 시 연결된 메시지도 함께 삭제(cascade)
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    role = Column(String)  # "user" 또는 "ai"
    content = Column(Text)  # 원본 대화 내용
    summary = Column(String)  # 우측/좌측 패널에 띄울 단답식 요약본
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("DebateSession", back_populates="messages")