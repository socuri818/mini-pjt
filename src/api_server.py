"""POST /query 서버 (제출 규약).

실행(프로젝트 루트에서):
    uvicorn src.api_server:app --port 8000
현재 사용자(러너)는 요청 본문이 아니라 환경 변수 CURRENT_RUNNER_ID 로 고정한다.
    PowerShell:  $env:CURRENT_RUNNER_ID = "R001"
호출 예:
    Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post -ContentType "application/json; charset=utf-8" `
        -Body '{"question": "이번 주 훈련표 짜줘"}'
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.agent import answer_question, build_agent

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    """요청 본문: question 필드 하나."""

    question: str


def create_app(agent_factory: Optional[Callable[[], Awaitable[Any]]] = None) -> FastAPI:
    """FastAPI 앱을 만든다. agent_factory 로 에이전트 생성 방식을 바꿀 수 있다. (테스트에서 가짜 모델 주입)

    기본값은 Bedrock 모델과 MCP 서버를 쓰는 실제 에이전트다. USE_MCP=0 이면 MCP 없이 로컬 도구를 쓴다.
    """
    async def default_factory():
        return await build_agent(use_mcp=os.environ.get("USE_MCP", "1") != "0")

    factory = agent_factory or default_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = await factory()      # 서버가 뜰 때 한 번만 만든다
        yield

    app = FastAPI(title="마라톤 훈련 코치", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """웹 UI (정적 HTML 한 장). 질문은 같은 서버의 POST /query 로 보낸다."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict:
        """서버 상태 확인."""
        return {"status": "ok"}

    @app.post("/query")
    async def query(request: QueryRequest):
        """질문을 받아 {answer, contexts, trace} 를 돌려준다."""
        question = request.question.strip()
        if not question:
            return JSONResponse(status_code=422, content={"detail": "question 이 비어 있습니다."})
        try:
            return await answer_question(question, app.state.agent)
        except Exception as exc:  # noqa: BLE001 - 내부 오류 내용은 로그에만 남기고 사용자에게는 노출하지 않는다
            logger.exception("질의 처리 중 오류")
            return JSONResponse(status_code=500, content={"detail": f"처리 중 오류가 발생했습니다 ({type(exc).__name__})"})

    return app


app = create_app()
