"""러닝 기록 MCP 서버 (stdio). 에이전트가 MCP 클라이언트로 이 서버의 도구를 호출한다.

실행: 프로젝트 루트에서 `python -m src.mcp_server`  (에이전트가 자동으로 띄우므로 보통 직접 실행하지 않는다)

도구 시그니처에 러너 ID 인자가 없다. 조회 대상은 항상 환경 변수 CURRENT_RUNNER_ID 의 러너 하나라서,
모델이 다른 러너 ID 를 넣어 타인 기록을 조회하는 일이 구조적으로 불가능하다. (가드레일 3)
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src import settings
from src.tools import load_recent_runs

mcp = FastMCP("running-records")


@mcp.tool()
def get_recent_runs(weeks: int = 4) -> dict:
    """현재 러너의 최근 러닝 기록(교육용 더미 데이터)과 주별 합계를 조회한다.

    Args:
        weeks: 최근 몇 주 분량을 조회할지 (기본 4)
    """
    return load_recent_runs(settings.current_runner_id(), weeks)


if __name__ == "__main__":
    mcp.run(transport="stdio")
