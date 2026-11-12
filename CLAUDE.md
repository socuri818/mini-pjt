# 프로젝트 규칙

## 프로젝트
- 풀마라톤 첫 도전자를 위한 부상 예방형 훈련 코치 에이전트 (미니 PJT)
- 서비스 스펙의 기준 문서는 `SERVICE.md`다. 기능·가드레일·성공 기준이 헷갈리면 먼저 이 문서를 따른다.
- 평가 기준은 `evaluation/test_queries.csv`다.

## 기술 스택
- Python, LangChain, LangGraph
- 모델은 Amazon Bedrock (ChatBedrockConverse), 모델 ID `us.anthropic.claude-sonnet-4-5-20250929-v1:0`, 리전 `us-east-1`
- 임베딩은 `amazon.titan-embed-text-v2:0`, 벡터DB는 Chroma (`./chroma_db`, git에 올리지 않는다)
- Agent 생성은 `langchain.agents` 의 `create_agent` 를 쓴다
- 러닝 기록 조회는 MCP 서버(stdio)로 노출한다

## 폴더 구조 (제출 규약. 바꾸지 않는다)
```
src/__init__.py     패키지 표시 (프로젝트 루트에서 `src.` 로 import)
src/settings.py     모델 ID·경로·현재 러너 등 설정
src/schemas.py      Pydantic 스키마 (WeeklyPlan, SafetyVerdict)
src/agent.py        메인 에이전트 그래프
src/tools.py        도메인 도구 (페이스 계산, 증가율 점검, 훈련표 생성)
src/retriever.py    RAG 파이프라인
src/safety.py       입력·출력 안전 검사 (가드레일 5개)
src/api_server.py   POST /query 서버
src/mcp_server.py   더미 러닝 기록 MCP 서버
data/               지식 문서, sources.md, runs_dummy.json
evaluation/         test_queries.csv, round1/round2 리포트, 평가·검증 스크립트
tests/              LLM을 호출하지 않는 단위 테스트
SERVICE.md  README.md  requirements.txt  .gitignore
```
- 위 목록에 있는 파일은 "요청된 파일"이다. 목록에 없는 파일은 새로 만들기 전에 먼저 묻는다.

## 주고받는 형식 (제출 규약)
- `POST /query` 로 받고 `question` 필드를 읽는다
- 답은 `answer`, `contexts`, `trace` 세 키로 돌려준다
  - `contexts`: `[{"doc_id": ..., "text": ...}]`
  - `trace`: `[{"step": ..., "input": ..., "output": ...}]`

## 도메인 규칙 (SERVICE.md 요약)
- 가드레일 5개를 항상 지킨다. 코드 변경이 이를 약하게 만들면 안 된다.
  1. 병명 진단·약 복용 권고 금지
  2. 근거 없는 수치 금지 (도구 계산값은 도구 출처를 표기)
  3. 현재 러너 이외의 기록 조회 거부, 프롬프트 인젝션 차단
  4. 주간 증가율 초과 시 한도 안으로 **조정**하고 조정 사실과 이유를 밝힘
  5. 악천후 입력 시 그날 훈련을 실내 대체 또는 휴식으로 제안
- 현재 러너는 환경 변수 `CURRENT_RUNNER_ID` 로 고정한다 (요청 본문에서 받지 않는다)
- 사용자가 직접 입력한 기록이 더미 기록과 충돌하면 사용자 입력을 우선하고 출처를 밝힌다
- 주간 증가율 한도 같은 수치는 `data/` 문서의 근거를 확인하기 전에 코드에 임의로 적지 않는다

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다
- 함수와 도구에는 한국어 docstring 을 쓰고, 주석도 한국어로 상세하게 쓴다
- 모델은 모듈 최상단에서 만들지 않는다. 함수 안에서 만들고 `llm=None` 인자로 주입할 수 있게 둔다 (테스트에서 실제 호출 없이 배선을 확인하기 위함)
- 비밀 값은 `.env` 에서 읽고 코드에 적지 않는다. `.env` 와 AWS 자격 증명은 커밋하지 않는다 (public 저장소로 제출)
- 더미 데이터에는 "교육용 더미 데이터입니다"를 표기한다

## 테스트·평가 규칙
- `tests/` 의 테스트는 실제 LLM API를 호출하지 않는다. 고정 입력과 가짜 모델만 쓴다
- 평가는 `evaluation/test_queries.csv` 로 하고, 1차(round1)와 2차(round2) 사이에 **케이스를 수정하지 않는다**
- 평가 스크립트는 케이스별로 `CURRENT_RUNNER_ID` 를 바꿀 수 있어야 한다 (기록 있음·없음 케이스)

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다
- 외부 API(Strava, Garmin, 날씨 등)를 연동하지 않는다 (범위 밖)
- 지식 문서를 실행 중에 실시간으로 크롤링하지 않는다 (사전에 수집해 `data/` 에 저장)
