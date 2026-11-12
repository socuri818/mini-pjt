# 풀마라톤 첫 도전자를 위한 부상 예방형 훈련 코치

풀마라톤을 처음 준비하는 직장인 러너를 위한 훈련 코치 에이전트입니다. 목표 기록, 가능한 요일, 최근 러닝 기록을 반영해 **한 주 또는 대회까지 여러 주의 훈련표**를 만들고, 훈련·부상 예방·영양 질문에는 **수집한 기관 문서의 근거와 출처**를 붙여 답합니다. 위험한 요청(증가율 초과, 악천후, 통증, 타인 기록 조회)은 시스템 규칙으로 걸러 안전하게 조정합니다.

- 서비스 명세: [SERVICE.md](SERVICE.md)
- 프로젝트 규칙: [CLAUDE.md](CLAUDE.md)
- 평가 결과 보고서: [evaluation/round1_report.md](evaluation/round1_report.md)

## 실행 방법

```
pip install -r requirements.txt
python -m src.retriever --rebuild          # 수집 문서를 벡터DB(chroma_db/)에 적재 (최초 1회)
uvicorn src.api_server:app --port 8000     # 서버 기동
```

- 웹 UI: http://127.0.0.1:8000/ (질문 입력 채팅 화면)
- API: `POST /query` with `{"question": "..."}` → `{"answer", "contexts", "trace"}`
- 상태 확인: `GET /health`

AWS Bedrock 자격 증명(`AWS_*` 환경 변수 또는 `~/.aws`)과 리전 `us-east-1`의 Claude Sonnet 4.5, Titan Embeddings v2 사용 권한이 필요합니다. 자격 증명은 `.env`나 환경 변수로 주며 저장소에 올리지 않습니다.

현재 러너는 환경 변수 `CURRENT_RUNNER_ID`로 고정합니다(기본 R001). 질문 본문으로 러너를 바꾸지 않습니다.

### 테스트

```
python -m unittest discover -s tests -v    # 가드레일 단위 테스트 18건 (LLM 호출 없음)
```

### 평가

```
python evaluation/run_eval.py --judge --repeat 3 --runner with_records=R001 --runner no_records=R099 \
  --server-cmd "python -m uvicorn src.api_server:app --port 8000" --out evaluation/results/round1
python evaluation/ragas_eval.py --base-url http://127.0.0.1:8000 --out evaluation/results/ragas1
```

평가 결과 원자료(`evaluation/results/`)는 로컬 산출물이라 저장소에 없습니다.

## 구조

```
src/
  agent.py        에이전트 (LangChain create_agent, 도구 호출 루프, 응답 조립, trace·contexts 수집)
  tools.py        도구: 페이스 계산, 증가율 점검, 한 주·여러 주 훈련표, 러닝 기록 계산
  retriever.py    RAG: 문서 적재(Chroma), retrieve_docs 도구
  safety.py       입력·출력 안전 검사 (결정적 규칙, 가드레일 5개)
  mcp_server.py   러닝 기록 조회 MCP 서버 (stdio)
  judge.py        LLM-as-Judge 채점 체인 (고정 루브릭, temperature 0)
  api_server.py   FastAPI: POST /query, GET /, GET /health
  schemas.py      Pydantic 스키마 (DayPlan, WeeklyPlan, TrainingPlan, SafetyVerdict)
  settings.py     설정·경로·모델 생성 (모델 객체는 함수 안에서 만든다)
  static/         웹 UI (index.html 한 장)
data/
  guides/         수집 문서 17건 (한국어 요약, 출처·이용 조건은 sources.md)
  runs_dummy.json 교육용 더미 러닝 기록 (R001, R002, R099)
evaluation/
  test_queries.csv        평가 질문 12건 (positive 5, negative 2, edge 3, guardrail 2)
  run_eval.py             HTTP로 질문을 보내 도구 호출·Judge 판정 (3회 과반)
  ragas_eval.py           RAGAS 4개 지표 측정
  ragas_references.csv    RAGAS 정답 참고문 7건 (사람 검토 필요)
  round1_report.md        1차 평가 보고서
tests/
  test_guardrails.py      가드레일 단위 테스트
```

## 동작 흐름

```
질문 → 입력 안전 검사 (인젝션·타인 기록·통증/약 차단, 악천후 힌트)
     → 에이전트 (retrieve_docs · get_recent_runs · calc_pace_zones · build_weekly_plan · build_training_plan …)
     → 출력 안전 검사 (약 권고·병명 단정 차단, 근거 없는 수치 차단)
     → 훈련표 표·근거 출처·더미 데이터 안내를 코드로 덧붙임
     → answer, contexts, trace
```

- **contexts**: 검색한 문서 조각과 도구 결과(훈련표·페이스·기록)를 함께 담습니다. 답변의 수치는 도구에서 나오므로 근거로 함께 싣습니다.
- **trace**: 입력 검사, 도구 호출과 결과, 출력 검사를 순서대로 기록합니다.

## 훈련 계획 규칙 (요약)

- **증가율**: 직전 주 대비 주 10% 이내. 넘는 요청은 한도 안으로 조정하고 이유를 밝힙니다. (근거: NHS 문서. Harvard는 이 규칙의 근거가 많지 않다고 밝힘)
- **여러 주 계획**: 증량 주 뒤에 BHF 테이퍼 가이드(최고 주간 대비 75%·50%·25~30%·20~30%)를 적용합니다.
- **고강도**: 러닝 요일이 3일 이상이면 주 중반에 임계 강도 인터벌(3×4분 → 4×4분 → 4×5분 → 4×6분, BHF 초보 계획 기준)을 넣습니다. 템포 러닝은 문서에 처방이 없어 넣지 않습니다.
- **페이스**: 목표 기록이 있으면 목표 기준(이지 +60~90초/km, 인터벌 −30~−15초/km), 없으면 최근 4주 기록의 분포로 계산합니다. 두 보정값은 문서 근거가 없는 설계상 기본값이며 응답에 표기합니다.
- **보급·수분**: 문서가 말하지 않는 세부(젤 개수, km 간격, 물 ml, 전해질 시점)는 "문서에 없다"고 밝힙니다.

## 평가 결과

### 통과율 (12개 문항, 문항당 3회 실행, 과반 통과 기준)

| 측정 | 전체 | 실행 단위 | 비고 |
|---|---|---|---|
| 1차 (round1, 2026-11-11) | **12/12 (100%)** | 35/36 | 같은 12건으로 개발 중 개선을 거친 뒤의 수치 |
| 회귀 점검 (2026-11-12) | 11/12 (92%) | 33/36 | #3 롱런 목적 검색 변동 (당시 원인) |
| 재채점 2 (검색 규칙 적용) | 12/12 (100%) | 32/36 | #3 회복. 나머지 4회는 Bedrock 오류 3회·Judge 오류 1회 |
| 2차 (round2, 2026-11-12, us-west-2) | 11/12 (92%) | 33/36 | #6 신발 질문 회귀 (3회 모두 검색 미호출). 리전만 변경 (상세: [round2_report.md](evaluation/round2_report.md)) |

### RAGAS (7개 문항: #1, #2, #3, #4, #5, #9, #10)

| 지표 | 1차 측정 (개선 전) | 검색 개선 후 (문서 10건) | 2차 (문서 16건) |
|---|---|---|---|
| context_recall | 0.798 | 0.964 | 0.964 |
| context_precision | 0.545 | 0.632 | 0.675 |
| faithfulness | 0.816 | 0.784 | **0.786** |
| answer_relevancy | 0.345 | 0.421 | 0.491 |

- faithfulness 기준 0.8은 측정마다 0.78~0.82에서 움직이고, 2차 측정(0.786)은 **기준에 못 미칩니다.**
- 샘플이 7개이고 평가 LLM이 에이전트와 같은 모델이라 점수가 후할 수 있습니다.
- answer_relevancy는 긴 한국어 답변에 낮게 나오는 경향이 있어 이 값만으로 품질을 판단하기 어렵습니다.

## 회고

**잘 된 것**
- 테스트 질문을 실제 Bedrock 모델로 여러 번 돌리며 답변 구조 문제(중복된 표, 가정 목록 반복, 제목 중복)를 코드 후처리로 고쳤습니다.
- 가드레일은 결정적 규칙으로 두어 LLM 없이 테스트할 수 있게 했습니다.
- 검색이 불안정하던 원인을 측정으로 찾았습니다. 모델이 만드는 검색어가 매번 달라 #3·#4가 흔들렸고, 검색어 규칙과 결과 누적 수정으로 context_recall이 0.80에서 0.96으로 올랐습니다.

**시도했지만 실패하거나 한계가 있었던 것**
- 검색어 규칙만 넣었을 때는 recall이 오히려 떨어졌습니다. 원인이 훈련표 요청의 검색어 변동이라는 것을 측정 뒤에 찾았습니다.
- 여러 번 검색하면 마지막 결과가 앞 결과를 덮어쓰는 버그가 있었습니다. 프롬프트가 아니라 코드 결함이었습니다.
- 동시에 여러 번 검색하면 Chroma 클라이언트가 충돌했습니다. 클라이언트를 한 번만 만들고 검색을 직렬화해 고쳤습니다.
- 공식 측정에 쓴 Bedrock 모델이 일일 토큰 한도에 여러 번 걸려, 같은 조건으로 다시 재지 못한 측정이 있습니다.

**남은 한계**
- 평가 세트가 12건이고, 12/12는 같은 케이스로 개발하며 고친 뒤의 결과입니다. 처음 보는 질문에 대한 성능이 아닙니다.
- 규칙 2(근거 없는 수치) 필터는 수치만 봅니다. "대회 전날 완전 휴식" 같은 문서 밖 일반론은 통과합니다. 차단되면 답변 전체가 일반 안내문으로 바뀝니다.
- 검색에 관련도 기준이 없어 질문과 무관한 조각이 contexts에 들어올 수 있습니다(context_precision 0.63).
- 인터벌 페이스 보정, 롱런 비중 40%는 문서 근거가 없는 설계상 기본값입니다.
- 공공누리 제4유형(변경금지) 문서 2건(질병관리청)을 요약해 두었습니다. 공개 제출 전에 인용·링크 중심으로 줄이는 것을 검토해야 합니다.

## 남은 과제
- 2차 평가(round2)는 측정 완료입니다(11/12, 개선 없음). 상세는 `evaluation/round2_report.md`에 있습니다.
- 3차(round3)는 12/12, 실행 36/36입니다. 다만 #5와 #6은 실패를 보고 고친 뒤의 결과입니다.
- 보지 않은 질문 8건(H1~H8)은 3차 코드로 8/8, 실행 24/24 통과했습니다(`evaluation/holdout_report.md`). 표본이 작고, 근거 없는 출처 표시와 계산 수치 문장 문제가 발견됐습니다.
- 신발 질문 답변의 "근거 문서" 줄에 무관한 문서가 붙는 문제가 있습니다. 검색 관련도 임계값을 넣어 해결할지 결정이 필요합니다.
- faithfulness 0.8 확보: 문서에 없는 해석(#4, #5 등)을 줄이는 방법을 검토합니다.
- #5 페이스 질문은 문서 검색 없이 계산 결과만 쓰는 경우가 있어 faithfulness가 낮게 나왔습니다. 개선안은 round2_report.md 7절에 있습니다.
- 구현하지 않은 패턴 항목(HITL, 미들웨어, Supervisor, 메모리 등)과 Docker 설정은 추가 여부를 결정해야 합니다.

## 핵심 코드 위치

- 에이전트 흐름과 응답 조립: [src/agent.py](src/agent.py) (`answer_question`, `_collect`, `_tool_context`)
- 훈련표 생성: [src/tools.py](src/tools.py) (`compute_weekly_plan`, `compute_multiweek_plan`, `compute_record_paces`)
- 가드레일: [src/safety.py](src/safety.py) (`check_input`, `check_output`)
- RAG: [src/retriever.py](src/retriever.py) (`build_index`, `retrieve`, `retrieve_docs`)
- API: [src/api_server.py](src/api_server.py)

## 데이터와 출처

- 지식 문서 17건의 출처 URL, 이용 조건, 읽은 범위는 [data/sources.md](data/sources.md)에 있습니다.
- 러닝 기록은 교육용 더미 데이터이며 응답마다 "교육용 더미 데이터입니다"를 표기합니다.
- 본 서비스는 의학적 진단이나 처방을 하지 않습니다. 통증이 있으면 전문의 상담을 권합니다.
