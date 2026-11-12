# Round 4 자체 평가 보고서 (4차)

풀마라톤 첫 도전자를 위한 부상 예방형 훈련 코치 · 평가 세트 `evaluation/test_queries.csv` 12건 (케이스 수정 없음)

## 1. 4차에서 바뀐 것 (3차 이후)

| 변경 | 내용 | 위치 |
|---|---|---|
| 관련도 기준 (상위 1건) | 상위 1건은 거리 1.30까지 허용, 나머지는 1.25 유지 | `src/retriever.py` `TOP_RESULT_MAX_DISTANCE` |
| faithfulness 규칙 | 답변의 기간·계획이 도구 결과에 없으면 "제안"으로 표시하고, 근거 없는 판단을 지어내지 않음 | `src/agent.py` 원칙 2 |

상위 1건 기준(1.30)은 #5와 보지 않은 질문 H4의 거리 값을 보고 정했습니다. 따라서 이 두 질문은 이제 **보지 않은 질문이 아닙니다.**

## 2. 측정 조건

| 항목 | 3차 | **4차** |
|---|---|---|
| 평가 명령 | `run_eval.py --judge --repeat 3` | 동일 |
| 에이전트·Judge 모델 | Sonnet 4.5 (`us.` 프로필) | Sonnet 4.5 (**`global.` 프로필**) |
| 리전 | us-west-2 | **ap-northeast-2** |
| 평가 케이스 | 12건 | 12건 (수정 없음) |
| 수집 문서 | 17건 | 17건 |

4차는 리전과 모델 프로필이 바뀌었습니다. 같은 Sonnet 4.5이지만 엔드포인트가 다르므로, 코드 변경의 효과와 엔드포인트 차이를 완전히 분리할 수 없습니다.

## 3. 평가 결과 (12건, 문항당 3회, 과반 통과)

| 차수 | 전체 | 실행 단위 |
|---|---|---|
| 1차 | 12/12 | 35/36 |
| 2차 | 11/12 | 33/36 |
| 3차 | 12/12 | 36/36 |
| **4차** | **12/12** | **36/36**, 오류 0 |

4차 문항별 결과는 모두 3/3입니다. 3차와 같은 결과입니다.

| # | 카테고리 | 3차 | 4차 | 호출 도구 (4차) |
|---|---|---|---|---|
| 1 | positive | 3/3 | 3/3 | build_weekly_plan, calc_pace_zones, get_recent_runs, retrieve_docs |
| 2 | positive | 3/3 | 3/3 | build_weekly_plan, get_recent_runs, retrieve_docs |
| 3 | positive | 3/3 | 3/3 | retrieve_docs |
| 4 | positive | 3/3 | 3/3 | retrieve_docs |
| 5 | positive | 3/3 | 3/3 | calc_pace_zones, retrieve_docs |
| 6 | negative | 3/3 | 3/3 | retrieve_docs |
| 7 | negative | 3/3 | 3/3 | (없음) |
| 8 | edge | 3/3 | 3/3 | get_recent_runs |
| 9 | edge | 3/3 | 3/3 | build_weekly_plan, check_weekly_increase, retrieve_docs |
| 10 | edge | 3/3 | 3/3 | build_weekly_plan, get_recent_runs, retrieve_docs |
| 11 | guardrail | 3/3 | 3/3 | (입력 검사에서 차단) |
| 12 | guardrail | 3/3 | 3/3 | (입력 검사에서 차단) |

## 4. RAGAS (7개 문항: #1, #2, #3, #4, #5, #9, #10)

| 지표 | 1차 측정 | 3차 | **4차** |
|---|---|---|---|
| context_recall | 0.798 | 0.964 | **0.964** |
| context_precision | 0.545 | 0.739 | **0.739** |
| faithfulness | 0.816 | 0.763 | **0.799** |
| answer_relevancy | 0.345 | 0.326 | **0.394** |

- **faithfulness는 3차보다 올랐습니다**(0.763→0.799). 그러나 기준 0.8에는 **못 미칩니다**. 측정마다 0.78~0.82 사이에서 흔들려서, 0.8을 확정적으로 넘었다고 말할 수 없습니다.
- **answer_relevancy도 올랐습니다**(0.326→0.394). 이전 차수보다는 나아졌지만 낮은 편입니다.
- **#5는 여전히 약합니다.** 검색 근거가 1건뿐이고 faithfulness가 0.357입니다. 상위 1건 기준을 올렸는데도 #5 질문의 검색 결과가 충분히 늘지 않았습니다. Judge는 #5를 통과시켰지만, RAGAS는 근거가 부족하다고 봤습니다.

## 5. 판단

**Judge 기준으로는 3차와 같습니다.** 12/12, 36/36이며 새로 깨진 문항은 없습니다. 4차 변경은 Judge 판정을 바꾸지 않았습니다.

**RAGAS 기준으로는 소폭 개선**입니다. faithfulness와 answer_relevancy가 올랐습니다. 다만 faithfulness는 기준에 못 미칩니다.

**이 결과에는 다음 한계가 있습니다.**
- **같은 12건으로 고친 뒤의 결과입니다.** 3차와 4차의 프롬프트 수정은 2차에서 실패한 #5, #6을 보고 고친 것이고, 4차 기준값은 #5와 H4의 거리를 보고 정했습니다.
- **보지 않은 질문 8건(H1~H8)은 4차 코드로 다시 측정하지 않았습니다.** 3차 코드로는 8/8이었습니다. 4차 변경이 새 질문에서 어떻게 작동하는지는 확인되지 않았습니다.
- **리전과 모델 프로필이 바뀌었습니다.** 같은 모델이지만 엔드포인트가 달라 응답이 미세하게 다를 수 있습니다.
- **각 문항은 3회씩이라 1~2회 차이는 우연일 수 있습니다.** RAGAS는 7개 문항 1회씩 측정했습니다.
- 평가 모델(Judge, RAGAS)이 에이전트와 같은 모델입니다.

## 6. 측정 과정에서 있었던 무효 시도
4차는 처음 두 번 무효였습니다. us-west-2와 us-east-1 모두 Bedrock 일일 호출 한도를 넘어서, 36회 중 대부분이 서버 오류 또는 Judge 오류로 끝났습니다. 두 번째 시도 결과는 같은 이름의 파일을 덮어썼습니다. 현재 `round4.json`은 ap-northeast-2에서 나온 **유효한 결과**입니다(오류 0건, 36/36).

## 7. 다음에 필요한 것
1. **보지 않은 질문으로 4차 코드 검증.** H1~H8 또는 새 질문으로 다시 측정해 일반화 여부를 확인합니다.
2. **#5 근거 부족 해결.** 페이스 질문의 검색어가 문서와 멀어서 근거가 버려지는 문제입니다. 검색어를 "훈련 강도 이지 인터벌"처럼 문서 표현에 맞추거나, 페이스 질문에 한해 기준을 달리 적용하는 방법이 있습니다.
3. **faithfulness 0.8 확보.** 0.799는 기준에 거의 닿았지만, 확정적으로 넘지 못했습니다.
4. **출처 표시 문제.** 근거가 없는 질문(H4)에서도 "근거 문서" 줄에 무관한 문서가 붙을 수 있습니다. 답변에서 실제로 쓴 문서만 출처로 표시하도록 바꿔야 합니다.

## 8. 재현
```
BEDROCK_REGION=ap-northeast-2 BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  python evaluation/run_eval.py --judge --repeat 3 --runner with_records=R001 --runner no_records=R099 \
  --server-cmd "python -m uvicorn src.api_server:app --port 8000" --out evaluation/results/round4
BEDROCK_REGION=ap-northeast-2 BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  python evaluation/ragas_eval.py --out evaluation/results/ragas_r4
```
원자료: `evaluation/results/round4.md|json`, `evaluation/results/ragas_r4.md|json` (로컬 산출물, gitignore 대상)
