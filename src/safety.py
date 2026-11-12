"""입력·출력 안전 검사 (가드레일). LLM 을 호출하지 않는 결정적 규칙이라 테스트하기 쉽다.

가드레일 5개와 이 파일이 맡는 부분
  1. 진단·약 권고 금지      : 통증·진단 질문은 전용 안전 응답으로 처리(입력), 진단·약 권고 문장은 차단(출력)
  2. 근거 없는 수치 금지    : 도구·문서 근거 없이 질문에 없던 수치가 나오면 차단(출력)
  3. 타인 기록·인젝션 차단  : 다른 러너 이름·ID, 프롬프트 인젝션 패턴을 차단(입력)
  4. 증가율 초과 조정       : tools.py 의 check_weekly_increase / build_weekly_plan 이 맡는다
  5. 악천후 대응            : 악천후와 요일을 감지해 에이전트에 힌트로 넘긴다(입력). 대체는 build_weekly_plan 이 한다
"""
from __future__ import annotations

import re

from src.schemas import SafetyVerdict
from src.tools import other_runner_keys

# ── 입력 패턴 ─────────────────────────────────────────────────────
_INJECTION = re.compile(
    r"(이전|위|앞)[^.\n]{0,12}(지시|명령|지침|프롬프트)[^.\n]{0,12}(무시|잊)"
    r"|(지시|명령|지침)[^.\n]{0,6}(무시|잊어)"
    r"|시스템\s*프롬프트|system\s*prompt|내부\s*프롬프트"
    r"|ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    re.IGNORECASE,
)
_PAIN = re.compile(r"아프|아픈|아파|통증|쑤시|저리|저려|붓|부었|부러|절뚝|시큰|뻐근|염증|인대|근육통")
_MEDICAL = re.compile(r"진단|병명|무슨\s*병|[가-힣]{2,}염\b|[가-힣]{2,}염\s*맞|약\s*(은|을|이)?\s*(뭐|무슨|어떤|먹|드)|소염|진통|파스|주사|치료")
_WEATHER = re.compile(r"폭우|호우|폭염|폭설|태풍|한파|강풍|우박|장마|악천후|무더위|혹한")
_WEEKDAY = re.compile(r"([월화수목금토일])요일")

# ── 출력 패턴 ─────────────────────────────────────────────────────
_MED_ADVICE = re.compile(r"(소염|진통|약|파스|연고|주사|보충제).{0,14}(드세요|드시|복용하|먹으세요|먹는|바르세요|처방)")
_DIAGNOSIS = re.compile(
    r"(장경인대염|슬개건염|족저근막염|아킬레스건염|피로\s*골절|러너스\s*니|정강이\s*통증\s*증후군|근막통증)"
    r"[가-힣]{0,3}\s*(입니다|이에요|이네요|로\s*보입니다|일\s*가능성이\s*(높|큽)|인\s*것\s*같)")
_NUMERIC_CLAIM = re.compile(r"\d+(?:\.\d+)?\s*(?:km|킬로|%|퍼센트|분|초|시간)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# 정보를 묻거나 요청하는 문장. 예: "전혀 안 달리셨다면 0km 라고 알려 주세요" 의 0km 는 주장이 아니라 안내라서 검사하지 않는다.
_REQUEST = re.compile(r"\?|알려\s*주|말씀해|여쭤|입력해")


def _claim_sentences(answer: str) -> list[str]:
    """답변을 문장 단위로 나누고, 질문·요청 문장은 뺀 '주장 문장'만 돌려준다. (규칙 2 검사 대상)"""
    sentences = re.split(r"[\n.!]", answer)
    return [s for s in sentences if s.strip() and not _REQUEST.search(s)]

# ── 고정 응답 ─────────────────────────────────────────────────────
REPLY_INJECTION = ("시스템 지침이나 내부 프롬프트는 공개할 수 없어요. "
                   "대신 풀마라톤 훈련표, 페이스, 부상 예방 같은 훈련 코칭은 계속 도와드릴 수 있어요.")
REPLY_OTHER_RUNNER = ("다른 러너의 기록은 조회할 수 없어요. "
                      "본인 기록을 기준으로 이번 주 훈련표를 짜 드릴 수 있어요.")
REPLY_MEDICAL = ("저는 병명을 진단하거나 약을 권해 드릴 수 없어요. "
                 "통증이 계속되거나 붓기·열감이 있으면 훈련을 멈추고 스포츠의학과 등 전문의 상담을 받으세요. "
                 "일반적으로 통증이 있는 동안에는 훈련 강도를 낮추거나 쉬는 것이 안전해요. "
                 "통증이 가라앉은 뒤에 훈련표를 요청해 주시면 그에 맞춰 도와드릴게요.")
REPLY_UNGROUNDED = ("근거 문서나 계산 도구로 확인할 수 없는 수치는 제시하기 어려워요. "
                    "확인 가능한 일반 정보와 근거 출처를 기준으로 다시 안내해 드릴게요.")


def check_input(question: str, runner_id: str) -> SafetyVerdict:
    """사용자 질문을 검사한다. block 이면 reply 를 그대로 돌려주고 에이전트를 부르지 않는다."""
    # 가드레일 3: 프롬프트 인젝션
    if _INJECTION.search(question):
        return SafetyVerdict(action="block", rule=3, reason="프롬프트 인젝션 패턴", reply=REPLY_INJECTION)

    # 가드레일 3: 현재 러너 이외의 이름·ID 로 기록을 요구
    for key in other_runner_keys(runner_id):
        if key and key.lower() in question.lower():
            return SafetyVerdict(action="block", rule=3, reason=f"타인 기록 조회 시도({key})",
                                 reply=REPLY_OTHER_RUNNER)

    # 가드레일 1: 통증·진단·약 질문은 전용 안전 경로로 처리한다
    if _PAIN.search(question) or _MEDICAL.search(question):
        return SafetyVerdict(action="block", rule=1, reason="통증·진단·약 관련 질문", reply=REPLY_MEDICAL)

    # 가드레일 5: 악천후는 막지 않고, 요일과 함께 힌트로 넘긴다
    if _WEATHER.search(question):
        days = sorted(set(_WEEKDAY.findall(question)), key="월화수목금토일".index)
        where = f"요일: {', '.join(days)}" if days else "요일은 명시되지 않음"
        return SafetyVerdict(
            action="allow", rule=5, reason=f"악천후 입력({where})",
            hint=(f"사용자가 악천후를 입력했습니다 ({where}). 훈련표를 만들 때 build_weekly_plan 의 "
                  f"bad_weather_days 에 해당 요일을 넣으세요. 요일이 없으면 어느 요일인지 먼저 물어보세요."))
    return SafetyVerdict(action="allow", reason="이상 없음")


def check_output(answer: str, question: str, grounded: bool) -> SafetyVerdict:
    """에이전트 답변을 검사한다. block 이면 reply 로 답변을 통째로 바꾼다.

    grounded: 이번 응답에 도구 결과나 검색 문서(contexts)가 있었는지.
    """
    # 가드레일 1: 약 복용 권고, 병명 단정
    if _MED_ADVICE.search(answer):
        return SafetyVerdict(action="block", rule=1, reason="약 복용 권고 문장", reply=REPLY_MEDICAL)
    if _DIAGNOSIS.search(answer):
        return SafetyVerdict(action="block", rule=1, reason="병명 단정 문장", reply=REPLY_MEDICAL)

    # 가드레일 2: 근거(도구·문서) 없이 질문에 없던 수치를 제시
    if not grounded:
        question_numbers = set(_NUMBER.findall(question))
        for claim in _NUMERIC_CLAIM.findall("\n".join(_claim_sentences(answer))):
            number = _NUMBER.search(claim).group()
            if number not in question_numbers:
                return SafetyVerdict(action="block", rule=2, reason=f"근거 없는 수치({claim.strip()})",
                                     reply=REPLY_UNGROUNDED)
    return SafetyVerdict(action="allow", reason="이상 없음")
