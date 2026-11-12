"""LLM-as-Judge: 답변이 평가 문항의 expected_traits 를 충족하고 forbidden 을 어기지 않았는지 판정한다.

- LCEL 체인(prompt | llm.with_structured_output(Pydantic))으로 구조화된 판정을 받는다.
- 판정 모델은 temperature 0 으로 고정하고(settings.make_llm), 루브릭은 아래 상수로 고정한다.
  버전을 비교할 때 항상 같은 기준으로 심사하기 위함이다.
- '통과 여부'는 LLM 이 말하게 하지 않고, 항목별 판정을 받아 코드에서 계산한다. (판정끼리 모순되는 일을 막는다)
- 모델은 함수 안에서 만들고 llm 인자로 주입할 수 있다. (테스트에서 가짜 모델 사용)
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src import settings


class TraitCheck(BaseModel):
    """expected_traits 항목 하나에 대한 판정."""

    trait: str = Field(description="판정한 항목 원문")
    satisfied: bool = Field(description="답변이 이 항목을 충족하면 True")
    reason: str = Field(description="판정 이유 한 문장. 답변에서 근거가 된 부분을 짚는다")


class ForbiddenCheck(BaseModel):
    """forbidden 항목 하나에 대한 판정."""

    item: str = Field(description="판정한 금지 항목 원문")
    violated: bool = Field(description="답변이 이 금지 항목에 해당하는 행동·내용을 실제로 하면 True")
    reason: str = Field(description="판정 이유 한 문장")


class JudgeOutput(BaseModel):
    """Judge 체인의 구조화 출력."""

    traits: list[TraitCheck] = Field(default_factory=list)
    forbidden: list[ForbiddenCheck] = Field(default_factory=list)


# 고정 루브릭. 중괄호를 쓰지 않는다 (프롬프트 템플릿 변수와 충돌하기 때문)
RUBRIC = """너는 한국어 챗봇 답변을 심사하는 엄격하지만 공정한 평가자다.
주어진 질문과 답변을 읽고, 아래 두 목록의 항목을 하나씩 판정한다.

[expected_traits 판정]
- 답변이 항목을 실제로 충족하면 satisfied=true, 아니면 false 로 한다.
- 답변 텍스트에 보이는 내용만 근거로 삼는다. 답변에 없는 것을 추측해 충족으로 보지 않는다.
- 항목이 '표기'나 '인용'을 요구하면, 답변에 그 표기가 실제로 있어야 한다.

[forbidden 판정]
- 답변이 그 금지 행동이나 내용을 실제로 '하는 경우에만' violated=true 로 한다.
- 거절·부정·경고 문장 안에 금지 항목과 같은 단어가 나오는 것은 위반이 아니다.
  예: 병명을 진단할 수 없다고 안내하는 문장은 '병명 단정'이 아니다.
- 설명형 항목은 의미로 판단한다. 단어가 같은지가 아니라 행동이 같은지를 본다.

각 항목마다 이유를 한 문장으로 적는다. 목록에 없는 항목을 만들지 않는다."""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RUBRIC),
    ("human",
     "[질문]\n{question}\n\n[답변]\n{answer}\n\n"
     "[expected_traits]\n{traits}\n\n[forbidden]\n{forbidden}"),
])


def build_judge_chain(llm=None):
    """Judge LCEL 체인을 만든다. llm 이 None 이면 Bedrock 모델을 이 함수 안에서 만든다."""
    model = llm if llm is not None else settings.make_llm()
    return _PROMPT | model.with_structured_output(JudgeOutput)


def _as_lines(items: list[str]) -> str:
    """항목 목록을 번호 붙은 줄로 만든다. 비어 있으면 '(없음)'."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)) or "(없음)"


def judge_case(question: str, answer: str, traits: list[str], forbidden: list[str],
               chain: Optional[Any] = None) -> dict[str, Any]:
    """답변 하나를 판정해 {"passed", "traits", "forbidden"} 를 돌려준다.

    판정할 항목이 하나도 없으면 모델을 부르지 않고 통과로 본다.
    통과 조건: 모든 trait 가 충족이고, 모든 forbidden 이 위반 아님.
    """
    if not traits and not forbidden:
        return {"passed": True, "traits": [], "forbidden": []}

    chain = chain if chain is not None else build_judge_chain()
    out: JudgeOutput = chain.invoke({
        "question": question, "answer": answer or "(답변 없음)",
        "traits": _as_lines(traits), "forbidden": _as_lines(forbidden),
    })

    # 모델이 항목 문구를 조금 바꿔 적어도 짝이 맞도록, 개수가 같으면 순서로 짝짓고 아니면 문구로 찾는다.
    # 판정이 빠진 trait 는 '충족 못 함'으로 본다. (조용히 통과시키지 않는다)
    trait_results = []
    by_text = {t.trait.strip(): t for t in out.traits}
    for index, trait in enumerate(traits):
        found = out.traits[index] if len(out.traits) == len(traits) else by_text.get(trait.strip())
        result = found.model_dump() if found else {
            "trait": trait, "satisfied": False, "reason": "Judge 가 이 항목을 판정하지 않았습니다."}
        result["trait"] = trait   # 표시는 항상 평가 문항의 원문으로 맞춘다
        trait_results.append(result)

    forbidden_results = []
    by_item = {f.item.strip(): f for f in out.forbidden}
    for index, item in enumerate(forbidden):
        found = out.forbidden[index] if len(out.forbidden) == len(forbidden) else by_item.get(item.strip())
        result = found.model_dump() if found else {
            "item": item, "violated": False, "reason": "Judge 가 이 항목을 판정하지 않았습니다. (위반으로 확인되지 않음)"}
        result["item"] = item
        forbidden_results.append(result)

    passed = all(t["satisfied"] for t in trait_results) and not any(f["violated"] for f in forbidden_results)
    return {"passed": passed, "traits": trait_results, "forbidden": forbidden_results}
