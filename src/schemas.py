"""Pydantic 스키마 모음. 훈련표와 안전 판정의 구조를 한곳에 정의한다."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# 요일별 활동 종류
Activity = Literal["러닝", "실내 대체", "휴식"]


class DayPlan(BaseModel):
    """하루치 훈련 계획."""

    day: str = Field(description="요일 한 글자 (월~일)")
    activity: Activity = Field(description="러닝 / 실내 대체 / 휴식")
    distance_km: Optional[float] = Field(default=None, description="러닝 거리(km). 휴식·실내 대체는 None")
    intensity: str = Field(description="강도 (이지, 롱런, 저강도, 휴식 등)")
    pace: Optional[str] = Field(default=None, description="권장 페이스 (예: 8:07~8:37 /km)")
    note: str = Field(default="", description="비고")


class WeeklyPlan(BaseModel):
    """한 주 단위 훈련표. build_weekly_plan 도구의 구조화 출력이다."""

    summary: str = Field(description="한 줄 요약 (주간 총 거리, 러닝 횟수)")
    days: list[DayPlan] = Field(description="월~일 7일치 계획")
    total_km: float = Field(description="야외 러닝 주간 총 거리(km)")
    adjustments: list[str] = Field(default_factory=list, description="안전 규칙에 따라 조정한 내용과 이유")
    assumptions: list[str] = Field(default_factory=list, description="사용자가 주지 않아 가정한 값")
    sources: list[str] = Field(default_factory=list, description="근거 문서 출처")
    provisional_params: dict = Field(default_factory=dict, description="문서 근거 확인 전의 임시 기본값")


class PlanWeek(BaseModel):
    """여러 주 계획 중 한 주."""

    week: int = Field(description="1부터 시작하는 주차")
    label: str = Field(description="이 주의 성격 (예: 증량 주, 대회 3주 전, 대회 주간)")
    target_km: float = Field(description="이 주의 목표 주간 거리(km)")
    plan: WeeklyPlan = Field(description="이 주의 요일별 훈련표")


class TrainingPlan(BaseModel):
    """대회까지 여러 주의 훈련 계획. build_training_plan 도구의 구조화 출력이다."""

    summary: str = Field(description="한 줄 요약 (주 수, 주차별 거리 흐름)")
    weeks: list[PlanWeek] = Field(description="주차별 계획")
    assumptions: list[str] = Field(default_factory=list, description="가정과 주의할 점")
    sources: list[str] = Field(default_factory=list, description="근거 문서 출처")
    provisional_params: dict = Field(default_factory=dict, description="수치 기준의 근거 구분")


class SafetyVerdict(BaseModel):
    """입력·출력 안전 검사 결과."""

    action: Literal["allow", "block"] = Field(description="block 이면 reply 를 그대로 사용자에게 돌려준다")
    rule: Optional[int] = Field(default=None, description="적용된 가드레일 번호 (1~5)")
    reason: str = Field(default="", description="판정 사유 (trace 기록용)")
    reply: str = Field(default="", description="block 일 때 사용자에게 줄 응답")
    hint: str = Field(default="", description="allow 일 때 에이전트에게 덧붙일 참고 (예: 악천후 요일)")
