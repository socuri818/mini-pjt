"""도메인 도구: 페이스 계산, 증가율 점검, 훈련표 생성, 러닝 기록 조회.

구성
  - compute_* / load_*  : 순수 함수. LLM 없이 테스트할 수 있다.
  - @tool 래퍼          : 에이전트가 부르는 도구. 이름은 SERVICE.md 의 도구 표와 같다.

[중요] 아래 기본값 중 주간 증가율 한도(10%)는 수집 문서(01 NHS, 02 Harvard)로 근거를 확인했다.
       이지 페이스 보정과 롱런 비중은 문서 근거가 없는 이 서비스의 설계상 기본값이다.
       값을 바꿀 때는 이 한 곳만 고치고, 도구 출력의 provisional_params 가 이 구분을 함께 싣는다.
       (가드레일 2: 근거 없는 수치 금지 → 근거 유무를 출력에 항상 함께 싣는다)
"""
from __future__ import annotations

import json
import statistics
from typing import Any, Optional

from langchain_core.tools import tool

from src import settings
from src.schemas import DayPlan, PlanWeek, TrainingPlan, WeeklyPlan

# ── 기본값 ────────────────────────────────────────────────────────
WEEKLY_INCREASE_LIMIT_PCT = 10.0          # 직전 주 대비 주간 증가율 한도(%) — 문서 근거 있음(01, 02)
EASY_PACE_OFFSET_SEC = (60, 90)           # 이지 페이스 = 마라톤 목표 페이스 + 이 범위(초/km) — 설계상 가정
INTERVAL_PACE_OFFSET_SEC = (-30, -15)     # 인터벌 페이스 = 마라톤 목표 페이스 + 이 범위(초/km, 더 빠름) — 설계상 가정
LONG_RUN_SHARE = 0.4                      # 주간 거리 중 롱런이 차지하는 비율 — 설계상 가정
PROVISIONAL_PARAMS = {
    "weekly_increase_limit_pct": WEEKLY_INCREASE_LIMIT_PCT,
    "easy_pace_offset_sec_per_km": list(EASY_PACE_OFFSET_SEC),
    "interval_pace_offset_sec_per_km": list(INTERVAL_PACE_OFFSET_SEC),
    "long_run_share": LONG_RUN_SHARE,
    "status": "주 10% 증가 한도는 수집 문서(NHS·Harvard) 근거, 이지·인터벌 페이스 보정과 롱런 비중은 "
              "이 서비스의 설계상 기본값(문서 근거 없음)",
}

# 테이퍼(감량) 주간 비율: 04번 문서(BHF 테이퍼 가이드)의 '최고 훈련 주간 대비' 표. 범위는 중간값을 쓴다.
TAPER_STEPS = [(0.75, "대회 3주 전"), (0.50, "대회 2주 전"),
               (0.275, "대회 1주 전"),     # 문서: 25~30%
               (0.25, "대회 주간")]        # 문서: 20~30%
# 요일을 말하지 않고 주당 러닝 횟수만 말했을 때 쓰는 기본 요일 배치 (설계상 가정, 휴식일이 사이에 끼도록 배치)
RUN_DAY_PRESETS = {2: ["수", "일"], 3: ["월", "수", "일"], 4: ["월", "수", "금", "일"], 5: ["월", "화", "목", "금", "일"]}
# 임계 강도 인터벌 진행 (세트 수, 세트당 분): 03번 문서(BHF 초보 계획) "3×4분 → 4×4분 → 4×5분 → 4×6분".
# 문서에 페이스 수치와 주차별 배치는 없으므로 강도는 '임계 강도'라고만 표기하고, 주차는 계획 순서대로 한 단계씩 올린다.
INTERVAL_LADDER = [(3, 4), (4, 4), (4, 5), (4, 6)]
MAX_PLAN_WEEKS = 12                     # 한 번에 만들 수 있는 최대 주 수 (설계상 제한)
BHF_BEGINNER_PLAN_WEEKS = 14              # 03번 문서(BHF 초보 풀마라톤 계획)의 구성 주 수

MARATHON_KM = 42.195                      # 풀마라톤 거리 (사실값)
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
DEFAULT_AVAILABLE_DAYS = ["월", "수", "금", "일"]


# ══════════════════════════════════════════════════════════════════
# 순수 함수
# ══════════════════════════════════════════════════════════════════
def _fmt_pace(seconds_per_km: float) -> str:
    """초/km 를 'M:SS' 문자열로 바꾼다."""
    minutes, seconds = divmod(round(seconds_per_km), 60)
    return f"{minutes}:{seconds:02d}"


def normalize_days(days: Optional[list[str]]) -> list[str]:
    """'수요일', '수' 같은 입력을 요일 한 글자로 통일하고 월~일 순으로 정렬한다."""
    picked = {str(d).strip()[:1] for d in (days or []) if str(d).strip()[:1] in DAYS}
    return [d for d in DAYS if d in picked]


def compute_pace_zones(goal_hours: float) -> dict[str, Any]:
    """목표 완주 시간(시간 단위)으로 마라톤 목표 페이스, 이지 페이스 구간, 인터벌 페이스 구간을 계산한다."""
    if goal_hours is None or goal_hours <= 0:
        return {"needs_info": True, "message": "목표 완주 시간(시간 단위)이 필요합니다. 예: 5시간이면 5"}
    marathon_pace = goal_hours * 3600 / MARATHON_KM
    low, high = EASY_PACE_OFFSET_SEC
    fast, slow = INTERVAL_PACE_OFFSET_SEC
    return {
        "source": "calc_pace_zones (계산 도구)",
        "goal_hours": goal_hours,
        "marathon_pace_per_km": _fmt_pace(marathon_pace),
        "easy_pace_per_km": f"{_fmt_pace(marathon_pace + low)}~{_fmt_pace(marathon_pace + high)}",
        "interval_pace_per_km": f"{_fmt_pace(marathon_pace + fast)}~{_fmt_pace(marathon_pace + slow)}",
        "provisional_params": {"easy_pace_offset_sec_per_km": list(EASY_PACE_OFFSET_SEC),
                               "interval_pace_offset_sec_per_km": list(INTERVAL_PACE_OFFSET_SEC),
                               "status": PROVISIONAL_PARAMS["status"]},
    }


def compute_weekly_increase(prev_week_km: Optional[float], requested_km: float) -> dict[str, Any]:
    """직전 주 실제 거리 대비 요청 거리가 증가율 한도를 넘는지 점검하고, 넘으면 한도로 조정한다."""
    if prev_week_km is None or prev_week_km <= 0:
        return {"needs_info": True,
                "message": "직전 주 실제 주행거리를 알 수 없습니다. 사용자에게 지난주에 몇 km 달렸는지 물어보세요."}
    allowed = round(prev_week_km * (1 + WEEKLY_INCREASE_LIMIT_PCT / 100), 1)
    exceeded = requested_km > allowed
    return {
        "source": "check_weekly_increase (계산 도구)",
        "prev_week_km": prev_week_km,
        "requested_km": requested_km,
        "allowed_max_km": allowed,
        "exceeded": exceeded,
        "adjusted_km": allowed if exceeded else requested_km,
        "limit_pct": WEEKLY_INCREASE_LIMIT_PCT,
        "provisional_params": {"weekly_increase_limit_pct": WEEKLY_INCREASE_LIMIT_PCT,
                               "status": PROVISIONAL_PARAMS["status"]},
    }


def _interval_note(reps: int, minutes: int) -> str:
    """인터벌 세션 설명. 90초 회복은 문서에서 3×4분 세션에만 명시되어 있어 그 경우에만 적는다."""
    rest = "사이 90초 조깅 회복" if (reps, minutes) == INTERVAL_LADDER[0] else "사이 조깅 회복"
    return f"임계 강도 {reps}×{minutes}분, {rest}"


def compute_weekly_plan(
    prev_week_km: Optional[float],
    requested_weekly_km: Optional[float] = None,
    available_days: Optional[list[str]] = None,
    goal_hours: Optional[float] = None,
    bad_weather_days: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    interval: Optional[tuple[int, int]] = None,
    record_paces: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """안전 규칙을 적용한 한 주 단위 훈련표(WeeklyPlan)를 만든다.

    - 직전 주 거리가 없으면 계획을 만들지 않고 질문이 필요하다고 알린다. (거리를 임의로 가정하지 않는다)
    - 요청 거리가 증가율 한도를 넘으면 한도 안으로 조정하고, 조정 사실을 adjustments 에 남긴다.
    - 악천후 요일은 야외 러닝 대신 '실내 대체'로 바꾸고, 나머지 요일 계획은 유지한다.
    """
    if prev_week_km is None or prev_week_km <= 0:
        return {"needs_info": True, "missing": ["prev_week_km"],
                "message": "직전 주 실제 주행거리가 필요합니다. 사용자에게 지난주에 몇 km 달렸는지 물어보세요."}

    assumptions: list[str] = []
    adjustments: list[str] = []

    days = normalize_days(available_days)
    if not days:
        days = list(DEFAULT_AVAILABLE_DAYS)
        assumptions.append("가능한 요일이 입력되지 않아 월·수·금·일로 가정했습니다.")
    if requested_weekly_km is None or requested_weekly_km <= 0:
        requested_weekly_km = prev_week_km
        assumptions.append(f"요청 거리가 없어 직전 주와 같은 {prev_week_km:g}km로 계획했습니다.")

    # 증가율 점검: 넘으면 한도 안으로 조정한다 (가드레일 4)
    increase = compute_weekly_increase(prev_week_km, requested_weekly_km)
    total = increase["adjusted_km"]
    if increase["exceeded"]:
        adjustments.append(
            f"요청하신 {requested_weekly_km:g}km는 직전 주 {prev_week_km:g}km 대비 증가율 한도"
            f"({increase['limit_pct']:g}%, 최대 {increase['allowed_max_km']:g}km)를 넘어 "
            f"{increase['adjusted_km']:g}km로 조정했습니다. 급격한 증가는 부상 위험을 키우기 때문입니다.")

    # 거리 배분: 마지막 가능 요일을 롱런으로, 나머지는 균등 분배
    shares = {d: (1.0 if len(days) == 1 else (1 - LONG_RUN_SHARE) / (len(days) - 1)) for d in days}
    long_day = days[-1]
    if len(days) > 1:
        shares[long_day] = LONG_RUN_SHARE
    distances = {d: round(total * s, 1) for d, s in shares.items()}
    distances[long_day] = round(distances[long_day] + (total - sum(distances.values())), 1)  # 반올림 오차 보정

    # 이지·조깅 페이스: 사용자가 말한 목표 기록이 있으면 그 계산값(사용자 입력 우선), 없으면 최근 러닝 기록에서 구한다.
    easy_pace = interval_pace = None
    pace_from_goal = bool(goal_hours)
    if goal_hours:
        zones = compute_pace_zones(goal_hours)
        easy_pace = zones.get("easy_pace_per_km")
        interval_pace = zones.get("interval_pace_per_km")   # 목표 기록 기준이면 이지·인터벌 모두 같은 기준에서 계산
        assumptions.append(
            f"이지·인터벌 페이스는 목표 {goal_hours:g}시간(마라톤 페이스 {zones.get('marathon_pace_per_km')}/km) 기준으로 "
            f"계산했습니다. 이지는 마라톤 페이스보다 {EASY_PACE_OFFSET_SEC[0]}~{EASY_PACE_OFFSET_SEC[1]}초/km 느리게, "
            f"인터벌은 {-INTERVAL_PACE_OFFSET_SEC[1]}~{-INTERVAL_PACE_OFFSET_SEC[0]}초/km 빠르게 잡은 설계상 기본값입니다.")
    elif record_paces:
        easy_pace, interval_pace = record_paces["easy"], record_paces["interval"]
        assumptions.append(
            f"이지·조깅 페이스는 최근 {record_paces['weeks']}주 러닝 기록 {record_paces['n_runs']}회의 "
            "중간 구간(페이스 25~75% 지점)으로 잡았습니다.")
    else:
        assumptions.append("목표 기록과 러닝 기록이 없어 페이스는 제시하지 않았습니다.")

    bad_days = set(normalize_days(bad_weather_days))
    plans: list[DayPlan] = []
    for day in DAYS:
        if day not in days:
            plans.append(DayPlan(day=day, activity="휴식", intensity="휴식", note="회복일"))
        elif day in bad_days:
            plans.append(DayPlan(
                day=day, activity="실내 대체", intensity="저강도",
                note="악천후로 야외 러닝 대신 러닝머신·크로스트레이닝 또는 휴식을 권합니다."))
            adjustments.append(f"{day}요일은 악천후 입력에 따라 야외 러닝을 실내 대체로 바꿨습니다.")
        else:
            plans.append(DayPlan(
                day=day, activity="러닝", distance_km=distances[day],
                intensity="롱런" if (day == long_day and len(days) > 1) else "이지",
                pace=easy_pace, note=""))

    # 고강도(임계 강도 인터벌) 세션: 러닝 요일이 3일 이상일 때 두 번째 러닝 요일(주 중반)에 한 번 넣는다.
    if interval and len(days) >= 3:
        quality_day = days[1]
        for p in plans:
            if p.day == quality_day and p.activity == "러닝":
                p.intensity, p.note = "인터벌(임계 강도)", _interval_note(*interval)
                assumptions.append(
                    f"고강도 훈련: BHF 초보 계획의 임계 강도 인터벌을 주 중반({quality_day}요일) 러닝에 넣었습니다. "
                    "통증이나 피로가 있으면 이지 러닝으로 바꾸세요.")
                if interval_pace:
                    p.pace = f"인터벌 {interval_pace}" + (f" / 회복 조깅 {easy_pace}" if easy_pace else "")
                    how = ("목표 기록에서 계산한" if pace_from_goal else "최근 기록 중 빠른 쪽 25% 구간으로 잡은")
                    assumptions.append(
                        f"인터벌 페이스는 {how} 이 서비스의 설계상 기본값입니다. 문서에 임계 강도 페이스 수치가 없어 "
                        "실제 임계 강도와 다를 수 있으니 몸 상태에 맞게 조절하세요.")

    outdoor = round(sum(p.distance_km or 0 for p in plans), 1)
    runs = sum(1 for p in plans if p.activity == "러닝")
    plan = WeeklyPlan(
        summary=f"주간 야외 러닝 {outdoor:g}km, {runs}회",
        days=plans, total_km=outdoor, adjustments=adjustments, assumptions=assumptions,
        sources=list(sources or []), provisional_params=PROVISIONAL_PARAMS)
    return plan.model_dump()


def compute_multiweek_plan(
    weeks_until_race: Optional[int],
    prev_week_km: Optional[float],
    available_days: Optional[list[str]] = None,
    goal_hours: Optional[float] = None,
    sources: Optional[list[str]] = None,
    runs_per_week: Optional[int] = None,
    include_intervals: bool = True,
    record_paces: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """대회까지 여러 주의 훈련 계획(TrainingPlan)을 만든다.

    - 앞쪽 주는 직전 주에서 매주 증가율 한도(10%)만큼 늘리는 증량 주, 마지막 최대 4주는 테이퍼 주다.
    - 테이퍼 비율은 04번 문서(BHF)의 '최고 훈련 주간 대비' 값이다. 최고 주간은 마지막 증량 주(없으면 직전 주)다.
    - 매주 한 주 단위 훈련표(compute_weekly_plan)를 만들어 안전 규칙이 그대로 적용된다.
    """
    if prev_week_km is None or prev_week_km <= 0:
        return {"needs_info": True, "missing": ["prev_week_km"],
                "message": "직전 주 실제 주행거리가 필요합니다. 사용자에게 지난주에 몇 km 달렸는지 물어보세요."}
    if weeks_until_race is None or int(weeks_until_race) < 1:
        return {"needs_info": True, "missing": ["weeks_until_race"],
                "message": "대회까지 남은 주 수가 필요합니다. 예: 한 달 정도 남았으면 4"}

    notes: list[str] = []
    if not normalize_days(available_days) and runs_per_week in RUN_DAY_PRESETS:
        available_days = RUN_DAY_PRESETS[runs_per_week]
        notes.append(f"달릴 요일을 말씀하지 않으셔서 주 {runs_per_week}회를 {'·'.join(available_days)}요일로 배치했습니다.")
    weeks = int(weeks_until_race)
    if weeks > MAX_PLAN_WEEKS:
        notes.append(f"한 번에 {MAX_PLAN_WEEKS}주까지만 계획을 만들 수 있어 {weeks}주 중 마지막 {MAX_PLAN_WEEKS}주를 계획했습니다.")
        weeks = MAX_PLAN_WEEKS
    taper = TAPER_STEPS[-min(len(TAPER_STEPS), weeks):]
    build_n = weeks - len(taper)

    # 주차별 (라벨, 목표 거리, 인터벌 세션). 증량 주는 인터벌을 한 단계씩 올리고, 테이퍼 주는 BHF 테이퍼 가이드를 따른다:
    # 3주 전은 약간만 줄여 유지, 2주 전·1주 전은 세트 수를 절반으로(강도는 유지), 대회 주간은 인터벌 없이 짧은 가속만.
    targets: list[tuple[str, float, Optional[tuple[int, int]]]] = []
    current = float(prev_week_km)
    step = 0
    for k in range(build_n):
        current = round(current * (1 + WEEKLY_INCREASE_LIMIT_PCT / 100), 1)
        step = min(k, len(INTERVAL_LADDER) - 1)
        targets.append((f"증량 주 (직전 주 대비 +{WEEKLY_INCREASE_LIMIT_PCT:g}%)", current,
                        INTERVAL_LADDER[step] if include_intervals else None))
    peak = current                          # 마지막 증량 주, 없으면 직전 주가 최고 주간
    peak_interval = INTERVAL_LADDER[step]   # 증량 주가 없으면 입문 단계(3×4분)에서 시작한다고 가정
    for ratio, name in taper:
        if not include_intervals or name == "대회 주간":
            session = None
        elif name == "대회 3주 전":
            session = peak_interval
        else:
            session = (-(-peak_interval[0] // 2), peak_interval[1])   # 세트 수 절반(올림), 세트당 시간은 유지
        targets.append((f"{name} (최고 주간의 {ratio * 100:g}%)", round(peak * ratio, 1), session))

    plan_weeks: list[PlanWeek] = []
    shared: list[str] = []
    prev = float(prev_week_km)
    for index, (label, target, session) in enumerate(targets, start=1):
        weekly = compute_weekly_plan(prev, target, available_days, goal_hours, None, sources, session, record_paces)
        shared += weekly["assumptions"]
        if label.startswith("대회 주간"):   # 대회 요일을 모르므로 롱런 대신 가벼운 이지 러닝만 둔다
            for day in weekly["days"]:
                if day["intensity"] == "롱런":
                    day["intensity"], day["note"] = "이지", "대회 일정에 맞춰 조정하세요. 끝에 짧은 가속 몇 번."
        weekly["assumptions"] = []           # 가정은 계획 전체에 한 번만 보여 준다
        plan_weeks.append(PlanWeek(week=index, label=label, target_km=weekly["total_km"], plan=WeeklyPlan(**weekly)))
        prev = weekly["total_km"]

    assumptions = list(dict.fromkeys(shared)) + notes
    assumptions.append(
        f"주차별 거리: 증량 주는 직전 주에서 매주 최대 {WEEKLY_INCREASE_LIMIT_PCT:g}%씩 늘렸고(NHS 문서), "
        "마지막 주들은 BHF 테이퍼 가이드(최고 주간 대비 75%·50%·25~30%·20~30%, 범위는 중간값)를 적용했습니다.")
    if len(taper) < len(TAPER_STEPS):
        assumptions.append("남은 기간이 짧아 테이퍼 가이드의 일부 주차만 적용했습니다.")
    assumptions.append("대회 요일을 몰라 대회 주간은 가벼운 이지 러닝만 배치했습니다. 대회일에 맞춰 조정이 필요합니다.")
    if include_intervals:
        assumptions.append(
            "인터벌 진행: BHF 초보 계획의 임계 강도 인터벌 진행(3×4분 → 4×4분 → 4×5분 → 4×6분)을 따랐습니다. "
            "증량 주가 없으면 입문 단계(3×4분)에서 시작한다고 가정했고, 테이퍼 주는 BHF 가이드대로 세트 수를 줄이되 강도는 유지했습니다. "
            "문서에 페이스 수치가 없어 '임계 강도'로만 표기했고, 템포 러닝은 문서에 시간·강도 처방이 없어 넣지 않았습니다.")
    if weeks < BHF_BEGINNER_PLAN_WEEKS:
        assumptions.append(
            f"수집 문서(BHF)의 초보 풀마라톤 계획은 {BHF_BEGINNER_PLAN_WEEKS}주 구성입니다. 남은 {weeks}주가 충분한 준비 기간인지는 "
            "문서로 확인하지 못했으니, 현재 주간 거리가 낮다면 완주 중심으로 목표를 잡고 필요하면 전문가와 상의하세요.")

    flow = " → ".join(f"{w.target_km:g}km" for w in plan_weeks)
    plan = TrainingPlan(
        summary=f"{weeks}주 훈련 계획 (주간 거리: {flow})",
        weeks=plan_weeks, assumptions=assumptions, sources=list(sources or []), provisional_params=PROVISIONAL_PARAMS)
    return plan.model_dump()


def _table_lines(plan: dict[str, Any]) -> list[str]:
    """WeeklyPlan(dict)의 요일별 표 줄들."""
    lines = ["| 요일 | 활동 | 거리 | 강도 | 페이스 | 비고 |", "|---|---|---|---|---|---|"]
    for d in plan["days"]:
        distance = f"{d['distance_km']:g}km" if d.get("distance_km") is not None else "-"
        lines.append(f"| {d['day']} | {d['activity']} | {distance} | {d['intensity']} | "
                     f"{d.get('pace') or '-'} | {d.get('note') or ''} |")
    return lines


def plan_to_markdown(plan: dict[str, Any]) -> str:
    """훈련표(dict)를 마크다운으로 바꾼다. 표는 코드가 만들어 LLM 이 표를 바꾸지 못하게 한다.

    한 주 계획(WeeklyPlan)과 여러 주 계획(TrainingPlan, 'weeks' 키)을 모두 처리한다.
    """
    if "weeks" in plan:
        lines = [f"**{plan['summary']}**"]
        for w in plan["weeks"]:
            lines += ["", f"**{w['week']}주차 · {w['label']} — {w['target_km']:g}km**", ""] + _table_lines(w["plan"])
            for adj in w["plan"].get("adjustments", []):
                lines.append(f"- {adj}")
        if plan.get("assumptions"):
            lines += ["", "**가정과 주의할 점**"] + [f"- {item}" for item in plan["assumptions"]]
        if plan.get("provisional_params"):
            lines += ["", f"※ 수치 기준: {plan['provisional_params']['status']}."]
        return "\n".join(lines)
    lines = [f"**{plan['summary']}**", ""] + _table_lines(plan)
    for title, key in (("조정", "adjustments"), ("가정", "assumptions")):
        if plan.get(key):
            lines += ["", f"**{title}**"] + [f"- {item}" for item in plan[key]]
    if plan.get("provisional_params"):
        lines += ["", f"※ 수치 기준: {plan['provisional_params']['status']}."]
    return "\n".join(lines)


def load_runners() -> list[dict[str, Any]]:
    """더미 러너 목록을 읽는다. 파일이 없으면 빈 목록."""
    if not settings.RUNS_PATH.exists():
        return []
    return json.loads(settings.RUNS_PATH.read_text(encoding="utf-8")).get("runners", [])


def other_runner_keys(current_runner_id: str) -> list[str]:
    """현재 러너를 제외한 다른 러너의 이름과 ID 목록. 타인 기록 조회 차단(가드레일 3)에 쓴다."""
    keys: list[str] = []
    for runner in load_runners():
        if runner["runner_id"] != current_runner_id:
            keys += [runner["runner_id"], runner["name"]]
    return keys


def _pace_to_sec(text: str) -> Optional[float]:
    """'6:40' 같은 페이스 문자열을 초/km 로 바꾼다. 형식이 맞지 않으면 None."""
    try:
        minutes, seconds = str(text).split(":")
        return int(minutes) * 60 + int(seconds)
    except (ValueError, AttributeError):
        return None


def compute_record_paces(runner_id: str, weeks: int = 4, min_runs: int = 4) -> Optional[dict[str, Any]]:
    """러너의 최근 기록에서 이지·조깅 페이스와 인터벌 기준 페이스를 구한다. 기록이 부족하면 None.

    기록에는 러닝별 페이스만 있고 이지/인터벌 구분이 없으므로, 임의의 보정값 없이 기록의 분포만 쓴다.
      - 이지·조깅: 페이스 25~75% 지점 사이 (가장 많이 달린 중간 구간)
      - 인터벌 기준: 가장 빠른 기록 ~ 25% 지점 (빠른 쪽 25%)
    """
    runner = next((r for r in load_runners() if r["runner_id"] == runner_id), None)
    if runner is None:
        return None
    window = max(1, weeks) * 7
    paces = [s for r in runner["runs"] if r["days_ago"] <= window
             if (s := _pace_to_sec(r.get("pace"))) is not None]
    if len(paces) < min_runs:
        return None
    paces.sort()                                             # 작을수록 빠름
    q1, _, q3 = statistics.quantiles(paces, n=4, method="inclusive")

    def span(fast: float, slow: float) -> str:
        return _fmt_pace(fast) if round(fast) == round(slow) else f"{_fmt_pace(fast)}~{_fmt_pace(slow)}"

    return {"easy": span(q1, q3), "interval": span(paces[0], q1), "weeks": weeks, "n_runs": len(paces),
            "source": "compute_record_paces (러닝 기록 계산)"}


def load_recent_runs(runner_id: str, weeks: int = 4) -> dict[str, Any]:
    """러너의 최근 러닝 기록과 주별 합계를 돌려준다. 더미 데이터이므로 항상 안내 문구를 싣는다."""
    runner = next((r for r in load_runners() if r["runner_id"] == runner_id), None)
    if runner is None:
        return {"notice": settings.DUMMY_NOTICE, "runner_id": runner_id, "runs": [],
                "message": "해당 러너의 기록을 찾을 수 없습니다."}
    window = max(1, weeks) * 7
    runs = [r for r in runner["runs"] if r["days_ago"] <= window]
    if not runs:
        return {"notice": settings.DUMMY_NOTICE, "runner_id": runner_id, "runs": [], "weekly_totals": [],
                "message": "저장된 러닝 기록이 없습니다. 사용자에게 지난주 주행거리를 직접 물어보세요."}
    weekly: dict[int, float] = {}
    for r in runs:
        week = (r["days_ago"] - 1) // 7 + 1          # 1 = 지난 7일, 2 = 그 전 7일 ...
        weekly[week] = round(weekly.get(week, 0) + r["distance_km"], 1)
    return {
        "notice": settings.DUMMY_NOTICE,
        "runner_id": runner_id,
        "weekly_totals": [{"weeks_ago": w, "total_km": km} for w, km in sorted(weekly.items())],
        "runs": runs,
    }


# ══════════════════════════════════════════════════════════════════
# 에이전트용 도구 (@tool 래퍼)
# ══════════════════════════════════════════════════════════════════
@tool
def calc_pace_zones(goal_hours: float) -> dict:
    """목표 완주 시간으로 풀마라톤 목표 페이스와 이지 페이스 구간을 계산한다.

    Args:
        goal_hours: 목표 완주 시간(시간 단위). 예: 5시간이면 5, 4시간 30분이면 4.5
    """
    return compute_pace_zones(goal_hours)


@tool
def check_weekly_increase(prev_week_km: float, requested_km: float) -> dict:
    """직전 주 실제 거리 대비 요청한 이번 주 거리가 증가율 한도를 넘는지 점검하고 안전한 거리를 돌려준다.

    Args:
        prev_week_km: 직전 주 실제 주행거리(km)
        requested_km: 이번 주에 하려는 주행거리(km)
    """
    return compute_weekly_increase(prev_week_km, requested_km)


@tool
def build_weekly_plan(
    prev_week_km: Optional[float] = None,
    requested_weekly_km: Optional[float] = None,
    available_days: Optional[list[str]] = None,
    goal_hours: Optional[float] = None,
    bad_weather_days: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    include_intervals: bool = True,
) -> dict:
    """안전 규칙(증가율 한도, 악천후 대체)을 적용한 한 주 단위 훈련표를 만든다.

    Args:
        prev_week_km: 직전 주 실제 주행거리(km). 모르면 None (그러면 사용자에게 먼저 물어봐야 한다)
        requested_weekly_km: 이번 주에 하고 싶다고 사용자가 말한 총 거리(km). 말하지 않았으면 None
        available_days: 달릴 수 있는 요일. 예: ["월", "수", "금", "일"]
        goal_hours: 목표 완주 시간(시간 단위). 없으면 None
        bad_weather_days: 사용자가 악천후라고 말한 요일. 예: ["수"]
        sources: 근거로 삼은 문서 출처 목록 (retrieve_docs 결과의 source)
        include_intervals: 주 중반에 임계 강도 인터벌(고강도) 세션을 넣을지. 기본 True, 사용자가 빼 달라고 하면 False
    """
    return compute_weekly_plan(prev_week_km, requested_weekly_km, available_days,
                               goal_hours, bad_weather_days, sources,
                               INTERVAL_LADDER[0] if include_intervals else None,
                               compute_record_paces(settings.current_runner_id()))


@tool
def build_training_plan(
    weeks_until_race: int,
    prev_week_km: Optional[float] = None,
    available_days: Optional[list[str]] = None,
    goal_hours: Optional[float] = None,
    sources: Optional[list[str]] = None,
    runs_per_week: Optional[int] = None,
    include_intervals: bool = True,
) -> dict:
    """대회까지 여러 주(예: 한 달 = 4주)의 훈련 계획을 주차별로 한 번에 만든다. 안전 규칙이 매주 적용된다.

    한 주만 필요하면 build_weekly_plan 을 쓴다. 한 달·몇 주 남았다는 말이 있거나 '전체 계획'을 원하면 이 도구를 쓴다.

    Args:
        weeks_until_race: 대회까지 남은 주 수. 한 달 정도면 4, 두 달 정도면 8
        prev_week_km: 직전 주 실제 주행거리(km). 모르면 None (그러면 사용자에게 먼저 물어봐야 한다)
        available_days: 사용자가 요일을 직접 말했을 때만 넣는다. 예: ["월", "수", "금", "일"]. 요일을 말하지 않았으면 None
        goal_hours: 목표 완주 시간(시간 단위). 없으면 None
        sources: 근거로 삼은 문서 출처 목록 (retrieve_docs 결과의 source)
        runs_per_week: 요일 없이 주당 러닝 횟수만 말했을 때의 횟수(2~5). "3~4회"처럼 범위면 작은 값. 없으면 None
        include_intervals: 주 중반에 임계 강도 인터벌(고강도) 세션을 넣을지. 기본 True, 사용자가 빼 달라고 하면 False
    """
    return compute_multiweek_plan(weeks_until_race, prev_week_km, available_days, goal_hours, sources,
                                  runs_per_week, include_intervals,
                                  compute_record_paces(settings.current_runner_id()))


@tool("get_recent_runs")   # MCP 서버의 도구와 같은 이름: trace·평가에서 어느 쪽이든 같은 도구로 보이게 한다
def get_recent_runs_local(weeks: int = 4) -> dict:
    """현재 러너의 최근 러닝 기록(더미)과 주별 합계를 조회한다. MCP 서버를 쓸 수 없을 때의 대체 도구."""
    return load_recent_runs(settings.current_runner_id(), weeks)
