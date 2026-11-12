"""평가 스크립트: test_queries.csv 의 문항마다 Agent 를 실행하고 자동 채점한다.

동작 방식
  - 서버의 POST /query (제출 규약)에 문항 input 을 보내 answer / contexts / trace 를 받는다.
  - 자동 채점은 두 가지만 한다.
      1) expected_tools : 적힌 도구를 모두 불렀는지 trace 로 확인한다.
      2) forbidden      : 항목이 answer 에 들어 있으면 실패로 본다.
  - expected_traits 는 자동 채점이 어려우므로 채점하지 않는다.
    답변과 나란히 결과 파일(.md)에 남겨서 사람이 확인한다.

사용 예
  # 서버가 이미 떠 있을 때 (모든 문항이 서버의 현재 러너로 실행된다)
  python evaluation/run_eval.py --base-url http://127.0.0.1:8000 --out evaluation/round1_report

  # 문항별 러너 전제를 지키려면 서버를 스크립트가 직접 띄운다
  python evaluation/run_eval.py --server-cmd "uvicorn src.api_server:app --port 8000" \
      --runner with_records=R001 --runner no_records=R099 --out evaluation/round1_report

주의
  - 표준 라이브러리만 쓴다. LLM 을 직접 호출하지 않고, 서버(Agent)만 호출한다.
  - 결과 .md 의 통과율은 '자동 검사(도구·금지어) 기준'이다. expected_traits 는 포함되지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

PASS, FAIL, ERROR = "PASS", "FAIL", "ERROR"
CATEGORIES = ["positive", "negative", "edge", "guardrail"]

# 도구 이름과 trace 의 step 이름이 다를 때의 별칭. (규약 예시 trace 의 step 은 "retrieve")
# 서버(api_server)가 trace 를 어떻게 남기는지 정해지면 여기를 맞춘다.
TOOL_STEP_ALIASES = {"retrieve_docs": {"retrieve"}}


# ──────────────────────────────────────────────────────────────
# 입력: CSV 읽기
# ──────────────────────────────────────────────────────────────
def load_cases(path: Path) -> list[dict]:
    """test_queries.csv 를 읽어 문항 목록으로 돌려준다. (엑셀 BOM 허용)"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        cases = list(csv.DictReader(f))
    if not cases:
        raise SystemExit(f"문항이 없습니다: {path}")
    return cases


def split_items(value: str | None) -> list[str]:
    """세미콜론으로 구분된 칸을 항목 목록으로 나눈다. 빈 칸은 빈 목록."""
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def runner_group(case: dict) -> str:
    """문항 note 의 'CURRENT_RUNNER_ID 전제' 를 읽어 필요한 러너 종류를 정한다.

    기록이 있는 러너 → "with_records", 기록이 없는 러너 → "no_records", 전제 없음 → "default".
    """
    note = case.get("note") or ""
    if "CURRENT_RUNNER_ID" not in note:
        return "default"
    if "기록이 없음" in note:
        return "no_records"
    if "기록이 있음" in note:
        return "with_records"
    return "default"


# ──────────────────────────────────────────────────────────────
# 서버 호출과 채점
# ──────────────────────────────────────────────────────────────
def ask(base_url: str, question: str, timeout: float) -> dict:
    """POST /query 를 호출해 응답 JSON 을 돌려준다. 규약 위반이면 ValueError."""
    body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    missing = [key for key in ("answer", "contexts", "trace") if key not in data]
    if missing:
        raise ValueError(f"응답 규약 위반: 없는 키 {missing}")
    return data


def called_tools(trace: list) -> set[str]:
    """trace 에서 불린 도구 이름 집합을 뽑는다.

    인정하는 형태: step == 이름, step == "tool:이름", 항목의 "tool" 또는 "name" 값.
    """
    names: set[str] = set()
    for entry in trace or []:
        if not isinstance(entry, dict):
            continue
        step = str(entry.get("step", ""))
        names.add(step)
        if step.startswith("tool:"):
            names.add(step[len("tool:"):])
        for key in ("tool", "name"):
            if entry.get(key):
                names.add(str(entry[key]))
    return names


def normalize(text: str) -> str:
    """금지어 비교용 정규화: 소문자로 바꾸고 공백을 모두 없앤다."""
    return "".join(str(text).casefold().split())


def check_once(case: dict, response: dict, substring_forbidden: bool = True) -> dict:
    """응답 한 번을 자동 채점한다 (expected_tools 와 forbidden 만)."""
    answer = str(response.get("answer", ""))
    trace = response.get("trace") or []
    tools = called_tools(trace)

    expected = split_items(case.get("expected_tools"))
    missing_tools = [
        tool for tool in expected
        if tool not in tools and not (TOOL_STEP_ALIASES.get(tool, set()) & tools)
    ]

    # --judge 를 켜면 금지 항목은 Judge 가 의미로 판정한다. 문자열 매칭은 거절 문장 속 단어를 위반으로 오판하기 때문이다.
    normalized_answer = normalize(answer)
    forbidden_hits = [
        item for item in split_items(case.get("forbidden"))
        if normalize(item) in normalized_answer
    ] if substring_forbidden else []

    reasons = []
    if missing_tools:
        reasons.append(f"호출 안 된 도구: {', '.join(missing_tools)}")
    if forbidden_hits:
        reasons.append(f"금지 항목 포함: {', '.join(forbidden_hits)}")
    return {
        "status": FAIL if reasons else PASS,
        "detail": " / ".join(reasons),
        "answer": answer,
        "contexts": len(response.get("contexts") or []),
        "tools_called": sorted(name for name in tools if name),
        "missing_tools": missing_tools,
        "forbidden_hits": forbidden_hits,
        "trace": trace,   # 실패 원인 진단용으로 결과에 그대로 남긴다
    }


def load_judge():
    """--judge 일 때만 LLM Judge 를 불러온다. 프로젝트 루트를 import 경로에 넣고 src.judge 를 쓴다."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.judge import build_judge_chain, judge_case

    chain = build_judge_chain()   # 같은 체인을 문항마다 재사용한다
    return lambda case, answer: judge_case(
        case["input"], answer, split_items(case.get("expected_traits")),
        split_items(case.get("forbidden")), chain)


def apply_judge(run: dict, case: dict, judge_fn) -> None:
    """실행 결과에 Judge 판정을 붙이고, 불통과면 상태를 FAIL 로 바꾼다. Judge 오류는 ERROR 로 기록한다."""
    try:
        verdict = judge_fn(case, run["answer"])
    except Exception as exc:  # noqa: BLE001 - Judge 한 번의 실패가 전체 채점을 멈추면 안 된다
        run["judge"] = {"error": f"{type(exc).__name__}: {exc}"}
        run["status"] = ERROR
        run["detail"] = f"Judge 오류: {type(exc).__name__}: {exc}"
        return
    run["judge"] = verdict
    if not verdict["passed"]:
        failing = [t["trait"] for t in verdict["traits"] if not t["satisfied"]]
        failing += [f"금지 위반: {f['item']}" for f in verdict["forbidden"] if f["violated"]]
        reason = f"Judge 불통과: {'; '.join(failing)}"
        run["status"] = FAIL
        run["detail"] = f"{run['detail']} / {reason}" if run["detail"] else reason


def run_case(case: dict, base_url: str, repeat: int, timeout: float, judge_fn=None) -> dict:
    """문항 하나를 repeat 번 실행하고, 과반이 통과하면 통과로 본다. judge_fn 이 있으면 Judge 판정도 반영한다."""
    runs = []
    for _ in range(repeat):
        started = time.time()
        try:
            run = check_once(case, ask(base_url, case["input"], timeout), substring_forbidden=judge_fn is None)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
            run = {"status": ERROR, "detail": f"{type(exc).__name__}: {exc}", "answer": "",
                   "contexts": 0, "tools_called": [], "missing_tools": [], "forbidden_hits": [], "trace": []}
        run["latency_s"] = round(time.time() - started, 2)
        run["auto_status"] = run["status"]          # 도구 호출(과 문자열 금지어)만으로 낸 판정. Judge 반영 전
        if judge_fn is not None and run["status"] != ERROR:
            apply_judge(run, case, judge_fn)
        runs.append(run)

    auto_passed = sum(1 for run in runs if run["auto_status"] == PASS)
    passed = sum(1 for run in runs if run["status"] == PASS)
    if passed * 2 > len(runs):
        status = PASS
    elif all(run["status"] == ERROR for run in runs):
        status = ERROR
    else:
        status = FAIL
    return {
        "id": case["id"],
        "category": case["category"],
        "input": case["input"],
        "status": status,
        "auto_status": PASS if auto_passed * 2 > len(runs) else FAIL,
        "passed_runs": passed,
        "runs": runs,
        "expected_traits": split_items(case.get("expected_traits")),
        "forbidden": split_items(case.get("forbidden")),
        "expected_tools": split_items(case.get("expected_tools")),
        "note": case.get("note") or "",
        "runner_group": runner_group(case),
    }


# ──────────────────────────────────────────────────────────────
# 서버 기동(선택): 러너 전제를 지키기 위해 CURRENT_RUNNER_ID 를 바꿔 가며 띄운다
# ──────────────────────────────────────────────────────────────
def wait_for_port(base_url: str, timeout: float) -> bool:
    """서버 포트가 열릴 때까지 기다린다."""
    parsed = urlparse(base_url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def stop_server(process: subprocess.Popen) -> None:
    """띄운 서버를 종료한다. (Windows 는 shell 아래 자식 프로세스까지 정리한다)"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def evaluate(cases: list[dict], args: argparse.Namespace) -> list[dict]:
    """문항을 러너 그룹별로 묶어 실행한다. 결과는 원래 문항 순서로 돌려준다."""
    results: dict[str, dict] = {}
    judge_fn = load_judge() if getattr(args, "judge", False) else None

    if not args.server_cmd:
        # 서버가 이미 떠 있다고 보고 전부 한 번에 돌린다. 러너 전제는 확인하지 못한다.
        for case in cases:
            result = run_case(case, args.base_url, args.repeat, args.timeout, judge_fn)
            if result["runner_group"] != "default":
                result["warning"] = (f"러너 전제({result['runner_group']})를 확인하지 못했습니다. "
                                     "서버의 CURRENT_RUNNER_ID 를 직접 맞춰야 합니다.")
            results[case["id"]] = result
    else:
        groups: "OrderedDict[str, list[dict]]" = OrderedDict()
        for case in cases:
            groups.setdefault(runner_group(case), []).append(case)
        runner_ids = dict(item.split("=", 1) for item in args.runner)

        for group, group_cases in groups.items():
            env = dict(os.environ)
            if group in runner_ids:
                env["CURRENT_RUNNER_ID"] = runner_ids[group]
            warning = ""
            if group != "default" and group not in runner_ids:
                warning = f"--runner {group}=<러너ID> 가 없어 서버 기본 러너로 실행했습니다."
            print(f"[서버 기동] 러너 그룹={group} 문항={[c['id'] for c in group_cases]}", flush=True)
            process = subprocess.Popen(args.server_cmd, shell=True, env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                if not wait_for_port(args.base_url, args.startup_timeout):
                    raise SystemExit(f"서버가 {args.startup_timeout:.0f}초 안에 열리지 않았습니다: {args.base_url}")
                for case in group_cases:
                    result = run_case(case, args.base_url, args.repeat, args.timeout, judge_fn)
                    if warning:
                        result["warning"] = warning
                    results[case["id"]] = result
            finally:
                stop_server(process)

    return [results[case["id"]] for case in cases]


# ──────────────────────────────────────────────────────────────
# 집계와 출력
# ──────────────────────────────────────────────────────────────
def summarize(results: list[dict], key: str = "status") -> dict:
    """카테고리별·전체 통과 수를 센다. key 는 'status'(최종) 또는 'auto_status'(Judge 반영 전)."""
    by_category = {cat: {"total": 0, "passed": 0} for cat in CATEGORIES}
    for result in results:
        bucket = by_category.setdefault(result["category"], {"total": 0, "passed": 0})
        bucket["total"] += 1
        if result[key] == PASS:
            bucket["passed"] += 1
    total = sum(b["total"] for b in by_category.values())
    passed = sum(b["passed"] for b in by_category.values())
    return {"total": total, "passed": passed, "by_category": by_category}


def rate(passed: int, total: int) -> str:
    """통과율 문자열. 문항이 없으면 '-'."""
    return f"{passed}/{total} ({passed / total * 100:.0f}%)" if total else "-"


def print_summary(summary: dict, results: list[dict], judged: bool = False) -> None:
    """콘솔에 카테고리별 통과율과 전체 통과율을 출력한다. judged 면 도구 검사만의 결과도 함께 보여 준다."""
    title = ("최종 결과 (도구 호출 + LLM Judge: expected_traits·forbidden 판정)" if judged
             else "자동 검사 결과 (expected_tools + forbidden, expected_traits 는 제외)")
    print(f"\n=== {title} ===")
    for category, bucket in summary["by_category"].items():
        if bucket["total"]:
            print(f"  {category:<10} {rate(bucket['passed'], bucket['total'])}")
    print(f"  {'전체':<10} {rate(summary['passed'], summary['total'])}")
    if judged:
        tools_only = summarize(results, "auto_status")
        print(f"  (참고) 도구 호출 검사만: {rate(tools_only['passed'], tools_only['total'])}")
    not_passed = [r for r in results if r["status"] != PASS]
    if not_passed:
        print("\n  통과하지 못한 문항:")
        for r in not_passed:
            detail = next((run["detail"] for run in r["runs"] if run["detail"]), "")
            print(f"   - #{r['id']} [{r['category']}] {r['status']}: {detail}")
    warned = [r for r in results if r.get("warning")]
    if warned:
        print("\n  경고:")
        for r in warned:
            print(f"   - #{r['id']}: {r['warning']}")


def write_markdown(path: Path, summary: dict, results: list[dict], meta: dict) -> None:
    """사람이 expected_traits 를 답변과 나란히 볼 수 있는 보고서를 쓴다."""
    lines = [
        "# 평가 결과 (" + ("도구 호출 + LLM Judge" if meta.get("judge") else "자동 검사 + 사람 검토용") + ")", "",
        f"- 실행 시각: {meta['started']}",
        f"- 대상: {meta['base_url']}  /  문항 수: {summary['total']}  /  문항당 실행 횟수: {meta['repeat']}",
        "- 통과 판정: 문항당 실행의 **과반**이 통과하면 PASS",
    ] + ([
        "- 판정 방식: expected_tools 호출 여부는 trace 로 확인하고, **expected_traits 와 forbidden 은 LLM Judge 가 판정**한다.",
        "  (고정 루브릭, temperature 0. 판정 모델은 에이전트와 같은 모델이라 자기 평가 편향 가능성이 있으므로 사람이 표본 검토한다.)", "",
        "## 통과율 (도구 호출 + LLM Judge)", "",
    ] if meta.get("judge") else [
        "- 자동 검사는 expected_tools 호출 여부와 forbidden 포함 여부만 본다.",
        "  **expected_traits 는 자동 채점하지 않으며, 아래 체크리스트로 사람이 확인한다.**", "",
        "## 통과율 (자동 검사 기준)", "",
    ]) + [
        "| 카테고리 | 통과 | 통과율 |", "|---|---|---|",
    ]
    for category, bucket in summary["by_category"].items():
        if bucket["total"]:
            pct = bucket["passed"] / bucket["total"] * 100
            lines.append(f"| {category} | {bucket['passed']}/{bucket['total']} | {pct:.0f}% |")
    overall = summary["passed"] / summary["total"] * 100 if summary["total"] else 0
    lines += [f"| **전체** | **{summary['passed']}/{summary['total']}** | **{overall:.0f}%** |", "",
              "## 문항별 결과", ""]

    for r in results:
        lines += [f"### #{r['id']} [{r['category']}] {r['status']}", "",
                  f"- 질문: {r['input']}",
                  f"- 자동 검사: {r['status']} (통과한 실행 {r['passed_runs']}/{len(r['runs'])})"]
        if r.get("warning"):
            lines.append(f"- ⚠ {r['warning']}")
        lines.append(f"- expected_tools: {', '.join(r['expected_tools']) or '(없음)'}")
        for index, run in enumerate(r["runs"], start=1):
            lines += ["",
                      f"**실행 {index}** — {run['status']}, 호출된 도구: "
                      f"{', '.join(run['tools_called']) or '(없음)'}, contexts {run['contexts']}건, "
                      f"{run['latency_s']}초" + (f", 사유: {run['detail']}" if run["detail"] else ""),
                      "", "> " + (run["answer"].replace("\n", "\n> ") or "(답변 없음)")]
            if run["status"] != PASS and run.get("trace"):
                # 통과하지 못한 실행은 단계별 trace 를 함께 남겨 원인(어느 단계에서 막혔는지)을 볼 수 있게 한다
                lines += ["", "<details><summary>trace (진단용)</summary>", ""]
                for step in run["trace"]:
                    out = str(step.get("output", "")).replace("\n", " ")
                    lines.append(f"- `{step.get('step')}` → {out[:160]}")
                lines += ["", "</details>"]
        judged = next((run["judge"] for run in r["runs"] if run.get("judge") and "error" not in run["judge"]), None)
        if judged:
            # [x] = 충족(또는 위반 없음), [ ] = 불충족(또는 위반). 표시는 Judge 판정이고 사람이 표본 확인한다.
            lines += ["", "**Judge 판정** (첫 판정 기준, 사람이 표본 확인)"]
            lines += [f"- [{'x' if t['satisfied'] else ' '}] {t['trait']} — {t['reason']}" for t in judged["traits"]]
            lines += [f"- [{' ' if f['violated'] else 'x'}] 금지 안 함: {f['item']} — {f['reason']}"
                      for f in judged["forbidden"]]
        else:
            lines += ["", "**사람이 확인할 expected_traits** (답변이 아래를 모두 만족하는가)"]
            lines += [f"- [ ] {trait}" for trait in r["expected_traits"]]
            if r["forbidden"]:
                lines += ["", "**참고: forbidden 항목** (문자열이 아닌 설명형 항목은 자동 검사로 잡히지 않을 수 있으니 직접 확인)"]
                lines += [f"- [ ] 포함되지 않음: {item}" for item in r["forbidden"]]
        if r["note"]:
            lines += ["", f"_note: {r['note']}_"]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="test_queries.csv 로 Agent 를 실행하고 채점한다.")
    parser.add_argument("--csv", type=Path, default=here / "test_queries.csv", help="평가 문항 CSV")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="서버 주소")
    parser.add_argument("--repeat", type=int, default=1, help="문항당 실행 횟수 (과반 통과 기준)")
    parser.add_argument("--timeout", type=float, default=120, help="요청 1건의 제한 시간(초)")
    parser.add_argument("--out", type=Path, default=None,
                        help="결과 파일 경로(확장자 제외). .md 와 .json 두 개가 생긴다")
    parser.add_argument("--server-cmd", default=None,
                        help="서버 실행 명령. 주면 러너 그룹마다 CURRENT_RUNNER_ID 를 바꿔 서버를 다시 띄운다")
    parser.add_argument("--runner", action="append", default=[], metavar="그룹=러너ID",
                        help="예: with_records=R001, no_records=R099 (여러 번 지정 가능)")
    parser.add_argument("--startup-timeout", type=float, default=60, help="서버 기동 대기 시간(초)")
    parser.add_argument("--judge", action="store_true",
                        help="LLM Judge 로 expected_traits·forbidden 을 판정한다 (문항·실행마다 모델 호출이 1회 늘어난다)")
    args = parser.parse_args()

    for item in args.runner:
        if "=" not in item:
            parser.error(f"--runner 는 그룹=러너ID 형식이어야 합니다: {item}")
    if args.repeat < 1:
        parser.error("--repeat 는 1 이상이어야 합니다")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = datetime.now()
    out_base = args.out or (here / "results" / f"eval_{started:%Y%m%d-%H%M%S}")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.csv)
    print(f"문항 {len(cases)}건, 문항당 {args.repeat}회, 대상 {args.base_url}", flush=True)
    results = evaluate(cases, args)
    summary = summarize(results)
    print_summary(summary, results, judged=args.judge)

    meta = {"started": started.strftime("%Y-%m-%d %H:%M:%S"), "base_url": args.base_url,
            "repeat": args.repeat, "judge": args.judge}
    write_markdown(out_base.with_suffix(".md"), summary, results, meta)
    out_base.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n저장: {out_base.with_suffix('.md')}\n      {out_base.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
