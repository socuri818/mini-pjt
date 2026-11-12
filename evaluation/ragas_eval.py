"""RAGAS 4개 지표 측정 스크립트.

측정 지표
  - context_recall    : 정답 참고문의 내용이 contexts 에 얼마나 들어 있는가
  - context_precision : contexts 중 정답에 필요한 조각이 위쪽에 있는가
  - faithfulness      : 답변의 주장이 contexts 로 뒷받침되는가
  - answer_relevancy  : 답변이 질문에 맞는가 (임베딩 사용)

방법
  1) 이미 실행 중인 서버의 POST /query 에 질문을 보내 answer / contexts 를 받는다.
     contexts 에는 검색한 문서 조각과 도구 결과(훈련표·페이스·기록)가 함께 들어 있다.
  2) 시스템이 답변 끝에 덧붙이는 '근거 문서:' 줄과 더미 데이터 안내 줄은 평가용 답변에서 뺀다. (모델이 쓴 문장만 평가)
  3) evaluation/ragas_references.csv 의 정답 참고문과 함께 RAGAS 로 4개 지표를 계산한다.

실행 (프로젝트 루트, 서버를 먼저 띄운 상태 / 기본 러너 R001):
    python evaluation/ragas_eval.py --base-url http://127.0.0.1:8000 --out evaluation/results/ragas1

주의
  - 평가용 LLM 은 에이전트와 같은 Bedrock 모델이라 점수가 후할 수 있다.
  - 정답 참고문은 수집 문서와 도구 규칙에서 작성한 것이므로 사람이 검토해야 한다.
  - 지표마다 LLM 을 여러 번 부르므로 Bedrock 호출 한도에 유의한다 (--max-workers 로 동시 호출 수 조절).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import types
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 지표 이름(RAGAS 내부 이름) → 보고서에 쓰는 이름
METRIC_LABELS = {
    "context_recall": "context_recall",
    "llm_context_precision_with_reference": "context_precision",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
}
TRAILER_PATTERNS = (r"^근거 문서:", r"^※ 러닝 기록은")   # 시스템이 덧붙이는 줄


def patch_langchain_community() -> None:
    """ragas 가 import 하는 Vertex AI 모듈이 설치된 langchain-community 에 없을 때 빈 대체 모듈을 끼운다.

    ragas 의 지표 계산은 Vertex AI 를 쓰지 않는다. (공유 가상환경의 패키지를 바꾸지 않으려는 임시 호환 처리)
    """
    for name, attr in (("langchain_community.chat_models.vertexai", "ChatVertexAI"),
                       ("langchain_community.llms.vertexai", "VertexAI")):
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            setattr(module, attr, type(attr, (), {}))
            sys.modules[name] = module


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def post_query(base_url: str, question: str, timeout: int) -> dict:
    """서버의 POST /query 를 호출한다."""
    body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/query", data=body,
                                     headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def clean_answer(answer: str) -> str:
    """시스템이 덧붙인 출처 줄과 더미 데이터 안내 줄을 뺀다."""
    kept = [line for line in answer.splitlines()
            if not any(re.match(pattern, line.strip()) for pattern in TRAILER_PATTERNS)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(clean) / len(clean), 3) if clean else None


def score_text(value) -> str:
    return "측정 실패" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 4개 지표 측정")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="실행 중인 서버 주소")
    parser.add_argument("--csv", default=str(ROOT / "evaluation" / "test_queries.csv"), help="평가 질문 CSV")
    parser.add_argument("--refs", default=str(ROOT / "evaluation" / "ragas_references.csv"), help="정답 참고문 CSV")
    parser.add_argument("--ids", default="", help="측정할 문항 id (쉼표로 구분). 비우면 참고문이 있는 문항 전부")
    parser.add_argument("--timeout", type=int, default=180, help="/query 호출 제한 시간(초)")
    parser.add_argument("--max-workers", type=int, default=2, help="RAGAS 동시 LLM 호출 수")
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "ragas1"),
                        help="결과 저장 경로(확장자 제외). .md 와 .json 으로 저장")
    args = parser.parse_args()

    patch_langchain_community()
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)
    from ragas.run_config import RunConfig

    from src import settings

    cases = {row["id"]: row for row in read_csv(Path(args.csv))}
    refs = {row["id"]: row for row in read_csv(Path(args.refs))}
    wanted = [i.strip() for i in args.ids.split(",") if i.strip()] or list(refs)
    ids = [i for i in wanted if i in refs and i in cases]
    if not ids:
        print("측정할 문항이 없습니다. (--ids 와 참고문 CSV 를 확인하세요)")
        return 2

    # 1) 질문을 보내 answer / contexts 를 모은다
    samples, rows = [], []
    for case_id in ids:
        question = cases[case_id]["input"]
        print(f"[#{case_id}] /query 호출 중 …", flush=True)
        try:
            data = post_query(args.base_url, question, args.timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"  호출 실패: {type(exc).__name__}: {exc}")
            return 1
        contexts = [c.get("text", "") for c in data.get("contexts", []) if c.get("text")]
        answer = clean_answer(data.get("answer", ""))
        rows.append({"id": case_id, "question": question, "answer": answer, "contexts": data.get("contexts", []),
                     "reference": refs[case_id]["reference"]})
        samples.append(SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts,
                                        reference=refs[case_id]["reference"]))
        print(f"  contexts {len(contexts)}건 (검색 문서 + 도구 결과), 답변 {len(answer)}자")

    # 2) RAGAS 로 4개 지표를 계산한다
    llm = LangchainLLMWrapper(settings.make_llm())
    embeddings = LangchainEmbeddingsWrapper(settings.make_embeddings())
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    print(f"RAGAS 계산 중 (샘플 {len(samples)}건 × 지표 {len(metrics)}개) …", flush=True)
    result = evaluate(EvaluationDataset(samples=samples), metrics=metrics, llm=llm, embeddings=embeddings,
                      run_config=RunConfig(max_workers=args.max_workers, timeout=300, max_retries=3),
                      show_progress=False, raise_exceptions=False)
    frame = result.to_pandas()

    # 3) 정리해서 저장한다
    per_case = []
    for index, row in enumerate(rows):
        scores = {}
        for key, label in METRIC_LABELS.items():
            value = frame.iloc[index].get(key) if key in frame.columns else None
            scores[label] = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
        per_case.append({"id": row["id"], "question": row["question"], "scores": scores,
                         "contexts": len(row["contexts"]), "answer": row["answer"]})
    summary = {label: mean([c["scores"][label] for c in per_case]) for label in METRIC_LABELS.values()}
    failed = {label: sum(1 for c in per_case if c["scores"][label] is None) for label in METRIC_LABELS.values()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "base_url": args.base_url,
            "model": settings.MODEL_ID, "embedding": settings.EMBEDDING_MODEL_ID, "samples": len(per_case),
            "ids": ids}
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary, "failed": failed, "cases": per_case},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# RAGAS 측정 결과", "",
             f"- 실행 시각: {meta['started']}", f"- 대상: {args.base_url} / 샘플 {len(per_case)}건 (문항 {', '.join('#' + i for i in ids)})",
             f"- 평가 LLM: {settings.MODEL_ID} (에이전트와 같은 모델, 자기 평가 편향 가능성)",
             f"- 임베딩: {settings.EMBEDDING_MODEL_ID}",
             "- contexts: 검색한 문서 조각 + 도구 결과(훈련표·페이스·기록). 시스템이 덧붙인 출처·더미 안내 줄은 답변에서 제외",
             "- 정답 참고문: evaluation/ragas_references.csv (수집 문서와 도구 규칙에서 작성, 사람 검토 필요)", "",
             "## 평균", "", "| 지표 | 평균 | 측정 실패 샘플 |", "|---|---|---|"]
    for label, value in summary.items():
        lines.append(f"| {label} | {score_text(value)} | {failed[label]} |")
    lines += ["", "## 문항별", "", "| # | context_recall | context_precision | faithfulness | answer_relevancy | contexts |",
              "|---|---|---|---|---|---|"]
    for c in per_case:
        s = c["scores"]
        lines.append(f"| {c['id']} | {score_text(s['context_recall'])} | {score_text(s['context_precision'])} | "
                     f"{score_text(s['faithfulness'])} | {score_text(s['answer_relevancy'])} | {c['contexts']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== RAGAS 평균 ===")
    for label, value in summary.items():
        print(f"  {label:<18} {score_text(value)}  (측정 실패 {failed[label]}건)")
    print(f"\n저장: {out.with_suffix('.md')}\n      {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
