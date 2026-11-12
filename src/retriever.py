"""RAG 파이프라인: data/guides/*.md 를 Chroma 에 적재하고, 질문과 관련된 문서 조각을 검색한다.

문서 형식(권장): 첫 줄 '# 제목', 이어서 '출처: 기관명', 'URL: ...' 줄을 두면 검색 결과의 source 에 쓰인다.

색인 만들기:  python -m src.retriever --rebuild
문서가 없거나 색인이 비어 있으면 검색은 빈 목록을 돌려준다. (근거가 없을 때 지어내지 않기 위함)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

from src import settings

CHUNK_CHARS = 600   # 문단을 이 길이 안팎으로 묶어 한 조각으로 만든다


def _parse_doc(path: Path) -> tuple[str, str, str]:
    """md 파일에서 (제목, 출처, 본문)을 뽑는다. 헤더 줄이 없으면 파일 이름을 제목으로 쓴다."""
    text = path.read_text(encoding="utf-8")
    title = next((m.group(1).strip() for m in re.finditer(r"^#\s+(.+)$", text, re.M)), path.stem)
    org = re.search(r"^출처:\s*(.+)$", text, re.M)
    url = re.search(r"^URL:\s*(\S+)", text, re.M)
    source = " / ".join(x.group(1).strip() for x in (org, url) if x) or title
    return title, source, text


def split_chunks(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """빈 줄 기준 문단을 limit 글자 안팎으로 묶어 조각을 만든다."""
    chunks, current = [], ""
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if current and len(current) + len(paragraph) > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_documents(guides_dir: Path | None = None) -> list[Document]:
    """data/guides 의 md 문서를 조각 단위 Document 로 읽는다."""
    guides_dir = guides_dir or settings.GUIDES_DIR
    if not guides_dir.exists():
        return []
    docs: list[Document] = []
    for path in sorted(guides_dir.glob("*.md")):
        title, source, text = _parse_doc(path)
        for index, chunk in enumerate(split_chunks(text)):
            docs.append(Document(page_content=chunk,
                                 metadata={"doc_id": f"{path.name}#{index}", "title": title, "source": source}))
    return docs


_STORE = None                       # 프로세스 안에서 한 번만 여는 Chroma 벡터DB
_STORE_LOCK = threading.Lock()      # 모델이 retrieve_docs 를 동시에 여러 번 부르면 도구가 여러 스레드에서 실행된다
_SEARCH_LOCK = threading.Lock()


def _vectorstore(fresh: bool = False):
    """Chroma 벡터DB 를 연다. 임베딩 모델은 여기서(함수 안에서) 만든다.

    Chroma 클라이언트는 동시에 만들면 서로 충돌해 검색이 실패하므로(KeyError 등), 한 번만 만들어 재사용한다.
    fresh=True 면 색인을 다시 만든 뒤처럼 새로 연다.
    """
    global _STORE
    with _STORE_LOCK:
        if fresh or _STORE is None:
            from langchain_chroma import Chroma

            _STORE = Chroma(collection_name=settings.COLLECTION_NAME, embedding_function=settings.make_embeddings(),
                            persist_directory=str(settings.CHROMA_DIR))
        return _STORE


def build_index(rebuild: bool = False) -> int:
    """문서를 읽어 Chroma 에 적재하고 적재한 조각 수를 돌려준다. rebuild 면 기존 색인을 지우고 다시 만든다."""
    docs = load_documents()
    if not docs:
        return 0
    if rebuild and settings.CHROMA_DIR.exists():
        shutil.rmtree(settings.CHROMA_DIR)
    store = _vectorstore(fresh=True)
    store.add_documents(docs, ids=[d.metadata["doc_id"] for d in docs])
    return len(docs)


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """질문과 가까운 문서 조각 k 개를 [{doc_id, text, source}] 로 돌려준다. 색인이 없으면 빈 목록."""
    if not settings.CHROMA_DIR.exists():
        return []
    store = _vectorstore()
    with _SEARCH_LOCK:                  # 동시 호출은 한 번에 하나씩 검색한다 (검색은 수백 ms라 지연이 거의 없다)
        found = store.similarity_search(query, k=k)
    return [{"doc_id": d.metadata.get("doc_id", ""), "text": d.page_content, "source": d.metadata.get("source", "")}
            for d in found]


@tool
def retrieve_docs(query: str) -> str:
    """풀마라톤 훈련·부상 예방·테이퍼링·영양 등 수집된 기관 문서에서 질문과 관련된 근거를 검색한다.

    결과는 JSON 문자열이다. 빈 목록([])이면 수집 문서에 근거가 없다는 뜻이므로 지어내지 말고 그렇게 안내한다.

    Args:
        query: 검색할 질문 또는 핵심 키워드
    """
    try:
        return json.dumps(retrieve(query), ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 - 검색 실패가 응답 전체를 죽이지 않게 한다
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="지식 문서를 Chroma 에 적재한다.")
    parser.add_argument("--rebuild", action="store_true", help="기존 색인을 지우고 다시 만든다")
    args = parser.parse_args()
    count = build_index(rebuild=args.rebuild)
    print(f"적재한 조각 수: {count}" if count else f"적재할 문서가 없습니다: {settings.GUIDES_DIR}")
