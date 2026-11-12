"""설정과 경로 모음. 비밀 값(자격 증명)은 .env / 환경 변수에서 읽고 코드에 적지 않는다."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# 프로젝트 루트 (이 파일은 src/ 아래에 있으므로 두 단계 위)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 이 파일 위치에서 상위 폴더로 올라가며 가장 가까운 .env 를 찾는다. (수강 저장소 스크립트와 같은 방식)
# 프로젝트 안에 .env 를 복사해 두지 않아도 되고, 없으면 오류 없이 넘어가 기본 자격 증명 체인을 쓴다.
# 비밀 값은 코드에 적지 않으며, .env 는 커밋하지 않는다.
load_dotenv(find_dotenv())

# LangSmith 추적은 .env 의 LANGSMITH_TRACING 설정을 따르되, 프로젝트 이름은 이 프로젝트 전용으로 고정한다.
# (상위 .env 에는 수강 중 쓰던 day4/day6/day7-practice 프로젝트명이 들어 있어, 분리하지 않으면 trace 가 거기에 섞인다)
LANGSMITH_PROJECT = "mini-pjt"
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# ── 모델 (저장소 고정값) ──────────────────────────────────────────
# 기본값은 평가에 쓴 모델이다. 호출 한도에 걸렸을 때만 환경 변수 BEDROCK_MODEL_ID 로 임시로 바꿔 띄운다.
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID") or "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
# 리전은 기본 us-east-1 이다. 이 리전의 호출 한도가 찼을 때만 환경 변수 BEDROCK_REGION 으로 바꾼다 (모델 ID 는 그대로).
REGION = os.environ.get("BEDROCK_REGION") or "us-east-1"

# ── 경로 ─────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
GUIDES_DIR = DATA_DIR / "guides"            # 수집한 지식 문서(md)
RUNS_PATH = DATA_DIR / "runs_dummy.json"    # 교육용 더미 러닝 기록
CHROMA_DIR = PROJECT_ROOT / "chroma_db"     # 벡터DB (git 에 올리지 않는다)
COLLECTION_NAME = "marathon_guides"

# 더미 데이터를 사용자에게 보여 줄 때 반드시 붙이는 문구
DUMMY_NOTICE = "교육용 더미 데이터입니다"

# CURRENT_RUNNER_ID 환경 변수가 없을 때 쓰는 기본 러너 (더미 데이터의 기록 있는 러너)
DEFAULT_RUNNER_ID = "R001"


def current_runner_id() -> str:
    """현재 사용자(러너) ID 를 돌려준다.

    API 요청 본문에는 사용자 정보가 없으므로, 서버 설정(환경 변수 CURRENT_RUNNER_ID)으로 고정한다.
    """
    return os.environ.get("CURRENT_RUNNER_ID", "").strip() or DEFAULT_RUNNER_ID


def make_llm():
    """Bedrock 채팅 모델을 만든다. 모듈 최상단이 아니라 함수 안에서 만든다. (테스트에서 llm 주입 가능)"""
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def make_embeddings():
    """Titan 임베딩 모델을 만든다. 함수 안에서 만들어 불필요한 AWS 호출을 피한다."""
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, region_name=REGION)
