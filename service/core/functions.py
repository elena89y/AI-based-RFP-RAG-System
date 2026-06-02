from __future__ import annotations

import datetime
import json
import os
import re
import unicodedata
from pathlib import Path

import chromadb
import numpy as np
try:
    import ollama as ollama_client
except Exception:  # pragma: no cover - optional runtime dependency
    ollama_client = None
from openai import OpenAI
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer

REPO_PATH = Path(__file__).resolve().parents[1]
DEFAULT_BASE_PATH = REPO_PATH if (REPO_PATH / "data").exists() else REPO_PATH.parent
BASE_PATH = Path(os.getenv("RFP_BASE_PATH", str(DEFAULT_BASE_PATH))).expanduser()
DATA_PATH = BASE_PATH / "data"
CHROMA_PATH = str(BASE_PATH / "chroma_seol_qwen3")
FILES_PATH = DATA_PATH / "files_advanced"
CHUNKS_PATH = DATA_PATH / "chroma_export.json"
META_PATH = DATA_PATH / "data_list_advanced.xlsx"
BOOKMARKS_PATH = DATA_PATH / "bookmarks.json"

COLLECTION_NAME = "rfp_seol_chunks_qwen3"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Generation
TEMPERATURE = 0.0
MAX_TOKENS = 1000
MAX_CONTEXT_TOKENS = 4500

# Retrieval
RETRIEVAL_EVAL_K = 20
ANSWER_TOP_K = 10
DENSE_CANDIDATES = 60
BM25_CANDIDATES = 60
RRF_CANDIDATES = 40

USE_RERANKER = False

NUMERIC_CHUNK_BOOST = 0.12
AGENCY_CHUNK_BOOST = 0.05
EVIDENCE_CHUNK_BOOST = 0.20
BUSINESS_OVERLAP_BOOST = 0.10
CORE_AMOUNT_BOOST = 0.60
WRONG_BUSINESS_PENALTY = 0.15
SOURCE_AUTOFILL_TOP_N = 3
MAX_HISTORY_MESSAGES = 24
HISTORY_WINDOW_FOR_REWRITE = 12

USE_DETERMINISTIC_BUDGET_ANSWER = True
USE_DETERMINISTIC_MULTI_AMOUNT_CALC = True
USE_DETERMINISTIC_DURATION_ANSWER = True

# Embedding
EMBED_BATCH_SIZE = 8
EMBED_TEXT_MAX_CHARS = 3000
EMBED_MAX_SEQ_LENGTH = 1024

# Runtime state
LLM_MODEL = "gemma4:e4b"
openai_client: OpenAI | None = None
emb_model: SentenceTransformer | None = None
chroma_client = None
collection = None
bm25_index: BM25Okapi | None = None

all_chunks: list[dict] = []
chunk_map: dict[str, dict] = {}
agency_list: list[str] = []

INITIALIZED = False
INITIALIZATION_ERROR: Exception | None = None
ACTIVE_AGENCY_FILTER: str | None = None


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)) if text else ""


def _detect_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def prepare_embedding_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:EMBED_TEXT_MAX_CHARS]


def tokenize(text: str) -> list[str]:
    text = nfc(text).lower()
    return re.findall(r"[가-힣A-Za-z0-9]+", text)


def chat_ollama(
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    _ensure_initialized()
    assert openai_client is not None

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    except TypeError:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return response.choices[0].message.content or ""


def _load_chunks() -> tuple[list[dict], dict[str, dict], list[str]]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "ids" in raw:
        chunks = [
            {
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "chunk_type": meta.get("chunk_type", "normal"),
            }
            for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
    else:
        chunks = raw

    cmap = {c["chunk_id"]: c for c in chunks}
    agencies = sorted(
        {
            nfc(c.get("metadata", {}).get("발주기관", ""))
            for c in chunks
            if c.get("metadata", {}).get("발주기관")
        }
    )
    return chunks, cmap, agencies


def _ensure_initialized() -> None:
    global INITIALIZED
    global INITIALIZATION_ERROR
    global LLM_MODEL
    global openai_client
    global emb_model
    global chroma_client
    global collection
    global bm25_index
    global all_chunks
    global chunk_map
    global agency_list

    if INITIALIZED:
        return

    if INITIALIZATION_ERROR is not None:
        raise RuntimeError("core/functions.py 초기화 실패") from INITIALIZATION_ERROR

    try:
        all_chunks, chunk_map, agency_list = _load_chunks()

        openai_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        device = _detect_device()
        emb_model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
            device=device,
        )
        try:
            emb_model.max_seq_length = min(emb_model.max_seq_length, EMBED_MAX_SEQ_LENGTH)
        except Exception:
            pass

        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        existing = [c.name for c in chroma_client.list_collections()]
        if not existing:
            raise RuntimeError("ChromaDB 컬렉션이 없습니다.")

        chosen = COLLECTION_NAME
        counts: dict[str, int] = {}
        for name in existing:
            try:
                counts[name] = chroma_client.get_collection(name).count()
            except Exception:
                counts[name] = 0

        if counts.get(chosen, 0) <= 0:
            chosen = max(counts, key=counts.get)
            if counts[chosen] <= 0:
                raise RuntimeError("사용 가능한 ChromaDB 컬렉션이 없습니다.")

        collection = chroma_client.get_collection(name=chosen)

        bm25_corpus = [tokenize(c.get("text", "")) for c in all_chunks]
        bm25_index = BM25Okapi(bm25_corpus)

        INITIALIZED = True
    except Exception as e:
        INITIALIZATION_ERROR = e
        raise


def set_agency_filter(agency: str | None) -> None:
    global ACTIVE_AGENCY_FILTER
    if agency in (None, "", "전체"):
        ACTIVE_AGENCY_FILTER = None
    else:
        ACTIVE_AGENCY_FILTER = agency


def get_agency_options() -> list[str]:
    _ensure_initialized()
    return agency_list[:]


AGENCY_ALIASES = {
    "가스공사": "한국가스공사",
    "한국 가스공사": "한국가스공사",
    "수자원공사": "한국수자원공사",
    "k-water": "한국수자원공사",
    "kwater": "한국수자원공사",
    "수은": "한국수출입은행",
    "수출입은행": "한국수출입은행",
    "수출입 은행": "한국수출입은행",
    "gkl": "그랜드코리아레저(주)",
    "그랜드코리아레저": "그랜드코리아레저(주)",
    "그랜드코리아레져": "그랜드코리아레저(주)",
    "그렌드코리아레져": "그랜드코리아레저(주)",
    "경희대학교산학협력단": "경희대학교산학협력단",
    "경희대학교 산학협력단": "경희대학교산학협력단",
    "굥희머학교": "경희대학교산학협력단",
    "경희대 산학협력단": "경희대학교산학협력단",
    "경희대학교": "경희대학교",
    "경희대": "경희대학교",
    "인천공항운영서비스": "인천공항운영서비스(주)",
    "인천공항운영서비스㈜": "인천공항운영서비스(주)",
    "인천공항운영서비스 주": "인천공항운영서비스(주)",
    "서울시립대": "서울시립대학교",
    "서울시륍대": "서울시립대학교",
    "남서울대": "남서울대학교",
    "남서율대햑교": "남서울대학교",
    "gist": "광주과학기술원",
    "지스트": "광주과학기술원",
}

NUMERIC_QUERY_WORDS = [
    "예산",
    "금액",
    "사업비",
    "소요예산",
    "총사업비",
    "용역금액",
    "용역예산",
    "발주금액",
    "추정가격",
    "기초금액",
    "배정액",
    "원",
    "만원",
    "억 원",
    "억원",
    "억",
    "%",
    "비율",
    "차액",
    "합계",
    "총액",
    "얼마",
]

CORE_AMOUNT_WORDS = [
    "용역금액",
    "용역예산",
    "사업예산",
    "사업금액",
    "사업비",
    "총사업비",
    "소요예산",
    "기초금액",
    "추정가격",
    "배정예산",
    "배정액",
    "계약금액",
]

LOW_PRIORITY_AMOUNT_WORDS = [
    "입찰가격",
    "평가",
    "평점",
    "배점",
    "실적",
    "유사사업",
    "수행실적",
    "평가기준",
    "가격평가",
    "정량평가",
    "정성평가",
    "참여율",
    "지분율",
    "하도급",
    "보증금율",
]

BUDGET_TEXT_WORDS = CORE_AMOUNT_WORDS + LOW_PRIORITY_AMOUNT_WORDS

REGION_QUERY_WORDS = ["과업 대상 지역", "대상 지역", "대상지역", "지자체", "어느 지역", "지역"]
REGION_TEXT_WORDS = [
    "과업 대상 지역",
    "대상지역",
    "대상 지역",
    "과업대상",
    "과업 대상",
    "대상 지",
    "위치",
    "용인시",
    "성남시",
    "광주시",
    "하남시",
    "지자체",
    "공급 지역",
    "급수구역",
    "계획 공급 지역",
]

DURATION_QUERY_WORDS = ["기간", "개월", "일정", "안정화", "추진 일정", "전체 몇 개월"]
DURATION_TEXT_WORDS = [
    "사업기간",
    "용역기간",
    "계약체결일",
    "착수일",
    "개월",
    "안정화",
    "추진일정",
    "정식 오픈",
    "구축 및 개발",
    "통합테스트",
]

EQUIPMENT_QUERY_WORDS = ["서버", "장비", "인프라", "스토리지", "SAN", "DB", "AP"]
EQUIPMENT_TEXT_WORDS = [
    "ECR",
    "시스템 장비",
    "인프라",
    "서버",
    "스토리지",
    "SAN",
    "스위치",
    "S/4 HANA",
    "S/4HANA",
    "DB 서버",
    "AP 서버",
    "백업",
    "LDM",
    "외부연계",
    "운영 DB",
    "품질 DB",
    "BW DB",
    "웹서버",
    "WAS",
]

MONEY_REGEX = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:원|천원|만\s*원|만원|백만\s*원|백만원|억\s*원|억원|억|%|퍼센트)"
)

STOP_TERMS = {
    "한국",
    "사업",
    "용역",
    "구축",
    "시스템",
    "프로젝트",
    "예산",
    "금액",
    "얼마",
    "전체",
    "관련",
    "이번",
    "연도",
    "추진",
    "발주",
    "타당성",
    "조사",
    "기본계획",
    "수립",
    "정보",
    "운영",
    "기능",
    "개선",
    "고도화",
    "기간",
    "과업",
    "대상",
    "지역",
    "포함",
    "정확히",
    "대략",
    "무엇",
    "어떤",
    "주요",
    "신규",
}


def normalize_for_agency_match(text: str) -> str:
    text = nfc(str(text)).lower()
    text = re.sub(r"[\s_()\[\]{}·\-.,/「」『』'\"“”‘’㈜]", "", text)
    return text


def normalize_for_keyword(text: str) -> str:
    text = nfc(str(text)).lower()
    text = re.sub(r"[\s_()\[\]{}·\-.,/「」『』'\"“”‘’㈜]", "", text)
    return text


def resolve_agency_name(name: str) -> str | None:
    _ensure_initialized()
    if not name:
        return None

    target_norm = normalize_for_agency_match(name)

    for agency in agency_list:
        if normalize_for_agency_match(agency) == target_norm:
            return agency

    for agency in agency_list:
        agency_norm = normalize_for_agency_match(agency)
        if target_norm and (target_norm in agency_norm or agency_norm in target_norm):
            return agency

    return None


def detect_agencies_from_query(query: str) -> list[str]:
    _ensure_initialized()
    q_norm = normalize_for_agency_match(query)
    found: list[str] = []

    for alias, canonical in AGENCY_ALIASES.items():
        alias_norm = normalize_for_agency_match(alias)
        if alias_norm and alias_norm in q_norm:
            resolved = resolve_agency_name(canonical)
            if resolved:
                found.append(resolved)

    for agency in agency_list:
        agency_norm = normalize_for_agency_match(agency)
        if agency_norm and agency_norm in q_norm:
            found.append(agency)

    unique: list[str] = []
    seen = set()

    found = sorted(found, key=lambda x: len(normalize_for_agency_match(x)), reverse=True)

    for agency in found:
        key = normalize_for_agency_match(agency)
        if any(key in normalize_for_agency_match(u) and key != normalize_for_agency_match(u) for u in unique):
            continue
        if key not in seen:
            unique.append(agency)
            seen.add(key)

    return unique


def get_agency_filter(query: str) -> tuple[str | None, float]:
    _ensure_initialized()

    detected = detect_agencies_from_query(query)
    if detected:
        return detected[0], 100.0

    query_words = re.findall(r"[가-힣A-Za-z0-9]+", nfc(query))
    if not query_words:
        return None, 0.0

    candidate = " ".join(query_words)
    best = process.extractOne(candidate, agency_list, scorer=fuzz.WRatio, score_cutoff=75)
    if best:
        name, score, _ = best
        return name, float(score)

    return None, 0.0


def detect_agency_from_query(query: str) -> str | None:
    agencies = detect_agencies_from_query(query)
    return agencies[0] if agencies else None


def has_any_word(text: str, words: list[str]) -> bool:
    t = str(text).lower()
    return any(w.lower() in t for w in words)


def is_numeric_query(query: str) -> bool:
    q = nfc(str(query))
    return any(w in q for w in NUMERIC_QUERY_WORDS) or bool(MONEY_REGEX.search(q))


def chunk_text_pool(chunk: dict) -> str:
    meta = chunk.get("metadata", {}) or {}
    return "\n".join(
        [
            str(chunk.get("text", "")),
            str(chunk.get("numeric_summary", "")),
            str(meta.get("numeric_summary", "")),
            str(meta.get("section", "")),
            str(meta.get("사업명", "")),
            str(meta.get("발주기관", "")),
            str(meta.get("파일명", "")),
        ]
    )


def is_core_amount_line(text: str) -> bool:
    return bool(MONEY_REGEX.search(str(text))) and has_any_word(text, CORE_AMOUNT_WORDS)


def is_low_priority_amount_line(text: str) -> bool:
    return bool(MONEY_REGEX.search(str(text))) and has_any_word(text, LOW_PRIORITY_AMOUNT_WORDS)


def chunk_has_core_amount_signal(chunk: dict) -> bool:
    return is_core_amount_line(chunk_text_pool(chunk))


def chunk_has_money_signal(chunk: dict) -> bool:
    text = chunk_text_pool(chunk)
    return bool(MONEY_REGEX.search(text)) and has_any_word(text, BUDGET_TEXT_WORDS)


def extract_business_terms(query: str) -> list[str]:
    q = nfc(str(query))
    quoted = re.findall(r"[\"'‘’“”](.*?)[\"'‘’“”]", q)
    terms = []

    for item in quoted:
        for t in re.findall(r"[가-힣A-Za-z0-9+]+", item):
            if len(t) >= 2 and t not in STOP_TERMS:
                terms.append(t)

    for t in re.findall(r"[가-힣A-Za-z0-9+]+", q):
        if len(t) >= 2 and t not in STOP_TERMS:
            terms.append(t)

    agency_norms = [normalize_for_keyword(a) for a in detect_agencies_from_query(q)]

    cleaned = []
    seen = set()

    for t in terms:
        nt = normalize_for_keyword(t)
        if not nt:
            continue

        if any(nt in a or a in nt for a in agency_norms):
            continue

        if nt not in seen:
            cleaned.append(t)
            seen.add(nt)

    return cleaned[:20]


def business_overlap_score(query: str, chunk: dict) -> float:
    terms = extract_business_terms(query)

    if not terms:
        return 0.0

    meta = chunk.get("metadata", {}) or {}
    target_text = " ".join(
        [
            str(meta.get("사업명", "")),
            str(meta.get("파일명", "")),
            str(meta.get("section", "")),
            str(chunk.get("text", ""))[:500],
        ]
    )

    target_norm = normalize_for_keyword(target_text)

    hit = 0
    for t in terms:
        nt = normalize_for_keyword(t)
        if nt and nt in target_norm:
            hit += 1

    return hit / max(len(terms), 1)


def is_probably_wrong_business(query: str, chunk: dict) -> bool:
    terms = extract_business_terms(query)

    if len(terms) < 3:
        return False

    agencies = detect_agencies_from_query(query)
    if not agencies:
        return False

    meta = chunk.get("metadata", {}) or {}
    chunk_agency = normalize_for_agency_match(meta.get("발주기관", ""))
    agency_hit = any(chunk_agency == normalize_for_agency_match(a) for a in agencies)

    if not agency_hit:
        return False

    if chunk_has_core_amount_signal(chunk):
        return False

    return business_overlap_score(query, chunk) < 0.08


def get_query_signal_types(query: str) -> set[str]:
    q = nfc(str(query))
    signals = set()

    if is_numeric_query(q):
        signals.add("numeric")
    if has_any_word(q, REGION_QUERY_WORDS):
        signals.add("region")
    if has_any_word(q, DURATION_QUERY_WORDS):
        signals.add("duration")
    if has_any_word(q, EQUIPMENT_QUERY_WORDS):
        signals.add("equipment")

    return signals


def chunk_matches_query_signal(query: str, chunk: dict) -> bool:
    signals = get_query_signal_types(query)
    text = chunk_text_pool(chunk)

    if "numeric" in signals and chunk_has_money_signal(chunk):
        return True
    if "region" in signals and has_any_word(text, REGION_TEXT_WORDS):
        return True
    if "duration" in signals and has_any_word(text, DURATION_TEXT_WORDS):
        return True
    if "equipment" in signals and has_any_word(text, EQUIPMENT_TEXT_WORDS):
        return True

    return False


def make_source(meta: dict) -> str:
    agency = meta.get("발주기관", "")
    biz = meta.get("사업명", "")
    fname = meta.get("파일명", "")
    section = meta.get("section", "")
    page = meta.get("page", "")

    loc = f"§{section}" if section else (f"p.{page}" if page else "")
    loc_str = f" | {loc}" if loc else ""

    if agency or biz:
        return f"[{agency} — {biz}{loc_str}]"
    return f"[{fname}{loc_str}]"


def apply_contextual_boost(query: str, chunk: dict, score: float, agency_filter: str | None = None) -> float:
    boosted = float(score)

    overlap = business_overlap_score(query, chunk)
    if overlap > 0:
        boosted += BUSINESS_OVERLAP_BOOST * overlap

    if chunk_matches_query_signal(query, chunk):
        boosted += EVIDENCE_CHUNK_BOOST

    if is_numeric_query(query) and chunk_has_core_amount_signal(chunk):
        boosted += CORE_AMOUNT_BOOST
    elif is_numeric_query(query) and chunk_has_money_signal(chunk):
        boosted += NUMERIC_CHUNK_BOOST

    if agency_filter:
        meta = chunk.get("metadata", {}) or {}
        if normalize_for_agency_match(meta.get("발주기관", "")) == normalize_for_agency_match(agency_filter):
            boosted += AGENCY_CHUNK_BOOST

    if is_probably_wrong_business(query, chunk):
        boosted -= WRONG_BUSINESS_PENALTY

    return boosted


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return scores


def _retrieve_single(
    query: str,
    top_k: int = RETRIEVAL_EVAL_K,
    agency_filter: str | None = None,
) -> list[dict]:
    _ensure_initialized()
    assert collection is not None
    assert bm25_index is not None

    query = nfc(str(query))

    where_filter = {"발주기관": agency_filter} if agency_filter else None
    q_emb = get_embeddings([query])[0]
    dense_n = min(DENSE_CANDIDATES, collection.count())

    dense_kwargs = {
        "query_embeddings": [q_emb],
        "n_results": dense_n,
        "include": ["documents", "metadatas", "distances"],
    }

    if where_filter:
        available = sum(
            1
            for c in all_chunks
            if normalize_for_agency_match(c.get("metadata", {}).get("발주기관", ""))
            == normalize_for_agency_match(agency_filter)
        )
        if available <= 0:
            return []

        dense_kwargs["where"] = where_filter
        dense_kwargs["n_results"] = max(1, min(dense_n, available))

    dense_res = collection.query(**dense_kwargs)
    dense_ids = dense_res.get("ids", [[]])[0]

    bm25_scores = bm25_index.get_scores(tokenize(query))

    if agency_filter:
        a_norm = normalize_for_agency_match(agency_filter)
        filtered_idx = [
            i
            for i, c in enumerate(all_chunks)
            if normalize_for_agency_match(c.get("metadata", {}).get("발주기관", "")) == a_norm
        ]
    else:
        filtered_idx = list(range(len(all_chunks)))

    bm25_ranked = sorted(
        [(i, bm25_scores[i]) for i in filtered_idx],
        key=lambda x: x[1],
        reverse=True,
    )[:BM25_CANDIDATES]

    bm25_ids = [all_chunks[i]["chunk_id"] for i, _ in bm25_ranked]

    rrf_scores = reciprocal_rank_fusion([dense_ids, bm25_ids])
    candidate_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:RRF_CANDIDATES]

    candidates = [chunk_map[cid] for cid in candidate_ids if cid in chunk_map]

    ranked = sorted(
        [(c, rrf_scores.get(c["chunk_id"], 0.0)) for c in candidates],
        key=lambda x: x[1],
        reverse=True,
    )

    adjusted = []
    for chunk, score in ranked:
        new_score = apply_contextual_boost(query, chunk, score, agency_filter)
        adjusted.append((chunk, new_score))

    adjusted = sorted(adjusted, key=lambda x: x[1], reverse=True)

    results: list[dict] = []
    for chunk, score in adjusted[:top_k]:
        item = dict(chunk)
        item["metadata"] = dict(chunk.get("metadata", {}) or {})
        item["score"] = round(float(score), 6)
        item["source"] = make_source(item["metadata"])
        item["business_overlap"] = round(business_overlap_score(query, item), 4)
        item["is_core_amount"] = bool(chunk_has_core_amount_signal(item))
        results.append(item)

    return results


def find_evidence_chunks_from_same_docs(
    query: str,
    retrieved_chunks: list[dict],
    max_add: int = 8,
) -> list[dict]:
    if not retrieved_chunks:
        return []

    target_files: list[str] = []
    for c in retrieved_chunks[:12]:
        meta = c.get("metadata", {}) or {}
        fname = meta.get("파일명", "")
        if fname and fname not in target_files:
            target_files.append(fname)

    evidence = []

    for fname in target_files:
        same_doc_chunks = [c for c in all_chunks if c.get("metadata", {}).get("파일명", "") == fname]

        scored = []

        for c in same_doc_chunks:
            text = chunk_text_pool(c)
            cc = dict(c)
            cc["metadata"] = dict(c.get("metadata", {}) or {})
            cc["source"] = make_source(cc["metadata"])

            score = 0.0

            if is_numeric_query(query) and chunk_has_core_amount_signal(c):
                score += 6.0
            elif is_numeric_query(query) and chunk_has_money_signal(c):
                score += 2.0

            if has_any_word(query, REGION_QUERY_WORDS) and has_any_word(text, REGION_TEXT_WORDS):
                score += 4.0
            if has_any_word(query, DURATION_QUERY_WORDS) and has_any_word(text, DURATION_TEXT_WORDS):
                score += 4.0
            if has_any_word(query, EQUIPMENT_QUERY_WORDS) and has_any_word(text, EQUIPMENT_TEXT_WORDS):
                score += 4.0

            score += business_overlap_score(query, cc) * 1.5

            if is_probably_wrong_business(query, cc):
                score -= 0.3

            if score > 0:
                cc["score"] = round(score, 6)
                cc["business_overlap"] = round(business_overlap_score(query, cc), 4)
                cc["is_core_amount"] = bool(chunk_has_core_amount_signal(c))
                scored.append(cc)

        scored = sorted(scored, key=lambda x: x.get("score", 0), reverse=True)
        evidence.extend(scored[:3])

    seen = set()
    unique = []

    for c in evidence:
        cid = c.get("chunk_id")
        if cid and cid not in seen:
            unique.append(c)
            seen.add(cid)

    return unique[:max_add]


def merge_retrieved_lists(lists: list[list[dict]], top_k: int) -> list[dict]:
    merged = []
    seen = set()

    max_len = max((len(lst) for lst in lists), default=0)

    for i in range(max_len):
        for lst in lists:
            if i >= len(lst):
                continue

            item = lst[i]
            cid = item.get("chunk_id")

            if cid and cid not in seen:
                merged.append(item)
                seen.add(cid)

            if len(merged) >= top_k:
                return merged

    return merged[:top_k]


def retrieve(query: str, top_k: int = RETRIEVAL_EVAL_K, verbose: bool = False) -> list[dict]:
    _ensure_initialized()
    query = nfc(str(query))

    if ACTIVE_AGENCY_FILTER:
        base = _retrieve_single(query, top_k=top_k, agency_filter=ACTIVE_AGENCY_FILTER)
        evidence = find_evidence_chunks_from_same_docs(query, base, max_add=8)
        return merge_retrieved_lists([evidence, base], top_k=top_k)

    agencies = detect_agencies_from_query(query)

    if verbose:
        print(f"  감지된 발주기관: {agencies if agencies else '없음'}")
        print(f"  signal types: {get_query_signal_types(query)}")
        print(f"  business terms: {extract_business_terms(query)}")

    result_sets = []

    if len(agencies) >= 2:
        for agency in agencies:
            result_sets.append(_retrieve_single(query, top_k=max(top_k, ANSWER_TOP_K), agency_filter=agency))

        result_sets.append(_retrieve_single(query, top_k=top_k, agency_filter=None))
        base = merge_retrieved_lists(result_sets, top_k=top_k)
    elif len(agencies) == 1:
        base = _retrieve_single(query, top_k=top_k, agency_filter=agencies[0])
        if not base:
            base = _retrieve_single(query, top_k=top_k, agency_filter=None)
    else:
        base = _retrieve_single(query, top_k=top_k, agency_filter=None)

    evidence = find_evidence_chunks_from_same_docs(query, base, max_add=8)
    final = merge_retrieved_lists([evidence, base], top_k=top_k)

    return final


SYSTEM_PROMPT_V7E = """당신은 RFP(입찰 공고) 문서 전문 분석가입니다.
반드시 아래 규칙을 따른다.

[최우선 원칙]
1. 답변은 반드시 제공된 컨텍스트와 질문에 직접 제시된 수치만 사용한다.
2. 컨텍스트에 없는 문서 사실은 추측하지 않는다.
3. 질문 자체에 금액, 비율, 수량이 명시되어 있고 단순 산술 계산만 요구하는 경우에는 그 수치를 사용하여 계산할 수 있다.
4. 단, 문서에 존재해야만 알 수 있는 사실은 반드시 컨텍스트에 근거가 있어야 한다.
5. 출력 형식은 항상 **답변**, **근거**, **출처** 세 구역을 유지한다.

[거부 규칙]
컨텍스트에 질문이 요구하는 문서 기반 정보가 없으면:
- **답변** 첫 줄을 반드시 "제공된 문서에서 해당 정보를 찾을 수 없습니다."로 시작한다.
- 그래도 **근거**, **출처** 구역은 생략하지 않는다.
- **근거**에는 어떤 문서는 검색되었고, 어떤 핵심 정보가 없었는지 간단히 적는다.
- **출처**에는 확인한 문서의 발주기관, 사업명, 파일명을 적는다.
- 일반 지식, 상식, 추정, 가능성 표현으로 보완하지 않는다.

[추정 금지]
다음 표현은 사용하지 않는다:
- "가능성이 높습니다"
- "추정됩니다"
- "일반적으로"
- "대부분의 경우"
- "추가 확인이 필요합니다" 단독 표현
- "공식 문서를 참조해야 합니다" 단독 표현
문서에 있으면 문서 내용을 답하고, 없으면 없다고 답한다.

[예산/금액/수치 인식 규칙]
다음 표현은 예산 또는 금액의 핵심 후보로 본다:
- 용역금액
- 용역예산
- 사업예산
- 사업금액
- 사업비
- 총사업비
- 소요예산
- 기초금액
- 추정가격
- 배정예산
- 배정액
- 계약금액

금액 질문에서는 위 표현이 포함된 줄을 가장 우선한다.
입찰가격 평가식, 실적 기준 금액, 배점 기준 금액, 유사사업 수행실적 금액은 해당 사업의 예산 답변 후보보다 후순위로 본다.
특히 "100분의 80", "100분의 70", "5억원 이상 실적", "1억원 이상 ~ 2억원 미만" 같은 표현은 평가 기준일 수 있으므로, 용역금액/사업예산으로 직접 답하지 않는다.

[금액 미기재/비공개 규칙]
문서에 예산이 미기재, 비공개, 확인 불가라고 되어 있으면 금액을 생성하지 않는다.
다른 유사 사업의 금액, 입찰 기준 금액, 실적 기준 금액, 평가 기준 금액을 해당 사업 예산으로 대체하지 않는다.
질문 대상 사업의 예산이 명시되지 않았으면 "제공된 문서에서 해당 사업의 예산 금액을 찾을 수 없습니다."라고 답한다.

[질문 제시 수치 계산 규칙]
질문 자체에 구체적인 금액, 비율, 수량이 명시된 경우:
1. 해당 수치는 컨텍스트 확인 없이 산술 계산에 사용할 수 있다.
2. 이 경우 계산 결과 옆에 "(질문 제시 수치 기반)"이라고 표기한다.
3. 다만 질문 속 수치가 어떤 문서의 공식 예산인지 검증해야 하는 질문이면 컨텍스트 근거가 필요하다.
4. 질문이 "A는 1억, B는 1.5억일 때 합계는?"처럼 수치를 직접 제공하면 합산할 수 있다.

[수치 계산 규칙]
1. 계산 전 원문 수치 또는 질문 제시 수치를 먼저 명시한다.
2. 계산식을 간단히 제시한다.
3. 분수는 반드시 정확히 변환한다.
4. 단위는 가능한 한 원(₩) 기준으로 통일한다.
5. 억원, 만원 표기는 괄호로 원화 병기한다.
6. 계산 결과는 반올림 여부를 명확히 한다.
7. 계산식은 실제 연산과 일치해야 한다.

[기간 질문 규칙]
기간 질문에서는 문서에 명시된 개월 수, 시작일, 종료일만 사용한다.
"상반기", "오픈 이후", "안정화 활동", "일반적으로" 같은 표현을 임의로 개월 수로 환산하지 않는다.
명시 기간이 없으면 계산하지 않는다.
사업기간, 용역기간, 구축기간, 안정화 기간이 서로 다르면 항목별로 구분한다.

[다중 기관/다중 사업 비교 규칙]
두 개 이상의 기관 또는 사업을 비교, 합산, 차액 계산하는 질문에서는:
1. 각 기관/사업별로 컨텍스트에 정보가 있는지 먼저 분리한다.
2. 필요한 정보가 모두 있을 때만 계산한다.
3. 일부 정보만 있으면 어떤 정보가 있고 없는지 명확히 말한다.

[문서 검토 규칙]
1. 컨텍스트의 [정답 후보 요약]을 먼저 확인한다.
2. 그 다음 [문서 1]부터 [문서 N]까지 검토한다.
3. 하나의 문서만 보고 성급히 답하지 않는다.
4. 핵심 용어, 기술명, 기관명, 사업명, 항목명은 원문 표현을 우선 사용한다.
5. 문서 제목만 맞고 본문에 답이 없으면 답이 있다고 보지 않는다.
6. 질문의 사업명과 다른 사업 문서의 금액을 답으로 사용하지 않는다.

[출력 형식]
반드시 아래 형식만 사용한다.

**답변**
질문에 직접 답한다. 계산이 있으면 원문 수치, 계산식, 결과를 포함한다.

**근거**
[문서 N] 형식으로 근거를 적는다.

**출처**
발주기관 — 사업명 | 파일명
출처를 특정할 수 없으면 "해당 없음"이라고 적는다.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_V7E


TOOL_INSTRUCTION = """
[도구 사용 규칙]
직접 계산하거나 추측하지 말고, 아래 상황에서는 반드시 해당 도구를 호출하라.

- 예산 차액·합계·비율·퍼센트 계산      → calculate
- 원화 금액을 억원/만원 단위로 변환      → format_currency
- 날짜 간격 또는 사업 기간 계산          → date_diff
- VAT 포함/제외 금액 계산               → vat_calculator
- 사업의 발주기관·예산·기간 등 핵심 정보 → extract_entities
- 유사 사업 추천 또는 다른 사업과 비교   → related_docs_finder
"""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "수학 수식을 계산합니다. 예산 차액, 합계, 비율 등 숫자 계산이 필요하면 반드시 이 도구를 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수식 (예: '14107009000 - 11270000000', '5031000000 * 0.1')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_currency",
            "description": "원화 금액을 억원/만원 한국식 단위로 변환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "변환할 금액 (원 단위 정수)",
                    }
                },
                "required": ["amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_diff",
            "description": "두 날짜 사이의 일수와 개월 수를 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "시작 날짜 (YYYY-MM-DD 또는 YYYY.MM.DD)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "종료 날짜 (YYYY-MM-DD 또는 YYYY.MM.DD)",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vat_calculator",
            "description": "VAT(부가가치세) 포함/제외 금액을 계산합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "기준 금액 (원 단위)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["add", "remove"],
                        "description": "add: VAT 포함 금액 계산 / remove: VAT 제외 금액 계산",
                    },
                },
                "required": ["amount", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": (
                "RFP 문서에서 발주기관, 사업명, 예산, 기간, 공고번호 등 핵심 정보를 구조화하여 추출합니다. "
                "사용자가 특정 사업의 기본 정보 요약을 요청할 때 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "정보를 추출할 사업명 또는 발주기관 키워드",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "related_docs_finder",
            "description": (
                "현재 질문과 유사한 다른 RFP 사업을 검색하여 반환합니다. "
                "유사 사업 추천, 비교 분석, '다른 비슷한 사업' 요청 시 사용하세요."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "유사 사업을 찾을 검색어 또는 사업명",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 유사 사업 수 (기본값: 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def tool_calculate(expression: str) -> str:
    try:
        expr = re.sub(r"[^0-9\+\-\*\/\.\(\)\s]", "", str(expression))
        result = eval(expr, {"__builtins__": {}}, {})
        won = int(round(result))

        if won >= 100_000_000:
            return f"계산 결과: {won:,}원 (약 {won / 100_000_000:.2f}억 원)"
        if won >= 10_000:
            return f"계산 결과: {won:,}원 (약 {won / 10_000:,.0f}만 원)"
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"


def tool_format_currency(amount) -> str:
    try:
        won = int(float(amount))
        if won >= 100_000_000:
            return f"{won:,}원 (약 {won / 100_000_000:,.2f}억 원)"
        if won >= 10_000:
            return f"{won:,}원 (약 {won / 10_000:,.0f}만 원)"
        return f"{won:,}원"
    except Exception as e:
        return f"변환 오류: {e}"


def tool_date_diff(start: str, end: str) -> str:
    try:
        for fmt in ["%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"]:
            try:
                s = datetime.datetime.strptime(start.strip(), fmt)
                e = datetime.datetime.strptime(end.strip(), fmt)
                days = (e - s).days
                months = round(days / 30.44, 1)
                return f"기간: {days}일 (약 {months}개월)"
            except ValueError:
                continue
        return "날짜 형식 오류 (YYYY-MM-DD 또는 YYYY.MM.DD 사용)"
    except Exception as e:
        return f"계산 오류: {e}"


def tool_vat_calculator(amount, mode: str) -> str:
    try:
        won = int(float(amount))
        if mode == "add":
            total = int(won * 1.1)
            return f"VAT 포함 금액: {total:,}원 (약 {total / 100_000_000:.2f}억 원)"
        base = int(won / 1.1)
        return f"VAT 제외 금액: {base:,}원 (약 {base / 100_000_000:.2f}억 원)"
    except Exception as e:
        return f"계산 오류: {e}"


def tool_extract_entities(query: str) -> str:
    try:
        chunks = retrieve(query, top_k=5)
        if not chunks:
            return "관련 문서를 찾을 수 없습니다."

        results = []
        seen = set()

        for c in chunks:
            meta = c.get("metadata", {}) or {}
            agency = meta.get("발주기관", "")
            biz = meta.get("사업명", "")
            filename = meta.get("파일명", "")
            text = c.get("text", "")

            key = f"{agency}_{biz}"
            if key in seen:
                continue
            seen.add(key)

            budget = "미확인"
            for pattern in [
                r"(?:용역금액|용역예산|사업예산|사업금액|사업비|총사업비|소요예산|기초금액|계약금액)\s*[：:]\s*([\d,]+원?)",
                r"([\d,]+)\s*원\s*\(",
                r"금\s*([\d,]+)\s*원",
            ]:
                m = re.search(pattern, text)
                if m:
                    raw = m.group(1).replace(",", "").replace("원", "")
                    try:
                        won = int(raw)
                        budget = f"{won:,}원" + (
                            f" (약 {won / 100_000_000:.1f}억)" if won >= 100_000_000 else ""
                        )
                    except Exception:
                        budget = m.group(1)
                    break

            duration = "미확인"
            m = re.search(r"(\d+)\s*개월", text)
            if m:
                duration = f"{m.group(1)}개월"

            notice_no = "미확인"
            m = re.search(r"공고\s*번호\s*[：:]\s*([A-Za-z0-9\-]+)", text)
            if m:
                notice_no = m.group(1)

            results.append(
                f"[{len(results) + 1}]\n"
                f"  발주기관 : {agency}\n"
                f"  사업명   : {biz}\n"
                f"  예산     : {budget}\n"
                f"  기간     : {duration}\n"
                f"  공고번호 : {notice_no}\n"
                f"  파일명   : {filename}"
            )
            if len(results) >= 3:
                break

        return "\n\n".join(results) if results else "관련 정보를 추출할 수 없습니다."
    except Exception as e:
        return f"엔티티 추출 오류: {e}"


def tool_related_docs_finder(query: str, top_k: int = 5) -> str:
    try:
        _ensure_initialized()
        assert collection is not None

        query_emb = get_embeddings([query])[0]
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k * 3, collection.count()),
            include=["metadatas", "distances"],
        )

        seen = {}
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            agency = meta.get("발주기관", "")
            biz = meta.get("사업명", "")
            key = f"{agency}_{biz}"
            sim = round(1 - dist, 4)

            if key not in seen or seen[key]["유사도"] < sim:
                seen[key] = {
                    "발주기관": agency,
                    "사업명": biz,
                    "파일명": meta.get("파일명", ""),
                    "유사도": sim,
                }

        top = sorted(seen.values(), key=lambda x: x["유사도"], reverse=True)[:top_k]
        if not top:
            return "유사 사업을 찾을 수 없습니다."

        lines = [f"유사 사업 TOP {len(top)}\n"]
        for i, r in enumerate(top, 1):
            lines.append(
                f"[{i}] {r['발주기관']} — {r['사업명']}\n"
                f"     유사도: {r['유사도']} | 파일: {r['파일명']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"유사 사업 검색 오류: {e}"


def execute_tool(name: str, args: dict) -> str:
    dispatch = {
        "calculate": lambda a: tool_calculate(a.get("expression", "")),
        "format_currency": lambda a: tool_format_currency(a.get("amount", 0)),
        "date_diff": lambda a: tool_date_diff(a.get("start_date", ""), a.get("end_date", "")),
        "vat_calculator": lambda a: tool_vat_calculator(a.get("amount", 0), a.get("mode", "add")),
        "extract_entities": lambda a: tool_extract_entities(a.get("query", "")),
        "related_docs_finder": lambda a: tool_related_docs_finder(a.get("query", ""), a.get("top_k", 5)),
    }

    fn = dispatch.get(name)
    return fn(args) if fn else f"알 수 없는 도구: {name}"


def chat_ollama_with_tools(
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 800,
    use_tools: bool = True,
) -> str:
    if ollama_client is None:
        raise RuntimeError("`ollama` 패키지가 설치되지 않았습니다. `pip install ollama` 후 다시 시도하세요.")

    msgs = list(messages)
    tools = TOOL_SCHEMAS if use_tools else None
    max_iter = 5
    tool_log = []

    for _ in range(max_iter):
        kwargs = {
            "model": model,
            "messages": msgs,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            kwargs["tools"] = tools

        response = ollama_client.chat(**kwargs)
        msg = response.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            if tool_log:
                print(f"  🔧 Tool 사용: {tool_log}")
            return msg.content or ""

        msgs.append({"role": "assistant", "content": msg.content or ""})

        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = (
                dict(tc.function.arguments)
                if hasattr(tc.function.arguments, "items")
                else tc.function.arguments
            )
            result = execute_tool(fn_name, fn_args)
            tool_log.append(f"{fn_name} → {str(result)[:60]}")
            msgs.append({"role": "tool", "content": str(result)})

    return msg.content or ""


def convert_cheonwon(text: str) -> str:
    def replacer(m):
        raw = m.group(1).replace(",", "")
        won = int(raw) * 1000
        orig = m.group(1)
        return f"{orig}천원 ({won:,}원)"

    return re.sub(r"([\d,]+)\s*천원", replacer, str(text))


def normalize_answer_format(answer: str) -> str:
    text = str(answer).strip()

    text = re.sub(r"\*\*\s*답변\s*:?\s*\*\*", "**답변**", text)
    text = re.sub(r"\*\*\s*근거\s*:?\s*\*\*", "**근거**", text)
    text = re.sub(r"\*\*\s*출처\s*:?\s*\*\*", "**출처**", text)

    if "**답변**" not in text:
        text = f"**답변**\n{text}"
    if "**근거**" not in text:
        text += "\n\n**근거**\n해당 없음"
    if "**출처**" not in text:
        text += "\n\n**출처**\n해당 없음"

    return text.strip()


def extract_section_local(text: str, name: str) -> str:
    pattern = rf"\*\*\s*{re.escape(name)}\s*:?\s*\*\*\s*\n?(.*?)(?=\n\s*\*\*\s*(?:답변|근거|출처)\s*:?\s*\*\*|\Z)"
    m = re.search(pattern, str(text), re.DOTALL)
    return m.group(1).strip() if m else ""


def replace_section_local(text: str, name: str, new_content: str) -> str:
    ans = extract_section_local(text, "답변")
    gnd = extract_section_local(text, "근거")
    src = extract_section_local(text, "출처")

    if name == "답변":
        ans = new_content
    elif name == "근거":
        gnd = new_content
    elif name == "출처":
        src = new_content

    return f"**답변**\n{ans}\n\n**근거**\n{gnd}\n\n**출처**\n{src}".strip()


def is_missing_section_content(content: str) -> bool:
    c = str(content).strip()
    return c in ("", "해당 없음", "없음", "—", "-")


def parse_money_to_won_local(text: str) -> list[int]:
    text = str(text)
    values = []

    patterns = [
        (r"(\d[\d,]*(?:\.\d+)?)\s*억\s*원", 100_000_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*억원", 100_000_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*억", 100_000_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*백만\s*원", 1_000_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*백만원", 1_000_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*만\s*원", 10_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*만원", 10_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*천\s*원", 1_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*천원", 1_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*원", 1),
    ]

    for pat, unit in patterns:
        for m in re.finditer(pat, text):
            raw = m.group(1).replace(",", "")
            try:
                values.append(int(round(float(raw) * unit)))
            except Exception:
                pass

    return sorted(set(values))


def format_won_amount(won: int) -> str:
    won = int(won)

    if won >= 100_000_000:
        eok = won / 100_000_000
        return f"{won:,}원 (약 {eok:,.2f}억 원)"

    if won >= 10_000:
        man = won / 10_000
        return f"{won:,}원 (약 {man:,.0f}만 원)"

    return f"{won:,}원"


def extract_question_numbers(query: str) -> list[str]:
    patterns = [
        r"\d[\d,]*(?:\.\d+)?\s*억\s*원?",
        r"\d[\d,]*(?:\.\d+)?\s*억",
        r"\d[\d,]*(?:\.\d+)?\s*만\s*원",
        r"\d[\d,]*(?:\.\d+)?\s*만원",
        r"\d[\d,]*(?:\.\d+)?\s*천\s*원",
        r"\d[\d,]*(?:\.\d+)?\s*천원",
        r"\d[\d,]*(?:\.\d+)?\s*원",
        r"\d[\d,]*(?:\.\d+)?\s*%",
        r"\d+\s*분의\s*\d+",
    ]

    found = []
    for p in patterns:
        found.extend(re.findall(p, str(query)))

    return list(dict.fromkeys([x.strip() for x in found if x.strip()]))


def line_priority(query: str, line: str) -> int:
    line = str(line)

    if bool(MONEY_REGEX.search(line)) and has_any_word(line, ["용역금액", "용역예산"]):
        return 0
    if bool(MONEY_REGEX.search(line)) and has_any_word(line, ["사업예산", "사업금액", "사업비", "소요예산"]):
        return 1
    if bool(MONEY_REGEX.search(line)) and has_any_word(
        line,
        ["총사업비", "기초금액", "추정가격", "배정예산", "배정액", "계약금액"],
    ):
        return 2
    if has_any_word(line, REGION_TEXT_WORDS):
        return 3
    if has_any_word(line, DURATION_TEXT_WORDS):
        return 4
    if has_any_word(line, EQUIPMENT_TEXT_WORDS):
        return 5
    if bool(MONEY_REGEX.search(line)) and has_any_word(line, LOW_PRIORITY_AMOUNT_WORDS):
        return 9
    if bool(MONEY_REGEX.search(line)):
        return 6

    return 8


def extract_candidate_lines(query: str, text: str, max_lines: int = 10) -> list[str]:
    text = str(text)
    raw_lines = re.split(r"[\n\r]+", text)
    candidates = []

    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue

        ok = False

        if is_numeric_query(query) and (MONEY_REGEX.search(line) or has_any_word(line, BUDGET_TEXT_WORDS)):
            ok = True
        if has_any_word(query, REGION_QUERY_WORDS) and has_any_word(line, REGION_TEXT_WORDS):
            ok = True
        if has_any_word(query, DURATION_QUERY_WORDS) and has_any_word(line, DURATION_TEXT_WORDS):
            ok = True
        if has_any_word(query, EQUIPMENT_QUERY_WORDS) and has_any_word(line, EQUIPMENT_TEXT_WORDS):
            ok = True

        if ok:
            candidates.append(line)

    if not candidates:
        compact = re.sub(r"\s+", " ", text).strip()

        if is_numeric_query(query) and (MONEY_REGEX.search(compact) or has_any_word(compact, BUDGET_TEXT_WORDS)):
            candidates.append(compact[:600])
        elif has_any_word(query, REGION_QUERY_WORDS) and has_any_word(compact, REGION_TEXT_WORDS):
            candidates.append(compact[:600])
        elif has_any_word(query, DURATION_QUERY_WORDS) and has_any_word(compact, DURATION_TEXT_WORDS):
            candidates.append(compact[:600])
        elif has_any_word(query, EQUIPMENT_QUERY_WORDS) and has_any_word(compact, EQUIPMENT_TEXT_WORDS):
            candidates.append(compact[:600])

    candidates = list(dict.fromkeys([c for c in candidates if c]))

    candidates = sorted(candidates, key=lambda x: (line_priority(query, x), -len(x)))

    return candidates[:max_lines]


def build_evidence_card(query: str, context_docs: list[dict]) -> str:
    cards = []
    qnums = extract_question_numbers(query)

    if qnums:
        cards.append("[질문에 직접 제시된 수치]\n" + "\n".join(f"- {n}" for n in qnums))

    for i, doc in enumerate(context_docs, start=1):
        meta = doc.get("metadata", {}) or {}

        text = "\n".join(
            [
                str(doc.get("numeric_summary", "")),
                str(meta.get("numeric_summary", "")),
                str(doc.get("text", "")),
            ]
        )

        lines = extract_candidate_lines(query, text, max_lines=10)

        if lines:
            cards.append(
                f"[문서 {i} 핵심 후보]\n"
                f"발주기관: {meta.get('발주기관', '')}\n"
                f"사업명: {meta.get('사업명', '')}\n"
                f"파일명: {meta.get('파일명', '')}\n"
                + "\n".join(f"- {line}" for line in lines[:10])
            )

    if not cards:
        return ""

    return "[정답 후보 요약]\n" + "\n\n".join(cards)


def is_simple_budget_query(query: str) -> bool:
    q = nfc(str(query))

    budget_words = ["예산", "금액", "사업비", "소요 예산", "소요예산", "용역금액", "얼마"]
    block_words = [
        "차액",
        "합계",
        "더한",
        "합산",
        "총 자금",
        "상회",
        "비율",
        "%",
        "3분의",
        "2분의",
        "1/3",
        "2/3",
        "기간",
        "개월",
        "지역",
        "지자체",
        "서버",
        "장비",
        "인프라",
        "둘 중",
        "비교",
        "명시된 사업",
    ]

    return any(w in q for w in budget_words) and not any(w in q for w in block_words)


def is_multi_amount_calc_query(query: str) -> bool:
    q = nfc(str(query))
    calc_words = ["차액", "합계", "합산", "더한", "총액", "총 자금"]
    return is_numeric_query(q) and any(w in q for w in calc_words)


def is_duration_query(query: str) -> bool:
    return has_any_word(query, DURATION_QUERY_WORDS)


def get_target_agency_norms(query: str) -> set[str]:
    return {normalize_for_agency_match(a) for a in detect_agencies_from_query(query)}


def extract_core_amount_candidates(query: str, answer_chunks: list[dict]) -> list[dict]:
    candidates = []
    target_agencies = get_target_agency_norms(query)

    for doc_idx, doc in enumerate(answer_chunks, start=1):
        meta = doc.get("metadata", {}) or {}
        agency_norm = normalize_for_agency_match(meta.get("발주기관", ""))

        if target_agencies and agency_norm not in target_agencies:
            continue

        if is_probably_wrong_business(query, doc):
            continue

        text = "\n".join(
            [
                str(doc.get("numeric_summary", "")),
                str(meta.get("numeric_summary", "")),
                str(doc.get("text", "")),
            ]
        )

        for line in re.split(r"[\n\r]+", text):
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue

            if not is_core_amount_line(line):
                continue
            if is_low_priority_amount_line(line):
                continue

            won_values = parse_money_to_won_local(line)
            won_values = [v for v in won_values if v >= 10_000]

            if not won_values:
                continue

            overlap = business_overlap_score(query, doc)
            priority = line_priority(query, line)

            score = 0.0
            score += max(0, 10 - priority)
            score += overlap * 5
            score += float(doc.get("score", 0)) * 0.1

            candidates.append(
                {
                    "doc_idx": doc_idx,
                    "line": line,
                    "won": max(won_values),
                    "score": score,
                    "source": make_source(meta),
                    "agency": meta.get("발주기관", ""),
                    "business": meta.get("사업명", ""),
                    "filename": meta.get("파일명", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "business_overlap": overlap,
                }
            )

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    unique = []
    seen = set()

    for c in candidates:
        key = (c["agency"], c["business"], c["won"])
        if key not in seen:
            unique.append(c)
            seen.add(key)

    return unique


def try_build_deterministic_budget_answer(query: str, answer_chunks: list[dict]) -> str | None:
    if not USE_DETERMINISTIC_BUDGET_ANSWER:
        return None

    if not is_simple_budget_query(query):
        return None

    candidates = extract_core_amount_candidates(query, answer_chunks)

    if not candidates:
        return None

    best = candidates[0]

    if best["score"] < 1.0:
        return None

    amount_text = format_won_amount(best["won"])

    return (
        f"**답변**\n"
        f"{best['business']}의 예산은 **{amount_text}**입니다.\n\n"
        f"**근거**\n"
        f"[문서 {best['doc_idx']}] {best['line']}\n\n"
        f"**출처**\n"
        f"{best['agency']} — {best['business']} | {best['filename']}"
    )


def try_build_deterministic_multi_amount_answer(query: str, answer_chunks: list[dict]) -> str | None:
    if not USE_DETERMINISTIC_MULTI_AMOUNT_CALC:
        return None

    if not is_multi_amount_calc_query(query):
        return None

    candidates = extract_core_amount_candidates(query, answer_chunks)

    if len(candidates) < 2:
        return None

    agency_best = {}
    for c in candidates:
        key = c["agency"] or c["business"]
        if key not in agency_best or c["score"] > agency_best[key]["score"]:
            agency_best[key] = c

    selected = list(agency_best.values())

    if len(selected) < 2:
        return None

    selected = sorted(selected, key=lambda x: x["score"], reverse=True)[:2]

    a, b = selected[0], selected[1]

    if "차액" in query:
        diff = abs(a["won"] - b["won"])
        answer_line = (
            f"{a['business']}의 금액은 {format_won_amount(a['won'])}, "
            f"{b['business']}의 금액은 {format_won_amount(b['won'])}입니다. "
            f"두 금액의 차액은 **{format_won_amount(diff)}**입니다."
        )
        calc_line = f"{a['won']:,}원 - {b['won']:,}원 = {diff:,}원"
    else:
        total = a["won"] + b["won"]
        answer_line = (
            f"{a['business']}의 금액은 {format_won_amount(a['won'])}, "
            f"{b['business']}의 금액은 {format_won_amount(b['won'])}입니다. "
            f"두 금액의 합계는 **{format_won_amount(total)}**입니다."
        )
        calc_line = f"{a['won']:,}원 + {b['won']:,}원 = {total:,}원"

    return (
        f"**답변**\n"
        f"{answer_line}\n\n"
        f"**근거**\n"
        f"[문서 {a['doc_idx']}] {a['line']}\n"
        f"[문서 {b['doc_idx']}] {b['line']}\n"
        f"계산식: {calc_line}\n\n"
        f"**출처**\n"
        f"{a['agency']} — {a['business']} | {a['filename']}\n"
        f"{b['agency']} — {b['business']} | {b['filename']}"
    )


def try_build_deterministic_duration_answer(query: str, answer_chunks: list[dict]) -> str | None:
    if not USE_DETERMINISTIC_DURATION_ANSWER:
        return None

    if not is_duration_query(query):
        return None

    duration_patterns = [
        r"(?:사업기간|용역기간|구축기간|수행기간)\s*[:：]?\s*([0-9]+)\s*개월",
        r"총\s*([0-9]+)\s*개월",
        r"([0-9]+)\s*개월",
    ]

    candidates = []

    for doc_idx, doc in enumerate(answer_chunks, start=1):
        meta = doc.get("metadata", {}) or {}
        text = str(doc.get("text", ""))

        if is_probably_wrong_business(query, doc):
            continue

        lines = re.split(r"[\n\r]+", text)

        for line in lines:
            clean = re.sub(r"\s+", " ", line).strip()
            if not clean:
                continue

            if not has_any_word(clean, DURATION_TEXT_WORDS):
                continue

            for pat in duration_patterns:
                m = re.search(pat, clean)
                if m:
                    months = int(m.group(1))
                    candidates.append(
                        {
                            "months": months,
                            "line": clean,
                            "doc_idx": doc_idx,
                            "agency": meta.get("발주기관", ""),
                            "business": meta.get("사업명", ""),
                            "filename": meta.get("파일명", ""),
                            "score": business_overlap_score(query, doc) + float(doc.get("score", 0)) * 0.01,
                        }
                    )

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    return (
        f"**답변**\n"
        f"{best['business']}의 명시된 기간은 **{best['months']}개월**입니다.\n\n"
        f"**근거**\n"
        f"[문서 {best['doc_idx']}] {best['line']}\n\n"
        f"**출처**\n"
        f"{best['agency']} — {best['business']} | {best['filename']}"
    )


def filter_chunks_by_target_agency(query: str, chunks: list[dict]) -> list[dict]:
    agencies = detect_agencies_from_query(query)

    if not agencies:
        return chunks

    target_norms = {normalize_for_agency_match(a) for a in agencies}

    kept = []
    removed = []

    for c in chunks:
        meta = c.get("metadata", {}) or {}
        agency_norm = normalize_for_agency_match(meta.get("발주기관", ""))

        if agency_norm in target_norms:
            kept.append(c)
        else:
            removed.append(c)

    if len(kept) >= 3:
        return kept + removed

    return chunks


def select_answer_chunks(query: str, retrieved_chunks: list[dict], top_k: int = ANSWER_TOP_K) -> list[dict]:
    if not retrieved_chunks:
        return []

    chunks = filter_chunks_by_target_agency(query, retrieved_chunks)

    priority = []
    seen = set()

    def add_chunk(c):
        cid = c.get("chunk_id")
        if cid and cid not in seen:
            priority.append(c)
            seen.add(cid)

    agencies = detect_agencies_from_query(query)

    if len(agencies) >= 2:
        for agency in agencies:
            a_norm = normalize_for_agency_match(agency)
            agency_chunks = [
                c
                for c in chunks
                if normalize_for_agency_match(c.get("metadata", {}).get("발주기관", "")) == a_norm
            ]
            agency_chunks = sorted(
                agency_chunks,
                key=lambda c: (
                    not chunk_has_core_amount_signal(c),
                    -business_overlap_score(query, c),
                    -float(c.get("score", 0)),
                ),
            )

            if agency_chunks:
                add_chunk(agency_chunks[0])

    if is_numeric_query(query):
        core_chunks = [c for c in chunks if chunk_has_core_amount_signal(c)]
        core_chunks = sorted(
            core_chunks,
            key=lambda c: (
                -business_overlap_score(query, c),
                is_probably_wrong_business(query, c),
                -float(c.get("score", 0)),
            ),
        )
        for c in core_chunks[:5]:
            add_chunk(c)

    signal_chunks = [c for c in chunks if chunk_matches_query_signal(query, c)]
    signal_chunks = sorted(
        signal_chunks,
        key=lambda c: (
            -business_overlap_score(query, c),
            is_probably_wrong_business(query, c),
            -float(c.get("score", 0)),
        ),
    )
    for c in signal_chunks[:5]:
        add_chunk(c)

    for c in chunks:
        add_chunk(c)
        if len(priority) >= top_k:
            break

    return priority[:top_k]


def build_context_text(
    context_docs: list[dict],
    query: str = "",
    max_chars: int = MAX_CONTEXT_TOKENS * 3,
) -> tuple[str, str]:
    parts = []
    total = 0

    evidence_card = build_evidence_card(query, context_docs)
    if evidence_card:
        parts.append(evidence_card)

    for i, doc in enumerate(context_docs, start=1):
        meta = doc.get("metadata", {}) or {}
        numeric = doc.get("numeric_summary") or meta.get("numeric_summary", "")

        header = (
            f"[문서 {i}]\n"
            f"출처: {doc.get('source', make_source(meta))}\n"
            f"발주기관: {meta.get('발주기관', '')}\n"
            f"사업명: {meta.get('사업명', '')}\n"
            f"파일명: {meta.get('파일명', '')}\n"
        )

        if numeric:
            header += f"수치요약: {numeric}\n"

        block = header + "본문:\n" + str(doc.get("text", ""))

        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 300:
                parts.append(block[:remain])
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts), evidence_card


def make_source_block(answer_chunks: list[dict], top_n: int = SOURCE_AUTOFILL_TOP_N) -> str:
    rows = []
    seen = set()

    for c in answer_chunks[:top_n]:
        meta = c.get("metadata", {}) or {}

        agency = meta.get("발주기관", "")
        biz = meta.get("사업명", "")
        fname = meta.get("파일명", "")

        row = f"{agency} — {biz} | {fname}".strip()

        if row and row not in seen:
            rows.append(row)
            seen.add(row)

    return "\n".join(rows) if rows else "해당 없음"


def make_ground_block_from_evidence(evidence_card: str, max_lines: int = 5) -> str:
    if not evidence_card:
        return "해당 없음"

    lines = []
    for line in evidence_card.splitlines():
        line = line.strip()
        if line.startswith("- "):
            lines.append(line)

    if not lines:
        return "해당 없음"

    return "\n".join(lines[:max_lines])


def autofill_ground_and_source(answer: str, answer_chunks: list[dict], evidence_card: str) -> str:
    text = normalize_answer_format(answer)

    ground = extract_section_local(text, "근거")
    if is_missing_section_content(ground):
        text = replace_section_local(text, "근거", make_ground_block_from_evidence(evidence_card))

    source = extract_section_local(text, "출처")
    if is_missing_section_content(source):
        text = replace_section_local(text, "출처", make_source_block(answer_chunks))

    return text


def ask(query: str, history: list[dict] | None = None) -> dict:
    _ensure_initialized()
    if history is None:
        history = []

    history_for_prompt = history[-HISTORY_WINDOW_FOR_REWRITE:] if history else []

    retrieved_chunks = retrieve(query, top_k=RETRIEVAL_EVAL_K)
    answer_chunks = select_answer_chunks(query, retrieved_chunks, top_k=ANSWER_TOP_K)

    context_text, evidence_card = build_context_text(answer_chunks, query=query)

    deterministic_answer = None
    deterministic_type = None

    deterministic_answer = try_build_deterministic_multi_amount_answer(query, answer_chunks)
    if deterministic_answer is not None:
        deterministic_type = "multi_amount"

    if deterministic_answer is None:
        deterministic_answer = try_build_deterministic_budget_answer(query, answer_chunks)
        if deterministic_answer is not None:
            deterministic_type = "budget"

    if deterministic_answer is None:
        deterministic_answer = try_build_deterministic_duration_answer(query, answer_chunks)
        if deterministic_answer is not None:
            deterministic_type = "duration"

    if deterministic_answer is not None:
        answer = deterministic_answer
        used_deterministic = True
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT + TOOL_INSTRUCTION}]
        messages += history_for_prompt
        messages.append({
            "role": "user",
            "content": f"[참고 문서]\n{context_text}\n\n[질문]\n{query}",
        })

        answer = chat_ollama_with_tools(
            messages=messages,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            use_tools=True,
        )
        used_deterministic = False

    answer = convert_cheonwon(answer)
    answer = normalize_answer_format(answer)
    answer = autofill_ground_and_source(answer, answer_chunks, evidence_card)

    new_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]

    if len(new_history) > MAX_HISTORY_MESSAGES:
        new_history = new_history[-MAX_HISTORY_MESSAGES:]

    return {
        "query": query,
        "answer": answer,
        "contexts": [d.get("text", "") for d in answer_chunks],
        "chunks": retrieved_chunks,
        "answer_chunks": answer_chunks,
        "evidence_card": evidence_card,
        "used_deterministic": used_deterministic,
        "deterministic_type": deterministic_type,
        "history": new_history,
    }


AMBIGUOUS_PATTERNS = [
    "그 ",
    "그럼",
    "그중",
    "그것",
    "해당",
    "그 부분",
    "거기서",
    "위의",
    "앞서",
    "방금",
    "거기",
    "그쪽",
    "그 연장선상",
    "말씀하신",
    "연장선상",
]

FOLLOWUP_HINTS = [
    "그 ",
    "그럼",
    "그건",
    "그것",
    "거기",
    "이건",
    "이것",
    "이 사업",
    "해당",
    "위 내용",
    "앞서",
    "방금",
    "추가로",
    "그러면",
]

GENERIC_CONTEXT_TERMS = {
    "자격",
    "자격증",
    "원본",
    "제출",
    "서류",
    "필요",
    "요건",
    "조건",
    "예산",
    "금액",
    "얼마",
    "기간",
    "일정",
    "방식",
    "방법",
    "가능",
    "여부",
    "어떻게",
    "무엇",
    "뭐",
    "이유",
    "근거",
}
GENERIC_CONTEXT_TERMS_NORM = {normalize_for_keyword(x) for x in GENERIC_CONTEXT_TERMS}

TYPO_HINTS = [
    "굥희",
    "머학교",
    "샨합",
    "혁렵",
    "증뵤",
    "시쓰탬",
    "샤업",
    "에싼",
    "그렌드",
    "레져",
    "쥬",
    "츄진",
    "구룹",
    "씨스탬",
    "구쭉",
    "입너",
    "서울시륍",
    "죙단",
    "하눈",
    "남서율",
    "대햑교",
    "즁보",
    "운용",
]

TYPO_REPLACEMENTS = {
    "굥희머학교": "경희대학교",
    "굥희": "경희",
    "머학교": "대학교",
    "샨합혁렵댠": "산학협력단",
    "샨합": "산학",
    "혁렵": "협력",
    "댠": "단",
    "증뵤시쓰탬": "정보시스템",
    "증뵤": "정보",
    "시쓰탬": "시스템",
    "씨스탬": "시스템",
    "샤업": "사업",
    "에싼": "예산",
    "에산": "예산",
    "얼마져": "얼마입니까",
    "잇쟌아효": "",
    "잇쟌아": "",
    "그거": "",
    "그렌드코리아레져": "그랜드코리아레저",
    "그렌드": "그랜드",
    "레져": "레저",
    "쥬": "주",
    "츄진": "추진",
    "구룹웨에": "그룹웨어",
    "구룹웨어": "그룹웨어",
    "구쭉": "구축",
    "입너가": "입니까",
    "입너": "입니",
    "서울시륍대": "서울시립대학교",
    "죙단분석": "종단분석",
    "하눈": "하는",
    "남서율대햑교": "남서울대학교",
    "대햑교": "대학교",
    "즁보": "정보",
    "운용": "운영",
}

REWRITE_SYSTEM_PROMPT = """
당신은 RFP 검색 쿼리 최적화 전문가입니다.

역할:
1. 모호한 지시어가 포함된 질문은 대화 히스토리를 참고하여 독립 질문으로 재작성한다.
2. 오타, 비문, 발음식 표기가 포함된 질문은 정확한 한국어로 교정한다.
3. 발주기관명, 사업명, 시스템명은 검색에 유리하도록 공식 명칭에 가깝게 복원한다.
4. 질문에 금액, 예산, 기간, 수량, 비율 등이 있으면 그 조건을 유지한다.
5. 재작성된 질문 한 문장만 출력한다.

출력 규칙:
- 반드시 질문 한 문장만 출력한다.
- 설명, 답변, 근거, 출처를 쓰지 않는다.
"""


def rule_based_query_cleanup(query: str) -> str:
    text = nfc(str(query))

    for wrong, right in sorted(TYPO_REPLACEMENTS.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(wrong, right)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^그\s+", "", text).strip()
    text = text.replace("있잖아요", "")
    text = text.replace("잇잖아요", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_ambiguous_or_typo(query: str) -> bool:
    q = nfc(str(query))

    if any(p in q for p in AMBIGUOUS_PATTERNS):
        return True
    if any(t in q for t in TYPO_HINTS):
        return True
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ]", q):
        return True

    cleaned = rule_based_query_cleanup(q)
    if cleaned != q:
        return True

    return False


def clean_rewritten_query(text: str, fallback: str) -> str:
    text = nfc(str(text)).strip()

    if not text:
        return fallback

    invalid = ["**답변**", "**근거**", "**출처**", "답변:", "근거:", "출처:"]
    if any(m in text for m in invalid):
        return fallback

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return fallback

    text = lines[0]
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text).strip()

    if len(text) > 400:
        return fallback

    return text if text else fallback


def has_explicit_topic(query: str) -> bool:
    q = nfc(str(query))
    if detect_agencies_from_query(q):
        return True

    terms = [t for t in extract_business_terms(q) if len(normalize_for_keyword(t)) >= 2]
    filtered = []
    for t in terms:
        nt = normalize_for_keyword(t)
        is_generic = any((g and (g in nt or nt in g)) for g in GENERIC_CONTEXT_TERMS_NORM)
        if not is_generic:
            filtered.append(t)
    return len(filtered) >= 2


def is_followup_like_query(query: str) -> bool:
    q = nfc(str(query)).strip()
    if not q:
        return False
    if any(h in q for h in FOLLOWUP_HINTS):
        return True
    return len(q) <= 18


def find_recent_user_topic(history: list[dict]) -> str:
    for item in reversed(history):
        if item.get("role") != "user":
            continue
        text = nfc(str(item.get("content", ""))).strip()
        if not text:
            continue
        if has_explicit_topic(text):
            return text
    return ""


def contextualize_query(query: str, history: list[dict], force: bool = False) -> str:
    q = nfc(str(query)).strip()
    if not q or not history:
        return q

    if has_explicit_topic(q):
        return q

    if not force and not is_followup_like_query(q):
        return q

    topic = find_recent_user_topic(history)
    if not topic:
        return q

    q_norm = normalize_for_keyword(q)
    topic_norm = normalize_for_keyword(topic)
    if q_norm and q_norm in topic_norm:
        return q

    return f"{q} (이전 대화 맥락: {topic[:220]})"


def rewrite_query(query: str, history: list[dict]) -> str:
    corrected = rule_based_query_cleanup(query)

    if corrected != query and len(corrected) >= 8:
        return corrected

    if not is_ambiguous_or_typo(query):
        return query

    messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]

    if history:
        messages.append(
            {
                "role": "user",
                "content": "이전 대화 히스토리입니다. 후속 질문의 지시어 해소에만 참고하세요.\n"
                + "\n".join([f"{h['role']}: {h['content']}" for h in history[-HISTORY_WINDOW_FOR_REWRITE:]]),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "다음 질문을 RFP 문서 검색에 적합한 독립 질문 한 문장으로 재작성하세요.\n"
                f"원문 질문: {query}\n"
                f"1차 보정 질문: {corrected}"
            ),
        }
    )

    try:
        rewritten = chat_ollama(
            messages=messages,
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=120,
        )
        return clean_rewritten_query(rewritten, corrected)
    except Exception:
        return corrected


def get_embeddings(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    _ensure_initialized()
    assert emb_model is not None

    prepared = [prepare_embedding_text(t) for t in texts]
    return emb_model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


class ConversationManager:
    def __init__(self):
        self.history: list[dict] = []

    def ask(self, query: str) -> dict:
        result = ask(query, history=self.history)
        self.history = result["history"]
        result["original_query"] = query
        result["rewritten_query"] = query
        return result

    def reset(self) -> None:
        self.history = []


class ConversationManagerV2(ConversationManager):
    def ask(self, query: str) -> dict:
        rewritten = rewrite_query(query, self.history)
        force_context = is_followup_like_query(query)
        contextual = contextualize_query(rewritten, self.history, force=force_context)
        result = ask(contextual, history=self.history)
        result["original_query"] = query
        result["rewritten_query"] = rewritten
        result["contextual_query"] = contextual
        self.history = result["history"]
        return result


def parse_answer(answer: str) -> str:
    text = str(answer)
    m = re.search(
        r"\*\*\s*답변\s*:?\s*\*\*\s*\n?(.*?)(?=\n\s*\*\*\s*(?:근거|출처)\s*:?\s*\*\*|\Z)",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else text.strip()


def get_section(text: str, name: str) -> str:
    pattern = rf"\*\*\s*{re.escape(name)}\s*:?\s*\*\*\s*\n?(.*?)(?=\n\s*\*\*\s*(?:답변|근거|출처)\s*:?\s*\*\*|\Z)"
    m = re.search(pattern, str(text), re.DOTALL)
    return m.group(1).strip() if m else "—"


__all__ = [
    "BASE_PATH",
    "DATA_PATH",
    "CHROMA_PATH",
    "FILES_PATH",
    "CHUNKS_PATH",
    "META_PATH",
    "BOOKMARKS_PATH",
    "COLLECTION_NAME",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "SYSTEM_PROMPT_V7E",
    "TOOL_INSTRUCTION",
    "TOOL_SCHEMAS",
    "ask",
    "retrieve",
    "get_embeddings",
    "get_agency_filter",
    "chat_ollama_with_tools",
    "execute_tool",
    "parse_answer",
    "get_section",
    "ConversationManager",
    "ConversationManagerV2",
    "set_agency_filter",
    "get_agency_options",
    "rule_based_query_cleanup",
    "TYPO_REPLACEMENTS",
    "INITIALIZED",
    "INITIALIZATION_ERROR",
]
