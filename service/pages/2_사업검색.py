from __future__ import annotations

import re

import chromadb
import streamlit as st

from core.functions import CHROMA_PATH, COLLECTION_NAME, get_embeddings

st.set_page_config(page_title="유사 사업 추천", page_icon="🔎", layout="wide")


def parse_money_value(value) -> int | None:
    if value is None:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        try:
            return int(float(s))
        except ValueError:
            return None

    patterns = [
        (r"(\d[\d,]*(?:\.\d+)?)\s*억\s*원?", 100_000_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*만원", 10_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*천원", 1_000),
        (r"(\d[\d,]*(?:\.\d+)?)\s*원", 1),
    ]
    for pat, unit in patterns:
        m = re.search(pat, s)
        if m:
            try:
                return int(float(m.group(1).replace(",", "")) * unit)
            except ValueError:
                return None

    return None


def format_won(value) -> str:
    v = parse_money_value(value)
    if v is None:
        return "미기재"
    if v >= 100_000_000:
        return f"{v:,}원 ({v / 100_000_000:.2f}억)"
    return f"{v:,}원"


@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    names = [c.name for c in client.list_collections()]
    if not names:
        raise RuntimeError("ChromaDB 컬렉션이 없습니다.")

    if COLLECTION_NAME in names:
        return client.get_collection(COLLECTION_NAME)

    # fallback: 최대 개수 컬렉션
    max_name = names[0]
    max_count = -1
    for name in names:
        count = client.get_collection(name).count()
        if count > max_count:
            max_count = count
            max_name = name
    return client.get_collection(max_name)


def search_similar_projects(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    emb = get_embeddings([query])[0]

    res = collection.query(
        query_embeddings=[emb],
        n_results=60,
        include=["metadatas", "distances"],
    )

    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    merged = {}
    for meta, dist in zip(metas, dists):
        meta = meta or {}
        biz = str(meta.get("사업명", "")).strip()
        agency = str(meta.get("발주기관", "")).strip()
        if not biz:
            continue

        key = (agency, biz)
        similarity = max(0.0, 1.0 - float(dist or 0.0))

        if key not in merged or similarity > merged[key]["similarity"]:
            merged[key] = {
                "사업명": biz,
                "발주기관": agency or "미기재",
                "예산": format_won(meta.get("사업금액", "")),
                "similarity": similarity,
            }

    items = sorted(merged.values(), key=lambda x: x["similarity"], reverse=True)
    return items[:top_k]


st.title("유사 사업 추천")
st.caption("사업명 또는 키워드 입력 후 유사 사업 Top 5")

if "queued_query" not in st.session_state:
    st.session_state.queued_query = ""

query = st.text_input("사업명/키워드", placeholder="예: 학사정보시스템 고도화")

if st.button("유사 사업 찾기", use_container_width=True) and query.strip():
    with st.spinner("유사도 검색 중..."):
        try:
            items = search_similar_projects(query.strip(), top_k=5)
        except Exception as e:
            st.error(f"검색 실패: {e}")
            st.stop()

    if not items:
        st.info("유사 사업을 찾지 못했습니다.")
    else:
        for i, item in enumerate(items, start=1):
            with st.container(border=True):
                st.markdown(f"### {i}. {item['사업명']}")
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**발주기관**\n\n{item['발주기관']}")
                c2.markdown(f"**예산**\n\n{item['예산']}")
                c3.markdown(f"**유사도**\n\n{item['similarity']:.4f}")

                if st.button("메인 챗에서 질의", key=f"sim_to_main_{i}"):
                    st.session_state.queued_query = f"{item['사업명']}의 핵심 조건 요약해줘"
                    try:
                        st.switch_page("app.py")
                    except Exception:
                        st.info("사이드바에서 메인 페이지(app.py)로 이동하면 자동으로 질의됩니다.")
