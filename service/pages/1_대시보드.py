from __future__ import annotations

import json
import re

import plotly.express as px
import pandas as pd
import streamlit as st

from core.functions import CHUNKS_PATH, META_PATH

st.set_page_config(page_title="예산 분석 대시보드", page_icon="📊", layout="wide")


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

    num = re.sub(r"[^0-9.]", "", s)
    if num:
        try:
            return int(float(num))
        except ValueError:
            return None
    return None


def bucketize_budget(v: int | None) -> str:
    if v is None:
        return "미기재"
    if v < 100_000_000:
        return "1억 미만"
    if v < 500_000_000:
        return "1~5억"
    if v < 1_000_000_000:
        return "5~10억"
    return "10억 이상"


@st.cache_data(show_spinner=False)
def load_meta_df():
    import pandas as pd

    if not META_PATH.exists():
        return pd.DataFrame()

    df = pd.read_excel(META_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        c_low = c.lower()
        if c == "발주기관" or "발주기관" in c:
            rename_map[c] = "발주기관"
        elif c in {"사업금액", "예산", "사업예산", "사업비", "용역금액"} or "금액" in c or "예산" in c:
            rename_map[c] = "사업금액"
        elif c == "파일형식" or "형식" in c:
            rename_map[c] = "파일형식"
        elif c == "chunk_type" or "chunk" in c_low:
            rename_map[c] = "chunk_type"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "발주기관" not in df.columns:
        df["발주기관"] = "미기재"
    if "사업금액" not in df.columns:
        df["사업금액"] = None
    if "파일형식" not in df.columns:
        df["파일형식"] = "unknown"

    df["발주기관"] = df["발주기관"].astype(str).str.strip().replace({"": "미기재"})
    df["파일형식"] = df["파일형식"].astype(str).str.lower().str.strip().replace({"": "unknown"})
    df["예산원"] = df["사업금액"].apply(parse_money_value)
    df["예산구간"] = df["예산원"].apply(bucketize_budget)
    return df


@st.cache_data(show_spinner=False)
def load_chunk_type_df():
    import pandas as pd

    if not CHUNKS_PATH.exists():
        return pd.DataFrame(columns=["발주기관", "chunk_type", "파일형식"])

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "metadatas" in raw:
        metas = raw["metadatas"]
    else:
        metas = [c.get("metadata", {}) for c in raw]

    rows = []
    for m in metas:
        m = m or {}
        rows.append(
            {
                "발주기관": str(m.get("발주기관", "미기재") or "미기재").strip(),
                "chunk_type": str(m.get("chunk_type", "unknown") or "unknown").strip().lower(),
                "파일형식": str(m.get("파일형식", "unknown") or "unknown").strip().lower(),
            }
        )

    return pd.DataFrame(rows)


st.title("예산 분석 대시보드")
st.caption("메타데이터 기반 통계 (LLM 호출 없음)")

try:
    df_meta = load_meta_df()
    df_chunks = load_chunk_type_df()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

if df_meta.empty and df_chunks.empty:
    st.error("표시할 데이터가 없습니다.")
    st.stop()

agencies = sorted(set(df_meta.get("발주기관", [])) | set(df_chunks.get("발주기관", [])))
agencies = [a for a in agencies if a]

selected_agency = st.selectbox("발주기관 필터", ["전체"] + agencies)

if selected_agency != "전체":
    if not df_meta.empty:
        df_meta = df_meta[df_meta["발주기관"] == selected_agency]
    if not df_chunks.empty:
        df_chunks = df_chunks[df_chunks["발주기관"] == selected_agency]

col1, col2 = st.columns(2)

with col1:
    st.subheader("발주기관별 평균 예산 (TOP 20)")
    if not df_meta.empty and df_meta["예산원"].notna().any():
        import pandas as pd

        avg_df = (
            df_meta[df_meta["예산원"].notna()]
            .groupby("발주기관", as_index=False)["예산원"]
            .mean()
            .sort_values("예산원", ascending=False)
            .head(20)
        )
        fig = px.bar(
            avg_df,
            x="예산원",
            y="발주기관",
            orientation="h",
            color="예산원",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=520, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("예산 데이터가 부족합니다.")

with col2:
    st.subheader("예산 구간별 사업 건수")
    if not df_meta.empty:
        order = ["1억 미만", "1~5억", "5~10억", "10억 이상", "미기재"]
        hist_df = (
            df_meta.groupby("예산구간", as_index=False)
            .size()
            .rename(columns={"size": "건수"})
        )
        hist_df["예산구간"] = pd.Categorical(hist_df["예산구간"], categories=order, ordered=True)
        hist_df = hist_df.sort_values("예산구간")
        fig = px.bar(hist_df, x="예산구간", y="건수", color="예산구간", color_discrete_sequence=px.colors.sequential.Blues)
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("표시할 메타 데이터가 없습니다.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("파일 형식 비율")
    src = df_meta if not df_meta.empty else df_chunks
    if not src.empty:
        pie_df = src.groupby("파일형식", as_index=False).size().rename(columns={"size": "건수"})
        fig = px.pie(pie_df, names="파일형식", values="건수", hole=0.0, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("파일 형식 데이터가 없습니다.")

with col4:
    st.subheader("chunk_type 분포")
    if not df_chunks.empty:
        donut_df = df_chunks.groupby("chunk_type", as_index=False).size().rename(columns={"size": "건수"})
        fig = px.pie(donut_df, names="chunk_type", values="건수", hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("chunk_type 데이터가 없습니다.")
