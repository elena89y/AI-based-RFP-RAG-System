from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from core.functions import retrieve

st.set_page_config(page_title="사업 일정 타임라인", page_icon="🗓️", layout="wide")

DATE_PATTERN = re.compile(
    r"(20\d{2}[.\-/년\s]+\d{1,2}[.\-/월\s]+\d{1,2}(?:일)?)"
)

DATE_LABELS = {
    "공고일": ["공고일", "공고일자", "공고기간 시작", "입찰공고"],
    "입찰 마감": ["입찰 마감", "마감일", "제출마감", "입찰서 제출마감", "마감"],
    "계약일": ["계약일", "계약체결일", "계약 예정일"],
    "착수일": ["착수일", "착수", "사업 시작"],
    "납기일": ["납기일", "완료일", "종료일", "검수일"],
}


def normalize_date_text(text: str) -> str:
    t = str(text).strip()
    t = t.replace("년", "-").replace("월", "-").replace("일", "")
    t = t.replace(".", "-").replace("/", "-")
    t = re.sub(r"\s+", "", t)
    return t


def parse_date(text: str) -> datetime | None:
    norm = normalize_date_text(text)
    try:
        return datetime.strptime(norm, "%Y-%m-%d")
    except ValueError:
        pass

    # 월/일 한 자리 보정
    m = re.match(r"(20\d{2})-(\d{1,2})-(\d{1,2})", norm)
    if m:
        y, mo, d = m.groups()
        try:
            return datetime(int(y), int(mo), int(d))
        except ValueError:
            return None

    return None


def extract_dates_from_chunks(chunks: list[dict]) -> dict[str, dict]:
    results = {k: {"date": None, "line": "미기재"} for k in DATE_LABELS}

    for chunk in chunks:
        text = str(chunk.get("text", ""))
        lines = [re.sub(r"\s+", " ", l).strip() for l in re.split(r"[\n\r]+", text) if l.strip()]

        for line in lines:
            for label, keys in DATE_LABELS.items():
                if results[label]["date"] is not None:
                    continue
                if not any(k in line for k in keys):
                    continue

                m = DATE_PATTERN.search(line)
                if not m:
                    continue

                dt = parse_date(m.group(1))
                if dt is None:
                    continue

                results[label] = {"date": dt, "line": line[:160]}

    return results


def make_timeline_df(date_map: dict[str, dict]):
    rows = []
    for label, info in date_map.items():
        dt = info["date"]
        if dt is None:
            continue
        rows.append(
            {
                "항목": label,
                "시작": dt,
                "종료": dt + timedelta(days=1),
                "상태": "확인됨",
            }
        )
    return pd.DataFrame(rows)


st.title("사업 일정 타임라인")
st.caption("사업명 검색 후 주요 일정(공고/마감/계약/착수/납기) 시각화")

query = st.text_input("사업명", placeholder="예: ITS 구축사업")

if st.button("타임라인 생성", use_container_width=True) and query.strip():
    with st.spinner("일정 정보 추출 중..."):
        chunks = retrieve(f"{query.strip()} 일정 공고일 마감일 계약일 착수일 납기일", top_k=25)

    if not chunks:
        st.warning("관련 문서를 찾지 못했습니다.")
        st.stop()

    date_map = extract_dates_from_chunks(chunks)
    df_timeline = make_timeline_df(date_map)

    st.subheader("일정 요약")
    summary_rows = []
    for label, info in date_map.items():
        dt = info["date"]
        summary_rows.append(
            {
                "항목": label,
                "날짜": dt.strftime("%Y-%m-%d") if dt else "미기재",
                "근거 라인": info["line"],
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if df_timeline.empty:
        st.info("타임라인에 표시할 날짜를 찾지 못했습니다.")
    else:
        fig = px.timeline(
            df_timeline,
            x_start="시작",
            x_end="종료",
            y="항목",
            color="상태",
            color_discrete_sequence=["#1f4b7a"],
        )
        fig.update_layout(height=420, showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
