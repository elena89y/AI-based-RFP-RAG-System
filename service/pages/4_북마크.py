import json

import streamlit as st

from core.functions import BOOKMARKS_PATH

st.set_page_config(page_title="북마크", page_icon="⭐", layout="wide")
st.title("즐겨찾기 / 북마크")

if "queued_query" not in st.session_state:
    st.session_state.queued_query = ""

if BOOKMARKS_PATH.exists():
    bookmarks = json.loads(BOOKMARKS_PATH.read_text(encoding="utf-8"))
else:
    bookmarks = []

st.caption(f"총 {len(bookmarks)}개")

if not bookmarks:
    st.info("저장된 북마크가 없습니다.")
else:
    for i, item in enumerate(bookmarks, start=1):
        title = item.get("사업명", "미기재")
        with st.expander(f"{i}. {title}"):
            st.write(f"저장시각: {item.get('저장시각', '')}")
            st.write(f"질문: {item.get('질문', '')}")
            st.write(f"요약: {item.get('답변요약', '')}")

            if st.button("메인 챗에서 열기", key=f"open_main_{i}"):
                st.session_state.queued_query = item.get("질문", "")
                try:
                    st.switch_page("app.py")
                except Exception:
                    st.info("사이드바에서 메인 페이지(app.py)로 이동하면 자동으로 질문이 입력됩니다.")
