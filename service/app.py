from __future__ import annotations

import html
import io
import json
import re
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from openai import OpenAI

from core.functions import (
    BOOKMARKS_PATH,
    CHUNKS_PATH,
    FILES_PATH,
    LLM_MODEL,
    ConversationManagerV2,
    get_agency_filter,
    get_section,
    parse_answer,
    retrieve,
    rule_based_query_cleanup,
    set_agency_filter,
)

APP_NAME = "Search RFP"
HOME_CHAT_ID = "__home__"

st.set_page_config(page_title=APP_NAME, page_icon="📘", layout="wide")

COMPARISON_HINTS = ["비교", "차이", "차액", "대비", "vs", "VS"]
STRATEGY_HINTS = ["전략", "어필"]
CHECKLIST_HINT = "체크리스트"
AGG_QUERY_HINTS = ["평균", "몇 개", "몇개", "개수", "건수", "합계", "최대", "최소", "총"]
SUGGESTED_PROMPTS = [
    "한국가스공사 관련 사업 예산 상위 3개를 알려줘",
    "인천공항운영서비스 사업의 계약기간과 입찰방식을 요약해줘",
    "경희대학교 산학협력단 사업 중 유사 사업을 추천해줘",
    "최근 대화 기준으로 제출 체크리스트를 만들어줘",
]

CHAT_ROOMS_PATH = BOOKMARKS_PATH.parent / "chat_rooms.json"
MAX_CHAT_ROOMS = 120
CHAT_TITLE_MAX_LEN = 36
MAX_MANAGER_HISTORY = 24
ESTIMATED_LLM_TOKENS_PER_SECOND = 9

CHECKLIST_CATEGORIES = {
    "필수 제출 서류": ["제출", "서류", "증명", "원본", "인감", "제안서", "첨부"],
    "자격 요건": ["자격", "요건", "참여", "실적", "인력", "보유"],
    "기술평가 기준": ["기술", "평가", "배점", "정량", "정성"],
    "일정": ["공고", "마감", "개찰", "제출기한", "착수", "계약일", "기간"],
    "주의사항": ["유의", "주의", "무효", "제한", "불가", "배제", "해지"],
}

RISK_KEYWORDS = {
    "high": [
        "계약 해지",
        "해지권",
        "무한 책임",
        "무제한 책임",
        "무제한 하자보수",
        "배상 한도 없음",
        "손해배상 전액",
    ],
    "medium": [
        "납기 30일",
        "짧은 납기",
        "단기간",
        "불명확",
        "범위 미정",
        "추가 비용",
        "하자보수",
    ],
    "low": ["표준", "관련 법령", "일반조건", "표준계약", "준수"],
}


def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def stream_text(text: str):
    for token in re.split(r"(\s+)", text):
        if token:
            yield token


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9_]+|[^\s]", re.UNICODE)


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_PATTERN.findall(str(text)))


def estimate_history_token_count(history: list[dict]) -> int:
    total = 0
    for item in history:
        total += estimate_token_count(item.get("content", ""))
    return total


def render_live_generation_panel(
    placeholder,
    elapsed_s: float,
    input_tokens: int,
    output_tokens: int,
    token_trace: list[int],
    done: bool = False,
    status_text: str | None = None,
) -> None:
    total_tokens = input_tokens + output_tokens
    speed = (output_tokens / elapsed_s) if elapsed_s > 0 else 0.0
    state_text = status_text or ("완료" if done else "생성 중")

    with placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("상태", state_text)
        c2.metric("소요 시간", f"{elapsed_s:.2f}s")
        c3.metric("출력 토큰(추정)", f"{output_tokens:,}")
        c4.metric("총 토큰(추정)", f"{total_tokens:,}", delta=f"{speed:.1f} tok/s")
        st.caption("실시간 토큰 증가 추이")
        if len(token_trace) >= 2:
            st.line_chart({"출력 토큰(추정)": token_trace}, height=130, width="stretch")


def stream_text_with_live_metrics(
    text: str,
    metrics_placeholder,
    input_tokens: int,
    stream_stats: dict,
    initial_output_tokens: int = 0,
):
    started = time.perf_counter()
    output_tokens = int(initial_output_tokens)
    token_trace: list[int] = [output_tokens]
    last_draw = -1.0

    for token in stream_text(text):
        output_tokens += estimate_token_count(token)
        elapsed = time.perf_counter() - started

        if (elapsed - last_draw) >= 0.08 or output_tokens <= 2:
            token_trace.append(output_tokens)
            render_live_generation_panel(
                placeholder=metrics_placeholder,
                elapsed_s=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                token_trace=token_trace[-80:],
                done=False,
            )
            last_draw = elapsed

        stream_stats["output_tokens"] = output_tokens
        stream_stats["stream_elapsed"] = elapsed
        yield token

    elapsed = time.perf_counter() - started
    stream_stats["output_tokens"] = output_tokens
    stream_stats["stream_elapsed"] = elapsed
    token_trace.append(output_tokens)
    render_live_generation_panel(
        placeholder=metrics_placeholder,
        elapsed_s=elapsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_trace=token_trace[-80:],
        done=True,
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_money_value(value) -> int | None:
    if value is None:
        return None

    s = normalize_text(value)
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
            raw = m.group(1).replace(",", "")
            try:
                return int(float(raw) * unit)
            except ValueError:
                return None

    num = re.sub(r"[^0-9.]", "", s)
    if num:
        try:
            return int(float(num))
        except ValueError:
            return None
    return None


def format_won(value: int | None) -> str:
    if value is None:
        return "미기재"
    won = int(value)
    if won >= 100_000_000:
        return f"{won:,}원 ({won / 100_000_000:.2f}억)"
    if won >= 10_000:
        return f"{won:,}원 ({won / 10_000:,.0f}만)"
    return f"{won:,}원"


def parse_won(value: str) -> str:
    return format_won(parse_money_value(value))


def extract_line_by_keywords(text: str, keywords: list[str], fallback: str = "미기재") -> str:
    for line in re.split(r"[\n\r]+", str(text)):
        line = normalize_text(line)
        if not line:
            continue
        if any(k in line for k in keywords):
            return line[:180]
    return fallback


def safe_chunk_text(chunk: dict) -> str:
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


def build_summary_card(answer_chunks: list[dict]) -> dict | None:
    if not answer_chunks:
        return None

    target = None
    for chunk in answer_chunks:
        meta = chunk.get("metadata", {}) or {}
        ctype = chunk.get("chunk_type") or meta.get("chunk_type", "")
        if str(ctype).lower() == "summary":
            target = chunk
            break

    if target is None:
        target = answer_chunks[0]

    meta = target.get("metadata", {}) or {}
    text = safe_chunk_text(target)

    return {
        "사업명": meta.get("사업명", "미기재") or "미기재",
        "발주기관": meta.get("발주기관", "미기재") or "미기재",
        "예산": parse_won(meta.get("사업금액", "")),
        "계약기간": extract_line_by_keywords(text, ["사업기간", "용역기간", "계약체결", "착수", "개월"], "미기재"),
        "입찰방식": extract_line_by_keywords(text, ["입찰방식", "계약방법", "낙찰", "협상"], "미기재"),
        "자격요건": extract_line_by_keywords(text, ["자격", "요건", "참여", "실적"], "미기재"),
    }


def extract_source_filenames(answer: str) -> list[str]:
    src = get_section(answer, "출처")
    files = re.findall(r"[\w\s\-가-힣\(\)\.]+\.(?:hwp|pdf)", src, flags=re.IGNORECASE)
    normalized = []
    seen = set()
    for f in files:
        name = f.strip()
        if name and name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def highlight_source_html(source_text: str) -> str:
    lines = [l.strip() for l in str(source_text).splitlines() if l.strip()]
    if not lines:
        return "<div style='color:#6b7280'>해당 없음</div>"

    items = []
    for line in lines:
        escaped = html.escape(line)
        files = re.findall(r"[\w\s\-가-힣\(\)\.]+\.(?:hwp|pdf)", line, flags=re.IGNORECASE)
        for fname in files:
            token = html.escape(fname.strip())
            escaped = escaped.replace(
                token,
                f"<span style='background:#e8eefb;color:#0f2d53;padding:2px 6px;border-radius:6px;font-weight:700'>{token}</span>",
            )
        items.append(f"<div style='margin-bottom:4px'>{escaped}</div>")

    return "\n".join(items)


def render_source_highlight(answer: str) -> None:
    source = get_section(answer, "출처")
    st.markdown("**출처 하이라이팅**")
    html_block = highlight_source_html(source)
    st.markdown(
        f"<div style='border:1px solid #d7e1ee;background:#f8fbff;padding:10px;border-radius:10px'>{html_block}</div>",
        unsafe_allow_html=True,
    )


def render_source_downloads(files: list[str], key_prefix: str) -> None:
    if not files:
        return

    st.markdown("**원본 파일 다운로드**")
    for idx, fname in enumerate(files):
        fpath = FILES_PATH / fname
        ext = fpath.suffix.lower()
        label = f"📄 {fname}" if ext == ".hwp" else f"📑 {fname}"

        if fpath.exists():
            with open(fpath, "rb") as fp:
                st.download_button(
                    label=label,
                    data=fp.read(),
                    file_name=fname,
                    mime="application/octet-stream",
                    key=f"{key_prefix}_dl_{idx}",
                )
        else:
            st.markdown(f"<span style='color:#9aa6b2'>{label} (파일 없음)</span>", unsafe_allow_html=True)


def render_answer_details(answer: str, risk: dict, source_files: list[str], key_prefix: str) -> None:
    with st.expander("답변 상세 보기", expanded=False):
        render_risk_box(risk)
        render_source_highlight(answer)
        render_source_downloads(source_files, key_prefix=key_prefix)


def load_bookmarks() -> list[dict]:
    try:
        if BOOKMARKS_PATH.exists():
            return json.loads(BOOKMARKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def save_bookmarks(bookmarks: list[dict]) -> None:
    BOOKMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOKMARKS_PATH.write_text(json.dumps(bookmarks, ensure_ascii=False, indent=2), encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clip_text(text: str, max_len: int = CHAT_TITLE_MAX_LEN) -> str:
    t = normalize_text(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def room_title_from_messages(messages: list[dict], fallback: str = "새 대화") -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = normalize_text(msg.get("content", ""))
            if content:
                return content
    return fallback


def room_has_content(room: dict) -> bool:
    return bool(room.get("messages") or room.get("records") or room.get("manager_history"))


def create_chat_room(title: str = "새 대화") -> dict:
    ts = now_stamp()
    return {
        "id": str(uuid4()),
        "title": title,
        "created_at": ts,
        "updated_at": ts,
        "messages": [],
        "records": [],
        "manager_history": [],
        "summary_card": None,
        "checklists": {},
    }


def normalize_chat_room(raw: dict) -> dict:
    base = create_chat_room(title="새 대화")
    if not isinstance(raw, dict):
        return base

    base["id"] = str(raw.get("id") or base["id"])
    base["title"] = str(raw.get("title") or base["title"])
    base["created_at"] = str(raw.get("created_at") or base["created_at"])
    base["updated_at"] = str(raw.get("updated_at") or base["updated_at"])
    base["messages"] = raw.get("messages", []) if isinstance(raw.get("messages", []), list) else []
    base["records"] = raw.get("records", []) if isinstance(raw.get("records", []), list) else []
    base["manager_history"] = (
        raw.get("manager_history", []) if isinstance(raw.get("manager_history", []), list) else []
    )
    base["summary_card"] = raw.get("summary_card")
    base["checklists"] = raw.get("checklists", {}) if isinstance(raw.get("checklists", {}), dict) else {}

    derived_title = room_title_from_messages(base["messages"], fallback="")
    if derived_title:
        base["title"] = derived_title
    else:
        base["title"] = normalize_text(base["title"]) or "새 대화"
    return base


def load_chat_rooms() -> list[dict]:
    try:
        if CHAT_ROOMS_PATH.exists():
            raw = json.loads(CHAT_ROOMS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return [r for r in [normalize_chat_room(item) for item in raw] if room_has_content(r)]
    except Exception:
        pass
    return []


def save_chat_rooms(chat_rooms: list[dict]) -> None:
    CHAT_ROOMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [normalize_chat_room(r) for r in chat_rooms[:MAX_CHAT_ROOMS] if room_has_content(r)]
    CHAT_ROOMS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def find_chat_room_index(room_id: str) -> int:
    for i, room in enumerate(st.session_state.chat_rooms):
        if room.get("id") == room_id:
            return i
    return -1


def get_active_chat_room() -> dict | None:
    if st.session_state.active_chat_id == HOME_CHAT_ID:
        return None
    idx = find_chat_room_index(st.session_state.active_chat_id)
    if idx < 0:
        return None
    return st.session_state.chat_rooms[idx]


def sync_home_session() -> None:
    st.session_state.messages = []
    st.session_state.records = []
    st.session_state.summary_card = None
    st.session_state.checklists = {}
    st.session_state.pending_original = ""
    st.session_state.pending_corrected = ""
    st.session_state.queued_query = ""
    st.session_state.editing_message_index = None
    st.session_state.editing_message_text = ""

    if "manager" not in st.session_state or st.session_state.manager is None:
        st.session_state.manager = ConversationManagerV2()
    st.session_state.manager.history = []


def sync_session_from_room(room: dict) -> None:
    st.session_state.messages = deepcopy(room.get("messages", []))
    st.session_state.records = deepcopy(room.get("records", []))
    st.session_state.summary_card = deepcopy(room.get("summary_card"))
    st.session_state.checklists = deepcopy(room.get("checklists", {}))
    st.session_state.pending_original = ""
    st.session_state.pending_corrected = ""
    st.session_state.queued_query = ""

    if "manager" not in st.session_state or st.session_state.manager is None:
        st.session_state.manager = ConversationManagerV2()
    st.session_state.manager.history = deepcopy(room.get("manager_history", []))


def sync_active_room_from_session(update_title: bool = True) -> None:
    if st.session_state.active_chat_id == HOME_CHAT_ID:
        return

    room = get_active_chat_room()
    if room is None:
        return

    room["messages"] = deepcopy(st.session_state.messages)
    room["records"] = deepcopy(st.session_state.records)
    room["summary_card"] = deepcopy(st.session_state.summary_card)
    room["checklists"] = deepcopy(st.session_state.checklists)
    room["manager_history"] = deepcopy(st.session_state.manager.history)
    room["updated_at"] = now_stamp()

    if update_title:
        room["title"] = room_title_from_messages(room["messages"], fallback=room.get("title", "새 대화"))

    save_chat_rooms(st.session_state.chat_rooms)


def switch_chat_room(room_id: str) -> None:
    idx = find_chat_room_index(room_id)
    if idx < 0:
        return
    sync_active_room_from_session(update_title=True)
    st.session_state.active_chat_id = room_id
    st.session_state.loaded_chat_id = room_id
    sync_session_from_room(st.session_state.chat_rooms[idx])


def switch_to_home() -> None:
    sync_active_room_from_session(update_title=True)
    st.session_state.active_chat_id = HOME_CHAT_ID
    st.session_state.loaded_chat_id = HOME_CHAT_ID
    sync_home_session()
    save_chat_rooms(st.session_state.chat_rooms)


def create_new_chat_room() -> None:
    switch_to_home()


def ensure_active_room_for_query() -> None:
    room = get_active_chat_room()
    if room is not None:
        return

    new_room = create_chat_room(title="새 대화")
    st.session_state.chat_rooms.insert(0, new_room)
    st.session_state.chat_rooms = st.session_state.chat_rooms[:MAX_CHAT_ROOMS]
    st.session_state.active_chat_id = new_room["id"]
    st.session_state.loaded_chat_id = new_room["id"]


def delete_chat_room(room_id: str) -> None:
    idx = find_chat_room_index(room_id)
    if idx < 0:
        return

    del st.session_state.chat_rooms[idx]
    if not st.session_state.chat_rooms:
        st.session_state.active_chat_id = HOME_CHAT_ID
        st.session_state.loaded_chat_id = HOME_CHAT_ID
        sync_home_session()
        save_chat_rooms(st.session_state.chat_rooms)
        return

    st.session_state.active_chat_id = st.session_state.chat_rooms[0]["id"]
    st.session_state.loaded_chat_id = st.session_state.active_chat_id
    sync_session_from_room(st.session_state.chat_rooms[0])
    save_chat_rooms(st.session_state.chat_rooms)


def resolve_page_link_by_order(order: int) -> str | None:
    pages_dir = Path(__file__).parent / "pages"
    pattern = re.compile(rf"^{order}_.+\.py$")
    matches = sorted([p for p in pages_dir.glob("*.py") if pattern.match(p.name)])
    if not matches:
        return None
    return str(Path("pages") / matches[0].name)


def calc_confidence(answer: str) -> str:
    src = get_section(answer, "출처").strip()
    ans = parse_answer(answer)
    has_source = src not in ("", "—", "해당 없음")
    has_number = bool(re.search(r"\d", ans))

    if has_source and has_number:
        return "🟢 높음"
    if has_source:
        return "🟡 보통"
    return "🔴 낮음"


def render_confidence_badge(confidence: str) -> None:
    color_map = {
        "🟢 높음": ("#e8f7ee", "#1f7a44"),
        "🟡 보통": ("#fff7e5", "#8a6200"),
        "🔴 낮음": ("#fdecec", "#a62f2f"),
    }
    bg, fg = color_map.get(confidence, ("#f2f4f8", "#4b5563"))
    st.markdown(
        f"""
        <div style="display:flex;justify-content:flex-end;margin-bottom:6px;">
          <span style="padding:4px 10px;border-radius:999px;background:{bg};color:{fg};font-weight:700;font-size:12px;">
            신뢰도 {confidence}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = load_bookmarks()
    if "agency_filter" not in st.session_state:
        st.session_state.agency_filter = "전체"
    st.session_state.agency_filter = "전체"
    set_agency_filter("전체")
    if "manager" not in st.session_state:
        st.session_state.manager = ConversationManagerV2()
    if "pending_original" not in st.session_state:
        st.session_state.pending_original = ""
    if "pending_corrected" not in st.session_state:
        st.session_state.pending_corrected = ""
    if "queued_query" not in st.session_state:
        st.session_state.queued_query = ""
    if "editing_message_index" not in st.session_state:
        st.session_state.editing_message_index = None
    if "editing_message_text" not in st.session_state:
        st.session_state.editing_message_text = ""
    if "chat_rooms" not in st.session_state:
        st.session_state.chat_rooms = load_chat_rooms()

    st.session_state.chat_rooms = [r for r in st.session_state.chat_rooms if room_has_content(r)]

    valid_ids = {r.get("id") for r in st.session_state.chat_rooms}
    valid_ids.add(HOME_CHAT_ID)
    if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in valid_ids:
        st.session_state.active_chat_id = st.session_state.chat_rooms[0]["id"] if st.session_state.chat_rooms else HOME_CHAT_ID

    if "loaded_chat_id" not in st.session_state or st.session_state.loaded_chat_id != st.session_state.active_chat_id:
        if st.session_state.active_chat_id == HOME_CHAT_ID:
            sync_home_session()
        else:
            room = get_active_chat_room()
            if room is None:
                st.session_state.active_chat_id = HOME_CHAT_ID
                sync_home_session()
            else:
                sync_session_from_room(room)
        st.session_state.loaded_chat_id = st.session_state.active_chat_id


def reset_chat_session(clear_records: bool = False) -> None:
    if clear_records:
        st.session_state.chat_rooms = []
        st.session_state.active_chat_id = HOME_CHAT_ID
        st.session_state.loaded_chat_id = HOME_CHAT_ID
        sync_home_session()
        save_chat_rooms(st.session_state.chat_rooms)
        return

    create_new_chat_room()


def add_bookmark(item: dict) -> None:
    bookmarks = st.session_state.bookmarks
    bookmarks.insert(0, item)
    st.session_state.bookmarks = bookmarks[:300]
    save_bookmarks(st.session_state.bookmarks)


@st.cache_resource(show_spinner=False)
def load_project_catalog() -> list[dict]:
    if not CHUNKS_PATH.exists():
        return []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "metadatas" in raw:
        metadatas = raw["metadatas"]
    else:
        metadatas = [c.get("metadata", {}) for c in raw]

    rows: dict[tuple[str, str, str], dict] = {}

    for meta in metadatas:
        meta = meta or {}
        agency = normalize_text(meta.get("발주기관", ""))
        business = normalize_text(meta.get("사업명", ""))
        filename = normalize_text(meta.get("파일명", ""))

        if not business and not filename:
            continue

        key = (agency, business, filename)
        budget = parse_money_value(meta.get("사업금액", ""))

        row = {
            "발주기관": agency or "미기재",
            "사업명": business or "미기재",
            "파일명": filename or "",
            "예산": budget,
        }

        if key not in rows:
            rows[key] = row
            continue

        prev = rows[key]
        if prev["예산"] is None and budget is not None:
            rows[key] = row

    return list(rows.values())


def detect_budget_condition(query: str):
    q = normalize_text(query)

    patterns = [
        (r"(\d+(?:\.\d+)?)\s*억\s*(이상|이하|초과|미만)", 100_000_000),
        (r"(\d+(?:\.\d+)?)\s*만원\s*(이상|이하|초과|미만)", 10_000),
        (r"(\d+(?:\.\d+)?)\s*원\s*(이상|이하|초과|미만)", 1),
    ]

    for pat, unit in patterns:
        m = re.search(pat, q)
        if not m:
            continue

        value = int(float(m.group(1)) * unit)
        op = m.group(2)

        def predicate(x, _value=value, _op=op):
            if x is None:
                return False
            if _op == "이상":
                return x >= _value
            if _op == "이하":
                return x <= _value
            if _op == "초과":
                return x > _value
            return x < _value

        return predicate, f"{m.group(1)}{('억' if unit==100_000_000 else '만원' if unit==10_000 else '원')} {op}"

    return None, ""


def is_aggregate_query(query: str) -> bool:
    q = normalize_text(query)
    return any(h in q for h in AGG_QUERY_HINTS)


def build_aggregate_answer(query: str, selected_agency: str) -> dict | None:
    if not is_aggregate_query(query):
        return None

    rows = load_project_catalog()
    if not rows:
        return None

    q = normalize_text(query)
    filtered = rows

    if selected_agency and selected_agency != "전체":
        filtered = [r for r in filtered if r["발주기관"] == selected_agency]

    agency_from_query, score = get_agency_filter(query)
    if agency_from_query and score >= 88:
        filtered = [r for r in filtered if r["발주기관"] == agency_from_query]

    pred, cond_text = detect_budget_condition(query)
    if pred:
        filtered = [r for r in filtered if pred(r["예산"])]

    numeric = [r for r in filtered if r["예산"] is not None]

    if not filtered:
        answer = (
            "**답변**\n조건에 맞는 사업을 찾지 못했습니다.\n\n"
            "**근거**\n현재 필터 또는 질의 조건으로 매칭된 사업이 없습니다.\n\n"
            "**출처**\n해당 없음"
        )
        return {"answer": answer, "top_rows": []}

    def source_lines(items: list[dict]) -> str:
        rows_out = []
        for r in items:
            rows_out.append(f"{r['발주기관']} — {r['사업명']} | {r['파일명']}")
        return "\n".join(rows_out) if rows_out else "해당 없음"

    top_rows = sorted(
        [r for r in filtered if r["예산"] is not None],
        key=lambda x: x["예산"],
        reverse=True,
    )[:5]

    cond_block = f" ({cond_text})" if cond_text else ""

    if "평균" in q and numeric:
        avg = int(sum(r["예산"] for r in numeric) / len(numeric))
        answer = (
            f"**답변**\n조건에 맞는 사업의 평균 예산은 **{format_won(avg)}**입니다{cond_block}.\n\n"
            f"**근거**\n집계 대상 {len(numeric)}건에 대해 예산 평균을 계산했습니다. 상위 사업: "
            + ", ".join([f"{r['사업명']}({format_won(r['예산'])})" for r in top_rows])
            + "\n\n**출처**\n"
            + source_lines(top_rows)
        )
        return {"answer": answer, "top_rows": top_rows}

    if ("합계" in q or "총" in q) and numeric:
        total = int(sum(r["예산"] for r in numeric))
        answer = (
            f"**답변**\n조건에 맞는 사업의 예산 합계는 **{format_won(total)}**입니다{cond_block}.\n\n"
            f"**근거**\n집계 대상 {len(numeric)}건의 예산을 합산했습니다. 상위 사업: "
            + ", ".join([f"{r['사업명']}({format_won(r['예산'])})" for r in top_rows])
            + "\n\n**출처**\n"
            + source_lines(top_rows)
        )
        return {"answer": answer, "top_rows": top_rows}

    if "최대" in q and numeric:
        best = max(numeric, key=lambda x: x["예산"])
        answer = (
            f"**답변**\n최대 예산 사업은 **{best['사업명']}**이며 예산은 **{format_won(best['예산'])}**입니다{cond_block}.\n\n"
            f"**근거**\n조건에 맞는 {len(numeric)}건을 비교했습니다.\n\n"
            f"**출처**\n{best['발주기관']} — {best['사업명']} | {best['파일명']}"
        )
        return {"answer": answer, "top_rows": [best] + top_rows[:4]}

    if re.search(r"몇\s*개|몇개|개수|건수", q):
        answer = (
            f"**답변**\n조건에 맞는 사업은 총 **{len(filtered)}개**입니다{cond_block}.\n\n"
            f"**근거**\n필터 기준으로 집계했으며, 예산 확인 가능 사업 {len(numeric)}건입니다. 상위 사업: "
            + ", ".join([f"{r['사업명']}({format_won(r['예산'])})" for r in top_rows])
            + "\n\n**출처**\n"
            + source_lines(top_rows)
        )
        return {"answer": answer, "top_rows": top_rows}

    return None


def should_compare(query: str, chunks: list[dict]) -> bool:
    if any(h in query for h in COMPARISON_HINTS):
        return True

    projects = []
    seen = set()
    for c in chunks[:12]:
        biz = str((c.get("metadata", {}) or {}).get("사업명", "")).strip()
        if biz and biz not in seen:
            projects.append(biz)
            seen.add(biz)
    return len(projects) >= 2


def should_checklist(query: str) -> bool:
    return CHECKLIST_HINT in query


def should_strategy(query: str) -> bool:
    return any(h in query for h in STRATEGY_HINTS)


def extract_project_info(project_name: str, chunks: list[dict]) -> dict:
    target = chunks[0] if chunks else {"metadata": {}, "text": ""}
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        ctype = c.get("chunk_type") or meta.get("chunk_type", "")
        if str(ctype).lower() in {"summary", "table"}:
            target = c
            break

    meta = target.get("metadata", {}) or {}
    text = safe_chunk_text(target)

    return {
        "사업명": meta.get("사업명", project_name) or project_name,
        "발주기관": meta.get("발주기관", "미기재") or "미기재",
        "예산": parse_won(meta.get("사업금액", "")),
        "계약기간": extract_line_by_keywords(text, ["사업기간", "용역기간", "계약체결", "착수", "개월"], "미기재"),
        "입찰방식": extract_line_by_keywords(text, ["입찰방식", "계약방법", "낙찰", "협상"], "미기재"),
    }


def build_comparison(query: str, chunks: list[dict]) -> dict | None:
    candidates = []
    seen = set()

    quoted = re.findall(r"[\"'“”‘’](.*?)[\"'“”‘’]", query)
    for q in quoted:
        qn = normalize_text(q)
        if qn and qn not in seen:
            candidates.append(qn)
            seen.add(qn)

    for c in chunks:
        biz = normalize_text((c.get("metadata", {}) or {}).get("사업명", ""))
        if biz and biz not in seen:
            candidates.append(biz)
            seen.add(biz)

    if len(candidates) < 2:
        return None

    selected = candidates[:2]
    details = []
    for name in selected:
        candidate_chunks = retrieve(f"{name} {query}", top_k=10)
        details.append(extract_project_info(name, candidate_chunks))

    return {"projects": selected, "details": details}


def render_comparison(comp: dict) -> None:
    st.markdown("### 사업 비교 분석")

    details = comp.get("details", [])
    if len(details) < 2:
        return

    c1, c2 = st.columns(2)

    def render_col(col, data: dict):
        with col:
            st.markdown(f"**{data.get('사업명', '미기재')}**")
            st.table(
                {
                    "항목": ["사업명", "발주기관", "예산", "계약기간", "입찰방식"],
                    "값": [
                        data.get("사업명", "미기재"),
                        data.get("발주기관", "미기재"),
                        data.get("예산", "미기재"),
                        data.get("계약기간", "미기재"),
                        data.get("입찰방식", "미기재"),
                    ],
                }
            )

    render_col(c1, details[0])
    render_col(c2, details[1])


def collect_lines_from_chunks(chunks: list[dict], limit: int = 200) -> list[str]:
    lines = []
    for chunk in chunks:
        for line in re.split(r"[\n\r]+", safe_chunk_text(chunk)):
            line = normalize_text(line)
            if not line or len(line) < 8:
                continue
            lines.append(line)
            if len(lines) >= limit:
                return lines
    return lines


def build_checklist(query: str, chunks: list[dict]) -> dict[str, list[str]]:
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            str((c.get("chunk_type") or (c.get("metadata", {}) or {}).get("chunk_type", "")).lower())
            not in {"summary", "table"},
            -float(c.get("score", 0) or 0),
        ),
    )

    lines = collect_lines_from_chunks(sorted_chunks, limit=250)

    categorized: dict[str, list[str]] = {k: [] for k in CHECKLIST_CATEGORIES}
    seen_by_cat: dict[str, set[str]] = {k: set() for k in CHECKLIST_CATEGORIES}

    for line in lines:
        for cat, keywords in CHECKLIST_CATEGORIES.items():
            if any(k in line for k in keywords):
                key = line[:150]
                if key not in seen_by_cat[cat]:
                    categorized[cat].append(line[:150])
                    seen_by_cat[cat].add(key)
                break

    for cat in categorized:
        categorized[cat] = categorized[cat][:8]

    return categorized


def render_checklist(query: str, checklist: dict[str, list[str]], key_prefix: str) -> None:
    st.markdown("### 입찰 체크리스트")

    for cat, items in checklist.items():
        st.markdown(f"**{cat}**")
        if not items:
            st.caption("- 관련 항목 미추출")
            continue

        for idx, item in enumerate(items):
            ck = f"{key_prefix}_{cat}_{idx}"
            st.checkbox(item, key=ck)

    st.session_state.checklists[query] = checklist


def detect_risks(chunks: list[dict]) -> dict:
    joined = "\n".join([safe_chunk_text(c) for c in chunks[:20]])
    found_high = [k for k in RISK_KEYWORDS["high"] if k in joined]
    found_medium = [k for k in RISK_KEYWORDS["medium"] if k in joined]
    found_low = [k for k in RISK_KEYWORDS["low"] if k in joined]

    if found_high:
        return {
            "level": "🔴 높음",
            "summary": "계약 해지권/무한 책임 성격의 조항 가능성이 있습니다.",
            "hits": found_high[:5],
        }
    if found_medium:
        return {
            "level": "🟡 주의",
            "summary": "짧은 납기 또는 범위 불명확 가능성이 있습니다.",
            "hits": found_medium[:5],
        }

    return {
        "level": "🟢 양호",
        "summary": "치명 위험 신호는 명확히 감지되지 않았습니다.",
        "hits": found_low[:3],
    }


def render_risk_box(risk: dict) -> None:
    level = risk.get("level", "🟢 양호")
    summary = risk.get("summary", "")
    hits = risk.get("hits", [])

    st.markdown("### 위험 요소 감지")
    st.markdown(f"**등급**: {level}")
    st.markdown(summary)
    if hits:
        st.markdown("감지 키워드: " + ", ".join(hits))


def build_strategy(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for idx, c in enumerate(chunks[:8], start=1):
        meta = c.get("metadata", {}) or {}
        context_parts.append(
            f"[문서 {idx}] {meta.get('발주기관','')} | {meta.get('사업명','')} | {meta.get('파일명','')}\n"
            + safe_chunk_text(c)[:900]
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "다음 RFP 문맥을 바탕으로 입찰 전략을 작성하라.\n"
        "형식:\n"
        "[강점 어필 포인트]\n- 3개\n"
        "[준비 방향]\n- 3개\n"
        "[주의사항]\n- 3개\n"
        "추측 금지, 문맥 근거 중심으로 작성.\n\n"
        f"질문: {query}\n\n"
        f"문맥:\n{context}"
    )

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=700,
        )
    except TypeError:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
        )

    return resp.choices[0].message.content or ""


def extract_strategy_section(text: str, name: str) -> str:
    pattern = rf"\[{re.escape(name)}\]\s*(.*?)(?=\n\[[^\]]+\]|\Z)"
    m = re.search(pattern, str(text), re.DOTALL)
    if not m:
        return "- 항목 생성 실패"
    return m.group(1).strip()


def render_strategy(strategy_text: str) -> None:
    st.markdown("### 입찰 전략 제안")
    p1 = extract_strategy_section(strategy_text, "강점 어필 포인트")
    p2 = extract_strategy_section(strategy_text, "준비 방향")
    p3 = extract_strategy_section(strategy_text, "주의사항")

    with st.expander("강점 어필 포인트", expanded=True):
        st.markdown(p1)
    with st.expander("준비 방향", expanded=True):
        st.markdown(p2)
    with st.expander("주의사항", expanded=True):
        st.markdown(p3)


def ask_with_live_timer(query: str, input_tokens_est: int, metrics_placeholder) -> tuple[dict, float, int]:
    manager = st.session_state.manager
    result_box: dict = {}
    token_trace: list[int] = [0]
    pending_output_tokens = 0

    def worker() -> None:
        try:
            result_box["result"] = manager.ask(query)
        except Exception as exc:
            result_box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    started = time.perf_counter()
    thread.start()

    while thread.is_alive():
        elapsed = time.perf_counter() - started
        pending_output_tokens = max(1, int(elapsed * ESTIMATED_LLM_TOKENS_PER_SECOND))
        token_trace.append(pending_output_tokens)
        render_live_generation_panel(
            placeholder=metrics_placeholder,
            elapsed_s=elapsed,
            input_tokens=input_tokens_est,
            output_tokens=pending_output_tokens,
            token_trace=token_trace[-80:],
            status_text="검색/생성 중",
        )
        time.sleep(0.2)

    thread.join()
    st.session_state.manager = manager

    if "error" in result_box:
        raise result_box["error"]

    elapsed = round(time.perf_counter() - started, 2)
    return result_box.get("result", {}), elapsed, pending_output_tokens


def run_query(query: str, display_query: str | None = None) -> None:
    ensure_active_room_for_query()

    user_text = display_query or query
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.markdown(user_text)

    answer = ""
    chunks: list[dict] = []
    answer_chunks: list[dict] = []
    checklist: dict[str, list[str]] = {}
    strategy_text = ""
    source_files: list[str] = []
    result: dict = {}
    elapsed = 0.0
    input_tokens_est = estimate_token_count(query) + estimate_history_token_count(st.session_state.manager.history)

    with st.chat_message("assistant"):
        live_metrics_placeholder = st.empty()
        stream_stats = {"output_tokens": 0, "stream_elapsed": 0.0}
        pending_output_tokens = 0

        with st.spinner("답변 생성 중..."):
            start = time.perf_counter()
            try:
                aggregate = build_aggregate_answer(query, st.session_state.agency_filter)

                if aggregate is not None:
                    answer = aggregate["answer"]
                    st.session_state.manager.history = (
                        st.session_state.manager.history
                        + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
                    )[-MAX_MANAGER_HISTORY:]

                    result = {
                        "query": query,
                        "answer": answer,
                        "chunks": chunks,
                        "answer_chunks": answer_chunks,
                        "history": st.session_state.manager.history,
                        "aggregate": aggregate,
                    }
                else:
                    result, elapsed, pending_output_tokens = ask_with_live_timer(
                        query,
                        input_tokens_est,
                        live_metrics_placeholder,
                    )
                    answer = result.get("answer", "")
                    chunks = result.get("chunks", [])
                    answer_chunks = result.get("answer_chunks", [])

                    if should_compare(query, chunks):
                        comp = build_comparison(query, chunks)
                        if comp:
                            render_comparison(comp)

                    if should_checklist(query):
                        checklist = build_checklist(query, chunks)
                        render_checklist(query, checklist, key_prefix=f"check_{len(st.session_state.records)}")

                    if should_strategy(query):
                        try:
                            strategy_text = build_strategy(query, answer_chunks or chunks)
                            render_strategy(strategy_text)
                        except Exception as e:
                            strategy_text = ""
                            st.warning(f"전략 제안 생성 실패: {e}")
            except Exception as e:
                answer = (
                    "**답변**\n현재 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n"
                    "**근거**\n모델 또는 검색 처리 단계에서 예외가 발생했습니다.\n\n"
                    "**출처**\n해당 없음"
                )
                st.session_state.manager.history = (
                    st.session_state.manager.history
                    + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
                )[-MAX_MANAGER_HISTORY:]
                result = {
                    "query": query,
                    "answer": answer,
                    "chunks": [],
                    "answer_chunks": [],
                    "history": st.session_state.manager.history,
                    "error": str(e),
                }
                st.warning("일시적 오류가 발생해 기본 안내 답변으로 전환했습니다.")

            if elapsed <= 0:
                elapsed = round(time.perf_counter() - start, 2)

        st.write_stream(
            stream_text_with_live_metrics(
                answer,
                metrics_placeholder=live_metrics_placeholder,
                input_tokens=input_tokens_est,
                stream_stats=stream_stats,
                initial_output_tokens=pending_output_tokens,
            )
        )

        confidence = calc_confidence(answer)
        render_confidence_badge(confidence)
        total_visible_elapsed = round(elapsed + float(stream_stats.get("stream_elapsed", 0.0)), 2)
        st.caption(f"응답시간: {elapsed}s (생성) / {total_visible_elapsed}s (표시 포함)")

        risk = detect_risks(chunks) if chunks else {"level": "🟢 양호", "summary": "집계 응답입니다.", "hits": []}
        source_files = extract_source_filenames(answer)
        render_answer_details(answer, risk, source_files, key_prefix=f"live_{len(st.session_state.messages)}")

        question_for_bookmark = user_text
        answer_summary = parse_answer(answer)[:180]

        if st.button("⭐ 북마크 저장", key=f"live_bookmark_{len(st.session_state.messages)}"):
            card = st.session_state.summary_card or {}
            add_bookmark(
                {
                    "사업명": card.get("사업명", "미기재"),
                    "질문": question_for_bookmark,
                    "답변요약": answer_summary,
                    "저장시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            st.success("북마크에 저장했습니다.")

    assistant_payload = {
        "role": "assistant",
        "content": answer,
        "confidence": confidence,
        "response_time": elapsed,
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": int(stream_stats.get("output_tokens", 0)),
        "total_tokens_est": int(input_tokens_est + int(stream_stats.get("output_tokens", 0))),
        "source_files": source_files,
        "raw": result,
        "question": question_for_bookmark,
        "risk": risk,
        "checklist": checklist,
        "strategy": strategy_text,
    }
    st.session_state.messages.append(assistant_payload)

    if answer_chunks:
        st.session_state.summary_card = build_summary_card(answer_chunks)

    st.session_state.records.append(
        {
            "질문": question_for_bookmark,
            "답변": answer,
            "출처": get_section(answer, "출처"),
            "신뢰도": confidence,
            "응답시간": elapsed,
            "입력토큰(추정)": input_tokens_est,
            "출력토큰(추정)": int(stream_stats.get("output_tokens", 0)),
            "총토큰(추정)": int(input_tokens_est + int(stream_stats.get("output_tokens", 0))),
            "위험등급": risk.get("level", ""),
            "체크리스트": checklist,
            "전략": strategy_text,
            "raw": result,
        }
    )
    sync_active_room_from_session(update_title=True)


def export_sidebar() -> None:
    st.sidebar.markdown("📥 내보내기")

    records = st.session_state.records
    json_bytes = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
    st.sidebar.download_button(
        "JSON",
        data=json_bytes,
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True,
    )

    try:
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "질문": r.get("질문", ""),
                    "답변": r.get("답변", ""),
                    "출처": r.get("출처", ""),
                    "신뢰도": r.get("신뢰도", ""),
                    "응답시간": r.get("응답시간", ""),
                    "입력토큰(추정)": r.get("입력토큰(추정)", ""),
                    "출력토큰(추정)": r.get("출력토큰(추정)", ""),
                    "총토큰(추정)": r.get("총토큰(추정)", ""),
                    "위험등급": r.get("위험등급", ""),
                }
                for r in records
            ]
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="history")
        buffer.seek(0)

        st.sidebar.download_button(
            "Excel",
            data=buffer,
            file_name="chat_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception:
        st.sidebar.caption("Excel 내보내기는 pandas/openpyxl 설치 후 활성화됩니다.")


def render_sidebar() -> None:
    if st.sidebar.button(APP_NAME, key="brand_home", use_container_width=True):
        switch_to_home()
        st.rerun()

    st.sidebar.markdown("#### 메뉴")
    if st.sidebar.button("홈", key="menu_home", use_container_width=True):
        switch_to_home()
        st.rerun()
    p1 = resolve_page_link_by_order(1)
    p2 = resolve_page_link_by_order(2)
    p3 = resolve_page_link_by_order(3)
    p4 = resolve_page_link_by_order(4)
    if p1:
        st.sidebar.page_link(p1, label="대시보드", icon="📊")
    if p2:
        st.sidebar.page_link(p2, label="사업검색", icon="🔎")
    if p3:
        st.sidebar.page_link(p3, label="타임라인", icon="🗓️")
    if p4:
        st.sidebar.page_link(p4, label="북마크", icon="⭐")

    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("➕ 새 대화", use_container_width=True, type="primary"):
            reset_chat_session(clear_records=False)
            st.rerun()
    with c2:
        if st.button("🧹 전체 초기화", use_container_width=True):
            reset_chat_session(clear_records=True)
            st.rerun()

    assistant_turns = sum(1 for m in st.session_state.messages if m.get("role") == "assistant")
    st.sidebar.caption(f"현재 대화 턴: {assistant_turns}")

    st.sidebar.markdown("#### 🗂️ 대화 목록")
    visible_rooms = [room for room in st.session_state.chat_rooms if room_has_content(room)]
    rooms = sorted(
        visible_rooms,
        key=lambda r: (r.get("updated_at", ""), r.get("created_at", "")),
        reverse=True,
    )

    if not rooms:
        st.sidebar.caption("대화를 시작하면 여기에 표시됩니다.")

    with st.sidebar.container(key="chat_room_list"):
        for room in rooms:
            room_id = room.get("id", "")
            room_title = normalize_text(room.get("title", "새 대화")) or "새 대화"
            is_active = room_id == st.session_state.active_chat_id

            c_room, c_del = st.columns([0.82, 0.18])
            with c_room:
                if st.button(
                    f"{'● ' if is_active else ''}{room_title}",
                    key=f"room_open_{room_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    sync_active_room_from_session(update_title=True)
                    switch_chat_room(room_id)
                    st.rerun()
            with c_del:
                if st.button(
                    "✕",
                    key=f"room_del_{room_id}",
                    use_container_width=True,
                    help="대화 삭제",
                ):
                    sync_active_room_from_session(update_title=True)
                    delete_chat_room(room_id)
                    st.rerun()

    st.sidebar.markdown("---")
    bookmarks = st.session_state.bookmarks
    st.sidebar.markdown(f"⭐ 즐겨찾기 ({len(bookmarks)}개)")

    for idx, b in enumerate(bookmarks[:3]):
        label = f"{idx + 1}. {b.get('사업명', '미기재')}"
        with st.sidebar.expander(label):
            st.caption(b.get("저장시각", ""))
            st.write(b.get("질문", ""))
            st.write(b.get("답변요약", ""))

    st.sidebar.markdown("---")
    export_sidebar()

    st.sidebar.markdown("---")
    st.sidebar.markdown("ℹ️ 시스템 상태")
    st.sidebar.markdown(f"Ollama: 🟢 {LLM_MODEL}")
    st.sidebar.markdown("ChromaDB: ✅ 연결 준비")


def render_summary_card() -> None:
    card = st.session_state.summary_card
    if not card:
        return

    with st.expander("핵심 조건 요약 카드", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**사업명**: {card.get('사업명', '미기재')}")
            st.markdown(f"**발주기관**: {card.get('발주기관', '미기재')}")
            st.markdown(f"**예산**: {card.get('예산', '미기재')}")
        with c2:
            st.markdown(f"**계약기간**: {card.get('계약기간', '미기재')}")
            st.markdown(f"**입찰방식**: {card.get('입찰방식', '미기재')}")
            st.markdown(f"**자격요건**: {card.get('자격요건', '미기재')}")


def render_starter_prompts() -> None:
    if st.session_state.messages:
        return

    st.markdown(
        """
        <div class='starter-wrap'>
          <div class='starter-title'>무엇을 도와드릴까요?</div>
          <div class='starter-sub'>아래 예시를 눌러 바로 질문을 시작할 수 있습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for idx, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[idx % 2]:
            if st.button(prompt, key=f"starter_{idx}", use_container_width=True):
                st.session_state.queued_query = prompt
                st.rerun()


def rebuild_records_from_messages(messages: list[dict]) -> list[dict]:
    records = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        answer = msg.get("content", "")
        records.append(
            {
                "질문": msg.get("question", ""),
                "답변": answer,
                "출처": get_section(answer, "출처"),
                "신뢰도": msg.get("confidence", ""),
                "응답시간": msg.get("response_time", ""),
                "입력토큰(추정)": msg.get("input_tokens_est", ""),
                "출력토큰(추정)": msg.get("output_tokens_est", ""),
                "총토큰(추정)": msg.get("total_tokens_est", ""),
                "위험등급": (msg.get("risk") or {}).get("level", "") if isinstance(msg.get("risk"), dict) else "",
                "체크리스트": msg.get("checklist", {}),
                "전략": msg.get("strategy", ""),
                "raw": msg.get("raw", {}),
            }
        )
    return records


def rebuild_manager_history_from_messages(messages: list[dict]) -> list[dict]:
    history = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if content:
            history.append({"role": role, "content": content})
    return history[-MAX_MANAGER_HISTORY:]


def queue_regeneration_from_user_message(user_index: int, query: str) -> None:
    prefix = st.session_state.messages[:user_index]
    st.session_state.messages = deepcopy(prefix)
    st.session_state.records = rebuild_records_from_messages(prefix)
    st.session_state.manager.history = rebuild_manager_history_from_messages(prefix)
    st.session_state.summary_card = None
    st.session_state.checklists = {}
    st.session_state.editing_message_index = None
    st.session_state.editing_message_text = ""
    st.session_state.queued_query = normalize_text(query)
    sync_active_room_from_session(update_title=True)
    st.rerun()


def render_user_message_actions(index: int, msg: dict) -> None:
    content = normalize_text(msg.get("content", ""))
    if not content:
        return

    c1, c2, c3 = st.columns([1.0, 0.18, 0.28])
    with c2:
        if st.button("수정", key=f"user_edit_{index}", use_container_width=True):
            st.session_state.editing_message_index = index
            st.session_state.editing_message_text = content
            st.rerun()
    with c3:
        if st.button("다시 생성", key=f"user_regen_{index}", use_container_width=True):
            queue_regeneration_from_user_message(index, content)

    if st.session_state.editing_message_index != index:
        return

    with st.form(f"user_edit_form_{index}"):
        edited = st.text_area(
            "메시지 수정",
            value=st.session_state.editing_message_text or content,
            label_visibility="collapsed",
            height=90,
        )
        f1, f2 = st.columns(2)
        with f1:
            submitted = st.form_submit_button("저장 후 재생성", use_container_width=True)
        with f2:
            cancelled = st.form_submit_button("취소", use_container_width=True)

    if submitted and normalize_text(edited):
        queue_regeneration_from_user_message(index, edited)
    if cancelled:
        st.session_state.editing_message_index = None
        st.session_state.editing_message_text = ""
        st.rerun()


def render_messages() -> None:
    for i, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            if role == "user":
                render_user_message_actions(i, msg)
            if role == "assistant":
                conf = msg.get("confidence")
                rt = msg.get("response_time")
                if conf or rt is not None:
                    render_confidence_badge(conf or "🟡 보통")
                    st.caption(f"응답시간: {rt if rt is not None else '-'}s")

                risk = msg.get("risk")
                if isinstance(risk, dict):
                    render_answer_details(
                        msg.get("content", ""),
                        risk,
                        msg.get("source_files", []),
                        key_prefix=f"hist_{i}",
                    )

                if st.button("⭐ 북마크 저장", key=f"hist_bookmark_{i}"):
                    card = st.session_state.summary_card or {}
                    add_bookmark(
                        {
                            "사업명": card.get("사업명", "미기재"),
                            "질문": msg.get("question", ""),
                            "답변요약": parse_answer(msg.get("content", ""))[:180],
                            "저장시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    st.success("북마크에 저장했습니다.")


def typo_confirmation_flow() -> bool:
    original = st.session_state.pending_original
    corrected = st.session_state.pending_corrected
    if not original or not corrected:
        return False

    st.info(f"💡 '{original}' → '{corrected}'로 이해했습니다. 맞나요?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("맞아요", use_container_width=True, key="confirm_typo_yes"):
            st.session_state.pending_original = ""
            st.session_state.pending_corrected = ""
            run_query(corrected, display_query=original)
            return True
    with c2:
        if st.button("다시 입력", use_container_width=True, key="confirm_typo_no"):
            st.session_state.pending_original = ""
            st.session_state.pending_corrected = ""
            st.warning("질문을 다시 입력해주세요.")
            return True

    return True


def run_queued_query_if_exists() -> None:
    queued = st.session_state.queued_query
    if queued:
        st.session_state.queued_query = ""
        run_query(queued)
        st.rerun()


def main() -> None:
    load_css()
    init_state()
    render_sidebar()

    st.title(APP_NAME)
    st.caption("RFP 문서를 기반으로 답변합니다.")

    if st.session_state.agency_filter and st.session_state.agency_filter != "전체":
        st.markdown(
            f"<span class='agency-badge'>현재 필터: {st.session_state.agency_filter}</span>",
            unsafe_allow_html=True,
        )

    run_queued_query_if_exists()

    render_summary_card()
    render_starter_prompts()
    render_messages()

    if typo_confirmation_flow():
        return

    user_input = st.chat_input("RFP에 대해 질문해 주세요")

    if user_input:
        corrected = rule_based_query_cleanup(user_input)

        if st.session_state.agency_filter == "전체":
            agency_name, score = get_agency_filter(user_input)
            if agency_name and score >= 88:
                st.info(f"💡 '{agency_name}' 기관을 자동 감지했습니다.")

        if corrected != user_input:
            st.session_state.pending_original = user_input
            st.session_state.pending_corrected = corrected
            st.rerun()
        else:
            run_query(user_input)


if __name__ == "__main__":
    main()
