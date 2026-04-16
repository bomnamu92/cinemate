import ast
import uuid

import streamlit as st

from retrieve import get_cinemate_response, query_store, set_model, _current_model

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"

st.set_page_config(
    page_title="CineMate",
    page_icon="🎬",
    layout="wide",
)

# ── Session state 초기화 ─────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []


# ── 유틸 함수 ────────────────────────────────────────
def parse_genres(genres_str) -> list[str]:
    try:
        result = ast.literal_eval(str(genres_str))
        return result if isinstance(result, list) else []
    except Exception:
        return []


def parse_cast(cast_str) -> list[str]:
    if not cast_str or str(cast_str) == "nan":
        return []
    return [c.strip() for c in str(cast_str).split("|") if c.strip()]


def display_movie_cards(movies: list):
    for movie in movies:
        with st.container(border=True):
            col_poster, col_info = st.columns([1, 3])

            # 포스터
            with col_poster:
                poster_path = movie.get("poster_path")
                if poster_path and str(poster_path) not in ("nan", "None", ""):
                    st.image(TMDB_IMAGE_BASE + str(poster_path), width=130)
                else:
                    st.markdown("### 🎬")

            # 영화 정보
            with col_info:
                title = movie.get("title", "")
                korean_title = movie.get("korean_title") or title
                year = movie.get("year")
                year_str = str(int(float(year))) if year and str(year) != "nan" else ""

                # 제목: 한국어 (English) - 원어(비라틴)
                def _is_latin(s: str) -> bool:
                    return all(ord(c) < 256 for c in s if c.isalpha())

                title_parts = [korean_title]
                if title and title != korean_title:
                    title_parts.append(f"({title})")
                if title and not _is_latin(title) and title != korean_title:
                    title_parts.append(f"- {title}")

                st.markdown(f"### {' '.join(title_parts)}")

                # 연도 · 평점 · 상영시간
                rating = movie.get("tmdb_vote_average") or movie.get("avg_rating") or 0
                runtime = movie.get("runtime")
                runtime_str = f" · {int(float(runtime))}분" if runtime and str(runtime) != "nan" else ""
                try:
                    rating_val = float(rating)
                    rating_display = f"⭐ {rating_val:.1f}"
                except Exception:
                    rating_display = ""

                st.markdown(f"**{year_str}** {rating_display}{runtime_str}")

                # 장르 배지
                genres = parse_genres(movie.get("genres", "[]"))
                if genres:
                    st.markdown(" ".join([f"`{g}`" for g in genres[:5]]))

                # 줄거리
                overview = movie.get("overview", "")
                if overview and str(overview) not in ("nan", "None", ""):
                    text = str(overview)
                    st.caption(text[:200] + ("..." if len(text) > 200 else ""))

                # 감독 · 출연
                director = movie.get("director", "")
                cast = parse_cast(movie.get("cast_top3", ""))
                parts = []
                if director and str(director) not in ("nan", "None", ""):
                    parts.append(f"🎬 **감독** {director}")
                if cast:
                    parts.append(f"🎭 **출연** {', '.join(cast)}")
                if parts:
                    st.markdown("  \n".join(parts))

                # 추천 이유
                reason = movie.get("reason", "")
                if reason:
                    st.info(f"💬 {reason}")


# ── 사이드바 ─────────────────────────────────────────
with st.sidebar:
    st.title("🎬 CineMate")
    st.caption("AI 영화 추천 챗봇")

    if st.button("+ 새 대화", use_container_width=True, type="primary"):
        old_id = st.session_state.session_id
        query_store.pop(old_id, None)  # 이전 세션 쿼리 조건 초기화
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("모델 설정")
    selected_model = st.selectbox(
        "OpenAI 모델",
        options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
        index=0,
        label_visibility="collapsed",
    )
    set_model(selected_model)
    st.caption(f"현재 모델: `{selected_model}`")

    st.divider()
    st.subheader("대화 목록")

    if not st.session_state.conversations:
        st.caption("아직 대화가 없어요.")
    else:
        for conv in reversed(st.session_state.conversations):
            preview = conv[:24] + ("..." if len(conv) > 24 else "")
            st.markdown(f"• {preview}")


# ── 메인 영역 ────────────────────────────────────────
st.title("CineMate 🎬")
st.caption("원하는 영화를 말씀해 주세요. 취향에 맞는 영화를 추천해 드릴게요!")

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        elif msg.get("movies"):
            display_movie_cards(msg["movies"])

# 입력
if prompt := st.chat_input("예) 가족이랑 보기 좋은 감동적인 영화 추천해줘"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt not in st.session_state.conversations:
        st.session_state.conversations.append(prompt)

    # 추천 생성 및 표시 (카드만)
    with st.chat_message("assistant"):
        with st.spinner("영화를 찾는 중..."):
            result = get_cinemate_response(prompt, st.session_state.session_id)
        display_movie_cards(result["movies"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": "",
        "movies": result["movies"],
    })
