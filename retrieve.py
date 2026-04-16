import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_classic.prompts import load_prompt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.vectorstores import FAISS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FAISS_PATH = "faiss_movies"
PROMPT_DIR = "prompts"
ENRICHED_CSV = "data/movies_enriched.csv"

# 영화 상세 정보 조회용 DataFrame (movie_id → row)
_movies_df = pd.read_csv(ENRICHED_CSV).set_index("movie_id")

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("벡터 수:", vectorstore.index.ntotal)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

class QueryParseResult(BaseModel):
    intent: str
    genres: List[str] = Field(default_factory=list)
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_rating: Optional[float] = None
    max_runtime: Optional[int] = None
    mood: List[str] = Field(default_factory=list)
    audience: List[str] = Field(default_factory=list)
    similar_to: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class PreferenceProfile(BaseModel):
    taste_summary: str
    preferred_genres: List[str] = Field(default_factory=list)
    preferred_moods: List[str] = Field(default_factory=list)
    preferred_themes: List[str] = Field(default_factory=list)

query_prompt = load_prompt(f"{PROMPT_DIR}/query_parsing.yaml", encoding="utf-8")
preference_prompt = load_prompt(f"{PROMPT_DIR}/preference_analysis.yaml", encoding="utf-8")
explanation_prompt = load_prompt(f"{PROMPT_DIR}/recommendation_explanation.yaml", encoding="utf-8")
followup_prompt = load_prompt(f"{PROMPT_DIR}/followup_update.yaml", encoding="utf-8")

query_parser = PydanticOutputParser(pydantic_object=QueryParseResult)
preference_parser = PydanticOutputParser(pydantic_object=PreferenceProfile)
text_parser = StrOutputParser()

# structured prompt에 format_instructions 주입
query_prompt = query_prompt.partial(
    format_instructions=query_parser.get_format_instructions()
)
preference_prompt = preference_prompt.partial(
    format_instructions=preference_parser.get_format_instructions()
)
followup_prompt = followup_prompt.partial(
    format_instructions=query_parser.get_format_instructions()
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

_SYSTEM_MESSAGE = (
    "당신은 CineMate라는 영화 추천 도우미입니다. "
    "과거 대화 히스토리를 참고하여 사용자의 취향과 직전 요청을 반영하세요. "
    "항상 한국어로 응답하세요.\n\n"
    "절대 규칙 (위반 금지):\n"
    "- '추천 결과'에 있는 영화만 소개하세요. 목록에 없는 영화는 절대 추가하거나 언급하지 마세요.\n"
    "- 영화 정보(감독, 줄거리, 연도 등)는 '추천 결과'에 있는 내용만 사용하세요. 임의로 만들지 마세요.\n"
    "- 조건에 완벽히 맞지 않더라도 주어진 영화를 모두 소개하세요. 스스로 거르지 마세요.\n\n"
    "출력 규칙:\n"
    "1. 추천 결과의 영화를 모두(3편) 빠짐없이 소개하세요.\n"
    "2. 각 영화 제목은 '한국어 제목 (English Title)' 형식으로 표시하세요.\n"
    "3. 연도와 추천 이유를 함께 작성하세요.\n"
    "4. 간결하고 읽기 쉽게 목록 형태로 작성하세요."
)

final_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_MESSAGE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "사용자 요청:\n{user_query}\n\n추천 결과:\n{recommended_movies_with_reasons}"),
])

def _build_chains(model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0)
    return {
        "query_parsing": query_prompt | llm | query_parser,
        "preference_analysis": preference_prompt | llm | preference_parser,
        "recommendation_explanation": explanation_prompt | llm | text_parser,
        "followup_update": followup_prompt | llm | query_parser,
        "memory_response": final_chat_prompt | llm | text_parser,
    }

_current_model = "gpt-4.1-mini"
_chains = _build_chains(_current_model)

query_parsing_chain              = _chains["query_parsing"]
preference_analysis_chain        = _chains["preference_analysis"]
recommendation_explanation_chain = _chains["recommendation_explanation"]
followup_update_chain            = _chains["followup_update"]
memory_response_chain            = _chains["memory_response"]

def set_model(model_name: str):
    """사용할 LLM 모델을 변경 (Streamlit 사이드바에서 호출)"""
    global _current_model, _chains
    global query_parsing_chain, preference_analysis_chain
    global recommendation_explanation_chain, followup_update_chain
    global memory_response_chain
    if model_name != _current_model:
        _current_model = model_name
        _chains = _build_chains(model_name)
        query_parsing_chain              = _chains["query_parsing"]
        preference_analysis_chain        = _chains["preference_analysis"]
        recommendation_explanation_chain = _chains["recommendation_explanation"]
        followup_update_chain            = _chains["followup_update"]
        memory_response_chain            = _chains["memory_response"]

# retrieval 함수

def retrieve_movies(user_query, k=5):
    return vectorstore.similarity_search(user_query, k=k)

def translate_titles(movies: list) -> list:
    """영화 제목을 한국어로 번역해서 korean_title 필드 추가 (LLM 1회 호출)"""
    titles = [m.get("title", "") for m in movies]
    prompt = (
        "다음 영화 제목들의 한국어 공식 제목을 알려주세요.\n"
        "형식: 번호. 한국어제목\n"
        "한국어 제목이 없으면 영어 제목 그대로 쓰세요.\n"
        "설명 없이 번호와 제목만 출력하세요.\n\n"
        + "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))
    )
    llm = ChatOpenAI(model=_current_model, temperature=0)
    result = llm.invoke(prompt).content.strip()

    korean_titles = []
    for line in result.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            # "1. 제목" 형태에서 제목만 추출
            korean_titles.append(line.split(". ", 1)[-1].strip())

    for i, movie in enumerate(movies):
        movie["korean_title"] = korean_titles[i] if i < len(korean_titles) else movie.get("title", "")

    return movies

# 재정렬 함수

def rerank_results(results, prefer_recent: bool = False):
    scored = []
    current_year = 2025
    for doc in results:
        avg_rating = doc.metadata.get("avg_rating", 0) or 0
        rating_count = doc.metadata.get("rating_count", 0) or 0
        year = doc.metadata.get("year") or 1900
        score = float(avg_rating) + min(float(rating_count) / 1000, 1.0)
        if prefer_recent:
            # 최신 영화일수록 최대 2점 가중치
            recency_bonus = min((float(year) - 1900) / (current_year - 1900), 1.0) * 2.0
            score += recency_bonus
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]

def _filter_by_audience(results, audience: list) -> list:
    """LLM 배치 호출로 audience에 부적합한 영화를 한 번에 제거"""
    if not audience or not results:
        return results

    audience_str = ", ".join(audience)
    movie_list = []
    for i, doc in enumerate(results):
        title = doc.metadata.get("title", "")
        genres = doc.metadata.get("genres", "")
        overview = doc.page_content[:400]
        movie_list.append(f"{i}. [{title}] 장르: {genres}\n내용: {overview}")

    movies_text = "\n\n".join(movie_list)
    prompt = (
        f"다음 영화 목록에서 '{audience_str}'에게 적합하지 않은 영화의 번호를 골라주세요.\n\n"
        "판단 기준: 폭력·공포·성인 내용·고어·성적 표현이 있으면 부적합.\n\n"
        f"영화 목록:\n{movies_text}\n\n"
        "출력 형식: 부적합한 영화 번호만 쉼표로 나열 (예: 0,3,7)\n"
        "부적합한 영화가 없으면 '없음'으로만 출력하세요. 설명 없이 번호만 출력하세요."
    )

    llm = ChatOpenAI(model=_current_model, temperature=0)
    result = llm.invoke(prompt).content.strip()
    print(f"[3] audience LLM 판정 : {result}")

    if result == "없음" or not result:
        return results

    try:
        excluded = {int(x.strip()) for x in result.split(",") if x.strip().isdigit()}
        filtered = [doc for i, doc in enumerate(results) if i not in excluded]
        return filtered if len(filtered) >= 3 else results
    except Exception:
        return results


# Memory component 추가

from langchain_community.chat_message_histories import ChatMessageHistory
store = {}
query_store = {}  # 세션별 마지막 parsed_query 저장

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# E2E 추천 함수

def build_recommendation_payload(user_query: str, session_id: str = None):
    # 1) 질의 구조화 — 후속 질문이면 이전 조건 이어받기
    previous_query = query_store.get(session_id) if session_id else None

    if previous_query:
        parsed_query = followup_update_chain.invoke({
            "previous_criteria": previous_query.model_dump_json(),
            "followup_query": user_query,
        })
    else:
        parsed_query = query_parsing_chain.invoke({
            "user_query": user_query
        })

    # 세션에 현재 파싱 결과 저장
    if session_id:
        query_store[session_id] = parsed_query

    # ── 로그 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[1] 쿼리         : {user_query}")
    print(f"[1] 후속질문 여부 : {previous_query is not None}")
    print(f"[1] parsed_query : {parsed_query}")

    # 2) 파싱 결과로 영어 검색 쿼리 구성 (임베딩 모델이 영어 전용)
    search_terms = (
        parsed_query.genres
        + parsed_query.keywords
        + parsed_query.mood
        + parsed_query.similar_to
        + parsed_query.audience  # audience 포함 → FAISS 후보 자체를 개선
    )
    enhanced_query = " ".join(search_terms) if search_terms else user_query
    print(f"[2] 검색 쿼리    : {enhanced_query}")

    # 3) 후보를 넉넉하게 검색 후 필터링
    results = retrieve_movies(enhanced_query, k=50)
    print(f"[3] FAISS 후보   : {len(results)}편")

    # 장르 필터
    if parsed_query.genres:
        target_genres = [g.lower() for g in parsed_query.genres]
        filtered = [
            r for r in results
            if any(g in str(r.page_content).lower() for g in target_genres)
        ]
        results = filtered if len(filtered) >= 3 else results
        print(f"[3] 장르 필터 후 : {len(results)}편 {parsed_query.genres}")

    # 연도 필터 (데이터셋 최대 연도 = 2018)
    DATASET_MAX_YEAR = 2018
    if parsed_query.min_year:
        min_year = min(parsed_query.min_year, DATASET_MAX_YEAR)
        for candidate_year in [min_year, min_year - 5, min_year - 10, 2000]:
            filtered = [
                r for r in results
                if (r.metadata.get("year") or 0) >= candidate_year
            ]
            if len(filtered) >= 3:
                results = filtered
                print(f"[3] 연도 필터 후 : {len(results)}편 (min_year={candidate_year})")
                break

    if parsed_query.max_year:
        filtered = [
            r for r in results
            if (r.metadata.get("year") or 9999) <= parsed_query.max_year
        ]
        results = filtered if len(filtered) >= 3 else results
        print(f"[3] max_year 후  : {len(results)}편 (max_year={parsed_query.max_year})")

    # audience 필터 — LLM이 부적합 영화 제거
    if parsed_query.audience:
        results = _filter_by_audience(results, parsed_query.audience)
        print(f"[3] audience 필터 후 : {len(results)}편 {parsed_query.audience}")

    results = rerank_results(results, prefer_recent=bool(parsed_query.min_year))
    print(f"[4] 최종 선택    :")
    for r in results[:3]:
        print(f"    - {r.metadata.get('title')} ({int(r.metadata.get('year') or 0)}) 감독: {r.metadata.get('director')}")

    # 3) 각 영화별 추천 이유 생성
    def _clean(text: str) -> str:
        """JSON 직렬화를 깨는 문자 제거"""
        return (
            str(text)
            .replace("\x00", "")          # null byte 제거
            .encode("utf-8", errors="ignore").decode("utf-8")  # 잘못된 유니코드 제거
            [:3000]                        # 너무 길면 잘라냄
        )

    movie_reasons = []
    for doc in results[:3]:
        reason = recommendation_explanation_chain.invoke({
            "user_query": _clean(user_query),
            "taste_profile": "",
            "movie_info": _clean(doc.page_content),
        })

        # FAISS 메타데이터에 없는 필드는 CSV에서 조회
        movie_id = doc.metadata.get("movie_id")
        enriched = {}
        if movie_id and movie_id in _movies_df.index:
            row = _movies_df.loc[movie_id]
            enriched = row.to_dict()

        movie_reasons.append({
            "title": enriched.get("title") or doc.metadata.get("title"),
            "year": enriched.get("year") or doc.metadata.get("year"),
            "reason": reason,
            "poster_path": enriched.get("poster_path"),
            "genres": enriched.get("genres") or doc.metadata.get("genres"),
            "avg_rating": enriched.get("avg_rating") or doc.metadata.get("avg_rating"),
            "tmdb_vote_average": enriched.get("tmdb_vote_average"),
            "overview": enriched.get("overview"),
            "director": enriched.get("director") or doc.metadata.get("director"),
            "cast_top3": enriched.get("cast_top3"),
            "runtime": enriched.get("runtime") or doc.metadata.get("runtime"),
        })

    movie_reasons = translate_titles(movie_reasons)

    return {
        "parsed_query": parsed_query,
        "movie_reasons": movie_reasons
    }

# RunnableWithMessageHistory로 감싸기

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

def invoke_cinemate_core(inputs: dict):
    user_query = inputs["user_query"]

    payload = build_recommendation_payload(user_query)

    return memory_response_chain.invoke({
        "history": inputs["history"],
        "user_query": user_query,
        "recommended_movies_with_reasons": json.dumps(
            payload["movie_reasons"],
            ensure_ascii=False,
            indent=2
        )
    })

cinemate_core = RunnableLambda(invoke_cinemate_core)

cinemate_with_memory = RunnableWithMessageHistory(
    cinemate_core,
    get_session_history,
    input_messages_key="user_query",
    history_messages_key="history",
)

# Streamlit용 통합 함수

from langchain_core.messages import HumanMessage, AIMessage

def get_cinemate_response(user_query: str, session_id: str) -> dict:
    """추천 텍스트와 영화 상세 정보를 함께 반환"""
    payload = build_recommendation_payload(user_query, session_id=session_id)

    history = get_session_history(session_id)

    response = memory_response_chain.invoke({
        "history": history.messages,
        "user_query": user_query,
        "recommended_movies_with_reasons": json.dumps(
            payload["movie_reasons"],
            ensure_ascii=False,
            indent=2
        )
    })

    history.add_message(HumanMessage(content=user_query))
    history.add_message(AIMessage(content=response))

    return {
        "response": response,
        "movies": payload["movie_reasons"]
    }


# End-to-End 실행 (직접 실행 시에만)

if __name__ == "__main__":
    session_id = "user-001"

    response1 = cinemate_with_memory.invoke(
        {"user_query": "가족이랑 보기 좋은 2시간 이하 감동적인 영화 추천해줘"},
        config={"configurable": {"session_id": session_id}}
    )

    print(response1)

    # 후속 질문

    response2 = cinemate_with_memory.invoke(
        {"user_query": "조금 더 가볍고 웃긴 영화로 바꿔줘"},
        config={"configurable": {"session_id": session_id}}
    )

    print(response2)
