import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
from search import MovieRecommender

st.set_page_config(page_title="영화 추천 사이트", page_icon="🎬", layout="centered")

st.title("🎬 영화 추천 사이트")
st.caption("보고 싶은 영화를 말로 설명해주세요. 한국어/영어 모두 가능합니다.")

@st.cache_resource
def load_recommender():
    return MovieRecommender(nprobe=64)

with st.spinner("모델 및 인덱스 로딩 중... (첫 실행 시 1~2분 소요)"):
    rec = load_recommender()

if 'mode' not in st.session_state:
    st.session_state.mode = 'search'
if 'keyword_data' not in st.session_state:
    st.session_state.keyword_data = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = None

# ── 키워드 검색 모드 ──────────────────────────────────────────────
if st.session_state.mode == 'keyword' and st.session_state.keyword_data:
    ks = st.session_state.keyword_data
    color = "🟢" if ks['sentiment'] == "긍정" else "🔴" if ks['sentiment'] == "부정" else "⚪"
    if st.button("← 메인으로 돌아가기", type="secondary"):
        st.session_state.mode = 'search'
        st.rerun()

    st.markdown("---")
    st.subheader(f"{color} #{ks['keyword']} — {ks['sentiment']} 리뷰 기반 추천")
    st.caption(f"'{ks['movie']}'에서 선택한 키워드로 유사 영화 검색")

    sc1, sc2 = st.columns(2)
    with sc1:
        sort_option = st.radio("정렬", ["추천 점수순", "최신순", "옛날순"], horizontal=True)
    with sc2:
        kw_year_range = st.slider("제작 연도", 1960, 2026, (1960, 2026), key="kw_year")

    with st.spinner("키워드 기반 검색 중..."):
        sent = ks['sentiment'] if ks['sentiment'] != "중립" else "긍정"
        kw_results = rec.search_by_keyword_sentiment(ks['keyword'], sent, top_k=20)

    kw_results = [r for r in kw_results if kw_year_range[0] <= int(r['year'] or 0) <= kw_year_range[1]]

    if sort_option == "최신순":
        kw_results.sort(key=lambda x: int(x['year'] or 0), reverse=True)
    elif sort_option == "옛날순":
        kw_results.sort(key=lambda x: int(x['year'] or 9999))

    if not kw_results:
        st.warning("관련 영화를 찾지 못했습니다.")
    else:
        for i, r in enumerate(kw_results[:5], 1):
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{i}. {r['title']}** ({r['year']})")
                with c2:
                    st.markdown(f"⭐ {r['avg_rating']} / 10")
                pos = r.get('pos_count', 0)
                neg = r.get('neg_count', 0)
                neu = r.get('neu_count', 0)
                st.caption(f"매칭 리뷰 {r['matched_reviews']}건 · 🟢긍정 {pos} · 🔴부정 {neg} · ⚪중립 {neu}")
                st.markdown("---")

# ── 일반 검색 모드 ────────────────────────────────────────────────
else:
    query = st.text_area(
        "어떤 영화를 찾고 계신가요?",
        placeholder="예) 가족을 잃은 슬픔을 다루면서도 희망적인 결말을 가진 영화\n예) a surreal thriller with an unreliable narrator",
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        year_range = st.slider("제작 연도", 1960, 2026, (1960, 2026))
    with col2:
        min_rating = st.slider("최소 평점", 0.0, 10.0, 0.0, step=0.5)

    if st.button("추천 받기", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("검색어를 입력해주세요.")
        else:
            with st.spinner("검색 중..."):
                results = rec.recommend(query.strip(), top_k=20,
                                        stage1_k=300, stage2_k=5000,
                                        max_negative_ratio=0.6)
            filtered = [
                r for r in results
                if year_range[0] <= int(r["year"] or 0) <= year_range[1]
                and r["avg_rating"] >= min_rating
            ][:5]
            st.session_state.last_results = filtered

    # 결과 표시
    filtered = st.session_state.last_results
    if filtered:
        st.markdown("---")
        st.subheader("추천 영화")
        for i, r in enumerate(filtered, 1):
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{i}. {r['title']}** ({r['year']})")
                with c2:
                    st.markdown(f"⭐ {r['avg_rating']} / 10")
                neg = r.get('negative_ratio', '-')
                st.caption(f"유사 리뷰 {r['similar_reviews']}건 · 부정 {neg}% · 추천 점수 {r['score']}")

                # 리뷰 발췌
                snippets = r.get('snippets', [])
                if snippets:
                    with st.expander("💬 리뷰 발췌 보기"):
                        for s in snippets:
                            icon = "🟢" if s['label'] == "긍정" else "🔴" if s['label'] == "부정" else "⚪"
                            st.markdown(f"{icon} **[{s['label']}]** {s['text']}")

                # 키워드 해시태그
                keywords = rec.extract_keywords(r['title'])
                if keywords:
                    st.caption("🟢 긍정 · 🔴 부정 · ⚪ 중립 — 클릭하면 유사 영화 검색")
                    top_kw = keywords[:4]
                    tag_cols = st.columns(min(len(top_kw), 4))
                    for ki, kw in enumerate(top_kw):
                        if kw['sentiment'] == "긍정":
                            label = f"🟢 #{kw['word']}"
                        elif kw['sentiment'] == "부정":
                            label = f"🔴 #{kw['word']}"
                        else:
                            label = f"⚪ #{kw['word']}"
                        with tag_cols[ki]:
                            if st.button(label, key=f"kb_{i}_{ki}", use_container_width=True):
                                st.session_state.mode = 'keyword'
                                st.session_state.keyword_data = {
                                    'keyword': kw['word'],
                                    'sentiment': kw['sentiment'],
                                    'movie': r['title']
                                }
                                st.rerun()

                    if len(keywords) > 4:
                        with st.expander(f"키워드 더보기 ({len(keywords)-4}개)"):
                            extra_cols = st.columns(4)
                            for ki, kw in enumerate(keywords[4:], start=4):
                                col_idx = (ki - 4) % 4
                                if kw['sentiment'] == "긍정":
                                    label = f"🟢 #{kw['word']}"
                                elif kw['sentiment'] == "부정":
                                    label = f"🔴 #{kw['word']}"
                                else:
                                    label = f"⚪ #{kw['word']}"
                                with extra_cols[col_idx]:
                                    if st.button(label, key=f"kb_{i}_{ki}", use_container_width=True):
                                        st.session_state.mode = 'keyword'
                                        st.session_state.keyword_data = {
                                            'keyword': kw['word'],
                                            'sentiment': kw['sentiment'],
                                            'movie': r['title']
                                        }
                                        st.rerun()

                st.markdown("---")
