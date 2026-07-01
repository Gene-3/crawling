import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
from keyword_service import KeywordService, KEYWORD_CATEGORIES, KW_KR

st.set_page_config(page_title="영화 키워드 추천", page_icon="🎬", layout="wide")

st.markdown("""
<style>
/* 헤더 */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero h1 { color: #e2e8f0; font-size: 2.2rem; margin: 0 0 0.4rem; }
.hero p  { color: #94a3b8; font-size: 1rem; margin: 0; }

/* 카테고리 라벨 */
.cat-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.2rem 0 0.4rem;
}

/* 선택된 키워드 칩 */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
.chip {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: #0f3460;
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 600;
}

/* 영화 카드 */
.movie-card {
    background: var(--secondary-background-color, #1e293b);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: rgba(255,255,255,0.18); }
.movie-title { font-size: 1.05rem; font-weight: 700; margin: 0 0 0.15rem; color: #f1f5f9; }
.movie-meta  { font-size: 0.82rem; color: #94a3b8; margin: 0; }

/* 감성 바 */
.bar-wrap { background: rgba(255,255,255,0.08); border-radius: 999px; height: 8px; overflow: hidden; margin: 0.5rem 0; }
.bar-fill  { height: 100%; border-radius: 999px; }

/* 예시 문장 */
.example {
    border-left: 3px solid;
    padding: 0.35rem 0.7rem;
    margin: 0.25rem 0;
    font-size: 0.82rem;
    color: #94a3b8;
    border-radius: 0 6px 6px 0;
    overflow-wrap: break-word;
    word-break: break-word;
    line-height: 1.4;
}

/* 뱃지 */
.badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.78rem;
    margin: 0.15rem 0.2rem 0.15rem 0;
}

/* 사이드바 */
[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_service():
    return KeywordService()


try:
    svc = load_service()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

genres = svc.get_genres()
if not genres:
    st.warning("장르 데이터가 없습니다.")
    st.stop()

# ── 사이드바 ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 장르 선택")
    GENRE_EMOJI = {
        'Action': '💥', 'Animation': '🎨', 'Comedy': '😂', 'Drama': '🎭',
        'Horror': '👻', 'Romance': '💕', 'Sci-Fi': '🚀', 'Thriller': '🔪',
    }
    selected_genre = st.radio(
        "",
        genres,
        format_func=lambda g: f"{GENRE_EMOJI.get(g, '🎬')} {g}",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;color:#64748b'>"
        "키워드를 여러 개 선택하면<br>모든 조건을 만족하는 영화를 추천합니다."
        "</div>",
        unsafe_allow_html=True,
    )

# 장르 바뀌면 키워드 초기화
if st.session_state.get('_genre') != selected_genre:
    st.session_state['selected_keywords'] = []
    st.session_state['_genre'] = selected_genre

sel = st.session_state.setdefault('selected_keywords', [])
genre_keywords = svc.get_genre_keywords(selected_genre)

# ── 히어로 헤더 ──────────────────────────────────────
emoji = GENRE_EMOJI.get(selected_genre, '🎬')
st.markdown(
    f"<div class='hero'>"
    f"<h1>{emoji} {selected_genre} 영화 키워드 추천</h1>"
    f"<p>원하는 분위기 키워드를 클릭하세요 — 긍정적으로 언급된 영화를 찾아드립니다</p>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── 키워드 버튼 ──────────────────────────────────────
if not genre_keywords:
    st.info("이 장르에 분석된 키워드가 없습니다.")
    st.stop()

for cat_name, cat_keywords in KEYWORD_CATEGORIES.items():
    available = [kw for kw in cat_keywords if kw in genre_keywords]
    if not available:
        continue
    st.markdown(f"<div class='cat-label'>{cat_name}</div>", unsafe_allow_html=True)
    cols = st.columns(min(len(available), 7))
    for i, kw in enumerate(available):
        kr = KW_KR.get(kw, kw)
        is_sel = kw in sel
        with cols[i % len(cols)]:
            label = f"✓ {kr}" if is_sel else kr
            if st.button(
                label,
                key=f"btn_{kw}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                if is_sel:
                    sel.remove(kw)
                else:
                    sel.append(kw)
                st.rerun()

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

# ── 선택 상태 표시 ────────────────────────────────────
if sel:
    chips = "".join(
        f"<span class='chip'>{KW_KR.get(k, k)}</span>" for k in sel
    )
    col_chips, col_reset = st.columns([6, 1])
    with col_chips:
        st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)
    with col_reset:
        if st.button("초기화", use_container_width=True):
            st.session_state['selected_keywords'] = []
            st.rerun()
else:
    st.info("위에서 키워드를 하나 이상 선택하세요.")
    st.stop()

# ── 결과 조회 ─────────────────────────────────────────
movies = svc.get_movies_multi(sel, genre=selected_genre, top_n=15)
kr_list = " + ".join(KW_KR.get(k, k) for k in sel)

st.markdown("---")
st.markdown(f"### 🏆 **{kr_list}** 긍정 추천 영화")

if not movies:
    st.warning("선택한 키워드를 모두 만족하는 영화가 없습니다. 키워드를 줄여보세요.")
    st.stop()

single = len(sel) == 1

for i, m in enumerate(movies, 1):
    rating_str = f"⭐ {m['avg_rating']:.1f}" if m['avg_rating'] > 0 else ""

    if single:
        pct = m['ratio'] * 100
        if pct >= 70:
            bar_color, text_color = "#22c55e", "#22c55e"
        elif pct >= 50:
            bar_color, text_color = "#f59e0b", "#f59e0b"
        else:
            bar_color, text_color = "#ef4444", "#ef4444"

        pos_ex = m.get('pos_example', '')
        neg_ex = m.get('neg_example', '')

        st.markdown(
            f"<div class='movie-card'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
            f"  <div>"
            f"    <div class='movie-title'>#{i} {m['title']}</div>"
            f"    <div class='movie-meta'>{m['year']} &nbsp;·&nbsp; {rating_str}</div>"
            f"  </div>"
            f"  <div style='text-align:right;min-width:90px'>"
            f"    <span style='font-size:1.4rem;font-weight:800;color:{text_color}'>{pct:.0f}%</span>"
            f"    <div style='font-size:0.75rem;color:#64748b'>긍정 {m['positive']}/{m['total']}</div>"
            f"  </div>"
            f"</div>"
            f"<div class='bar-wrap'>"
            f"  <div class='bar-fill' style='width:{min(pct,100):.0f}%;background:{bar_color}'></div>"
            f"</div>"
            + (f"<div class='example' style='border-color:#22c55e'>✅ {pos_ex}</div>" if pos_ex else "")
            + (f"<div class='example' style='border-color:#ef4444'>❌ {neg_ex}</div>" if neg_ex else "")
            + f"</div>",
            unsafe_allow_html=True,
        )
    else:
        badges = ""
        for b in m['keywords']:
            pct = b['ratio'] * 100
            if pct >= 70:
                bg, fg = "#14532d", "#86efac"
            elif pct >= 50:
                bg, fg = "#78350f", "#fde68a"
            else:
                bg, fg = "#7f1d1d", "#fca5a5"
            kr = KW_KR.get(b['keyword'], b['keyword'])
            badges += (
                f"<span class='badge' style='background:{bg};color:{fg}'>"
                f"{kr} <b>{pct:.0f}%</b> ({b['positive']}/{b['total']})</span>"
            )

        # 대표 리뷰 예시: 가장 긍정 비율 높은 키워드에서 가져옴
        best = max(m['keywords'], key=lambda b: b['ratio'])
        pos_ex = best.get('pos_example', '')

        st.markdown(
            f"<div class='movie-card'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
            f"  <div>"
            f"    <div class='movie-title'>#{i} {m['title']}</div>"
            f"    <div class='movie-meta'>{m['year']} &nbsp;·&nbsp; {rating_str}</div>"
            f"  </div>"
            f"  <div style='font-size:0.8rem;color:#64748b;min-width:80px;text-align:right'>"
            f"    Wilson {m['score']:.3f}"
            f"  </div>"
            f"</div>"
            f"<div style='margin-top:0.6rem'>{badges}</div>"
            + (f"<div class='example' style='border-color:#22c55e;margin-top:0.5rem'>✅ {pos_ex}</div>" if pos_ex else "")
            + f"</div>",
            unsafe_allow_html=True,
        )
