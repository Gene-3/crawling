# -*- coding: utf-8 -*-
"""
키워드 인덱스 사전 계산

학습된 모델로 상위 영화들의 키워드별 감성을 분석하고
장르 분류도 함께 수행한다.

출력:
  keyword_index.pkl  — 키워드별 영화 감성 통계
  genre_map.pkl      — 장르별 영화 목록 + 인기 키워드
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import pickle
import sqlite3
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, "review_texts.db")

TOP_MOVIES = 3000
REVIEWS_PER_MOVIE = 500

# 'scary'는 장르 분류에서 제외한다: "love is scary", "scary good" 등 관용적 용법이
# 로맨스/드라마를 Horror로 오분류시키기 때문. 측정 결과 scary를 빼도 진짜 공포영화
# 886/915(96.8%)는 horror/terrifying/gore 등 다른 신호로 유지되고, 탈락한 29개는
# 전부 비-공포(La La Land, To All the Boys, Dr. Strangelove 등) 오분류였다.
# (scary는 서비스 클릭 키워드로는 그대로 유지 — 여기서 빼는 건 장르 판별 용도뿐)
GENRE_KEYWORDS = {
    'Horror': ['horror', 'terrifying', 'creepy', 'haunting',
               'nightmare', 'ghost', 'zombie', 'slasher', 'gore'],
    'Comedy': ['comedy', 'funny', 'hilarious', 'humor', 'laugh',
               'comedic', 'humorous', 'slapstick'],
    'Romance': ['romance', 'romantic', 'love story', 'love interest',
                'chemistry'],
    'Drama': ['drama', 'dramatic'],
    'Action': ['action', 'fight scene', 'battle', 'explosion', 'combat',
               'stunt', 'warfare', 'shootout'],
    'Thriller': ['thriller', 'suspense', 'suspenseful', 'mystery',
                 'crime', 'detective', 'noir'],
    'Sci-Fi': ['sci-fi', 'science fiction', 'futuristic', 'alien',
               'spaceship', 'dystopian', 'robot'],
    'Animation': ['animated', 'animation', 'pixar', 'cartoon',
                  'disney', 'anime'],
}

_genre_patterns = {}
for _g, _kws in GENRE_KEYWORDS.items():
    _parts = []
    for kw in _kws:
        if ' ' in kw or '-' in kw:
            _parts.append(re.escape(kw))
        else:
            _parts.append(r'\b' + re.escape(kw) + r'\b')
    _genre_patterns[_g] = re.compile('|'.join(_parts), re.IGNORECASE)

_cjk_re = re.compile(r'[가-힯ぁ-鿿]')
_sent_split = re.compile(r'[.!?;\n]+')


# 배포 모델: v2 (긍정 8~9 / 부정 2~3). IMDB 사람 라벨 검증에서 v1보다 근소 우위 +
# precision/recall 균형이 좋아 서비스 인덱스를 v2로 빌드한다. v1로 되돌리려면 "_v2"를 ""로.
MODEL_SUFFIX = "_v3"


def load_model_and_config():
    model = load_model(os.path.join(MODEL_DIR, f"keyword_sentiment{MODEL_SUFFIX}.keras"))
    with open(os.path.join(MODEL_DIR, f"keyword_tokenizer{MODEL_SUFFIX}.pkl"), 'rb') as f:
        tok = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f"keyword_config{MODEL_SUFFIX}.pkl"), 'rb') as f:
        cfg = pickle.load(f)
    return model, tok, cfg


def extract_sentences(text, kw_pattern):
    if _cjk_re.search(text):
        return []
    results = []
    for sent in _sent_split.split(text):
        sent = sent.strip()
        words = sent.split()
        if len(words) < 5 or len(words) > 60:
            continue
        matches = kw_pattern.findall(sent)
        if matches:
            results.append((sent, [m.lower() for m in matches]))
    return results


def predict_batch(model, tok, cfg, sentences):
    if not sentences:
        return np.array([])
    seq = tok.texts_to_sequences(sentences)
    padded = pad_sequences(seq, maxlen=cfg['max_len'],
                           padding='post', truncating='post')
    return model.predict(padded, batch_size=512, verbose=0).flatten()


def main():
    print("모델 로드...")
    model, tok, cfg = load_model_and_config()
    keywords = cfg['keywords']
    kw_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b',
        re.IGNORECASE)

    print("movie_meta 로드...")
    with open(os.path.join(MODEL_DIR, "movie_meta.pkl"), 'rb') as f:
        movie_meta = pickle.load(f)

    # 리뷰 수 기준 상위 영화 선택
    sorted_movies = sorted(movie_meta.items(),
                           key=lambda x: x[1].get('review_count', 0),
                           reverse=True)[:TOP_MOVIES]
    top_keys = set()
    for idx, meta in sorted_movies:
        top_keys.add(f"{meta['title']}|||{meta['year']}")
    print(f"  상위 {len(top_keys)}개 영화 선택")

    print("review_meta 로드...")
    with open(os.path.join(DATA_DIR, "review_meta.pkl"), 'rb') as f:
        review_meta = pickle.load(f)

    # 상위 영화의 리뷰 인덱스 수집
    print("리뷰 인덱스 매핑...")
    movie_indices = defaultdict(list)
    for i, (key, _) in enumerate(review_meta):
        if key in top_keys and len(movie_indices[key]) < REVIEWS_PER_MOVIE:
            movie_indices[key].append(i)

    del review_meta

    # 결과 저장 구조
    keyword_stats = defaultdict(lambda: defaultdict(lambda: {
        'positive': 0, 'negative': 0, 'total': 0,
        'pos_example': '', 'neg_example': '',
    }))
    genre_counts = defaultdict(lambda: defaultdict(int))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    processed = 0
    for movie_key in top_keys:
        indices = movie_indices.get(movie_key, [])
        if not indices:
            continue

        placeholders = ','.join('?' * len(indices))
        cur.execute(
            f"SELECT id, text FROM reviews WHERE id IN ({placeholders})",
            indices)
        texts = {row[0]: row[1] for row in cur.fetchall() if row[1]}

        all_sentences = []

        for idx in indices:
            text = texts.get(idx, '')
            if not text:
                continue

            text_lower = text.lower()
            for genre, pattern in _genre_patterns.items():
                hits = len(pattern.findall(text_lower))
                if hits > 0:
                    genre_counts[movie_key][genre] += hits

            for sent, matched in extract_sentences(text, kw_pattern):
                for kw in matched:
                    all_sentences.append((kw, sent))

        if all_sentences:
            sent_texts = [s[1] for s in all_sentences]
            probs = predict_batch(model, tok, cfg, sent_texts)

            for (kw, sent_text), prob in zip(all_sentences, probs):
                stats = keyword_stats[kw][movie_key]
                stats['total'] += 1
                if prob >= 0.5:
                    stats['positive'] += 1
                    if not stats['pos_example'] or len(sent_text) > len(stats['pos_example']):
                        stats['pos_example'] = sent_text[:200]
                else:
                    stats['negative'] += 1
                    if not stats['neg_example'] or len(sent_text) > len(stats['neg_example']):
                        stats['neg_example'] = sent_text[:200]

        processed += 1
        if processed % 100 == 0:
            print(f"\r  처리: {processed}/{len(top_keys)}", end='', flush=True)

    conn.close()
    print(f"\r  완료: {processed}개 영화" + " " * 20)

    # 비율 계산
    for kw in keyword_stats:
        for mk in keyword_stats[kw]:
            s = keyword_stats[kw][mk]
            s['ratio'] = s['positive'] / s['total'] if s['total'] > 0 else 0.0

    # 장르 분류
    print("\n장르 분류...")
    genre_map = defaultdict(lambda: {'movies': [], 'top_keywords': []})
    for movie_key in top_keys:
        counts = genre_counts.get(movie_key, {})
        if not counts:
            continue
        review_count = len(movie_indices.get(movie_key, []))
        if review_count == 0:
            continue
        sorted_genres = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for genre, cnt in sorted_genres[:2]:
            if cnt >= review_count * 0.03:
                genre_map[genre]['movies'].append(movie_key)

    # 장르별 인기 키워드
    for genre in genre_map:
        genre_movies = set(genre_map[genre]['movies'])
        kw_totals = {}
        for kw in keywords:
            total = sum(keyword_stats[kw].get(mk, {}).get('total', 0)
                        for mk in genre_movies)
            if total >= 10:
                kw_totals[kw] = total
        genre_map[genre]['top_keywords'] = [
            kw for kw, _ in sorted(kw_totals.items(),
                                   key=lambda x: x[1], reverse=True)
        ]

    # 저장 (defaultdict → dict 변환)
    kw_out = {kw: dict(movies) for kw, movies in keyword_stats.items()}
    with open(os.path.join(MODEL_DIR, "keyword_index.pkl"), 'wb') as f:
        pickle.dump(kw_out, f)
    gm_out = {g: dict(v) for g, v in genre_map.items()}
    with open(os.path.join(MODEL_DIR, "genre_map.pkl"), 'wb') as f:
        pickle.dump(gm_out, f)

    print("\n장르별 영화 수:")
    for genre in sorted(genre_map.keys()):
        movies = genre_map[genre]['movies']
        kws = genre_map[genre]['top_keywords'][:5]
        print(f"  {genre}: {len(movies)}개 | 인기: {', '.join(kws)}")

    print(f"\n키워드별 커버 영화 수:")
    for kw in sorted(keyword_stats.keys()):
        count = len([mk for mk, s in keyword_stats[kw].items() if s['total'] >= 3])
        print(f"  {kw}: {count}개")

    print(f"\n저장 완료: keyword_index.pkl, genre_map.pkl")


if __name__ == '__main__':
    main()
