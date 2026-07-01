# -*- coding: utf-8 -*-
"""키워드 감성 기반 영화 추천 서비스"""
import os
import math
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')


def wilson_lower_bound(positive, total, z=1.96):
    """긍정 비율의 Wilson 신뢰구간 하한.

    단순 비율(positive/total)은 표본이 작을수록 과대평가된다.
    (예: 3/3=100%가 30/34=88%보다 위로 랭크됨)
    Wilson 하한은 표본 크기를 반영해 작은 표본에 페널티를 준다.
    """
    if total == 0:
        return 0.0
    phat = positive / total
    denom = 1 + z * z / total
    center = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (center - margin) / denom

# 클릭 가능한 키워드는 "긍정적으로 언급"이 성립하는 것만 노출한다.
# boring/predictable/overrated/disappointing은 불평 단어라 리뷰의 78~97%가 부정이고
# "칭찬받는 지루함" 같은 긍정 언급이 성립하지 않아 제외했다. (인덱스에는 남아 있으나
# UI에 노출하지 않음)
KEYWORD_CATEGORIES = {
    '분위기': ['scary', 'funny', 'romantic', 'sad', 'dark', 'intense', 'lighthearted'],
    '감정': ['thrilling', 'touching', 'heartwarming', 'disturbing', 'inspiring', 'emotional'],
    '스토리': ['original', 'twist', 'entertaining', 'masterpiece', 'underrated'],
    '연기/기술': ['acting', 'cinematography', 'soundtrack', 'dialogue', 'visuals',
                'suspenseful', 'hilarious', 'creepy'],
}

KW_KR = {
    'scary': '무서운', 'funny': '웃긴', 'romantic': '로맨틱', 'sad': '슬픈',
    'dark': '어두운', 'intense': '긴장감', 'lighthearted': '가벼운',
    'thrilling': '스릴', 'touching': '감동', 'heartwarming': '따뜻한',
    'disturbing': '불편한', 'inspiring': '영감', 'emotional': '감성적',
    'predictable': '예측가능', 'original': '독창적', 'twist': '반전',
    'boring': '지루한', 'entertaining': '재미있는', 'masterpiece': '명작',
    'overrated': '과대평가', 'underrated': '과소평가', 'disappointing': '실망',
    'acting': '연기', 'cinematography': '촬영', 'soundtrack': '음악',
    'dialogue': '대사', 'visuals': '영상미', 'suspenseful': '서스펜스',
    'hilarious': '폭소', 'creepy': '으스스한',
}


class KeywordService:
    def __init__(self):
        idx_path = os.path.join(MODEL_DIR, "keyword_index.pkl")
        genre_path = os.path.join(MODEL_DIR, "genre_map.pkl")
        meta_path = os.path.join(MODEL_DIR, "movie_meta.pkl")

        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                "keyword_index.pkl이 없습니다. training/build_keyword_index.py를 먼저 실행하세요.")

        with open(idx_path, 'rb') as f:
            self.keyword_index = pickle.load(f)
        with open(genre_path, 'rb') as f:
            self.genre_map = pickle.load(f)
        with open(meta_path, 'rb') as f:
            self.movie_meta = pickle.load(f)

        self.key_to_meta = {}
        for idx, meta in self.movie_meta.items():
            key = f"{meta['title']}|||{meta['year']}"
            self.key_to_meta[key] = meta

    def get_genres(self):
        return sorted(self.genre_map.keys())

    def get_genre_keywords(self, genre):
        info = self.genre_map.get(genre, {})
        return info.get('top_keywords', [])

    def get_movies(self, keyword, genre=None, top_n=20, min_mentions=5):
        kw_data = self.keyword_index.get(keyword, {})
        if not kw_data:
            return []

        if genre:
            genre_movies = set(self.genre_map.get(genre, {}).get('movies', []))
            candidates = {k: v for k, v in kw_data.items() if k in genre_movies}
        else:
            candidates = kw_data

        # Wilson 하한으로 정렬: 단순 비율이 아니라 표본 크기를 반영한 점수.
        # 작은 표본(3/3=100%)이 큰 표본(30/34=88%)을 이기는 문제를 방지.
        ranked = []
        for k, v in candidates.items():
            if v['total'] < min_mentions:
                continue
            score = wilson_lower_bound(v['positive'], v['total'])
            ranked.append((k, v, score))
        ranked.sort(key=lambda x: x[2], reverse=True)

        results = []
        for movie_key, stats, score in ranked[:top_n]:
            parts = movie_key.split('|||')
            title = parts[0]
            year = parts[1] if len(parts) > 1 else ''
            meta = self.key_to_meta.get(movie_key, {})
            results.append({
                'title': title,
                'year': year,
                'avg_rating': meta.get('avg_rating', 0),
                'positive': stats['positive'],
                'negative': stats['negative'],
                'total': stats['total'],
                'ratio': stats['ratio'],
                'score': round(score, 3),
                'pos_example': stats.get('pos_example', ''),
                'neg_example': stats.get('neg_example', ''),
            })
        return results

    def get_movies_multi(self, keywords, genre=None, top_n=20, min_mentions=5):
        """여러 키워드를 동시에 만족(AND)하는 영화 교집합.

        선택한 모든 키워드에서 (언급 min_mentions 이상, 긍정 비율 0.5 이상)인
        영화만 남긴다. 랭킹은 각 키워드 Wilson 점수의 최솟값 — 가장 약한
        항목을 기준으로 삼아 '모든 면에서 좋은' 영화가 위로 오게 한다.
        만족하는 영화가 없으면 빈 리스트.
        """
        keywords = [k for k in keywords if k in self.keyword_index]
        if not keywords:
            return []
        if len(keywords) == 1:
            return self.get_movies(keywords[0], genre, top_n, min_mentions)

        genre_movies = None
        if genre:
            genre_movies = set(self.genre_map.get(genre, {}).get('movies', []))

        # 키워드별로 조건을 만족하는 영화 집합
        per_kw = []
        for kw in keywords:
            ok = {}
            for mk, v in self.keyword_index[kw].items():
                if v['total'] < min_mentions or v['ratio'] < 0.5:
                    continue
                if genre_movies is not None and mk not in genre_movies:
                    continue
                ok[mk] = v
            per_kw.append(ok)

        # 교집합
        common = set(per_kw[0])
        for ok in per_kw[1:]:
            common &= set(ok)

        results = []
        for mk in common:
            parts = mk.split('|||')
            scores = []
            breakdown = []
            for i, kw in enumerate(keywords):
                v = per_kw[i][mk]
                scores.append(wilson_lower_bound(v['positive'], v['total']))
                breakdown.append({
                    'keyword': kw,
                    'positive': v['positive'],
                    'total': v['total'],
                    'ratio': v['ratio'],
                    'pos_example': v.get('pos_example', ''),
                })
            meta = self.key_to_meta.get(mk, {})
            results.append({
                'title': parts[0],
                'year': parts[1] if len(parts) > 1 else '',
                'avg_rating': meta.get('avg_rating', 0),
                'score': round(min(scores), 3),
                'keywords': breakdown,
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]
