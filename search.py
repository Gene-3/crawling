"""
영화 추천 검색 엔진
- Stage 1: 영화 평균 벡터로 후보 추림
- Stage 2: 리뷰 벡터로 정밀 검색 + 평점 반영
- Stage 3: GRU 감성분류로 부정 리뷰 탐지 → 부정 비율 높은 영화 제외
- Score = log(1 + 유사리뷰 수) * 평균 평점
"""
import os, re, sqlite3
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import faiss
import numpy as np
import pickle
import math
from FlagEmbedding import BGEM3FlagModel
from collections import defaultdict

OUT_DIR = r"D:\Lecture\_AIService26\movie_recommender"
DB_PATH = os.path.join(OUT_DIR, "review_texts.db")


class MovieRecommender:
    def __init__(self, nprobe: int = 64):
        print("모델 로딩 중...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')

        print("인덱스 로딩 중...")
        self.index_movie  = faiss.read_index(os.path.join(OUT_DIR, "faiss_movie.index"))
        self.index_review = faiss.read_index(os.path.join(OUT_DIR, "faiss_review.index"))
        self.index_review.nprobe = nprobe

        with open(os.path.join(OUT_DIR, "movie_meta.pkl"),  "rb") as f:
            self.movie_meta = pickle.load(f)

        with open(os.path.join(OUT_DIR, "review_meta.pkl"), "rb") as f:
            review_meta = pickle.load(f)

        self.review_movie_key = [r[0] for r in review_meta]
        self.review_rating    = [r[1] for r in review_meta]

        self.key_to_idx = {
            f"{v['title']}|||{v['year']}": k
            for k, v in self.movie_meta.items()
        }

        # SQLite 경로 저장 (조회 시마다 연결)
        self.db_path = DB_PATH

        # 감성분류 모델 로드
        self._load_sentiment_model()
        print("준비 완료!\n")

    def _get_review_texts(self, indices):
        """SQLite에서 리뷰 텍스트 조회"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        texts = {}
        for idx in indices:
            cur.execute("SELECT text FROM reviews WHERE id = ?", (idx,))
            row = cur.fetchone()
            if row:
                texts[idx] = row[0]
        conn.close()
        return texts

    def _load_sentiment_model(self):
        """학습된 Keras GRU 감성분류 모델 로드"""
        model_path  = os.path.join(OUT_DIR, "sentiment_64.keras")
        tok_path    = os.path.join(OUT_DIR, "sentiment_tokenizer_64.pkl")
        config_path = os.path.join(OUT_DIR, "sentiment_config_64.pkl")

        if not os.path.exists(model_path):
            print("  감성분류 모델 없음 — 감성 필터 비활성화")
            self.sentiment_model = None
            return

        print("감성분류 모델 로딩 중...")
        from tensorflow.keras.models import load_model
        self.sentiment_model = load_model(model_path)
        with open(tok_path, 'rb') as f:
            self.tokenizer_sent = pickle.load(f)
        with open(config_path, 'rb') as f:
            self.sent_config = pickle.load(f)
        print("  감성분류 모델 로드 완료!")

    def _predict_sentiment(self, texts):
        """텍스트 리스트 → 감성 라벨 (1=긍정, 0=중립, -1=부정)"""
        if not self.sentiment_model or not texts:
            return np.zeros(len(texts), dtype=int)

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        encoded = self.tokenizer_sent.texts_to_sequences(texts)
        padded = pad_sequences(encoded, maxlen=self.sent_config['max_len'],
                               padding='post', truncating='post')
        preds = self.sentiment_model.predict(padded, batch_size=512, verbose=0)
        neg_probs = preds[:, 0]  # 부정 확률

        # 부정 70%+ → -1, 긍정 40%- → 1, 나머지 → 0(중립)
        labels = np.zeros(len(texts), dtype=int)
        labels[neg_probs >= 0.7] = -1   # 부정
        labels[neg_probs <= 0.4] = 1    # 긍정
        return labels

    def _embed(self, text: str) -> np.ndarray:
        text = text[:2000]
        vec = self.model.encode(
            [text], batch_size=1, max_length=512,
            return_dense=True, return_sparse=False, return_colbert_vecs=False
        )['dense_vecs'][:, :256].astype('float32')
        faiss.normalize_L2(vec)
        return vec

    def recommend(self, query: str, top_k: int = 10,
                  stage1_k: int = 100, stage2_k: int = 2000,
                  max_negative_ratio: float = 0.6) -> list:
        """
        query              : 검색 쿼리 (한/영 모두 가능)
        top_k              : 최종 추천 영화 수
        stage1_k           : 1단계 후보 영화 수
        stage2_k           : 2단계 유사 리뷰 검색 수
        max_negative_ratio : 최대 부정 비율 (이상이면 제외)
        """
        qvec = self._embed(query)

        # ── Stage 1: 영화 평균 벡터로 후보 추리기 ───────────────
        _, movie_idxs = self.index_movie.search(qvec, stage1_k)
        candidate_keys = {
            f"{self.movie_meta[i]['title']}|||{self.movie_meta[i]['year']}"
            for i in movie_idxs[0] if i != -1
        }

        # ── Stage 2: 리뷰 벡터로 유사 리뷰 검색 ────────────────
        _, review_idxs = self.index_review.search(qvec, stage2_k)

        # 후보 영화 내 유사 리뷰 집계
        movie_reviews = defaultdict(list)
        for idx in review_idxs[0]:
            if idx == -1:
                continue
            key    = self.review_movie_key[idx]
            rating = self.review_rating[idx]
            if key in candidate_keys:
                movie_reviews[key].append((int(idx), rating))

        # ── Stage 3: 감성분류로 부정 리뷰 탐지 ─────────────────
        all_indices = []
        for reviews in movie_reviews.values():
            for idx, _ in reviews:
                all_indices.append(idx)

        idx_to_neg = {}
        if all_indices and self.sentiment_model:
            review_texts = self._get_review_texts(all_indices)
            ordered_texts = []
            ordered_indices = []
            for idx in all_indices:
                text = review_texts.get(idx, '')
                # 한국어 리뷰는 중립 처리
                if re.search(r'[가-힯ぁ-鿿]', text):
                    idx_to_neg[idx] = 0
                # 60단어 이하 짧은 리뷰는 중립 처리 (모델 판단 어려움)
                elif len(text.split()) <= 60:
                    idx_to_neg[idx] = 0
                else:
                    ordered_texts.append(text)
                    ordered_indices.append(idx)

            if ordered_texts:
                sentiments = self._predict_sentiment(ordered_texts)
                for idx, sent in zip(ordered_indices, sentiments):
                    idx_to_neg[idx] = int(sent)  # -1=부정, 0=중립, 1=긍정

        # ── 스코어 계산 (부정 비율 높으면 제외, 중립은 모수에서 제외) ──
        results = []
        for key, reviews in movie_reviews.items():
            total_count = len(reviews)
            pos_count = 0
            neg_count = 0
            ratings_list = []

            for idx, rating in reviews:
                sent = idx_to_neg.get(idx, 0)  # -1=부정, 0=중립, 1=긍정
                if sent == -1:
                    neg_count += 1
                elif sent == 1:
                    pos_count += 1
                if rating > 0:
                    ratings_list.append(rating)

            # 중립 포함, 전체 리뷰 대비 부정 비율 계산
            negative_ratio = neg_count / total_count if total_count > 0 else 0

            avg_r = sum(ratings_list) / len(ratings_list) if ratings_list else 5.0

            # 부정 비율 80% 이상이면 제외 (확실히 나쁜 것만)
            if negative_ratio >= 0.8:
                continue

            # 리뷰 발췌 수집 (최대 3개, 긴 리뷰 우선)
            snippets = []
            sorted_reviews = sorted(reviews, key=lambda x: len(
                (review_texts.get(x[0], '') if all_indices and self.sentiment_model else '')), reverse=True)
            for idx, rating in sorted_reviews:
                text = review_texts.get(idx, '') if all_indices and self.sentiment_model else ''
                if len(text.strip()) < 20:
                    continue
                sent = idx_to_neg.get(idx, 0)
                if sent == 1:
                    label = "긍정"
                elif sent == -1:
                    label = "부정"
                else:
                    label = "중립"
                snippet = text[:200].strip()
                if len(text) > 200:
                    snippet += "..."
                snippets.append({"label": label, "text": snippet})
                if len(snippets) >= 3:
                    break

            score = math.log1p(total_count) * avg_r
            title, year = key.split("|||") if "|||" in key else (key, "")
            results.append({
                "title"           : title,
                "year"            : year,
                "score"           : round(score, 3),
                "avg_rating"      : round(avg_r, 2),
                "similar_reviews" : total_count,
                "negative_ratio"  : round(negative_ratio * 100, 1),
                "snippets"        : snippets,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def extract_keywords(self, movie_title: str, top_n: int = 8) -> list:
        """영화 리뷰에서 감성별 키워드 추출 (명사/형용사만, 영화 제목 제외)"""
        from collections import Counter
        from nltk import pos_tag

        stopwords = {
            'the','a','an','is','was','are','were','be','been','being',
            'have','has','had','do','does','did','will','would','could',
            'should','may','might','can','shall','it','its','i','me','my',
            'you','your','he','him','his','she','her','we','us','our',
            'they','them','their','this','that','these','those','and','but',
            'or','so','if','then','than','too','very','just','not','no',
            'of','in','on','at','to','for','with','from','by','about',
            'into','through','during','before','after','above','below',
            'between','out','off','over','under','again','further','once',
            'all','each','every','both','few','more','most','other','some',
            'such','only','own','same','also','how','what','which','who',
            'whom','when','where','why','up','down','here','there','really',
            'one','two','much','many','well','even','like','movie','film',
            'watch','watched','watching','get','got','make','made','go',
            'going','see','seen','thing','way','time','good','great','bad',
            'best','first','last','still','think','know','feel','never',
            'show','shows','season','episode','series','part','scene',
            'character','characters','story','plot','end','ending',
            'love','loved','pretty','little','something','everything',
            'nothing','anything','always','enough','back','come','take',
            'year','years','new','old','real','right','left','long',
            'man','men','woman','women','people','guy','girl','life',
            'world','work','lot','bit','kind','day','point','fact',
            # 스페인어/프랑스어/포르투갈어 불용어
            'que','una','con','por','del','los','las','para','como',
            'mas','pero','sin','sur','les','des','est','une','pas',
        }

        key = None
        title_words = set()
        for k, v in self.movie_meta.items():
            if v['title'].lower() == movie_title.lower():
                key = f"{v['title']}|||{v['year']}"
                title_words = set(re.findall(r'[a-z]+', v['title'].lower()))
                break

        if not key:
            return []

        # 영화 제목 단어를 불용어에 추가
        all_stops = stopwords | title_words

        review_indices = [i for i, k in enumerate(self.review_movie_key) if k == key][:1000]
        if not review_indices:
            return []

        texts = self._get_review_texts(review_indices)

        # GRU로 감성 분류
        classify_texts = []
        classify_indices = []
        for idx in review_indices:
            text = texts.get(idx, '')
            if re.search(r'[가-힯ぁ-鿿]', text) or len(text.split()) <= 60:
                continue
            classify_texts.append(text)
            classify_indices.append(idx)

        idx_to_sent = {}
        if classify_texts and self.sentiment_model:
            sentiments = self._predict_sentiment(classify_texts)
            for idx, sent in zip(classify_indices, sentiments):
                idx_to_sent[idx] = int(sent)

        # 명사/형용사 태그
        allowed_tags = {
            'NN', 'NNS', 'NNP', 'NNPS',       # 명사
            'JJ', 'JJR', 'JJS',                 # 형용사
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # 동사
        }

        # 긍정/부정 리뷰별 단어 빈도 분리
        pos_words = []
        neg_words = []
        for idx in review_indices:
            text = texts.get(idx, '')
            if re.search(r'[가-힯ぁ-鿿]', text):
                continue
            words = re.findall(r'[a-z]+', text.lower())
            filtered = [w for w in words if w not in all_stops and len(w) > 3]
            tagged = pos_tag(filtered)
            nouns_adjs = [w for w, tag in tagged if tag in allowed_tags]
            sent = idx_to_sent.get(idx, 0)
            if sent == 1:
                pos_words.extend(nouns_adjs)
            elif sent == -1:
                neg_words.extend(nouns_adjs)

        pos_counter = Counter(pos_words)
        neg_counter = Counter(neg_words)

        # 감성 분류된 리뷰가 없으면 전체 리뷰에서 키워드만 추출
        if not pos_words and not neg_words:
            all_words = []
            for idx in review_indices:
                text = texts.get(idx, '')
                if re.search(r'[가-힯ぁ-鿿]', text):
                    continue
                words = re.findall(r'[a-z]+', text.lower())
                filtered = [w for w in words if w not in all_stops and len(w) > 3]
                tagged = pos_tag(filtered)
                all_words.extend([w for w, tag in tagged if tag in allowed_tags])
            all_counter = Counter(all_words)
            return [{"word": w, "sentiment": "중립", "count": c}
                    for w, c in all_counter.most_common(top_n)]

        # 전체 빈도 상위 키워드에서 감성 결정
        all_counter = Counter(pos_words + neg_words)
        results = []
        for word, count in all_counter.most_common(top_n * 3):
            if len(word) <= 3:
                continue
            pc = pos_counter.get(word, 0)
            nc = neg_counter.get(word, 0)
            if pc + nc < 2:
                continue
            sentiment = "긍정" if pc >= nc else "부정"
            results.append({"word": word, "sentiment": sentiment, "count": count})
            if len(results) >= top_n:
                break

        return results

    def search_by_keyword_sentiment(self, keyword: str, sentiment: str,
                                     top_k: int = 10, stage1_k: int = 300,
                                     stage2_k: int = 5000) -> list:
        """키워드로 유사 영화 검색 + 감성 정보 표시 (필터링 아님)"""
        # 1. SQLite 텍스트 매칭
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        total_rows = len(self.review_movie_key)
        cur.execute(
            "SELECT id, text FROM reviews WHERE text LIKE ? ORDER BY RANDOM() LIMIT 5000",
            (f'%{keyword}%',))

        candidate_indices = []
        candidate_texts = {}
        for row in cur.fetchall():
            idx = row[0]
            if idx < total_rows:
                candidate_indices.append(idx)
                candidate_texts[idx] = row[1]
        conn.close()

        # 2. FAISS 벡터 검색 추가
        qvec = self._embed(keyword)
        _, review_idxs = self.index_review.search(qvec, stage2_k)

        faiss_indices = [int(idx) for idx in review_idxs[0] if idx != -1]
        faiss_texts = self._get_review_texts(faiss_indices)
        existing = set(candidate_indices)
        for idx in faiss_indices:
            if idx not in existing:
                candidate_indices.append(idx)
                candidate_texts[idx] = faiss_texts.get(idx, '')

        # 3. GRU 감성분류 (정보 제공용, 필터링 아님)
        idx_to_sent = {}
        if self.sentiment_model:
            classify_texts = []
            classify_indices = []
            for idx in candidate_indices:
                text = candidate_texts.get(idx, '')
                if re.search(r'[가-힯ぁ-鿿]', text) or len(text.split()) <= 60:
                    idx_to_sent[idx] = 0
                else:
                    classify_texts.append(text)
                    classify_indices.append(idx)

            if classify_texts:
                sentiments = self._predict_sentiment(classify_texts)
                for idx, sent in zip(classify_indices, sentiments):
                    idx_to_sent[idx] = int(sent)

        # 4. 영화별 집계 (전체 포함 + 감성 건수 표시)
        movie_data = defaultdict(lambda: {'total': 0, 'pos': 0, 'neg': 0, 'neu': 0, 'ratings': []})
        for idx in candidate_indices:
            key = self.review_movie_key[idx]
            rating = self.review_rating[idx]
            d = movie_data[key]
            d['total'] += 1
            sent = idx_to_sent.get(idx, 0)
            if sent == 1:
                d['pos'] += 1
            elif sent == -1:
                d['neg'] += 1
            else:
                d['neu'] += 1
            if rating > 0:
                d['ratings'].append(rating)

        # 5. 점수 계산
        results = []
        for key, d in movie_data.items():
            avg_r = sum(d['ratings']) / len(d['ratings']) if d['ratings'] else 5.0
            score = math.log1p(d['total']) * avg_r
            title, year = key.split("|||") if "|||" in key else (key, "")
            results.append({
                "title": title,
                "year": year,
                "score": round(score, 3),
                "avg_rating": round(avg_r, 2),
                "matched_reviews": d['total'],
                "total_reviews": d['total'],
                "pos_count": d['pos'],
                "neg_count": d['neg'],
                "neu_count": d['neu'],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ── 실행 테스트 ──────────────────────────────────────────────────
if __name__ == "__main__":
    rec = MovieRecommender(nprobe=64)

    test_queries = [
        "가족을 잃은 슬픔을 다루면서도 희망적인 결말을 가진 영화",
        "time travel with complex narrative and philosophical themes",
        "기억을 잃은 여자가 나오는 로맨스 영화",
    ]

    for query in test_queries:
        print(f"\n{'='*55}")
        print(f"쿼리: {query}")
        print(f"{'='*55}")
        results = rec.recommend(query, top_k=5, stage1_k=300, stage2_k=5000)
        if not results:
            print("  결과 없음")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['year']}] {r['title']}")
            print(f"     점수: {r['score']} | 평점: {r['avg_rating']} | "
                  f"유사리뷰: {r['similar_reviews']}건 | 부정: {r['negative_ratio']}%")
