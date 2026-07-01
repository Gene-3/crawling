# 영화 키워드 감성 추천 서비스

Letterboxd 리뷰 **4,021만 건**을 기반으로 LSTM 감성 분류 모델을 학습하고,  
장르와 키워드를 선택하면 해당 키워드가 **긍정적으로 언급된** 영화를 추천하는 Streamlit 서비스입니다.

---

## 프로젝트 배경 — 기존 서비스에서 새 서비스로

### 기존 서비스 (의미 검색)
텍스트를 직접 입력하면 BGE-M3 임베딩 → FAISS 벡터 검색 → GRU 감성 필터로 영화를 찾는 구조였습니다.

```
텍스트 입력 → BGE-M3 임베딩 → FAISS 검색 → GRU 필터 → 추천
```

**한계:** 교수님 피드백 — "GRU 모델이 굳이 필요한가?" 벡터 검색이 이미 의미 유사 결과를 반환하므로 감성 필터의 실질적 개선 효과가 불분명했습니다.

### 새 서비스 (키워드 감성 추천)
LSTM이 핵심 역할을 하는 구조로 전환했습니다. 키워드 클릭 → 사전 계산된 감성 인덱스 조회 → Wilson 랭킹.

```
장르 + 키워드 선택 → keyword_index.pkl 조회 → Wilson 하한 랭킹 → 추천
```

LSTM이 리뷰 문장의 **맥락**을 분류합니다.  
예) "tries to be funny but fails" → 부정 / "unexpectedly funny in the best way" → 긍정  
단순 키워드 존재 여부가 아니라 감성 맥락을 구분하는 것이 핵심입니다.

---

## 환경 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.9 |
| TensorFlow / Keras | 2.20.0 / 3.10.0 |
| Streamlit | 1.x |
| NumPy | 1.26.4 |
| curl_cffi | 최신 |
| undetected-chromedriver | 최신 |

Anaconda 환경 `aiservice26` 기준입니다.

```bash
conda create -n aiservice26 python=3.9
conda activate aiservice26
pip install tensorflow==2.20.0 streamlit numpy==1.26.4
pip install curl_cffi undetected-chromedriver
```

---

## 파일 구조

```
movie_recommender/
│
├── [서비스]
│   ├── app_keyword.py              # 키워드 추천 Streamlit 앱 (메인 서비스)
│   ├── app.py                      # 의미 검색 Streamlit 앱 (기존 서비스)
│   ├── keyword_service.py          # Wilson 랭킹 추천 로직
│   └── search.py                   # 의미 검색 로직
│
├── [모델 — v3 배포용]
│   ├── keyword_sentiment_v3.keras  # LSTM 감성 분류 모델 (500k×2, 84%)
│   ├── keyword_best_v3.keras       # EarlyStopping 최적 체크포인트
│   ├── keyword_tokenizer_v3.pkl    # 토크나이저
│   └── keyword_config_v3.pkl       # 학습 설정 (max_len, vocab_size 등)
│
├── [사전 계산 인덱스]
│   ├── keyword_index.pkl           # 키워드별 영화 감성 통계 (서비스 핵심)
│   ├── genre_map.pkl               # 장르별 영화 목록
│   └── movie_meta.pkl              # 영화 메타데이터
│
├── [데이터 샘플]
│   └── review_sample.db            # 상위 500편 × 100건 리뷰 샘플 (50,000건)
│                                   # 원본: 40,217,722건 (review_texts.db, 12GB — 미포함)
│
├── [학습 스크립트]
│   ├── train_keyword_sentiment.py  # v1: 극단 평점(9~10/1~2), 80k×2
│   ├── train_keyword_sentiment_v2.py # v2: 완만 평점(8~9/2~3), 80k×2
│   └── train_keyword_sentiment_v3.py # v3: 완만 평점(8~9/2~3), 500k×2 ← 배포
│
├── [인덱스 빌드]
│   └── build_keyword_index.py      # LSTM으로 상위 3,000편 사전 추론
│
├── [크롤러]
│   └── crawl_letterboxd.py         # Letterboxd 영화 목록 + 리뷰 수집
│
├── [실행 파일]
│   ├── run_keyword.bat             # 키워드 추천 서비스 실행 (더블클릭)
│   └── run_search.bat              # 의미 검색 서비스 실행 (더블클릭)
│
└── [문서]
    ├── README.md
    └── technical_report.md
```

---

## 빠른 실행 (서비스 구동)

### 방법 1 — 배치 파일 (더블클릭)

`run_keyword.bat` 파일을 더블클릭하면 conda 환경 자동 활성화 후 브라우저가 열립니다.

### 방법 2 — 터미널

```bash
conda activate aiservice26
cd D:\Lecture\_AIService26\movie_recommender
streamlit run app_keyword.py
```

브라우저에서 `http://localhost:8501` 접속

> **주의:** `keyword_index.pkl`, `genre_map.pkl`, `movie_meta.pkl`, `keyword_sentiment_v3.keras`, `keyword_tokenizer_v3.pkl` 파일이 동일 폴더에 있어야 합니다.

---

## 데이터 수집 (크롤링)

### Phase 1 — 영화 목록 수집

```bash
# undetected-chromedriver로 Letterboxd 봇 탐지 우회
python crawl_letterboxd.py
```

- 수집 범위: 1960~2026년 연도별 인기 상위 400편씩
- 봇 탐지 우회: `undetected-chromedriver` (Chrome 패치 드라이버)
- 수집 항목: 영화 제목, 연도, URL → `movie_meta.pkl`

### Phase 2 — 리뷰 수집

- 도구: `curl_cffi AsyncSession` (TLS 핑거프린트 위장, 비동기)
- 각 영화 리뷰 페이지 순회하여 리뷰 텍스트 + 평점 수집
- 저장: `review_texts.db` (SQLite, 텍스트) + `review_meta.pkl` (평점/영화키)
- 최종 수집량: **40,217,722건** / ~18,000편

> 이 레포에는 원본 DB 대신 상위 500편 × 100건 샘플(`review_sample.db`, 14MB)이 포함됩니다.

---

## 데이터 전처리

`train_keyword_sentiment_v3.py` 내부에서 자동으로 처리됩니다.

```python
# 평점 기준으로 긍정/부정 분류
POS_LO, POS_HI = 8, 9   # 8~9점 → 긍정
NEG_LO, NEG_HI = 2, 3   # 2~3점 → 부정
TARGET_PER_CLASS = 500_000

# 전처리 흐름
review_texts.db + review_meta.pkl
    → 평점 기준 필터링 (극단값 제외: 10점, 1점 제외)
    → 클래스 균형 맞춤 (긍정 500k / 부정 500k)
    → 토크나이저 학습 (VOCAB_SIZE=30,000)
    → 패딩 (MAX_LEN=80)
    → 학습/테스트 9:1 분리
```

**평점 기준 선택 이유:**
- v1 (9~10 / 1~2): 극단 평점 → 트롤 리뷰 포함 가능, 부정 recall 편향
- v2/v3 (8~9 / 2~3): 완만한 기준 → recall 균형, 일반적인 감성 학습에 적합

---

## 모델 학습

```bash
conda activate aiservice26
python train_keyword_sentiment_v3.py
```

### 모델 구조

```python
Sequential([
    Embedding(30001, 64),
    SpatialDropout1D(0.15),   # 채널 단위 dropout — 임베딩 과적합 억제
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),  # 긍정 확률 출력
])
# optimizer: Adam(lr=0.001)
# loss: binary_crossentropy
# EarlyStopping(patience=3, restore_best_weights=True)
```

**GRU → LSTM 전환 이유:** TensorFlow 2.20 + Keras 3 환경에서 GRU loss가 0.693에 고착되는 버그 발생 → LSTM으로 교체 후 정상 학습.

### 버전별 성능 비교

| 버전 | 학습 데이터 | 평점 기준 | Accuracy | Recall(긍정) | Recall(부정) |
|------|------------|----------|----------|-------------|-------------|
| v1 | 80k × 2 | 9~10 / 1~2 | 84% | 0.746 | 0.856 |
| v2 | 80k × 2 | 8~9 / 2~3 | 84% | 0.853 | 0.830 |
| **v3** | **500k × 2** | **8~9 / 2~3** | **84%** | **0.84** | **0.84** |

v3에서 학습량 증가로 recall 완전 균형 달성. 절대 정확도 84%는 LSTM 아키텍처 상한선.

---

## 키워드 인덱스 사전 계산

```bash
# 약 30~60분 소요 (CPU 기준, 상위 3,000편)
python build_keyword_index.py
```

학습된 모델로 상위 3,000편의 리뷰 문장을 오프라인 추론하여 저장합니다.

```
review_texts.db
    → 상위 3,000편 선택 (리뷰 수 기준)
    → 키워드 포함 문장 추출 (5~60 단어, 영어만)
    → LSTM 배치 추론 (batch_size=512)
    → 영화별 키워드 긍정/부정 카운트 저장
    → Wilson 하한 점수 계산 준비
    → keyword_index.pkl 저장
    → 장르 분류 → genre_map.pkl 저장
```

런타임에는 모델 추론 없이 `keyword_index.pkl` 딕셔너리 조회만 수행합니다.

---

## Wilson 하한 랭킹

단순 긍정 비율 대신 Wilson 신뢰구간 하한을 사용합니다.

```python
def wilson_lower_bound(positive, total, z=1.96):
    # 3/3=100%보다 30/34=88%를 더 신뢰 가능한 결과로 처리
```

소표본(리뷰 3건 중 3건 긍정 = 100%)이 대표본(34건 중 30건 = 88%) 위에 노출되는 문제를 방지합니다.

---

## 서비스 사용법

1. **왼쪽 사이드바**에서 장르 선택 (Action / Comedy / Horror 등 8개)
2. **키워드 버튼** 클릭 (분위기 / 감정 / 스토리 / 연기·기술 카테고리, 26개)
3. 여러 개 선택하면 **AND 조건** 교집합 추천
4. 결과에서 키워드별 긍정 비율(%) 및 리뷰 예시 확인

---

## 한계

- 추천 풀이 리뷰 수 상위 **3,000편**으로 제한됨 (인기작 편향)
- 문장 단위 감성 분류 시 짧거나 모호한 문장은 오분류 가능
- 장르 분류는 키워드 빈도 기반이라 다의어로 인한 오분류 일부 존재
