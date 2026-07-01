# 영화 키워드 감성 추천 서비스

**22503110159 김광진 · AI 서비스 프로그래밍**

Letterboxd 리뷰 4,021만 건을 수집해 만든 영화 추천 서비스. BGE-M3 임베딩 기반 의미 검색(FAISS + GRU)으로 시작했다가, 지금은 LSTM 키워드 감성 분류 기반 추천으로 바뀌었습니다.

- 발표 자료: [`docs/발표자료_기존서비스.pdf`](docs/발표자료_기존서비스.pdf)
- 시연 영상: [`docs/시연영상.mp4`](docs/시연영상.mp4)
- 기술 보고서: [`docs/technical_report.pdf`](docs/technical_report.pdf)

---

## 프로젝트 구조

```
crawling/
│
├── app/                          # 현재 서비스 — 키워드 감성 추천 (실행 가능)
│   ├── app_keyword.py            # Streamlit 앱 진입점
│   ├── keyword_service.py        # 인덱스 조회 + Wilson 랭킹 로직
│   └── run_keyword.bat           # 더블클릭하면 서비스 실행
│
├── legacy/                       # 기존 서비스 — 의미 검색 (실행 불가, 코드 참고용)
│   ├── app.py                    # Streamlit 앱 진입점
│   ├── search.py                 # BGE-M3 임베딩 + FAISS 검색 + GRU 감성 필터 로직
│   ├── api_server.py             # 추천 로직을 Flask REST API로 노출하는 실험적 확장
│   └── run_search.bat            # 실행 스크립트
│
├── training/                     # 데이터 수집 및 모델 학습
│   ├── crawl_letterboxd.py           # Letterboxd 영화 목록·리뷰 크롤러
│   ├── train_keyword_sentiment.py    # v1 학습 스크립트 (긍정 9~10점 / 부정 1~2점 기준)
│   ├── train_keyword_sentiment_v2.py # v2 학습 스크립트 (긍정 8~9점 / 부정 2~3점 기준)
│   ├── train_keyword_sentiment_v3.py # v3 학습 스크립트 — 배포 중인 모델을 만든 스크립트
│   └── build_keyword_index.py        # 리뷰를 LSTM으로 분류해 keyword_index.pkl 생성
│
├── models/                       # 학습된 모델 및 인덱스
│   ├── keyword_sentiment_v3.keras    # 배포 중인 LSTM 감성 분류 모델 (정확도 84%)
│   ├── keyword_best_v3.keras         # 학습 중 val_loss 최저 시점 체크포인트
│   ├── keyword_tokenizer_v3.pkl      # 위 모델이 쓰는 토크나이저
│   ├── keyword_config_v3.pkl         # 위 모델의 학습 설정값(MAX_LEN, VOCAB_SIZE 등)
│   ├── keyword_index.pkl             # 키워드×영화별 긍정/부정 문장 수 (서비스가 읽는 핵심 데이터)
│   ├── genre_map.pkl                 # 장르별 영화 목록
│   ├── movie_meta.pkl                # 영화 제목·연도 등 메타데이터
│   ├── faiss_movie.index             # 영화 평균 벡터 인덱스 (search.py가 사용)
│   ├── faiss_review.index            # 리뷰 단위 벡터 인덱스 (search.py가 사용)
│   ├── faiss_movie_db.index          # 영화 평균 벡터 인덱스 (Oracle DB 버전, search_db.py가 사용 — search_db.py는 레포에 없음)
│   ├── faiss_review_db.index         # 리뷰 단위 벡터 인덱스 (Oracle DB 버전, 위와 동일)
│   ├── sentiment_64.keras            # 기존 서비스용 GRU 감성 모델
│   ├── sentiment_config_64.pkl       # 위 모델의 학습 설정값
│   └── sentiment_tokenizer_64.pkl    # 위 모델이 쓰는 토크나이저
│
├── data/
│   └── review_sample.db          # 원본 리뷰 DB가 너무 커서 대신 넣은 샘플 (상위 500편 × 100건, 14MB)
│
├── docs/
│   ├── technical_report.docx / .pdf   # 기술 보고서
│   ├── 발표자료_기존서비스.pdf
│   └── 시연영상.mp4
│
└── README.md
```

---

## 빠른 실행

### 현재 서비스 (키워드 감성 추천)

```bash
# 방법 1: 더블클릭
app/run_keyword.bat

# 방법 2: 터미널
conda activate aiservice26
streamlit run app/app_keyword.py
# → http://localhost:8501
```

실행에 필요한 파일(`keyword_index.pkl`, `genre_map.pkl`, `movie_meta.pkl`, `keyword_sentiment_v3.keras`, `keyword_tokenizer_v3.pkl`)은 전부 `models/`에 포함되어 있습니다.

### 기존 서비스 (의미 검색) — 실행 불가, 코드 참고용

```bash
legacy/run_search.bat
```

리뷰 메타데이터(`review_meta.pkl`, 약 1.45GB)와 원본 리뷰 DB(`review_texts.db`, 약 11.5GB)가 용량 문제로 레포에 없어서 동작하지 않습니다. 나머지 모델 파일(`models/faiss_movie.index`, `models/sentiment_64.keras` 등)은 전부 포함되어 있습니다. 아키텍처 참고용으로만 남겨둡니다.

`api_server.py`는 추천 로직을 Flask REST API로 노출하는 실험적 확장입니다. `search_db.py`(Oracle DB 연동 버전)가 레포에 없고 DB 연결 정보도 없어서 역시 실행되지 않습니다.

---

## 설치 및 클론

```bash
git clone https://github.com/Gene-3/crawling.git
cd crawling
git lfs pull   # FAISS 인덱스 파일 다운로드 (약 1GB)
```

### 환경 설치

```bash
conda create -n aiservice26 python=3.9
conda activate aiservice26
pip install tensorflow==2.20.0 streamlit numpy==1.26.4 scikit-learn==1.6.1
pip install curl_cffi undetected-chromedriver
```

---

## 서비스 변경 이력

| | 기존 서비스 (`legacy/`) | 현재 서비스 (`app/`) |
|---|---|---|
| 검색 방식 | 자유 텍스트 → BGE-M3 임베딩 → FAISS | 장르 + 키워드 버튼 선택 |
| 감성 모델 | GRU (64 units) | LSTM (64 units) |
| 랭킹 | 코사인 유사도 | Wilson 하한 (z=1.96) |
| 중단/전환 이유 | "GRU 감성 필터가 벡터 검색 결과와 실질적으로 중복된다"는 피드백. 실제로 GRU 필터를 빼고 벡터 검색만으로 추천해도 결과가 크게 다르지 않았음 | — |

같은 키워드라도 리뷰에서 어떤 맥락으로 쓰였는지에 따라 감성이 갈립니다.
- "tries to be funny but fails" → 부정
- "unexpectedly funny" → 긍정

단어가 리뷰에 있는지가 아니라, 그 단어가 긍정적으로 쓰였는지를 LSTM이 판단합니다.

### 현재 서비스 흐름

```
장르 + 키워드 선택
    → keyword_index.pkl 조회 (런타임 모델 추론 없음)
    → Wilson 하한 랭킹
    → 추천 결과 + 리뷰 예시
```

직접 측정한 결과, 서버 시작 시 pkl 로드에 약 100ms, 키워드 조회 1회에 1ms 미만이 걸립니다(20회 평균 0.35ms).

---

## 모델 학습

### LSTM 모델 구조

```python
Sequential([
    Embedding(30001, 64),
    SpatialDropout1D(0.15),   # 임베딩 채널 단위 dropout — 과적합 억제
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])
# Adam(lr=0.001) · binary_crossentropy · EarlyStopping(patience=3)
```

기존 서비스(의미 검색)에서는 GRU를 썼지만, 키워드 감성 분류에는 LSTM이 더 적합하다고 판단해 교체했습니다.

### 버전별 성능 비교

| 버전 | 학습 데이터 | 평점 기준 | Accuracy | Recall(긍정) | Recall(부정) | 비고 |
|------|-----------|---------|---------|------------|------------|------|
| v1 | 80k × 2 | 9~10 / 1~2 | 84% | 0.746 | 0.856 | 부정 편향 |
| v2 | 80k × 2 | 8~9 / 2~3 | 84% | 0.853 | 0.830 | 균형 개선 |
| v3 | 500k × 2 | 8~9 / 2~3 | 84% | 0.84 | 0.84 | 균형 달성, 배포 중 |

### 재학습 방법

```bash
conda activate aiservice26

# 1. 모델 학습
python training/train_keyword_sentiment_v3.py

# 2. 키워드 인덱스 재계산 (약 30~60분, 모델 변경 시에만 재실행)
python training/build_keyword_index.py
```

---

## 데이터 수집

### Phase 1 — 영화 목록
- 도구: `undetected-chromedriver` (Chrome 패치, 봇 탐지 우회)
- 범위: Letterboxd 1960~2026년 연도별 인기 상위 400편 × 66년
- 수집 항목: 영화 제목, 연도, URL → `models/movie_meta.pkl`

### Phase 2 — 리뷰 수집
- 도구: `curl_cffi AsyncSession` (TLS 핑거프린트 위장, 비동기)
- 결과: 40,217,722건 / 26,761편
- 저장: `review_texts.db` (SQLite, 약 11.5GB) + `review_meta.pkl` (평점/영화키, 약 1.45GB)
- 요청 간 0.15~0.25초 랜덤 지연, 오류 시 3~10초 대기 후 재시도

한국어(CJK) 리뷰 제외는 크롤링 단계가 아니라 `training/train_keyword_sentiment_v3.py`의 학습 데이터 수집 단계에서 처리합니다.

> 원본 DB는 용량 문제로 레포에 포함하지 않았습니다. 대신 상위 500편 × 100건 샘플(`data/review_sample.db`, 14MB)을 넣었습니다.

---

## 발생한 문제와 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| 문장 단위 레이블 노이즈 | 문장 추출 후 전체 평점을 레이블로 사용 | 리뷰 전문으로 학습 전환 |
| 소표본 랭킹 왜곡 | 3/3=100%가 30/34=88% 위에 표시 | Wilson 하한 적용 (z=1.96) |
| scary 장르 오분류 | "love is scary" 관용구로 로맨스→Horror 분류 | 장르 분류 키워드에서만 제외 |

---

## 서비스 사용법

1. `app/run_keyword.bat` 더블클릭 또는 `streamlit run app/app_keyword.py`
2. 왼쪽 사이드바에서 장르 선택 (Action · Animation · Comedy · Drama · Horror · Romance · Sci-Fi · Thriller)
3. 키워드 버튼 클릭 (분위기/감정/스토리/연기·기술, 총 26개)
4. 여러 개 선택 시 AND 조건으로 교집합만 추천
5. 결과에서 키워드별 긍정 비율(%)과 리뷰 예시 확인

---

## 한계

- 추천 풀이 리뷰 수 상위 3,000편으로 제한되어 인기작 편향이 있음
- 짧거나 모호한 문장에서 오분류 가능
- 장르 분류가 키워드 빈도 기반이라 다의어 오분류가 일부 남아 있음
