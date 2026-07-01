# 영화 추천 서비스

**22503110159 김광진 · AI 서비스 프로그래밍**

Letterboxd 리뷰 **4,021만 건** 수집 기반 영화 추천 서비스.  
기존 의미 검색 서비스에서 LSTM 키워드 감성 추천 서비스로 개선했습니다.

> 📄 발표 자료: [`발표자료_기존서비스.pdf`](발표자료_기존서비스.pdf)  
> 🎬 시연 영상: [`시연영상.mp4`](시연영상.mp4)

---

## 서비스 변경 이력

### ❌ 기존 서비스 — 의미 검색 (`app.py`)

```
텍스트 입력 → BGE-M3 임베딩 → FAISS 벡터 검색 → GRU 감성 필터 → 추천
```

**한계:**
- 교수님 피드백: "GRU 모델이 굳이 필요한가?"
- 벡터 검색이 이미 의미 유사 결과를 반환 → 감성 필터의 실질적 역할 불분명
- 자유 텍스트 입력 → 쿼리 품질에 결과가 크게 좌우됨

### ✅ 현재 서비스 — 키워드 감성 추천 (`app_keyword.py`)

```
장르 + 키워드 선택 → keyword_index.pkl 조회 → Wilson 하한 랭킹 → 추천
```

LSTM이 리뷰 문장의 **감성 맥락**을 분류하는 것이 핵심입니다.  
"tries to be funny but fails" → 부정 / "unexpectedly funny" → 긍정  
단순 키워드 존재 여부가 아니라 맥락을 구분합니다.

---

## 파일 구조

```
movie_recommender/
│
├── ✅ [현재 서비스 — 키워드 감성 추천]
│   ├── app_keyword.py              # Streamlit 메인 앱
│   ├── keyword_service.py          # Wilson 랭킹 추천 로직
│   ├── build_keyword_index.py      # 키워드 인덱스 사전 계산
│   ├── keyword_index.pkl           # 사전 계산 인덱스 (서비스 핵심)
│   ├── genre_map.pkl               # 장르별 영화 목록
│   ├── movie_meta.pkl              # 영화 메타데이터
│   ├── keyword_sentiment_v3.keras  # LSTM 모델 (84% accuracy)
│   ├── keyword_best_v3.keras       # EarlyStopping 체크포인트
│   ├── keyword_tokenizer_v3.pkl    # 토크나이저
│   ├── keyword_config_v3.pkl       # 학습 설정
│   ├── run_keyword.bat             # 실행 파일 (더블클릭)
│   └── train_keyword_sentiment_v3.py  # 학습 스크립트 (배포 버전)
│
├── ❌ [기존 서비스 — 의미 검색]
│   ├── app.py                      # Streamlit 의미 검색 앱
│   ├── search.py                   # FAISS 검색 로직
│   ├── api_server.py               # FastAPI 서버
│   └── run_search.bat              # 실행 파일
│
├── [공통 — 데이터 수집 / 학습]
│   ├── crawl_letterboxd.py         # Letterboxd 크롤러
│   ├── train_keyword_sentiment.py  # v1 학습 스크립트
│   ├── train_keyword_sentiment_v2.py  # v2 학습 스크립트
│   └── review_sample.db            # 리뷰 샘플 (상위 500편 × 100건)
│
└── [문서]
    ├── README.md
    ├── technical_report.md
    ├── 발표자료_기존서비스.pdf      # 기존 서비스 발표 자료
    └── 시연영상.mp4                 # 서비스 시연 영상
```

---

## 빠른 실행

### 현재 서비스 (키워드 감성 추천)
```bash
# 방법 1: 더블클릭
run_keyword.bat

# 방법 2: 터미널
conda activate aiservice26
streamlit run app_keyword.py
# → http://localhost:8501
```

### 기존 서비스 (의미 검색)
```bash
run_search.bat
# 또는
conda activate aiservice26
streamlit run app.py
```

> **현재 서비스 실행에 필요한 파일:** `keyword_index.pkl`, `genre_map.pkl`, `movie_meta.pkl`, `keyword_sentiment_v3.keras`, `keyword_tokenizer_v3.pkl` — 모두 레포에 포함되어 있습니다.

---

## 다운로드

```bash
git clone https://github.com/Gene-3/crawling.git
cd crawling
git lfs pull   # FAISS 인덱스 파일 다운로드 (약 1GB)
```

---

## 환경 설치

```bash
conda create -n aiservice26 python=3.9
conda activate aiservice26
pip install tensorflow==2.20.0 streamlit numpy==1.26.4 scikit-learn==1.6.1
pip install curl_cffi undetected-chromedriver
```

---

## 데이터 수집 (크롤링)

### Phase 1 — 영화 목록
- 도구: `undetected-chromedriver` (Chrome 패치, 봇 탐지 우회)
- 범위: Letterboxd 1960~2026년 연도별 인기 상위 400편 × 66년
- 수집 항목: 영화 제목, 연도, URL → `movie_meta.pkl`

### Phase 2 — 리뷰 수집
- 도구: `curl_cffi AsyncSession` (TLS 핑거프린트 위장, 비동기)
- 결과: **40,217,722건** / ~18,000편
- 저장: `review_texts.db` (SQLite, 12GB) + `review_meta.pkl` (평점/영화키, 1.5GB)
- 핵심 설계: Rate-limit 준수 · 재시도 로직 · CJK(한국어) 리뷰 필터링

> 원본 DB는 용량 문제로 미포함. 레포에는 상위 500편 × 100건 샘플(`review_sample.db`, 14MB) 포함.

---

## 데이터 전처리 및 학습

```bash
conda activate aiservice26
python train_keyword_sentiment_v3.py
```

### 전처리 흐름
```
review_texts.db + review_meta.pkl
    → 평점 기준 필터링: 긍정 8~9점 / 부정 2~3점 (극단값 제외)
    → 클래스 균형: 긍정 500,000건 / 부정 500,000건
    → 토크나이저 학습 (VOCAB=30,000)
    → 패딩 (MAX_LEN=80)
    → 학습/테스트 9:1 분리
```

### 모델 구조
```python
Sequential([
    Embedding(30001, 64),
    SpatialDropout1D(0.15),   # 채널 단위 dropout — 임베딩 과적합 억제
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])
# Adam(lr=0.001) · binary_crossentropy · EarlyStopping(patience=3)
```

> **GRU → LSTM 전환:** 1부 서비스에서는 GRU를 사용했으나, 키워드 감성 분류에는 LSTM이 더 적합하다고 판단해 교체

### 버전별 성능 비교

| 버전 | 학습 데이터 | 평점 기준 | Accuracy | Recall(긍정) | Recall(부정) | 비고 |
|------|------------|----------|----------|-------------|-------------|------|
| v1 | 80k × 2 | 9~10 / 1~2 | 84% | 0.746 | 0.856 | 부정 편향 |
| v2 | 80k × 2 | 8~9 / 2~3 | 84% | 0.853 | 0.830 | 균형 개선 |
| **v3** | **500k × 2** | **8~9 / 2~3** | **84%** | **0.84** | **0.84** | **균형 달성 (배포)** |

---

## 키워드 인덱스 사전 계산

```bash
# 약 30~60분 소요 (모델 변경 시에만 재실행)
python build_keyword_index.py
```

```
review_texts.db → 상위 3,000편 선택 → 키워드 포함 문장 추출
    → LSTM 배치 추론 (batch=512) → 긍정/부정 카운트
    → keyword_index.pkl + genre_map.pkl 저장
```

런타임에는 모델 추론 없이 pkl 딕셔너리 조회만 수행합니다 (응답 ≈ 0.1초).

---

## 발생한 문제와 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| 문장 단위 레이블 노이즈 | 문장 추출 후 전체 평점을 레이블로 사용 | 리뷰 전문으로 학습 전환 |
| 소표본 랭킹 왜곡 | 3/3=100%가 30/34=88% 위에 표시 | Wilson 하한 적용 (z=1.96) |
| scary 장르 오분류 | "love is scary" 관용구로 로맨스→Horror | 장르 분류 키워드에서만 제외 |

---

## 서비스 사용법

1. 왼쪽 사이드바에서 **장르** 선택 (8개)
2. **키워드 버튼** 클릭 (분위기/감정/스토리/연기·기술, 26개)
3. 여러 개 선택 시 **AND 조건** 교집합 추천
4. 결과에서 키워드별 긍정 비율(%) + 리뷰 예시 확인

---

## 한계

- 추천 풀이 리뷰 수 상위 **3,000편**으로 제한 (인기작 편향)
- 짧거나 모호한 문장에서 오분류 가능
- 장르 분류는 키워드 빈도 기반 → 다의어 오분류 일부 존재
