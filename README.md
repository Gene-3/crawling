# 🎬 영화 키워드 감성 추천 서비스

**22503110159 김광진 · AI 서비스 프로그래밍**

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> Letterboxd 리뷰 **4,021만 건** 수집 기반 영화 추천 서비스.  
> 기존 의미 검색(BGE-M3 + FAISS + GRU)에서 **LSTM 키워드 감성 추천**으로 개선.

📄 **발표 자료**: [`docs/발표자료_기존서비스.pdf`](docs/발표자료_기존서비스.pdf)  
🎬 **시연 영상**: [`docs/시연영상.mp4`](docs/시연영상.mp4)  
📑 **기술 보고서**: [`docs/technical_report.pdf`](docs/technical_report.pdf)

---

## 📁 프로젝트 구조

```
crawling/
│
├── 📂 app/                          # ✅ 현재 서비스 — 키워드 감성 추천
│   ├── app_keyword.py               #    Streamlit 메인 앱
│   ├── keyword_service.py           #    Wilson 랭킹 추천 로직
│   └── run_keyword.bat              #    실행 파일 (더블클릭)
│
├── 📂 legacy/                       # ❌ 기존 서비스 — 의미 검색 (참고용)
│   ├── app.py                       #    Streamlit 의미 검색 앱
│   ├── search.py                    #    FAISS 검색 로직
│   ├── api_server.py                #    Flask REST API 실험 (실행 불가)
│   └── run_search.bat               #    실행 파일
│
├── 📂 training/                     # 🧠 데이터 수집 및 모델 학습
│   ├── train_keyword_sentiment_v3.py  # 배포 버전 (500k×2)
│   ├── train_keyword_sentiment_v2.py  # v2 (80k×2)
│   ├── train_keyword_sentiment.py     # v1 (80k×2, 극단값)
│   ├── build_keyword_index.py         # 키워드 인덱스 사전 계산
│   └── crawl_letterboxd.py            # Letterboxd 크롤러
│
├── 📂 models/                       # 💾 학습된 모델 및 인덱스 (Git LFS)
│   ├── keyword_sentiment_v3.keras   #    LSTM 모델 (84% accuracy) — 배포
│   ├── keyword_best_v3.keras        #    EarlyStopping 체크포인트
│   ├── keyword_tokenizer_v3.pkl     #    토크나이저
│   ├── keyword_config_v3.pkl        #    학습 설정
│   ├── keyword_index.pkl            #    사전 계산 인덱스 (서비스 핵심)
│   ├── genre_map.pkl                #    장르별 영화 목록
│   ├── movie_meta.pkl               #    영화 메타데이터
│   ├── faiss_movie_db.index         #    FAISS 벡터 인덱스 (기존 서비스용)
│   ├── sentiment_64.keras           #    GRU 감성 모델 (기존 서비스용)
│   ├── sentiment_config_64.pkl      #    GRU 설정
│   └── sentiment_tokenizer_64.pkl   #    GRU 토크나이저
│
├── 📂 data/                         # 📊 데이터 파일
│   └── review_sample.db             #    리뷰 샘플 (상위 500편 × 100건, 14MB)
│
├── 📂 docs/                         # 📄 문서
│   ├── technical_report.docx / .pdf
│   ├── 발표자료_기존서비스.pdf
│   └── 시연영상.mp4
│
└── README.md
```

---

## 🚀 빠른 실행

### 현재 서비스 (키워드 감성 추천)

```bash
# 방법 1: 더블클릭
app/run_keyword.bat

# 방법 2: 터미널
conda activate aiservice26
streamlit run app/app_keyword.py
# → http://localhost:8501
```

> **실행에 필요한 파일** (모두 `models/` 폴더에 포함):  
> `keyword_index.pkl` · `genre_map.pkl` · `movie_meta.pkl`  
> `keyword_sentiment_v3.keras` · `keyword_tokenizer_v3.pkl`

### 기존 서비스 (의미 검색) — ⚠️ 실행 불가, 코드 참고용

```bash
legacy/run_search.bat
```

> ⚠️ `review_meta.pkl`(~1.45GB) 및 원본 리뷰 DB(`review_texts.db`, ~11.5GB)가  
> 용량 문제로 미포함되어 동작하지 않습니다. 아키텍처 참고용으로만 보존합니다.

---

## 📥 설치 및 클론

```bash
git clone https://github.com/Gene-3/crawling.git
cd crawling
git lfs pull   # 대용량 모델 파일 다운로드
```

### 환경 설치

```bash
conda create -n aiservice26 python=3.9
conda activate aiservice26
pip install tensorflow==2.20.0 streamlit numpy==1.26.4 scikit-learn==1.6.1
pip install curl_cffi undetected-chromedriver
```

---

## 🔄 서비스 변경 이력

| | 기존 서비스 (`legacy/`) | 현재 서비스 (`app/`) |
|---|---|---|
| **파일** | `app.py` | `app_keyword.py` |
| **검색 방식** | 자유 텍스트 → BGE-M3 임베딩 → FAISS | 장르 + 키워드 버튼 선택 |
| **감성 모델** | GRU (64 units) | LSTM (64 units) |
| **랭킹** | 코사인 유사도 | Wilson 하한 (z=1.96) |
| **한계** | 쿼리 품질 의존, 감성 필터 역할 불분명 | 추천 풀 상위 3,000편 제한 |

### 현재 서비스 흐름

```
장르 + 키워드 선택
    → keyword_index.pkl 조회 (응답 ≈ 0.1초, 런타임 추론 없음)
    → Wilson 하한 랭킹
    → 추천 결과 + 리뷰 예시
```

---

## 🧠 모델 학습

### LSTM 모델 구조

```python
Sequential([
    Embedding(30001, 64),
    SpatialDropout1D(0.15),   # 임베딩 과적합 억제
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])
# Adam(lr=0.001) · binary_crossentropy · EarlyStopping(patience=3)
```

### 버전별 성능 비교

| 버전 | 학습 데이터 | 평점 기준 | Accuracy | Recall(긍정) | Recall(부정) | 비고 |
|------|-----------|---------|---------|------------|------------|------|
| v1 | 80k × 2 | 9~10 / 1~2 | 84% | 0.746 | 0.856 | 부정 편향 |
| v2 | 80k × 2 | 8~9 / 2~3 | 84% | 0.853 | 0.830 | 균형 개선 |
| **v3** | **500k × 2** | **8~9 / 2~3** | **84%** | **0.84** | **0.84** | **균형 달성 (배포)** |

### 재학습 방법

```bash
conda activate aiservice26

# 1. 모델 학습 (약 수 시간)
python training/train_keyword_sentiment_v3.py

# 2. 키워드 인덱스 재계산 (약 30~60분)
python training/build_keyword_index.py
```

---

## 🕷️ 데이터 수집

### Phase 1 — 영화 목록
- **도구**: `undetected-chromedriver` (Chrome 패치, 봇 탐지 우회)
- **범위**: Letterboxd 1960~2026 연도별 인기 상위 400편 × 66년
- **결과**: 영화 제목·연도·URL → `models/movie_meta.pkl`

### Phase 2 — 리뷰 수집
- **도구**: `curl_cffi AsyncSession` (TLS 핑거프린트 위장, 비동기)
- **결과**: **40,217,722건** / ~18,000편
- **저장**: `review_texts.db` (SQLite, 12GB) + `review_meta.pkl` (평점/영화키, 1.5GB)
- **설계**: Rate-limit 준수 · 재시도 로직 · CJK(한국어) 리뷰 필터링

> 원본 DB는 용량 문제로 미포함. `data/review_sample.db`(14MB)에 상위 500편 × 100건 샘플 포함.

---

## 🛠️ 발생한 문제와 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| 문장 단위 레이블 노이즈 | 문장 추출 후 전체 평점을 레이블로 사용 | 리뷰 전문으로 학습 전환 |
| 소표본 랭킹 왜곡 | 3/3=100%가 30/34=88% 위에 표시 | Wilson 하한 적용 (z=1.96) |
| scary 장르 오분류 | "love is scary" 관용구로 로맨스→Horror | 장르 분류 키워드에서만 제외 |

---

## 📖 서비스 사용법

1. `app/run_keyword.bat` 더블클릭 또는 `streamlit run app/app_keyword.py`
2. 왼쪽 사이드바에서 **장르** 선택 (8개: Action · Animation · Comedy · Drama · Horror · Romance · Sci-Fi · Thriller)
3. **키워드 버튼** 클릭 (분위기/감정/스토리/연기·기술, 총 26개)
4. 여러 개 선택 시 **AND 조건** 교집합 추천
5. 결과에서 키워드별 긍정 비율(%) + 리뷰 예시 확인

---

## ⚠️ 한계

- 추천 풀이 리뷰 수 상위 **3,000편**으로 제한 (인기작 편향)
- 짧거나 모호한 문장에서 오분류 가능
- 장르 분류는 키워드 빈도 기반 → 다의어 오분류 일부 존재
