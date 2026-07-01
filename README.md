# 영화 키워드 감성 추천 서비스

Letterboxd 리뷰 4천만 건을 기반으로, 장르와 키워드를 선택하면 해당 키워드가 **긍정적으로 언급된** 영화를 추천하는 Streamlit 서비스입니다.

---

## 환경 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.9 |
| TensorFlow / Keras | 2.20.0 / 3.10.0 |
| Streamlit | 1.x |
| NumPy | 1.26.4 |

Anaconda 환경 `aiservice26` 기준으로 작성되었습니다.

---

## 파일 구조

```
movie_recommender/
├── app_keyword.py              # 키워드 추천 Streamlit 앱 (메인 서비스)
├── app.py                      # 의미 검색 Streamlit 앱
├── keyword_service.py          # 추천 서비스 로직 (Wilson 랭킹)
├── build_keyword_index.py      # 키워드 인덱스 사전 계산
│
├── keyword_sentiment_v3.keras  # 키워드 감성 LSTM 모델 (v3, 500k×2)
├── keyword_tokenizer_v3.pkl
├── keyword_config_v3.pkl
│
├── keyword_index.pkl           # 사전 계산된 키워드별 감성 통계
├── genre_map.pkl               # 장르별 영화 목록
├── movie_meta.pkl              # 영화 메타데이터
│
├── review_texts.db             # SQLite 리뷰 텍스트 (40,217,722건)
├── review_meta.pkl             # 리뷰 메타데이터 (영화키, 평점)
│
├── run_keyword.bat             # 키워드 추천 앱 실행 스크립트
└── run_search.bat              # 의미 검색 앱 실행 스크립트
```

---

## 실행 방법

### 방법 1 — 배치 파일 (더블클릭)

`run_keyword.bat` 파일을 더블클릭하면 브라우저가 자동으로 열립니다.

### 방법 2 — 터미널

```bash
conda activate aiservice26
cd D:\Lecture\_AIService26\movie_recommender
streamlit run app_keyword.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 인덱스 재빌드 (모델 변경 시)

`keyword_index.pkl`과 `genre_map.pkl`은 사전 계산 파일입니다.  
모델을 바꿨을 때만 아래 명령을 실행하세요 (약 30~60분 소요).

```bash
conda activate aiservice26
cd D:\Lecture\_AIService26\movie_recommender
python build_keyword_index.py
```

---

## 서비스 사용법

1. **왼쪽 사이드바**에서 장르 선택 (Action / Comedy / Horror 등 8개)
2. **키워드 버튼** 클릭 (분위기 / 감정 / 스토리 / 연기·기술 카테고리)
3. 여러 개 선택하면 **AND 조건**으로 교집합 추천
4. 결과에서 긍정 비율(%) 및 리뷰 예시 확인

---

## 모델 학습 스크립트

| 파일 | 설명 |
|------|------|
| `train_keyword_sentiment.py` | v1 — 극단 평점 (9~10 / 1~2), 80k×2 |
| `train_keyword_sentiment_v2.py` | v2 — 완만 평점 (8~9 / 2~3), 80k×2 |
| `train_keyword_sentiment_v3.py` | v3 — 완만 평점 (8~9 / 2~3), 500k×2 |

```bash
conda activate aiservice26
python train_keyword_sentiment_v3.py
```
