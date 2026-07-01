# -*- coding: utf-8 -*-
"""
키워드 감성 분류 LSTM 학습 v3 — 학습량 500k×2 확대판

v2(80k×2, 8~9/2~3)와 동일한 아키텍처·평점 기준.
학습 데이터를 500k×2(100만건)로 늘려 v1/v2와 성능 비교한다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import random
import pickle
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.join(BASE_DIR, '..')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR  = os.path.join(ROOT_DIR, 'data')
DB_PATH   = os.path.join(DATA_DIR, "review_texts.db")

VOCAB_SIZE = 30_000
MAX_LEN    = 80
EMBED_DIM  = 64
LSTM_UNITS = 64
BATCH      = 64
EPOCHS     = 15
TARGET_PER_CLASS = 500_000   # v2의 80k → 500k

POS_LO, POS_HI = 8, 9
NEG_LO, NEG_HI = 2, 3

KEYWORDS = [
    'scary', 'funny', 'romantic', 'sad', 'dark', 'intense', 'lighthearted',
    'thrilling', 'touching', 'heartwarming', 'disturbing', 'inspiring',
    'predictable', 'original', 'twist', 'boring', 'entertaining',
    'acting', 'cinematography', 'soundtrack', 'dialogue', 'visuals',
    'masterpiece', 'overrated', 'underrated', 'disappointing',
    'emotional', 'suspenseful', 'hilarious', 'creepy',
]

_cjk_re = re.compile(r'[ㄱ-鿿]')


def collect_reviews(review_ids, target):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    collected = []
    batch_size = 5000

    for start in range(0, len(review_ids), batch_size):
        if len(collected) >= target:
            break
        batch = review_ids[start:start + batch_size]
        placeholders = ','.join('?' * len(batch))
        cur.execute(
            f"SELECT id, text FROM reviews WHERE id IN ({placeholders})", batch)
        for _, text in cur.fetchall():
            if not text or _cjk_re.search(text) or len(text.split()) < 10:
                continue
            collected.append(text[:500])
            if len(collected) >= target:
                break
        if start % 100000 == 0:
            print(f"\r    수집: {len(collected):,}/{target:,}", end='', flush=True)

    conn.close()
    print(f"\r    완료: {len(collected):,}건" + " " * 20)
    return collected[:target]


def build_model(num_words):
    m = Sequential([
        Embedding(num_words, EMBED_DIM),
        SpatialDropout1D(0.15),
        LSTM(LSTM_UNITS),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid'),
    ])
    m.compile(loss='binary_crossentropy', optimizer=Adam(0.001),
              metrics=['accuracy'])
    return m


def main():
    print("review_meta 로드...")
    with open(os.path.join(DATA_DIR, "review_meta.pkl"), 'rb') as f:
        review_meta = pickle.load(f)
    print(f"  전체 리뷰: {len(review_meta):,}건")

    pos_ids, neg_ids = [], []
    for i, (_, r) in enumerate(review_meta):
        if POS_LO <= r <= POS_HI:
            pos_ids.append(i)
        elif NEG_LO <= r <= NEG_HI:
            neg_ids.append(i)
    print(f"  긍정({POS_LO}~{POS_HI}점): {len(pos_ids):,}건")
    print(f"  부정({NEG_LO}~{NEG_HI}점): {len(neg_ids):,}건")
    del review_meta

    random.seed(42)
    random.shuffle(pos_ids)
    random.shuffle(neg_ids)

    print("\n긍정 리뷰 수집...")
    pos_texts = collect_reviews(pos_ids, TARGET_PER_CLASS)
    print("부정 리뷰 수집...")
    neg_texts = collect_reviews(neg_ids, TARGET_PER_CLASS)

    n = min(len(pos_texts), len(neg_texts))
    pos_texts, neg_texts = pos_texts[:n], neg_texts[:n]

    texts  = pos_texts + neg_texts
    labels = np.array([1]*len(pos_texts) + [0]*len(neg_texts), dtype=np.float32)
    print(f"\n학습 데이터: {len(texts):,}건 (긍정 {len(pos_texts):,} / 부정 {len(neg_texts):,})")

    order = list(range(len(texts)))
    random.shuffle(order)
    texts  = [texts[i] for i in order]
    labels = labels[order]

    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels)
    print(f"학습: {len(tr_texts):,} / 테스트: {len(te_texts):,}")

    num_words = VOCAB_SIZE + 1
    tok = Tokenizer(num_words=num_words)
    tok.fit_on_texts(tr_texts)

    tr_X = pad_sequences(tok.texts_to_sequences(tr_texts),
                         maxlen=MAX_LEN, padding='post', truncating='post')
    te_X = pad_sequences(tok.texts_to_sequences(te_texts),
                         maxlen=MAX_LEN, padding='post', truncating='post')

    model = build_model(num_words)
    model.summary()

    ckpt_path = os.path.join(MODEL_DIR, 'keyword_best_v3.keras')
    model.fit(
        tr_X, tr_labels,
        epochs=EPOCHS, batch_size=BATCH, validation_split=0.1,
        callbacks=[
            EarlyStopping('val_loss', patience=3, restore_best_weights=True, verbose=1),
            ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]
    )

    model.load_weights(ckpt_path)
    te_pred = (model.predict(te_X, batch_size=512, verbose=0) >= 0.5).astype(int).flatten()
    print("\n=== 분류 성능 (v3: 500k×2, 8~9/2~3) ===")
    print(classification_report(te_labels.astype(int), te_pred, target_names=['부정', '긍정']))

    model.save(os.path.join(MODEL_DIR, "keyword_sentiment_v3.keras"))
    with open(os.path.join(MODEL_DIR, "keyword_tokenizer_v3.pkl"), 'wb') as f:
        pickle.dump(tok, f)
    with open(os.path.join(MODEL_DIR, "keyword_config_v3.pkl"), 'wb') as f:
        pickle.dump({
            'vocab_size': VOCAB_SIZE,
            'num_words': num_words,
            'max_len': MAX_LEN,
            'keywords': KEYWORDS,
            'pos_range': [POS_LO, POS_HI],
            'neg_range': [NEG_LO, NEG_HI],
            'target_per_class': TARGET_PER_CLASS,
        }, f)
    print("\n저장 완료: keyword_sentiment_v3.keras")


if __name__ == '__main__':
    main()
