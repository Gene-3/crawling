# -*- coding: utf-8 -*-
"""
키워드 맥락 감성 분류 LSTM 모델 학습

극단 평점(9-10: 긍정, 1-2: 부정) 리뷰 전문으로 일반 감성 분류기를 학습한다.
학습된 모델은 키워드가 포함된 문장에 적용하여 해당 키워드의 긍정/부정 맥락을 판별한다.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "review_texts.db")

VOCAB_SIZE = 30_000
MAX_LEN = 80
EMBED_DIM = 64
LSTM_UNITS = 64
BATCH = 64
EPOCHS = 15
TARGET_PER_CLASS = 80_000

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
    cur = conn.cursor()
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
        if start % 50000 == 0:
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
    with open(os.path.join(BASE_DIR, "review_meta.pkl"), 'rb') as f:
        review_meta = pickle.load(f)
    print(f"  전체 리뷰: {len(review_meta):,}건")

    pos_ids = []
    neg_ids = []
    for i, (_, r) in enumerate(review_meta):
        if r >= 9:
            pos_ids.append(i)
        elif r <= 2:
            neg_ids.append(i)
    print(f"  긍정(9-10점): {len(pos_ids):,}건")
    print(f"  부정(1-2점): {len(neg_ids):,}건")

    del review_meta

    random.seed(42)
    random.shuffle(pos_ids)
    random.shuffle(neg_ids)

    print("\n긍정 리뷰 수집...")
    pos_texts = collect_reviews(pos_ids, TARGET_PER_CLASS)
    print("부정 리뷰 수집...")
    neg_texts = collect_reviews(neg_ids, TARGET_PER_CLASS)

    texts = pos_texts + neg_texts
    labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts), dtype=np.float32)
    print(f"\n학습 데이터: {len(texts):,}건 (긍정 {len(pos_texts):,} / 부정 {len(neg_texts):,})")

    order = list(range(len(texts)))
    random.shuffle(order)
    texts = [texts[i] for i in order]
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

    ckpt_path = os.path.join(BASE_DIR, 'keyword_best.keras')
    model.fit(tr_X, tr_labels,
              epochs=EPOCHS, batch_size=BATCH, validation_split=0.1,
              callbacks=[
                  EarlyStopping('val_loss', patience=3, restore_best_weights=True),
                  ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
              ])

    model.load_weights(ckpt_path)
    te_pred = (model.predict(te_X, batch_size=512, verbose=0) >= 0.5).astype(int).flatten()
    print("\n분류 성능:")
    print(classification_report(te_labels, te_pred, target_names=['부정', '긍정']))

    examples = [
        "The acting was absolutely phenomenal and captivating",
        "The acting was terrible and wooden throughout",
        "So scary that I couldn't sleep for days, loved every moment",
        "It tried to be scary but completely failed",
        "The story was beautifully crafted and original",
        "Boring and predictable from start to finish",
        "The cinematography was breathtaking and stunning",
        "Disappointing sequel that ruined everything",
    ]
    print("\n문장 수준 일반화 테스트:")
    for text in examples:
        seq = tok.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        prob = model.predict(pad, verbose=0)[0][0]
        tag = "긍정" if prob >= 0.5 else "부정"
        print(f"  [{tag} {prob:.3f}] {text}")

    model.save(os.path.join(BASE_DIR, "keyword_sentiment.keras"))
    with open(os.path.join(BASE_DIR, "keyword_tokenizer.pkl"), 'wb') as f:
        pickle.dump(tok, f)
    with open(os.path.join(BASE_DIR, "keyword_config.pkl"), 'wb') as f:
        pickle.dump({
            'vocab_size': VOCAB_SIZE,
            'num_words': num_words,
            'max_len': MAX_LEN,
            'keywords': KEYWORDS,
        }, f)
    print("\n저장 완료: keyword_sentiment.keras, keyword_tokenizer.pkl, keyword_config.pkl")


if __name__ == '__main__':
    main()
