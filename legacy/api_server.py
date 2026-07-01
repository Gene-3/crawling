"""
Flask REST API wrapper for MovieRecommenderDB (Oracle DB 기반)
실행: python api_server.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, request, jsonify

app = Flask(__name__)
rec = None

try:
    from deep_translator import GoogleTranslator
    translator_available = True
except ImportError:
    translator_available = False


def get_recommender():
    global rec
    if rec is None:
        from search_db import MovieRecommenderDB
        rec = MovieRecommenderDB()
    return rec


def translate_snippets(results, target='ko'):
    if not translator_available or not results:
        return results

    texts = []
    indices = []
    for i, r in enumerate(results):
        for j, s in enumerate(r.get('snippets', [])):
            text = s.get('text', '').strip()
            if text:
                texts.append(text[:500])
                indices.append((i, j))

    if not texts:
        return results

    try:
        translator = GoogleTranslator(source='auto', target=target)
        batch_size = 20
        translated = []
        for b in range(0, len(texts), batch_size):
            batch = texts[b:b+batch_size]
            translated.extend(translator.translate_batch(batch))

        for k, (i, j) in enumerate(indices):
            if k < len(translated) and translated[k]:
                results[i]['snippets'][j]['text'] = translated[k]
    except Exception:
        pass

    return results


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    min_year = data.get('min_year', 1940)
    max_year = data.get('max_year', 2030)
    min_rating = data.get('min_rating', 0.0)
    translate = data.get('translate', None)

    if not query.strip():
        return jsonify({'success': False, 'message': '검색어를 입력해주세요'})

    r = get_recommender()
    results = r.recommend(query.strip(), top_k=max(top_k * 2, 20))

    filtered = [
        item for item in results
        if min_year <= int(item.get('year') or 0) <= max_year
        and item.get('avg_rating', 0) >= min_rating
    ][:top_k]

    if translate:
        filtered = translate_snippets(filtered, target=translate)

    return jsonify({'success': True, 'results': filtered})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'mode': 'db'})


if __name__ == '__main__':
    print("모델 사전 로딩 중...")
    get_recommender()
    print("API 서버 시작: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
