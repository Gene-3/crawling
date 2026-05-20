"""
Letterboxd 영화 리뷰 크롤러
- 연도별 상위 영화의 사용자 리뷰를 수집하여 CSV로 저장
- 영화 목록: undetected-chromedriver / 리뷰 수집: 비동기 curl_cffi
"""

import asyncio, csv, io, json, os, random, shutil, threading, time
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from curl_cffi.requests import AsyncSession
import undetected_chromedriver as uc

# ─── 기본 설정 ───────────────────────────────────────────
LB_USERNAME          = "ID"
LB_PASSWORD          = "PWD"
MOVIES_PER_YEAR      = 400
MAX_REVIEWS_PER_FILM = 0      # 영화당 최대 리뷰 수 (0 = 무제한)
CONCURRENT_MOVIES    = 2      # 동시 처리 영화 수
GLOBAL_SEM           = 12     # 전체 동시 요청 수 캡
RETRY_COUNT          = 3
SCRIPT_FOLDER        = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE            = os.path.join(SCRIPT_FOLDER, "letterboxd_reviews.csv")
LOGIN_COOKIE         = os.path.join(SCRIPT_FOLDER, "letterboxd_cookies.json")
should_stop          = threading.Event()
# ────────────────────────────────────────────────────────


def p(msg): print(msg, flush=True)

def convert_rating(text):
    """Letterboxd 별점(★★★½) → 숫자(0~10) 변환"""
    if not text or text == 'N/A':
        return 'N/A'
    full  = text.count('★')
    half  = 1 if '½' in text else 0
    score = (full + half * 0.5) * 2
    return str(score) if score > 0 else 'N/A'

def load_cookie_list():
    with open(LOGIN_COOKIE) as f:
        return json.load(f)


# ── 브라우저 (영화 목록용) ────────────────────────────────
def start_browser():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1280,800")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-position=-32000,-32000")
    return uc.Chrome(options=options, headless=False)

def is_logged_in(browser):
    try:
        return 'logged-out' not in browser.find_element(By.TAG_NAME, 'body').get_attribute('class')
    except:
        return False

def load_cookies(browser):
    if not os.path.exists(LOGIN_COOKIE):
        raise Exception("쿠키 파일 없음. letterboxd_login.py를 먼저 실행해주세요.")
    browser.get("https://letterboxd.com")
    time.sleep(2)
    with open(LOGIN_COOKIE) as f:
        for cookie in json.load(f):
            try:
                browser.add_cookie({k: v for k, v in cookie.items() if k not in ['expiry', 'sameSite']})
            except:
                continue
    browser.refresh()
    time.sleep(3)
    if is_logged_in(browser):
        p("쿠키로 로그인 완료\n")
        return True
    raise Exception("쿠키 만료. letterboxd_login.py를 다시 실행해주세요.")

def check_cloudflare(browser, url):
    src = browser.page_source
    if 'Just a moment' in src or 'cf-browser-verification' in src or '잠시만' in src:
        p("    Cloudflare 감지. 대기 후 재시도...")
        time.sleep(random.uniform(8, 12))
        browser.get(url)
        time.sleep(random.uniform(3, 5))

def get_movie_list(browser, year):
    """해당 연도 상위 영화 목록 수집 (브라우저)"""
    movies = []
    page   = 1
    while len(movies) < MOVIES_PER_YEAR:
        url = f"https://letterboxd.com/films/year/{year}/by/popular/page/{page}/"
        browser.get(url)
        time.sleep(random.uniform(0.8, 1.2))
        check_cloudflare(browser, url)

        soup  = BeautifulSoup(browser.page_source, 'lxml')
        items = soup.find_all('li', class_='posteritem')
        if not items:
            break

        for item in items:
            try:
                div   = item.find('div', class_='react-component')
                slug  = div['data-item-slug']
                title = div['data-item-name']
                movies.append({'title': title, 'slug': slug})
            except:
                continue

        next_btn = soup.find('a', class_='next')
        if not next_btn:
            break
        p(f"  -> {len(movies)}개 로드 중...")
        page += 1
        time.sleep(random.uniform(0.5, 0.8))

    p(f"  -> 총 {len(movies[:MOVIES_PER_YEAR])}개 영화 로드 완료")
    return movies[:MOVIES_PER_YEAR]


# ── 비동기 리뷰 수집 (curl_cffi) ─────────────────────────
async def _fetch(session, sem, url):
    async with sem:
        r = await session.get(url, impersonate="chrome120", timeout=15)
        await asyncio.sleep(random.uniform(0.15, 0.25))
        return r.text

def _parse_reviews(html, title, year):
    """HTML에서 리뷰 파싱"""
    reviews  = []
    has_more = False
    soup     = BeautifulSoup(html, 'lxml')
    boxes    = soup.find_all('article', class_='production-viewing')
    for box in boxes:
        try:
            body = box.find('div', class_='body-text')
            if not body:
                continue
            text = body.get_text(separator=' ').strip()
            if len(text) < 10:
                continue
            rating_el = box.find('span', class_='inline-rating')
            rating    = convert_rating(rating_el.get_text(strip=True)) if rating_el else 'N/A'
            reviews.append({'Year': year, 'Title': title,
                             'Review_Rating': rating, 'Review_Text': text})
        except:
            continue
    if soup.find('a', class_='next'):
        has_more = True
    return reviews, has_more

async def _collect(session, sem, slug, title, year):
    """영화 한 편 리뷰 수집 (글로벌 세마포어 공유)"""
    reviews       = []
    page          = 1
    pages_per_req = GLOBAL_SEM // CONCURRENT_MOVIES  # 영화당 동시 요청 수
    base_url      = f"https://letterboxd.com/film/{slug}/reviews/by/activity/"

    while not should_stop.is_set():
        urls    = [base_url if page + i == 1 else f"{base_url}page/{page + i}/"
                   for i in range(pages_per_req)]
        results = await asyncio.gather(*[_fetch(session, sem, u) for u in urls],
                                       return_exceptions=True)
        has_more      = False
        last_more     = None          # 마지막으로 성공한 페이지의 has_more
        for html in results:
            if isinstance(html, Exception):
                continue
            parsed, more = _parse_reviews(html, title, year)
            reviews.extend(parsed)
            last_more = more          # 순서 보장 → 마지막 값이 최종 페이지 기준
        if last_more is not None:
            has_more = last_more      # 중간 페이지 next 버튼에 속지 않음

        p(f"    [{title[:20]}] 페이지 {page}~{page + pages_per_req - 1}: 누적 {len(reviews)}개")

        if not has_more or (MAX_REVIEWS_PER_FILM and len(reviews) >= MAX_REVIEWS_PER_FILM):
            break
        page += pages_per_req

    return reviews


# ── 저장 ─────────────────────────────────────────────────
def backup_csv(total_count):
    milestone   = (total_count // 50000) * 50000
    backup_file = os.path.join(SCRIPT_FOLDER, f"letterboxd_reviews_backup_{milestone:07d}.csv")
    if not os.path.exists(backup_file):
        shutil.copy2(SAVE_FILE, backup_file)
        p(f"  [백업] {milestone:,}개 달성 → {os.path.basename(backup_file)} 생성")

def save_reviews(review_list):
    if not review_list:
        return
    df = pd.DataFrame(review_list)
    for col in ['Review_Text', 'Title']:
        df[col] = df[col].str.replace(r'[\r\n\t\x85\x0b\x0c]+', ' ', regex=True).str.strip()
    df = df.drop_duplicates(subset=['Title', 'Review_Text'])
    is_new_file = not os.path.exists(SAVE_FILE)
    tmp_file    = SAVE_FILE + '.tmp'
    buf  = io.StringIO()
    df.to_csv(buf, index=False, header=is_new_file, quoting=csv.QUOTE_ALL, lineterminator='\n')
    data = buf.getvalue().encode('utf-8-sig' if is_new_file else 'utf-8')
    with open(tmp_file, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    if is_new_file:
        os.replace(tmp_file, SAVE_FILE)
    else:
        with open(SAVE_FILE, 'ab') as f:
            with open(tmp_file, 'rb') as tmp:
                f.write(tmp.read())
            f.flush(); os.fsync(f.fileno())
        os.remove(tmp_file)


# ── 메인 실행 ────────────────────────────────────────────
async def _run_async(start_year, end_year, browser, cookie_list):
    """단일 이벤트 루프로 전체 크롤링 (CONCURRENT_MOVIES 동시 처리)"""
    loop        = asyncio.get_event_loop()
    total_saved = 0
    sem         = asyncio.Semaphore(GLOBAL_SEM)
    save_lock   = asyncio.Lock()

    already_done  = set()
    initial_count = 0
    if os.path.exists(SAVE_FILE) and os.path.getsize(SAVE_FILE) > 0:
        try:
            existing      = pd.read_csv(SAVE_FILE, encoding='utf-8-sig', usecols=['Title'], on_bad_lines='skip')
            already_done  = set(existing['Title'].unique())
            initial_count = len(existing)
            p(f"기존 수집: {len(already_done)}편 / {initial_count:,}개 건너뜀\n")
        except Exception:
            os.remove(SAVE_FILE)
            p("손상된 CSV 파일 제거. 처음부터 수집합니다.\n")

    async def process_movie(session, movie, idx, total):
        nonlocal total_saved
        title, slug = movie['title'], movie['slug']
        if title in already_done:
            p(f"  [{idx}/{total}] '{title}' - 이미 수집됨, 건너뜀")
            return
        p(f"  [{idx}/{total}] '{title}' 수집 중... (이번 세션: {total_saved}개)")
        for i in range(1, RETRY_COUNT + 1):
            if should_stop.is_set():
                return
            try:
                reviews = await _collect(session, sem, slug, title, year)
                async with save_lock:
                    save_reviews(reviews)
                    prev_total    = initial_count + total_saved
                    total_saved  += len(reviews)
                    total_in_file = initial_count + total_saved
                    already_done.add(title)
                    p(f"    -> '{title[:20]}' {len(reviews)}개 저장 완료 (이번 세션: {total_saved}개 / 전체: {total_in_file:,}개)")
                    if total_in_file // 50000 > prev_total // 50000:
                        backup_csv(total_in_file)
                return
            except Exception as e:
                p(f"    -> '{title[:20]}' 수집 실패 ({i}/{RETRY_COUNT}): {e}")
                if i == RETRY_COUNT:
                    p(f"    -> '{title}' 건너뜀")
                    already_done.add(title)
                else:
                    await asyncio.sleep(random.uniform(3, 5))

    async with AsyncSession() as session:
        for cookie in cookie_list:
            session.cookies.set(cookie['name'], cookie['value'],
                                domain=cookie.get('domain', 'letterboxd.com'))

        for year in range(start_year, end_year + 1):
            if should_stop.is_set():
                break
            p(f"\n[{year}년] 영화 목록 수집 중...")
            movies = []
            for i in range(1, RETRY_COUNT + 1):
                if should_stop.is_set():
                    break
                try:
                    movies = await loop.run_in_executor(None, get_movie_list, browser, year)
                    break
                except Exception as e:
                    p(f"  -> 목록 수집 실패 ({i}/{RETRY_COUNT}): {e}")
                    if i < RETRY_COUNT:
                        await asyncio.sleep(random.uniform(5, 10))

            # CONCURRENT_MOVIES 단위로 묶어서 동시 처리
            indexed = [(idx + 1, m) for idx, m in enumerate(movies)
                       if m['title'] not in already_done]
            for batch_start in range(0, len(indexed), CONCURRENT_MOVIES):
                if should_stop.is_set():
                    break
                batch = indexed[batch_start:batch_start + CONCURRENT_MOVIES]
                tasks = [
                    process_movie(session, movie, idx, len(movies))
                    for idx, movie in batch
                ]
                await asyncio.gather(*tasks)
                await asyncio.sleep(random.uniform(0.2, 0.4))

    return total_saved

def run_crawling(start_year, end_year, username, password):
    threading.Thread(
        target=lambda: [p("엔터 키를 누르면 안전하게 종료됩니다.\n"), input(), should_stop.set()],
        daemon=True
    ).start()

    browser     = start_browser()
    cookie_list = load_cookie_list()
    load_cookies(browser)

    total_saved = 0
    try:
        total_saved = asyncio.run(_run_async(start_year, end_year, browser, cookie_list))
    finally:
        try:
            browser.quit()
        except:
            pass
        p(f"\n종료. 총 {total_saved}개 리뷰 저장 완료.")
        p("다음 실행 시 이어서 수집됩니다.")


if __name__ == "__main__":
    p("Letterboxd 리뷰 크롤링 시작\n")
    run_crawling(1960, 2026, username=LB_USERNAME, password=LB_PASSWORD)
