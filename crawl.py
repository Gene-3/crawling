"""
IMDB 영화 리뷰 크롤러
- 연도별 상위 영화의 사용자 리뷰를 수집하여 CSV로 저장
- Amazon 계정으로 IMDB 로그인 후 Selenium으로 동적 페이지 크롤링
"""

import csv, io, json, os, random, shutil, threading, time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ─── 기본 설정 ───────────────────────────────────────────
AMAZON_EMAIL    = "id"
AMAZON_PASSWORD = "pwd"
MOVIES_PER_YEAR = 200
RETRY_COUNT     = 3
SCRIPT_FOLDER   = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE       = os.path.join(SCRIPT_FOLDER, "imdb_reviews_2014_to_2026.csv")
LOGIN_COOKIE    = os.path.join(SCRIPT_FOLDER, "imdb_cookies.json")
should_stop     = threading.Event()
# ────────────────────────────────────────────────────────


def p(msg): print(msg, flush=True)


# ── 브라우저 ─────────────────────────────────────────────
def start_browser(show_browser=False):
    options = Options()
    if not show_browser:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1280,800")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


# ── 로그인 ───────────────────────────────────────────────
def is_logged_in(browser):
    cookie_names = [c['name'] for c in browser.get_cookies()]
    return any(name in cookie_names for name in ['ubid-main', 'at-main', 'sess-at-main'])

def save_cookies(browser):
    with open(LOGIN_COOKIE, 'w') as f:
        json.dump(browser.get_cookies(), f)
    p("  쿠키 저장 완료")

def load_cookies(browser):
    """저장된 쿠키로 로그인 시도. 성공 시 True 반환"""
    if not os.path.exists(LOGIN_COOKIE):
        return False
    browser.get("https://www.imdb.com")
    time.sleep(2)
    with open(LOGIN_COOKIE) as f:
        for cookie in json.load(f):
            try:
                browser.add_cookie({**{k: v for k, v in cookie.items() if k not in ['expiry', 'sameSite']}, 'domain': '.imdb.com'})
            except:
                continue
    browser.refresh()
    time.sleep(3)
    if is_logged_in(browser):
        p("쿠키로 로그인 완료\n")
        return True
    p("쿠키 만료. 재로그인 시도...")
    return False

def auto_login(email, password):
    """Amazon 계정으로 IMDB 자동 로그인"""
    p("자동 로그인 중...")
    browser = start_browser(show_browser=True)
    browser.get("https://www.imdb.com/registration/signin/")
    time.sleep(random.uniform(3, 4))
    try:
        for selector in ['a[href*="amazon"]', '//a[contains(text(), "Amazon")]']:
            try:
                by = By.XPATH if selector.startswith('//') else By.CSS_SELECTOR
                WebDriverWait(browser, 5).until(EC.element_to_be_clickable((by, selector))).click()
                time.sleep(random.uniform(2, 3))
                break
            except:
                continue
        WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, 'ap_email'))).send_keys(email)
        try:
            browser.find_element(By.ID, 'continue').click()
            time.sleep(random.uniform(1, 2))
        except:
            pass
        WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.ID, 'ap_password'))).send_keys(password)
        browser.find_element(By.ID, 'signInSubmit').click()
        time.sleep(random.uniform(3, 4))
        if 'cvf' in browser.current_url or 'puzzle' in browser.page_source.lower():
            p("\n퍼즐 감지! 브라우저 창에서 직접 풀어주세요.")
            input("퍼즐 완료 후 엔터를 누르세요...")
        if is_logged_in(browser):
            save_cookies(browser)
            p("로그인 완료\n")
            browser.quit()
            return True
        p("로그인 실패.")
        browser.quit()
        return False
    except Exception as e:
        p(f"로그인 오류: {e}")
        browser.save_screenshot(os.path.join(SCRIPT_FOLDER, "login_error.png"))
        browser.quit()
        return False

def login(browser, email, password):
    """쿠키 로그인 먼저 시도, 실패 시 자동 로그인"""
    if load_cookies(browser):
        return
    for i in range(1, RETRY_COUNT + 1):
        p(f"재로그인 시도 ({i}/{RETRY_COUNT})...")
        if auto_login(email, password) and load_cookies(browser):
            return
        time.sleep(random.uniform(3, 5))
    raise Exception("로그인 실패. imdb_cookies.json을 수동으로 갱신해주세요.")


# ── 공통 유틸 ────────────────────────────────────────────
def click_more_button(browser, max_count=None):
    """더보기 버튼 반복 클릭 (max_count: 영화 목록용 / 없으면 리뷰 끝까지)"""
    last_count = 0
    while not should_stop.is_set():
        try:
            btn = WebDriverWait(browser, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.ipc-see-more__button'))
            )
            if 'more' not in btn.text.lower():
                break
            if max_count:
                page    = BeautifulSoup(browser.page_source, 'html.parser')
                current = len(page.find_all('li', class_='ipc-metadata-list-summary-item'))
                if current >= max_count:
                    break
                p(f"  -> 영화 목록 {current + 50}개 로드 중...")
                time.sleep(random.uniform(0.7, 1.2))
            else:
                browser.execute_script("arguments[0].click();", btn)
                time.sleep(random.uniform(0.7, 1.2))
                count = browser.execute_script("return document.querySelectorAll('article.user-review-item').length")
                if count // 100 > last_count // 100:
                    p(f"    리뷰 {count}개 로드 중...")
                    last_count = count
                continue
            browser.execute_script("arguments[0].click();", btn)
        except:
            break

def restart_browser(browser, email, password):
    """브라우저 재시작 후 재로그인"""
    try:
        browser.quit()
    except:
        pass
    new_browser = start_browser(show_browser=False)
    login(new_browser, email, password)
    return new_browser


# ── 데이터 수집 ──────────────────────────────────────────
def get_movie_list(browser, year):
    """해당 연도 상위 영화 목록 수집"""
    url = (f"https://www.imdb.com/search/title/"
           f"?title_type=feature&release_date={year}-01-01,{year}-12-31&sort=num_votes,desc")
    browser.get(url)
    time.sleep(random.uniform(2, 3))
    click_more_button(browser, max_count=MOVIES_PER_YEAR)
    page  = BeautifulSoup(browser.page_source, 'html.parser')
    rows  = page.find_all('li', class_='ipc-metadata-list-summary-item')[:MOVIES_PER_YEAR]
    p(f"  -> 총 {len(rows)}개 영화 로드 완료")
    movie_list = []
    for row in rows:
        try:
            title    = row.find('h3', class_='ipc-title__text').text.strip()
            if '. ' in title:
                title = title.split('. ', 1)[1]
            link     = row.find('a', class_='ipc-title-link-wrapper')['href']
            movie_id = link.split('/')[2].split('?')[0]
            movie_list.append({'title': title, 'movie_id': movie_id})
        except:
            continue
    return movie_list

def collect_reviews(browser, movie_id, movie_title, year):
    """영화 한 편의 모든 리뷰 수집"""
    review_url = f"https://www.imdb.com/title/{movie_id}/reviews/"
    browser.get(review_url)
    try:
        WebDriverWait(browser, 8).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'article.user-review-item')))
    except:
        time.sleep(random.uniform(2, 3))
    if not is_logged_in(browser):
        p("    세션 만료. 재로그인 중...")
        login(browser, AMAZON_EMAIL, AMAZON_PASSWORD)
        browser.get(review_url)
        time.sleep(random.uniform(2, 3))
    click_more_button(browser)
    page        = BeautifulSoup(browser.page_source, 'html.parser')
    review_list = []
    for box in page.find_all('article', class_='user-review-item'):
        try:
            rating  = box.find('span', class_='ipc-rating-star--rating')
            content = box.find('div',  class_='ipc-html-content-inner-div')
            if not content:
                continue
            text = content.get_text(separator=' ').strip()
            if len(text) < 10:
                continue
            review_list.append({'Year': year, 'Title': movie_title,
                                 'Review_Rating': rating.text.strip() if rating else 'N/A',
                                 'Review_Text': text})
        except:
            continue
    return review_list

def backup_csv(total_count):
    """5만 개 단위로 CSV 백업"""
    milestone   = (total_count // 50000) * 50000
    backup_file = os.path.join(SCRIPT_FOLDER, f"imdb_reviews_backup_{milestone:07d}.csv")
    if not os.path.exists(backup_file):
        shutil.copy2(SAVE_FILE, backup_file)
        p(f"  [백업] {milestone:,}개 달성 → {os.path.basename(backup_file)} 생성")

def save_reviews(review_list):
    """리뷰 데이터를 CSV에 추가 저장 (줄바꿈 제거 + 중복 제거)"""
    if not review_list:
        return
    df = pd.DataFrame(review_list)
    for col in ['Review_Text', 'Title']:
        df[col] = df[col].str.replace(r'[\r\n\t\x85\x0b\x0c]+', ' ', regex=True).str.strip()
    df = df.drop_duplicates(subset=['Title', 'Review_Text'])
    is_new_file = not os.path.exists(SAVE_FILE)
    tmp_file    = SAVE_FILE + '.tmp'
    # 메모리 직렬화 → tmp 저장(fsync) → 메인 파일 반영
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
def run_crawling(start_year, end_year, email, password):
    threading.Thread(
        target=lambda: [p("엔터 키를 누르면 안전하게 종료됩니다.\n"), input(), should_stop.set()],
        daemon=True
    ).start()

    browser     = start_browser(show_browser=False)
    total_saved = 0
    login(browser, email, password)

    already_done  = set()
    initial_count = 0
    if os.path.exists(SAVE_FILE):
        existing      = pd.read_csv(SAVE_FILE, encoding='utf-8-sig', usecols=['Title'], on_bad_lines='skip')
        already_done  = set(existing['Title'].unique())
        initial_count = len(existing)
        p(f"기존 수집: {len(already_done)}편 / {initial_count:,}개 건너뜀\n")

    try:
        for year in range(start_year, end_year + 1):
            if should_stop.is_set():
                break
            p(f"\n[{year}년] 영화 목록 수집 중...")
            movies = []
            for i in range(1, RETRY_COUNT + 1):
                if should_stop.is_set():
                    break
                try:
                    movies = get_movie_list(browser, year)
                    break
                except Exception as e:
                    p(f"  -> 목록 수집 실패 ({i}/{RETRY_COUNT}): {e}")
                    if i < RETRY_COUNT:
                        browser = restart_browser(browser, email, password)
                        time.sleep(random.uniform(5, 10))

            for idx, movie in enumerate(movies, 1):
                if should_stop.is_set():
                    break
                title, movie_id = movie['title'], movie['movie_id']
                if title in already_done:
                    p(f"  [{idx}/{len(movies)}] '{title}' - 이미 수집됨, 건너뜀")
                    continue
                p(f"  [{idx}/{len(movies)}] '{title}' 수집 중... (이번 세션: {total_saved}개)")
                for i in range(1, RETRY_COUNT + 1):
                    if should_stop.is_set():
                        break
                    try:
                        reviews       = collect_reviews(browser, movie_id, title, year)
                        save_reviews(reviews)
                        prev_total    = initial_count + total_saved
                        total_saved  += len(reviews)
                        total_in_file = initial_count + total_saved
                        already_done.add(title)
                        p(f"    -> {len(reviews)}개 저장 완료 (이번 세션: {total_saved}개 / 전체: {total_in_file:,}개)")
                        if total_in_file // 50000 > prev_total // 50000:
                            backup_csv(total_in_file)
                        break
                    except Exception as e:
                        p(f"    -> 수집 실패 ({i}/{RETRY_COUNT}): {e}")
                        if i == RETRY_COUNT:
                            p(f"    -> '{title}' 건너뜀")
                            already_done.add(title)
                        else:
                            browser = restart_browser(browser, email, password)
                            time.sleep(random.uniform(5, 10))
                time.sleep(random.uniform(1, 2))

    finally:
        try:
            browser.quit()
        except:
            pass
        p(f"\n종료. 총 {total_saved}개 리뷰 저장 완료.")
        p("다음 실행 시 이어서 수집됩니다.")


if __name__ == "__main__":
    p("IMDB 리뷰 크롤링 시작\n")
    run_crawling(2014, 2026, email=AMAZON_EMAIL, password=AMAZON_PASSWORD)
