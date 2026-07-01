@echo off
chcp 65001 > nul
echo 영화 의미 검색 서비스를 시작합니다...
echo [주의] 이 서비스는 CUDA GPU 및 review_texts.db(12GB) 가 필요합니다.
cd /d "%~dp0"

:: Anaconda 기본 경로 (사용자 폴더 기준)
set CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat
if not exist "%CONDA_ACTIVATE%" (
    echo [오류] Anaconda를 찾을 수 없습니다.
    echo       직접 실행: conda activate aiservice26 ^&^& streamlit run app.py
    pause
    exit /b 1
)

call "%CONDA_ACTIVATE%" aiservice26
streamlit run app.py
pause
