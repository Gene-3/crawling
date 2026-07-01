@echo off
chcp 65001 > nul
echo 영화 키워드 추천 서비스를 시작합니다...
cd /d "%~dp0"

:: Anaconda 기본 경로 (사용자 폴더 기준)
set CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat
if not exist "%CONDA_ACTIVATE%" (
    echo [오류] Anaconda를 찾을 수 없습니다.
    echo       직접 실행: conda activate aiservice26 ^&^& streamlit run app_keyword.py
    pause
    exit /b 1
)

call "%CONDA_ACTIVATE%" aiservice26
streamlit run app_keyword.py
pause
