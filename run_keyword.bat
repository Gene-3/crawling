@echo off
chcp 65001 > nul
echo 영화 키워드 추천 서비스를 시작합니다...
cd /d "%~dp0"
call conda activate aiservice26
streamlit run app_keyword.py
pause
