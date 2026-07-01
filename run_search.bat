@echo off
chcp 65001 > nul
echo 영화 의미 검색 서비스를 시작합니다...
cd /d "%~dp0"
call conda activate aiservice26
streamlit run app.py
pause
