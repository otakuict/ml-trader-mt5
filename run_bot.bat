@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python scripts\run_forever.py
pause
