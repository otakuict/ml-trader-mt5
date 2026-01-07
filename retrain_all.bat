@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate
python scripts\train_model.py
if errorlevel 1 exit /b 1
python scripts\backtest_bt.py
if errorlevel 1 exit /b 1
python scripts\predict_signal.py
endlocal
