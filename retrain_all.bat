@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate
set ERR=0
python scripts\train_model.py
if errorlevel 1 set ERR=1 & goto end
python scripts\backtest_bt.py
if errorlevel 1 set ERR=1 & goto end
python scripts\backtest_ml.py
if errorlevel 1 set ERR=1 & goto end
python scripts\predict_signal.py
:end
pause
endlocal & exit /b %ERR%
