@echo off
REM Double-click launcher for portfolio_scan.py.
REM cd to script dir so relative paths resolve; pause at end so the window
REM stays open long enough to read output and tracebacks.
cd /d "%~dp0"
python portfolio_scan.py
pause
