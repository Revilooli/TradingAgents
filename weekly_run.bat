@echo off
REM Double-click launcher for weekly_scan.py.
REM Anchors to %~dp0 so it runs from the project root regardless of where it's launched;
REM pauses at the end so output and tracebacks stay visible.
cd /d "%~dp0"
python weekly_scan.py
pause
