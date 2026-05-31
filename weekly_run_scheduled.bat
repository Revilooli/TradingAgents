@echo off
REM Non-interactive launcher for Windows Task Scheduler.
REM Differs from weekly_run.bat: no pause, output appended to reports\schedule.log
REM so you can see what happened the next time you check.
cd /d "%~dp0"
if not exist "reports" mkdir "reports"
python weekly_scan.py >> "reports\schedule.log" 2>&1
