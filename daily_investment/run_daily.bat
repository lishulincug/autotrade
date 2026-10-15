@echo off
cd /d "%~dp0"
python daily_analysis.py
exit /b %errorlevel%
