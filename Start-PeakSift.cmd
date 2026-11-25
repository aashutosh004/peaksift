@echo off
rem Double-click to start the PeakSift web app. It opens http://localhost:8000
rem in your browser. Keep this window open while you use PeakSift; close it
rem (or press Ctrl+C) to stop.
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo PeakSift isn't installed yet. Double-click Setup-PeakSift.cmd first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m peaksift serve
pause
