@echo off
rem Double-click to install PeakSift into this folder (nothing system-wide).
rem Needs Python 3.11 or 3.12. Safe to run again: it only adds what's missing.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
pause
