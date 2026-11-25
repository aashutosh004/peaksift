@echo off
rem PeakSift command line from this folder:  .\peaksift clip "<link>"   .\peaksift --help
rem Runs through python -m rather than an .exe launcher, which Windows Smart App
rem Control blocks because it isn't signed.
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run Setup-PeakSift.cmd first.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m peaksift %*
