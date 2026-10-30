@echo off
rem Double-click installer for Windows 10/11. If Windows SmartScreen says
rem "Windows protected your PC": click "More info", then "Run anyway".
title Installing Fourier Analyzer
echo Installing Fourier Analyzer. The first install downloads about 2 GB
echo (Python, the separation models) and takes 5-15 minutes.
echo.
if exist "%~dp0install.ps1" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $ProgressPreference='SilentlyContinue'; iex ((New-Object Net.WebClient).DownloadString('https://github.com/dylancleverdon/fourier-analyzer/releases/latest/download/install.ps1'))"
)
echo.
pause
