@echo off
REM ==========================================================================
REM  Chay tu dong toan bo: LiDAR -> ban do 2.5D -> test case -> planner -> REPORT.md
REM  Cach dung (double-click hoac tu cmd):
REM     run_auto.bat                       -> chay mac dinh ten "rellis_full"
REM     run_auto.bat ten_lan_chay [tham so them cho auto_pipeline.py]
REM  Vi du:
REM     run_auto.bat sample_quick --download-sample --seqs 00000 --stride 1 --per-map 3 --workers 4
REM  Bi tat giua chung? Chay lai dung lenh cu, se tu tiep tuc.
REM  Ket qua: "logs\testsuite\<ten>\REPORT.md"
REM ==========================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set NAME=%1
if "%NAME%"=="" set NAME=rellis_full
shift

set EXTRA=
:collect
if "%~1"=="" goto run
set EXTRA=%EXTRA% %1
shift
goto collect

:run
python -c "import numpy, scipy, matplotlib" 2>nul || python -m pip install -q -r requirements.txt
python auto_pipeline.py --name %NAME% %EXTRA%
echo.
echo Xong (exit %ERRORLEVEL%). Bao cao: "%~dp0logs\testsuite\%NAME%\REPORT.md"
pause
