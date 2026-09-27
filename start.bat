@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 (
    python "%~dp0start.py" %*
    goto finished
)
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 (
    py -3 "%~dp0start.py" %*
    goto finished
)
echo [ERROR] Python 3.11+ was not found. Install Python and add it to PATH.
set "result=1"
goto close
:finished
set "result=%errorlevel%"
:close
if "%~1"=="" pause
exit /b %result%
