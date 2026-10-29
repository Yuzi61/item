@echo off
REM ============================================================================
REM  RHyperLab launcher  (Anaconda Python 3.7)
REM
REM  How to use:
REM    1. Double-click this file. Keep the window OPEN while using the app
REM       (closing the window stops the server).
REM    2. First start takes about 10-15 seconds to load libraries, then the
REM       browser opens automatically at http://127.0.0.1:8050/
REM    3. If the browser says "cannot connect", wait a moment and press F5.
REM
REM  Interpreter: C:\ProgramData\Anaconda3\python.exe  (Python 3.7.0)
REM
REM  NOTE: keep this file ASCII-only (English). Chinese text in a .bat gets
REM        corrupted by code-page issues -- the previous version was lost
REM        exactly this way. Do NOT translate the echo lines below.
REM ============================================================================
setlocal
set TF_ENABLE_ONEDNN_OPTS=0
cd /d "%~dp0"

set "PYEXE=C:\ProgramData\Anaconda3\python.exe"
set "PORT=8050"

if not exist "%PYEXE%" goto nopython

REM ---- dependency self-check (imports 3 light packages only) ----
"%PYEXE%" -c "import dash, dash_bootstrap_components, plotly" >nul 2>&1
if errorlevel 1 goto nodeps

REM ---- port-occupied self-check ----
netstat -ano | findstr "LISTENING" | findstr ":%PORT% " >nul 2>&1
if not errorlevel 1 goto portbusy

echo.
echo   Python : %PYEXE%
echo   URL    : http://127.0.0.1:%PORT%/
echo.
echo   Starting server... first start takes 10-15 seconds.
echo   The browser will open automatically when ready.
echo   If the page says "cannot connect", wait a moment and press F5.
echo   To stop the app: close this window (or press Ctrl+C).
echo.

"%PYEXE%" index.py

echo.
echo   Server stopped. If there is any red error text above, take a screenshot.
pause
exit /b 0

:nopython
echo.
echo [ERROR] Python interpreter not found:
echo         %PYEXE%
echo         Check that Anaconda is installed, or edit the PYEXE path in this file.
echo.
pause
exit /b 1

:nodeps
echo.
echo [ERROR] Missing packages (dash / dash-bootstrap-components / plotly).
echo         Run this in a terminal:
echo           "%PYEXE%" -m pip install -r requirements.txt
echo.
pause
exit /b 1

:portbusy
echo.
echo [ERROR] Port %PORT% is already in use. Cannot start.
echo.
echo         Maybe another instance is already running. Just open:
echo           http://127.0.0.1:%PORT%/
echo.
echo         To free the port, find the PID then kill it:
echo           netstat -ano ^| findstr ":%PORT% "
echo           taskkill /PID ^<the PID you found^> /F
echo.
pause
exit /b 1
