@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM 🛡️ AEGIS THREAT DETECTION SYSTEM - Windows Batch Startup
REM ============================================================
REM Fully Automated Setup - Handles everything from A to Z
REM Double-click this file to start Aegis
REM ============================================================

title Aegis Threat Detection System - Automated Setup

REM Get script directory
cd /d "%~dp0"

REM Setup logging
set "LOG_FILE=%~dp0batch_log.txt"
echo ============================================================ >"%LOG_FILE%"
echo    AEGIS THREAT DETECTION SYSTEM >>"%LOG_FILE%"
echo ============================================================ >>"%LOG_FILE%"
echo    Fully Automated Setup ^& Launch >>"%LOG_FILE%"
echo ============================================================ >>"%LOG_FILE%"
echo. >>"%LOG_FILE%"
echo Log started at: %date% %time% >>"%LOG_FILE%"
echo Working directory: %CD% >>"%LOG_FILE%"
echo. >>"%LOG_FILE%"

echo.
echo ============================================================
echo    AEGIS THREAT DETECTION SYSTEM
echo ============================================================
echo    Fully Automated Setup ^& Launch
echo ============================================================
echo.

echo [1/7] Checking prerequisites...
echo [1/7] Checking prerequisites... >>"%LOG_FILE%"
echo.

REM Check Python
echo Checking Python...
python --version >>"%LOG_FILE%" 2>&1
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python not found! Please install Python 3.10+
    echo   ERROR: Python not found! Please install Python 3.10+ >>"%LOG_FILE%"
    echo   Download from: https://www.python.org/downloads/
    echo   Download from: https://www.python.org/downloads/ >>"%LOG_FILE%"
    echo.
    echo. >>"%LOG_FILE%"
    pause
    exit /b 1
)
echo   OK: Python found
echo   OK: Python found >>"%LOG_FILE%"
echo.

REM Check Node.js
echo Checking Node.js...
node --version >>"%LOG_FILE%" 2>&1
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Node.js not found! Please install Node.js 18+
    echo   ERROR: Node.js not found! Please install Node.js 18+ >>"%LOG_FILE%"
    echo   Download from: https://nodejs.org/
    echo   Download from: https://nodejs.org/ >>"%LOG_FILE%"
    echo.
    echo. >>"%LOG_FILE%"
    pause
    exit /b 1
)
echo   OK: Node.js found
echo   OK: Node.js found >>"%LOG_FILE%"
echo.

REM Check npm
echo Checking npm...
echo Checking npm... >>"%LOG_FILE%"
where npm >nul 2>&1
if errorlevel 1 (
    echo   ERROR: npm not found!
    echo   ERROR: npm not found! >>"%LOG_FILE%"
    echo.
    echo. >>"%LOG_FILE%"
    pause
    exit /b 1
)
for /f %%i in ('npm --version 2^>nul') do set NPM_VERSION=%%i
echo   OK: npm found (version: !NPM_VERSION!)
echo   OK: npm found (version: !NPM_VERSION!) >>"%LOG_FILE%"
echo.
echo. >>"%LOG_FILE%"

echo [2/7] Setting up virtual environment...
echo [2/7] Setting up virtual environment... >>"%LOG_FILE%"
echo.

if not exist ".venv" (
    echo   Creating virtual environment...
    echo   Creating virtual environment... >>"%LOG_FILE%"
    python -m venv .venv >>"%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo   ERROR: Failed to create virtual environment
        echo   ERROR: Failed to create virtual environment >>"%LOG_FILE%"
        echo   Please ensure Python is properly installed and in PATH
        echo   Please ensure Python is properly installed and in PATH >>"%LOG_FILE%"
        pause
        exit /b 1
    )
    if not exist ".venv\Scripts\python.exe" (
        echo   ERROR: Virtual environment created but activation script missing
        echo   ERROR: Virtual environment created but activation script missing >>"%LOG_FILE%"
        pause
        exit /b 1
    )
    echo   OK: Virtual environment created
    echo   OK: Virtual environment created >>"%LOG_FILE%"
) else (
    echo   OK: Virtual environment exists
    echo   OK: Virtual environment exists >>"%LOG_FILE%"
)

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo   Activating virtual environment...
    echo   Activating virtual environment... >>"%LOG_FILE%"
    call .venv\Scripts\activate.bat >>"%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo   WARNING: Virtual environment activation returned error, continuing...
        echo   WARNING: Virtual environment activation returned error, continuing... >>"%LOG_FILE%"
    ) else (
        echo   OK: Virtual environment activated
        echo   OK: Virtual environment activated >>"%LOG_FILE%"
    )
) else (
    echo   ERROR: Activation script not found at .venv\Scripts\activate.bat
    echo   ERROR: Activation script not found at .venv\Scripts\activate.bat >>"%LOG_FILE%"
    pause
    exit /b 1
)
echo.

echo [3/7] Upgrading pip...
echo [3/7] Upgrading pip... >>"%LOG_FILE%"
echo.
python -m pip install --upgrade pip --quiet >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo   WARNING: pip upgrade failed, trying without quiet mode...
    echo   WARNING: pip upgrade failed, trying without quiet mode... >>"%LOG_FILE%"
    python -m pip install --upgrade pip >>"%LOG_FILE%" 2>&1
)
echo   OK: pip upgraded
echo   OK: pip upgraded >>"%LOG_FILE%"
echo.

echo [4/7] Installing Python dependencies...
echo [4/7] Installing Python dependencies... >>"%LOG_FILE%"
echo   This may take 5-10 minutes on first run...
echo   This may take 5-10 minutes on first run... >>"%LOG_FILE%"
echo.
if not exist "requirements.txt" (
    echo   ERROR: requirements.txt not found!
    echo   ERROR: requirements.txt not found! >>"%LOG_FILE%"
    pause
    exit /b 1
)
python -m pip install -r requirements.txt >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo   ERROR: Failed to install Python dependencies
    echo   ERROR: Failed to install Python dependencies >>"%LOG_FILE%"
    echo   Please check your internet connection and try again
    echo   Please check your internet connection and try again >>"%LOG_FILE%"
    echo   See %LOG_FILE% for details
    pause
    exit /b 1
)
echo   OK: Python dependencies installed
echo   OK: Python dependencies installed >>"%LOG_FILE%"
echo.

echo [5/7] Installing frontend dependencies...
echo [5/7] Installing frontend dependencies... >>"%LOG_FILE%"
echo.
if not exist "frontend" (
    echo   ERROR: frontend directory not found!
    echo   ERROR: frontend directory not found! >>"%LOG_FILE%"
    pause
    exit /b 1
)

REM Store current directory and log file path before cd
set "ORIG_DIR=%CD%"
set "FRONTEND_DIR=%CD%\frontend"

cd /d "%FRONTEND_DIR%"
if errorlevel 1 (
    echo   ERROR: Could not change to frontend directory
    echo   ERROR: Could not change to frontend directory >>"%LOG_FILE%"
    pause
    exit /b 1
)
echo   Changed to frontend directory
echo   Changed to frontend directory >>"%LOG_FILE%"

if not exist "package.json" (
    echo   ERROR: package.json not found in frontend directory!
    echo   ERROR: package.json not found in frontend directory! >>"%LOG_FILE%"
    cd /d "%ORIG_DIR%"
    pause
    exit /b 1
)

echo   Checking for node_modules...
echo   Checking for node_modules... >>"%LOG_FILE%"

if exist "node_modules" goto :npm_exists

REM node_modules does not exist, install it
echo   Installing npm packages (first time - may take 2-3 minutes)...
echo   Installing npm packages (first time - may take 2-3 minutes)... >>"%LOG_FILE%"
call npm install >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo   ERROR: Failed to install frontend dependencies
    echo   ERROR: Failed to install frontend dependencies >>"%LOG_FILE%"
    echo   See %LOG_FILE% for details
    cd /d "%ORIG_DIR%"
    pause
    exit /b 1
)
echo   OK: Frontend dependencies installed
echo   OK: Frontend dependencies installed >>"%LOG_FILE%"
goto :npm_done

:npm_exists
echo   OK: Frontend dependencies already installed (node_modules found)
echo   OK: Frontend dependencies already installed (node_modules found) >>"%LOG_FILE%"

:npm_done

echo   Changing back to project root...
echo   Changing back to project root... >>"%LOG_FILE%"
cd /d "%ORIG_DIR%"
if errorlevel 1 (
    echo   WARNING: Could not return to original directory
    echo   WARNING: Could not return to original directory >>"%LOG_FILE%"
) else (
    echo   OK: Returned to project root
    echo   OK: Returned to project root >>"%LOG_FILE%"
)
echo.
echo. >>"%LOG_FILE%"

echo [6/7] Preparing ports...
echo [6/7] Preparing ports... >>"%LOG_FILE%"
echo.

REM Kill process on port 8000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >>"%LOG_FILE%" 2>&1
)
echo   OK: Port 8000 cleared
echo   OK: Port 8000 cleared >>"%LOG_FILE%"

REM Kill process on port 3000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >>"%LOG_FILE%" 2>&1
)
echo   OK: Port 3000 cleared
echo   OK: Port 3000 cleared >>"%LOG_FILE%"
echo.

timeout /t 2 /nobreak >nul

echo [7/7] Starting Aegis services...
echo [7/7] Starting Aegis services... >>"%LOG_FILE%"
echo.

REM Start Backend API in new window
echo   Starting Backend API on port 8000...
echo   Starting Backend API on port 8000... >>"%LOG_FILE%"
start "Aegis Backend API" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && echo Aegis Backend API Starting... && echo ============================================================ && python -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload && pause"

timeout /t 4 /nobreak >nul

REM Start Frontend in new window
echo   Starting Frontend on port 3000...
echo   Starting Frontend on port 3000... >>"%LOG_FILE%"
start "Aegis Frontend" cmd /k "cd /d "%~dp0frontend" && echo Aegis Frontend Starting... && echo ============================================================ && npm run dev && pause"

timeout /t 6 /nobreak >nul

echo.
echo ============================================================
echo    AEGIS IS STARTING!
echo ============================================================
echo.
echo    Dashboard:     http://localhost:3000
echo    API Docs:      http://localhost:8000/docs
echo    Health Check:  http://localhost:8000/health
echo.
echo    Test Credentials:
echo    Email:    demo@aegis.com
echo    Password: demo123
echo.
echo    Services are starting in separate windows...
echo    Please wait 10-15 seconds for full startup.
echo.
echo    Full log saved to: %LOG_FILE%
echo ============================================================
echo.

echo. >>"%LOG_FILE%"
echo ============================================================ >>"%LOG_FILE%"
echo    AEGIS IS STARTING! >>"%LOG_FILE%"
echo ============================================================ >>"%LOG_FILE%"

set /p OPEN_BROWSER="Open dashboard in browser? (Y/n): "
if /i not "%OPEN_BROWSER%"=="n" (
    timeout /t 5 /nobreak >nul
    start http://localhost:3000
    echo   Browser opened!
    echo   Browser opened! >>"%LOG_FILE%"
)

echo.
echo Press any key to close this window (services will keep running)...
echo Log file: %LOG_FILE%
pause >nul
exit /b 0
