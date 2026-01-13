# ============================================================
# 🛡️ AEGIS THREAT DETECTION SYSTEM - Windows PowerShell Startup
# ============================================================
# Fully Automated Setup - Handles everything from A to Z
# Usage: Right-click and "Run with PowerShell" or run in terminal:
#        .\start_aegis.ps1
# ============================================================

$ErrorActionPreference = "Continue"

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

# Colors for output
function Write-Banner {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "   🛡️  AEGIS THREAT DETECTION SYSTEM" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "   Fully Automated Setup & Launch" -ForegroundColor White
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step($step, $message) {
    Write-Host "[$step] $message" -ForegroundColor Yellow
}

function Write-Success($message) {
    Write-Host "  ✅ $message" -ForegroundColor Green
}

function Write-Error($message) {
    Write-Host "  ❌ $message" -ForegroundColor Red
}

function Write-Info($message) {
    Write-Host "  ℹ️  $message" -ForegroundColor Gray
}

Write-Banner

# ============================================================
# Step 1: Check Prerequisites
# ============================================================
Write-Step "1/7" "Checking prerequisites..."

$PREREQS_OK = $true

# Check Python
$pythonCmd = $null
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0 -or $pythonVersion -match "Python") {
        $pythonCmd = "python"
        Write-Success "Python found: $pythonVersion"
        
        # Check Python version
        $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
        if ($versionMatch) {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
                Write-Error "Python 3.10+ required. Found: $pythonVersion"
                $PREREQS_OK = $false
            }
        }
    }
} catch {
    Write-Error "Python not found!"
    $PREREQS_OK = $false
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    if ($LASTEXITCODE -eq 0 -or $nodeVersion -match "v\d") {
        Write-Success "Node.js found: $nodeVersion"
        
        # Check Node version
        $versionMatch = $nodeVersion -match "v(\d+)"
        if ($versionMatch) {
            $major = [int]$Matches[1]
            if ($major -lt 18) {
                Write-Error "Node.js 18+ required. Found: $nodeVersion"
                $PREREQS_OK = $false
            }
        }
    }
} catch {
    Write-Error "Node.js not found!"
    $PREREQS_OK = $false
}

# Check npm
try {
    $npmVersion = npm --version 2>&1
    if ($LASTEXITCODE -eq 0 -or $npmVersion -match "\d") {
        Write-Success "npm found: v$npmVersion"
    }
} catch {
    Write-Error "npm not found!"
    $PREREQS_OK = $false
}

if (-not $PREREQS_OK) {
    Write-Host ""
    Write-Error "Missing prerequisites! Please install:"
    Write-Info "  • Python 3.10+: https://www.python.org/downloads/"
    Write-Info "  • Node.js 18+: https://nodejs.org/"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# ============================================================
# Step 2: Setup Virtual Environment
# ============================================================
Write-Step "2/7" "Setting up Python virtual environment..."

$VENV_PATH = Join-Path $SCRIPT_DIR ".venv"

if (-not (Test-Path $VENV_PATH)) {
    Write-Info "Creating virtual environment (this may take a moment)..."
    & $pythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment"
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Success "Virtual environment created"
} else {
    Write-Success "Virtual environment exists"
}

# Activate virtual environment
$activateScript = Join-Path $VENV_PATH "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    try {
        & $activateScript
        Write-Success "Virtual environment activated"
    } catch {
        Write-Error "Could not activate virtual environment. Trying alternative method..."
        $env:VIRTUAL_ENV = $VENV_PATH
        $env:PATH = "$VENV_PATH\Scripts;$env:PATH"
    }
} else {
    Write-Error "Activation script not found"
    exit 1
}

# ============================================================
# Step 3: Upgrade pip
# ============================================================
Write-Step "3/7" "Upgrading pip..."

$pythonExe = Join-Path $VENV_PATH "Scripts\python.exe"
& $pythonExe -m pip install --upgrade pip --quiet 2>$null
Write-Success "pip upgraded"

# ============================================================
# Step 4: Install Python Dependencies
# ============================================================
Write-Step "4/7" "Installing Python dependencies..."

if (Test-Path "requirements.txt") {
    Write-Info "Installing packages from requirements.txt (this may take 5-10 minutes)..."
    & $pythonExe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Python dependencies"
        Write-Info "Retrying with verbose output..."
        & $pythonExe -m pip install -r requirements.txt --verbose
    } else {
        Write-Success "Python dependencies installed"
    }
} else {
    Write-Error "requirements.txt not found!"
    exit 1
}

# Verify critical packages
Write-Info "Verifying critical packages..."
$criticalPackages = @("fastapi", "uvicorn", "tensorflow", "scikit-learn", "pandas", "numpy")
$allInstalled = $true
foreach ($pkg in $criticalPackages) {
    $installed = & $pythonExe -m pip show $pkg 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Missing package: $pkg"
        $allInstalled = $false
    }
}
if ($allInstalled) {
    Write-Success "All critical packages verified"
}

# ============================================================
# Step 5: Install Frontend Dependencies
# ============================================================
Write-Step "5/7" "Installing frontend dependencies..."

$frontendPath = Join-Path $SCRIPT_DIR "frontend"
if (Test-Path $frontendPath) {
    Push-Location $frontendPath
    
    if (-not (Test-Path "node_modules")) {
        Write-Info "Installing npm packages (first time - may take 2-3 minutes)..."
        npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install frontend dependencies"
            Pop-Location
            exit 1
        }
        Write-Success "Frontend dependencies installed"
    } else {
        Write-Success "Frontend dependencies already installed"
    }
    
    Pop-Location
} else {
    Write-Error "Frontend directory not found!"
    exit 1
}

# ============================================================
# Step 6: Prepare Ports
# ============================================================
Write-Step "6/7" "Preparing ports..."

# Kill process on port 8000
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($port8000) {
    Write-Info "Clearing port 8000..."
    $port8000 | ForEach-Object { 
        try {
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        } catch { }
    }
    Start-Sleep -Seconds 2
}
Write-Success "Port 8000 ready"

# Kill process on port 3000
$port3000 = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if ($port3000) {
    Write-Info "Clearing port 3000..."
    $port3000 | ForEach-Object { 
        try {
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        } catch { }
    }
    Start-Sleep -Seconds 2
}
Write-Success "Port 3000 ready"

# ============================================================
# Step 7: Start Services
# ============================================================
Write-Step "7/7" "Starting Aegis services..."

# Start Backend API in new window
Write-Info "Starting Backend API on port 8000..."
$backendCmd = @"
`$ErrorActionPreference = 'Continue'
Set-Location '$SCRIPT_DIR'
`$env:VIRTUAL_ENV = '$VENV_PATH'
`$env:PATH = '$VENV_PATH\Scripts;' + `$env:PATH
Write-Host '🛡️ Aegis Backend API Starting...' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
& '$pythonExe' -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
Write-Host ''
Write-Host 'Backend stopped. Press any key to close...' -ForegroundColor Yellow
pause
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
Start-Sleep -Seconds 4

# Start Frontend in new window
Write-Info "Starting Frontend on port 3000..."
$frontendCmd = @"
Set-Location '$frontendPath'
Write-Host '🎨 Aegis Frontend Starting...' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
npm run dev
Write-Host ''
Write-Host 'Frontend stopped. Press any key to close...' -ForegroundColor Yellow
pause
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd
Start-Sleep -Seconds 6

# ============================================================
# Verify Services Started
# ============================================================
Write-Host ""
Write-Info "Verifying services..."

Start-Sleep -Seconds 3
$backendRunning = $false
$frontendRunning = $false

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    $backendRunning = $true
    Write-Success "Backend API is responding"
} catch {
    Write-Info "Backend may still be starting (this is normal)..."
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    $frontendRunning = $true
    Write-Success "Frontend is responding"
} catch {
    Write-Info "Frontend may still be starting (this is normal)..."
}

# ============================================================
# Success Message
# ============================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "   🎉 AEGIS IS STARTING!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "   🌐 Dashboard:     http://localhost:3000" -ForegroundColor White
Write-Host "   📚 API Docs:      http://localhost:8000/docs" -ForegroundColor White
Write-Host "   💚 Health Check:  http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "   Test Credentials:" -ForegroundColor Yellow
Write-Host "   📧 Email:    demo@aegis.com" -ForegroundColor Gray
Write-Host "   🔑 Password: demo123" -ForegroundColor Gray
Write-Host ""
Write-Host "   Services are starting in separate windows..." -ForegroundColor Cyan
Write-Host "   Please wait 10-15 seconds for full startup." -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Ask to open browser
$openBrowser = Read-Host "Open dashboard in browser? (Y/n)"
if ($openBrowser -ne "n" -and $openBrowser -ne "N") {
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:3000"
    Write-Success "Browser opened!"
}

Write-Host ""
Write-Host "Press Enter to close this window (services will keep running)..." -ForegroundColor Gray
Read-Host
