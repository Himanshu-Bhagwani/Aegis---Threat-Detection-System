# PowerShell script to start Aegis API server

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $scriptPath "..")

# Activate virtual environment if exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    . .\venv\Scripts\Activate.ps1
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AEGIS THREAT DETECTION API" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  Health:    http://localhost:8000/health" -ForegroundColor Green
Write-Host "  Endpoints:" -ForegroundColor Yellow
Write-Host "    - POST /risk/unified     - Unified risk scoring" -ForegroundColor Gray
Write-Host "    - POST /gps/score        - GPS spoofing detection" -ForegroundColor Gray
Write-Host "    - POST /login/score      - Login anomaly detection" -ForegroundColor Gray
Write-Host "    - POST /password/score   - Password risk assessment" -ForegroundColor Gray
Write-Host "    - POST /fraud/score      - Transaction fraud detection" -ForegroundColor Gray
Write-Host ""

# Start the server
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload


