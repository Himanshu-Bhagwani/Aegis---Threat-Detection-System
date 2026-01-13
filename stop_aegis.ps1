# ============================================================
# 🛡️ AEGIS - Stop Services Script (Windows)
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   🛑 Stopping Aegis Services" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$stoppedCount = 0

# Stop processes on port 8000 (Backend)
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($port8000) {
    Write-Host "  Stopping Backend API (port 8000)..." -ForegroundColor Yellow
    $pids = $port8000 | ForEach-Object { $_.OwningProcess } | Select-Object -Unique
    foreach ($pid in $pids) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host "    ✓ Stopped process $pid" -ForegroundColor Green
            $stoppedCount++
        } catch {
            Write-Host "    ⚠ Could not stop process $pid" -ForegroundColor Yellow
        }
    }
    Write-Host "  ✅ Backend stopped" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  Backend was not running" -ForegroundColor Gray
}

# Stop processes on port 3000 (Frontend)
$port3000 = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if ($port3000) {
    Write-Host "  Stopping Frontend (port 3000)..." -ForegroundColor Yellow
    $pids = $port3000 | ForEach-Object { $_.OwningProcess } | Select-Object -Unique
    foreach ($pid in $pids) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host "    ✓ Stopped process $pid" -ForegroundColor Green
            $stoppedCount++
        } catch {
            Write-Host "    ⚠ Could not stop process $pid" -ForegroundColor Yellow
        }
    }
    Write-Host "  ✅ Frontend stopped" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  Frontend was not running" -ForegroundColor Gray
}

# Also try to kill Python processes running uvicorn
Write-Host "  Checking for Python processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*fastapi*"
} | ForEach-Object {
    try {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        Write-Host "    ✓ Stopped Python process $($_.Id)" -ForegroundColor Green
        $stoppedCount++
    } catch { }
}

# Also try to kill Node processes running Next.js
Write-Host "  Checking for Node processes..." -ForegroundColor Yellow
Get-Process node -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*next*" -or $_.CommandLine -like "*dev*"
} | ForEach-Object {
    try {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        Write-Host "    ✓ Stopped Node process $($_.Id)" -ForegroundColor Green
        $stoppedCount++
    } catch { }
}

Write-Host ""
if ($stoppedCount -gt 0) {
    Write-Host "✅ Stopped $stoppedCount process(es)" -ForegroundColor Green
} else {
    Write-Host "ℹ️  No running Aegis services found" -ForegroundColor Gray
}
Write-Host "All Aegis services stopped." -ForegroundColor Green
Write-Host ""
