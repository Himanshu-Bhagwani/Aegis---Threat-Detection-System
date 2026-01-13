#!/bin/bash

# ============================================================
# 🛡️ AEGIS - Stop Services Script (Mac/Linux)
# ============================================================

echo ""
echo "============================================================"
echo "   🛑 Stopping Aegis Services"
echo "============================================================"
echo ""

stopped_count=0

# Stop processes on port 8000 (Backend)
if lsof -ti:8000 &>/dev/null; then
    echo "  Stopping Backend API (port 8000)..."
    pids=$(lsof -ti:8000)
    for pid in $pids; do
        kill -9 $pid 2>/dev/null && {
            echo "    ✓ Stopped process $pid"
            ((stopped_count++))
        } || echo "    ⚠ Could not stop process $pid"
    done
    echo "  ✅ Backend stopped"
else
    echo "  ℹ️  Backend was not running"
fi

# Stop processes on port 3000 (Frontend)
if lsof -ti:3000 &>/dev/null; then
    echo "  Stopping Frontend (port 3000)..."
    pids=$(lsof -ti:3000)
    for pid in $pids; do
        kill -9 $pid 2>/dev/null && {
            echo "    ✓ Stopped process $pid"
            ((stopped_count++))
        } || echo "    ⚠ Could not stop process $pid"
    done
    echo "  ✅ Frontend stopped"
else
    echo "  ℹ️  Frontend was not running"
fi

# Also try to kill Python processes running uvicorn
echo "  Checking for Python processes..."
pids=$(pgrep -f "uvicorn.*fastapi_app" 2>/dev/null || true)
for pid in $pids; do
    kill -9 $pid 2>/dev/null && {
        echo "    ✓ Stopped Python process $pid"
        ((stopped_count++))
    } || true
done

# Also try to kill Node processes running Next.js
echo "  Checking for Node processes..."
pids=$(pgrep -f "next.*dev" 2>/dev/null || true)
for pid in $pids; do
    kill -9 $pid 2>/dev/null && {
        echo "    ✓ Stopped Node process $pid"
        ((stopped_count++))
    } || true
done

echo ""
if [ $stopped_count -gt 0 ]; then
    echo "✅ Stopped $stopped_count process(es)"
else
    echo "ℹ️  No running Aegis services found"
fi
echo "All Aegis services stopped."
echo ""
