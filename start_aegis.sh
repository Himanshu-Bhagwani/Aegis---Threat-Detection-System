#!/bin/bash

# ============================================================
# 🛡️ AEGIS THREAT DETECTION SYSTEM - Mac/Linux Startup Script
# ============================================================
# Fully Automated Setup - Handles everything from A to Z
# Usage: chmod +x start_aegis.sh && ./start_aegis.sh
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Functions
print_banner() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}   🛡️  AEGIS THREAT DETECTION SYSTEM${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${WHITE}   Fully Automated Setup & Launch${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[$1] $2${NC}"
}

print_success() {
    echo -e "  ${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "  ${RED}❌ $1${NC}"
}

print_info() {
    echo -e "  ${GRAY}ℹ️  $1${NC}"
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Services stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

print_banner

# ============================================================
# Step 1: Check Prerequisites
# ============================================================
print_step "1/7" "Checking prerequisites..."

PREREQS_OK=true

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_success "Python found: $PYTHON_VERSION"
    
    # Check version
    PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    if [ $(echo "$PYTHON_VER 3.10" | awk '{print ($1 >= $2)}') -eq 0 ]; then
        print_error "Python 3.10+ required. Found: $PYTHON_VER"
        PREREQS_OK=false
    fi
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$(python --version 2>&1)
    print_success "Python found: $PYTHON_VERSION"
    
    PYTHON_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    if [ $(echo "$PYTHON_VER 3.10" | awk '{print ($1 >= $2)}') -eq 0 ]; then
        print_error "Python 3.10+ required. Found: $PYTHON_VER"
        PREREQS_OK=false
    fi
else
    print_error "Python not found! Please install Python 3.10+"
    print_info "Download from: https://www.python.org/downloads/"
    PREREQS_OK=false
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    print_success "Node.js found: $NODE_VERSION"
    
    # Check version
    NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -lt 18 ]; then
        print_error "Node.js 18+ required. Found: v$NODE_VER"
        PREREQS_OK=false
    fi
else
    print_error "Node.js not found! Please install Node.js 18+"
    print_info "Download from: https://nodejs.org/"
    PREREQS_OK=false
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version 2>&1)
    print_success "npm found: v$NPM_VERSION"
else
    print_error "npm not found!"
    PREREQS_OK=false
fi

if [ "$PREREQS_OK" = false ]; then
    echo ""
    exit 1
fi

# ============================================================
# Step 2: Setup Virtual Environment
# ============================================================
print_step "2/7" "Setting up Python virtual environment..."

VENV_PATH="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_PATH" ]; then
    print_info "Creating virtual environment (this may take a moment)..."
    $PYTHON_CMD -m venv .venv
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
    print_success "Virtual environment created"
else
    print_success "Virtual environment exists"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
print_success "Virtual environment activated"

# ============================================================
# Step 3: Upgrade pip
# ============================================================
print_step "3/7" "Upgrading pip..."

python -m pip install --upgrade pip --quiet 2>/dev/null || python -m pip install --upgrade pip
print_success "pip upgraded"

# ============================================================
# Step 4: Install Python Dependencies
# ============================================================
print_step "4/7" "Installing Python dependencies..."

if [ -f "requirements.txt" ]; then
    print_info "Installing packages from requirements.txt (this may take 5-10 minutes)..."
    python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        print_error "Failed to install Python dependencies"
        exit 1
    fi
    print_success "Python dependencies installed"
    
    # Verify critical packages
    print_info "Verifying critical packages..."
    python -c "import fastapi, uvicorn, tensorflow, sklearn, pandas, numpy" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "All critical packages verified"
    else
        print_error "Some packages may be missing"
    fi
else
    print_error "requirements.txt not found!"
    exit 1
fi

# ============================================================
# Step 5: Install Frontend Dependencies
# ============================================================
print_step "5/7" "Installing frontend dependencies..."

FRONTEND_PATH="$SCRIPT_DIR/frontend"

if [ -d "$FRONTEND_PATH" ]; then
    cd "$FRONTEND_PATH"
    
    if [ ! -d "node_modules" ]; then
        print_info "Installing npm packages (first time - may take 2-3 minutes)..."
        npm install
        if [ $? -ne 0 ]; then
            print_error "Failed to install frontend dependencies"
            cd "$SCRIPT_DIR"
            exit 1
        fi
        print_success "Frontend dependencies installed"
    else
        print_success "Frontend dependencies already installed"
    fi
    
    cd "$SCRIPT_DIR"
else
    print_error "Frontend directory not found!"
    exit 1
fi

# ============================================================
# Step 6: Prepare Ports
# ============================================================
print_step "6/7" "Preparing ports..."

# Kill process on port 8000
if lsof -ti:8000 &>/dev/null; then
    print_info "Clearing port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi
print_success "Port 8000 ready"

# Kill process on port 3000
if lsof -ti:3000 &>/dev/null; then
    print_info "Clearing port 3000..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi
print_success "Port 3000 ready"

# ============================================================
# Step 7: Start Services
# ============================================================
print_step "7/7" "Starting Aegis services..."

# Start Backend API
print_info "Starting Backend API on port 8000..."
cd "$SCRIPT_DIR"
source "$VENV_PATH/bin/activate"
python -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 3

# Check if backend started
if kill -0 $BACKEND_PID 2>/dev/null; then
    print_success "Backend API started (PID: $BACKEND_PID)"
else
    print_error "Backend API failed to start!"
    exit 1
fi

# Start Frontend
print_info "Starting Frontend on port 3000..."
cd "$FRONTEND_PATH"
npm run dev &
FRONTEND_PID=$!

sleep 5

# Check if frontend started
if kill -0 $FRONTEND_PID 2>/dev/null; then
    print_success "Frontend started (PID: $FRONTEND_PID)"
else
    print_error "Frontend failed to start!"
    exit 1
fi

cd "$SCRIPT_DIR"

# ============================================================
# Verify Services
# ============================================================
echo ""
print_info "Verifying services..."

sleep 3

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    print_success "Backend API is responding"
else
    print_info "Backend may still be starting (this is normal)..."
fi

if curl -s http://localhost:3000 >/dev/null 2>&1; then
    print_success "Frontend is responding"
else
    print_info "Frontend may still be starting (this is normal)..."
fi

# ============================================================
# Print Success Message
# ============================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   🎉 AEGIS IS RUNNING!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "   ${WHITE}🌐 Dashboard:     ${CYAN}http://localhost:3000${NC}"
echo -e "   ${WHITE}📚 API Docs:      ${CYAN}http://localhost:8000/docs${NC}"
echo -e "   ${WHITE}💚 Health Check:  ${CYAN}http://localhost:8000/health${NC}"
echo ""
echo -e "   ${YELLOW}Test Credentials:${NC}"
echo -e "   ${GRAY}📧 Email:    demo@aegis.com${NC}"
echo -e "   ${GRAY}🔑 Password: demo123${NC}"
echo ""
echo -e "${GREEN}============================================================${NC}"

# Open browser (platform-specific)
echo ""
read -p "Open dashboard in browser? (Y/n): " OPEN_BROWSER
if [[ ! "$OPEN_BROWSER" =~ ^[Nn]$ ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "http://localhost:3000"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "http://localhost:3000" 2>/dev/null || sensible-browser "http://localhost:3000" 2>/dev/null || echo "Please open http://localhost:3000 in your browser"
    fi
    print_success "Browser opened!"
fi

echo ""
echo -e "${GRAY}Press Ctrl+C to stop all services...${NC}"
echo ""

# Wait for user to stop
wait
