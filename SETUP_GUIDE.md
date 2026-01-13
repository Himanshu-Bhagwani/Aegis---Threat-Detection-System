# 🛡️ Aegis Threat Detection System - Setup Guide

## Overview

Aegis is a **multi-layer threat detection system** that identifies suspicious behavior across:
- 🛰️ **GPS Movement** - Detects location spoofing
- 🔐 **Login Patterns** - Identifies anomalous authentication
- 🔑 **Password Integrity** - Assesses password strength and breach risk
- 💳 **Transaction Fraud** - Flags suspicious transactions
- ⚡ **Unified Risk Score** - Combines all signals into one actionable score

---

## 🚀 Quick Start

### Windows
```powershell
# Double-click or run in PowerShell:
.\start_aegis.ps1
```

### Mac/Linux
```bash
# Make executable and run:
chmod +x start_aegis.sh
./start_aegis.sh
```

---

## 📋 Prerequisites

### Required Software
| Software | Version | Download |
|----------|---------|----------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| Git | Latest | [git-scm.com](https://git-scm.com/) |

### Verify Installation
```bash
python --version   # Should be 3.10+
node --version     # Should be 18+
npm --version      # Should be 9+
```

---

## 🔧 Manual Setup

### Step 1: Create Virtual Environment

**Windows (PowerShell):**
```powershell
cd aegis-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
cd aegis-main
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 4: Start Backend API
```bash
python -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

### Step 5: Start Frontend (New Terminal)
```bash
cd frontend
npm run dev
```

### Step 6: Open in Browser
- **Frontend Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🔑 Test Credentials

| Email | Password | Description |
|-------|----------|-------------|
| `demo@aegis.com` | `demo123` | Demo user account |
| `admin@aegis.com` | `admin123` | Admin account |
| `test@test.com` | `test123` | Test account |

Or create your own account via the signup page!

---

## 🧪 Testing Features

### 1. Login Anomaly Detection
Navigate to Dashboard → Login Anomaly Test
- Set **Hour of Day**: `3` (suspicious off-hours)
- Set **Failed Attempts**: `5` (brute force indicator)
- Check **New Device**: ✓
- Click **Analyze Login**

### 2. Password Strength Checker
Navigate to Dashboard → Password Strength
- Type: `password123` → Should show HIGH risk
- Type: `Xk9#mP2$vL7@nQ` → Should show LOW risk

### 3. Unified Risk Analysis
Click **Run Demo Analysis** to see combined threat assessment

### 4. API Testing (Swagger UI)
Visit http://localhost:8000/docs for interactive API testing

---

## 📁 Project Structure

```
aegis-main/
├── src/
│   ├── api/                 # FastAPI backend
│   │   ├── fastapi_app.py   # Main API application
│   │   └── routers/         # API route handlers
│   │       ├── gps.py       # GPS spoofing detection
│   │       ├── login.py     # Login anomaly detection
│   │       ├── password.py  # Password risk assessment
│   │       ├── fraud.py     # Transaction fraud detection
│   │       └── risk.py      # Unified risk scoring
│   ├── gps/                 # GPS detection models
│   ├── login/               # Login anomaly models
│   ├── passwords/           # Password scoring
│   └── fusion/              # Risk score fusion
├── frontend/                # Next.js dashboard
│   ├── app/                 # App router pages
│   ├── components/          # React components
│   └── lib/                 # API client utilities
├── models/                  # Pre-trained ML models
│   ├── gps/                 # GPS detection models
│   └── login/               # Login anomaly models
├── requirements.txt         # Python dependencies
├── start_aegis.ps1         # Windows startup script
├── start_aegis.sh          # Mac/Linux startup script
└── SETUP_GUIDE.md          # This file
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/login` | POST | User authentication |
| `/signup` | POST | User registration |
| `/gps/score` | POST | GPS spoofing detection |
| `/login/score` | POST | Login anomaly detection |
| `/password/score` | POST | Password risk assessment |
| `/fraud/score` | POST | Transaction fraud detection |
| `/risk/unified` | POST | Combined risk score |

---

## 🐛 Troubleshooting

### Port Already in Use

**Windows:**
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Kill process on port 3000
netstat -ano | findstr :3000
taskkill /F /PID <PID>
```

**Mac/Linux:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Virtual Environment Issues
```bash
# Delete and recreate
rm -rf .venv
python -m venv .venv
```

### TensorFlow/Keras Errors
The system gracefully falls back to sklearn models (Isolation Forest, GBM) if deep learning models have compatibility issues.

### Frontend Not Loading
```bash
cd frontend
rm -rf node_modules
npm install
npm run dev
```

---

## 📊 ML Models Used

| Layer | Models | Status |
|-------|--------|--------|
| GPS Detection | Isolation Forest, GBM, Autoencoder, CNN-RNN | ✅ Trained |
| Login Anomaly | Isolation Forest, GBM, Autoencoder | ✅ Trained |
| Password Risk | Rule-based + ML (if trained) | ✅ Working |
| Fraud Detection | XGBoost, Random Forest | ⚠️ Requires training |

---

## 🤝 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review API docs at http://localhost:8000/docs
3. Check browser console for frontend errors

---

Built with ❤️ using FastAPI, Next.js, TensorFlow, and scikit-learn

