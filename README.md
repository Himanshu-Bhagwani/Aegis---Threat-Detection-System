# 🛡️ Aegis Threat Detection System

<div align="center">

**A Unified Multi-Layer Security Platform**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)

*A comprehensive threat detection system that combines GPS spoofing detection, login anomaly analysis, password risk assessment, and transaction fraud detection into a unified risk score.*

</div>

---

## ✨ Features

### 🛰️ **GPS Spoofing Detection**
- Multiple ML models: Isolation Forest, Gradient Boosting, Autoencoder, CNN-RNN
- Dynamic trajectory generation and analysis
- Real-time location anomaly detection
- Rule-based fallback for reliability

### 🔐 **Login Anomaly Detection**
- LANL-trained models for authentication pattern analysis
- IP, user agent, and temporal pattern detection
- Isolation Forest, GBM, and Autoencoder models
- Heuristic-based scoring for edge cases

### 🔑 **Password Risk Assessment**
- XGBoost-based strength evaluation
- Breach database checking
- Pattern and complexity analysis
- Rule-based fallback scoring

### 💳 **Transaction Fraud Detection**
- Real-time transaction analysis
- Behavioral pattern recognition
- Risk probability scoring
- Multi-factor fraud indicators

### ⚡ **Unified Risk Scoring**
- Weighted fusion of all detection layers
- Comprehensive risk breakdown
- Actionable threat categories
- Real-time risk visualization

---

## 🚀 Quick Start

### One-Command Setup

**Windows:**
```powershell
# Double-click or run:
.\start_aegis.bat
# OR
.\start_aegis.ps1
```

**Mac/Linux:**
```bash
chmod +x start_aegis.sh
./start_aegis.sh
```

That's it! The script automatically:
- ✅ Checks prerequisites (Python 3.10+, Node.js 18+)
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Starts backend API (port 8000)
- ✅ Starts frontend dashboard (port 3000)
- ✅ Opens browser automatically

### Access URLs

Once running:
- 🌐 **Dashboard**: http://localhost:3000
- 📚 **API Docs**: http://localhost:8000/docs
- 💚 **Health Check**: http://localhost:8000/health

### Test Credentials

- **Email**: `demo@aegis.com`
- **Password**: `demo123`

---

## 📋 Prerequisites

| Software | Version | Download |
|----------|---------|----------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| npm | 9+ | (comes with Node.js) |

Verify installation:
```bash
python --version   # Should be 3.10+
node --version     # Should be 18+
npm --version      # Should be 9+
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend Dashboard                    │
│              (Next.js + React + TypeScript)             │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────┐
│              FastAPI Backend API                        │
│  ┌──────────┬──────────┬──────────┬──────────┐        │
│  │   GPS    │  Login   │ Password │  Fraud   │        │
│  │  Router  │  Router  │  Router  │  Router  │        │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘        │
│       │          │          │          │               │
│  ┌────▼─────┐ ┌──▼───┐ ┌───▼───┐ ┌───▼───┐           │
│  │ Detection│ │Detection│Detection│Detection│           │
│  │ Modules  │ │ Modules │ Modules │ Modules │           │
│  └────┬─────┘ └───┬───┘ └───┬───┘ └───┬───┘           │
│       └───────────┴─────────┴─────────┘                │
│                    │                                    │
│              ┌─────▼─────┐                             │
│              │   Risk    │                             │
│              │  Fusion   │                             │
│              │  Module   │                             │
│              └───────────┘                             │
└────────────────────────────────────────────────────────┘
```

### Detection Layers

1. **GPS Spoofing Detection**
   - Models: Isolation Forest, GBM, Autoencoder, CNN-RNN
   - Input: GPS coordinates, speed, heading, timestamps
   - Output: Spoof probability score

2. **Login Anomaly Detection**
   - Models: Isolation Forest, GBM, Autoencoder
   - Input: IP address, user agent, timestamp, login method
   - Output: Anomaly score and risk level

3. **Password Risk Assessment**
   - Models: XGBoost + Rule-based
   - Input: Password string
   - Output: Strength score, breach risk, recommendations

4. **Transaction Fraud Detection**
   - Models: XGBoost classifier
   - Input: Amount, location, payment method, merchant
   - Output: Fraud probability and risk level

5. **Unified Risk Scoring**
   - Combines all layer outputs
   - Weighted aggregation
   - Final risk score and breakdown

---

## 🔧 Manual Installation

If you prefer manual setup:

### 1. Clone Repository
```bash
git clone <repository-url>
cd aegis-main
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### 4. Run Services

**Terminal 1 - Backend:**
```bash
python -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## 📡 API Endpoints

### Authentication
- `POST /signup` - User registration
- `POST /login` - User authentication

### Detection Endpoints
- `POST /gps/score` - GPS spoofing detection
- `POST /login/score` - Login anomaly detection
- `POST /password/score` - Password risk assessment
- `POST /fraud/score` - Transaction fraud detection
- `POST /risk/unified` - Unified risk score

### System
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

See [API Documentation](http://localhost:8000/docs) for detailed request/response schemas.

---

## 📁 Project Structure

```
aegis-main/
├── src/
│   ├── api/
│   │   ├── routers/          # API endpoint routers
│   │   │   ├── gps.py       # GPS detection endpoints
│   │   │   ├── login.py     # Login detection endpoints
│   │   │   ├── password.py  # Password endpoints
│   │   │   ├── fraud.py     # Fraud detection endpoints
│   │   │   └── risk.py      # Unified risk endpoints
│   │   └── fastapi_app.py   # Main FastAPI application
│   ├── gps/                  # GPS spoofing detection
│   │   └── score_gps.py
│   ├── login/                # Login anomaly detection
│   │   └── score_login.py
│   ├── passwords/            # Password risk assessment
│   │   └── score_password.py
│   └── fusion/               # Unified risk scoring
│       └── risk_scoring.py
├── frontend/                 # Next.js frontend
│   ├── app/                 # App router pages
│   │   └── dashboard/       # Main dashboard
│   ├── lib/                 # API client
│   └── components/          # React components
├── models/                  # ML model artifacts (if available)
├── start_aegis.ps1         # Windows PowerShell setup
├── start_aegis.bat         # Windows Batch setup
├── start_aegis.sh          # Mac/Linux setup
├── stop_aegis.ps1          # Windows stop script
├── stop_aegis.sh           # Mac/Linux stop script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── SETUP_GUIDE.md          # Detailed setup guide
├── QUICK_START.md          # Quick reference
└── PROJECT_SUMMARY.md      # Complete project summary
```

---

## 🎯 Usage Examples

### GPS Spoofing Detection
```python
POST /gps/score
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "speed": 65.0,
  "heading": 90.0
}
```

### Login Anomaly Detection
```python
POST /login/score
{
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "timestamp": "2025-06-12T10:30:00Z",
  "login_method": "password"
}
```

### Password Risk Assessment
```python
POST /password/score
{
  "password": "MySecureP@ssw0rd123"
}
```

### Unified Risk Score
```python
POST /risk/unified
{
  "gps_data": {...},
  "login_data": {...},
  "password_data": {...},
  "fraud_data": {...}
}
```

---

## 🛑 Stopping Services

**Windows:**
```powershell
.\stop_aegis.ps1
```

**Mac/Linux:**
```bash
./stop_aegis.sh
```

Or manually:
```bash
# Kill backend (port 8000)
# Windows:
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Mac/Linux:
lsof -ti:8000 | xargs kill -9

# Kill frontend (port 3000)
# Windows:
netstat -ano | findstr :3000
taskkill /F /PID <PID>

# Mac/Linux:
lsof -ti:3000 | xargs kill -9
```

---

## 🐛 Troubleshooting

### Port Already in Use
Use the stop scripts above or manually kill processes on ports 8000/3000.

### Virtual Environment Issues
```bash
# Delete and recreate
rm -rf .venv  # or rmdir /s .venv on Windows
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### Model Loading Errors
The system gracefully falls back to available models. Missing ML models will use rule-based scoring.

### Frontend Not Loading
```bash
cd frontend
rm -rf node_modules
npm install
npm run dev
```

### Batch File Crashes (Windows)
- Check `batch_log.txt` for detailed error logs
- Ensure Python and Node.js are in PATH
- Run PowerShell as Administrator if needed

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation and setup instructions
- **[QUICK_START.md](QUICK_START.md)** - Quick reference guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete project overview and work done
- **[API Docs](http://localhost:8000/docs)** - Interactive API documentation (when running)

---

## 🔐 Security Notes

- This is a **demo/development system** with mock authentication
- For production use, implement proper authentication (JWT, OAuth, etc.)
- Replace in-memory user store with secure database
- Add rate limiting and input validation
- Use HTTPS in production
- Implement proper logging and monitoring

---

## 📊 ML Models

| Layer | Models | Status |
|-------|--------|--------|
| GPS Detection | Isolation Forest, GBM, Autoencoder, CNN-RNN | ✅ Trained |
| Login Anomaly | Isolation Forest, GBM, Autoencoder | ✅ Trained |
| Password Risk | XGBoost + Rule-based | ✅ Implemented |
| Fraud Detection | XGBoost | ✅ Implemented |

**Note**: ML models gracefully degrade to rule-based scoring if artifacts are missing.

---

## 📄 License

[Add your license information here]

---

## 🙏 Acknowledgments

- LANL dataset for login anomaly detection models
- TensorFlow/Keras for deep learning models
- FastAPI for the backend framework
- Next.js for the frontend framework

---

## 📞 Support

For issues and questions:
- Check `batch_log.txt` for error logs
- Review [SETUP_GUIDE.md](SETUP_GUIDE.md) for troubleshooting
- Check API docs at http://localhost:8000/docs when running

---

<div align="center">

**Built with ❤️ for comprehensive threat detection**

[🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🐛 Troubleshooting](#-troubleshooting)

</div>
