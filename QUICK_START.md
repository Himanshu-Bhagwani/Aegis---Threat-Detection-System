# 🚀 Quick Start Guide - Aegis Threat Detection System

## One-Click Setup & Run

All setup is now **fully automated**! Just run one script and everything will be configured.

---

## ⚡ Windows (Recommended)

### Option 1: PowerShell (Best)
```powershell
.\start_aegis.ps1
```

### Option 2: Batch File (Simple)
```
Double-click start_aegis.bat
```

---

## 🍎 Mac / 🐧 Linux

```bash
chmod +x start_aegis.sh
./start_aegis.sh
```

---

## 🛑 Stop Services

### Windows
```powershell
.\stop_aegis.ps1
```

### Mac/Linux
```bash
./stop_aegis.sh
```

---

## ✨ What Gets Automated

The scripts automatically handle:

1. ✅ **Prerequisites Check** - Verifies Python 3.10+ and Node.js 18+
2. ✅ **Virtual Environment** - Creates and activates Python venv
3. ✅ **pip Upgrade** - Updates pip to latest version
4. ✅ **Python Dependencies** - Installs all packages from requirements.txt
5. ✅ **Frontend Dependencies** - Installs all npm packages
6. ✅ **Port Cleanup** - Clears ports 8000 and 3000 if in use
7. ✅ **Service Startup** - Starts Backend API and Frontend
8. ✅ **Verification** - Checks if services are responding
9. ✅ **Browser Launch** - Optionally opens dashboard

**Everything from A to Z is automated!**

---

## 📋 First Time Setup

On a **completely fresh PC**, you only need to install:

1. **Python 3.10+** - Download from https://www.python.org/downloads/
   - ✅ Check "Add Python to PATH" during installation

2. **Node.js 18+** - Download from https://nodejs.org/
   - ✅ npm comes automatically with Node.js

That's it! The scripts handle everything else.

---

## 🔍 Troubleshooting

### Script won't run?
- **Windows PowerShell**: Right-click → "Run with PowerShell"
- **Linux/Mac**: Run `chmod +x start_aegis.sh` first

### Port already in use?
- Run `stop_aegis.ps1` or `stop_aegis.sh` first
- Or manually kill processes on ports 8000/3000

### Python/Node not found?
- Make sure they're added to PATH during installation
- Restart terminal after installing

### Dependencies fail to install?
- Check internet connection
- On Windows, try running PowerShell as Administrator
- Check if antivirus is blocking downloads

---

## 🌐 Access URLs

Once running:

- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

---

## 🔑 Test Login

- **Email**: `demo@aegis.com`
- **Password**: `demo123`

---

**That's it! Run the script and you're good to go!** 🎉

