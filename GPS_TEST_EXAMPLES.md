# 🛰️ GPS Test Coordinate Examples

## Quick Test Examples for GPS Spoofing Detection

Use these 3 examples to test different GPS spoofing scenarios in the dashboard:

---

### 📍 Example 1: **Normal Movement** (Low Risk)
**Location:** San Francisco, CA, USA

- **Latitude:** `37.7749`
- **Longitude:** `-122.4194`
- **Speed:** `60` km/h (normal city driving)
- **Heading:** `90` degrees (East)

**Expected Result:** ✅ Low spoofing probability - Normal vehicle movement

**Description:** Realistic GPS coordinates in San Francisco with normal driving speed. This should show a low risk of spoofing.

---

### ⚠️ Example 2: **Suspicious Movement** (Medium Risk)
**Location:** New York City, NY, USA

- **Latitude:** `40.7128`
- **Longitude:** `-74.0060`
- **Speed:** `500` km/h (extremely high speed)
- **Heading:** `180` degrees (South)

**Expected Result:** ⚠️ Medium to High spoofing probability - Impossible speed for ground vehicle

**Description:** Real NYC coordinates but with an unrealistic speed (500 km/h = 310 mph), which is impossible for a car. This should trigger spoofing detection.

---

### 🚨 Example 3: **Clear Spoofing** (High Risk)
**Location:** London, UK

- **Latitude:** `51.5074`
- **Longitude:** `-0.1278`
- **Speed:** `8000` km/h (teleportation-level speed)
- **Heading:** `270` degrees (West)

**Expected Result:** 🚨 High spoofing probability - Clear indication of GPS manipulation

**Description:** Coordinates are in London but the speed (8000 km/h ≈ 5000 mph) is faster than commercial aircraft! This is a clear sign of GPS spoofing or manipulation.

---

## 🎯 How to Use

1. **In the Dashboard:**
   - Click one of the preset buttons (1️⃣ Normal, 2️⃣ Suspicious, 3️⃣ Spoofed)
   - The coordinates will auto-fill
   - Click "Analyze GPS" to see the results

2. **Manual Entry:**
   - Copy coordinates from above
   - Paste into the GPS test panel
   - Click "Analyze GPS"

---

## 📊 What to Look For

- **Spoof Probability:** Higher = more likely spoofed
- **Confidence:** How reliable the detection is
- **Model Scores:** Individual model predictions
- **Status Badge:** Visual indicator (Normal/Spoofed)

---

## 🗺️ Real-World Coordinates Reference

| City | Latitude | Longitude | Use Case |
|------|----------|-----------|----------|
| San Francisco, CA | 37.7749 | -122.4194 | Normal urban movement |
| New York, NY | 40.7128 | -74.0060 | High-speed suspicious |
| London, UK | 51.5074 | -0.1278 | Impossible speed |
| Tokyo, Japan | 35.6762 | 139.6503 | International test |
| Sydney, Australia | -33.8688 | 151.2093 | Southern hemisphere |

---

**Tip:** Combine different speeds with these coordinates to test various scenarios!

