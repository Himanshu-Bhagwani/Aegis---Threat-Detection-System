# AEGIS Client SDK

Lightweight browser/Node tracking SDK for the **AEGIS Threat Detection** platform.

Collects login, GPS, device fingerprint, and transaction events — scores them in real-time via the AEGIS API — and fires a callback whenever risk crosses a threshold.

---

## Install

```bash
# npm
npm install @aegis/client-sdk

# yarn
yarn add @aegis/client-sdk

# CDN (no build step required)
<script src="https://cdn.example.com/aegis/1.0.0/aegis.min.js"></script>
```

---

## Quick start — vanilla JS

```html
<!DOCTYPE html>
<html>
<head>
  <script src="aegis.js"></script>
</head>
<body>
<script>
  // 1. Initialise once (e.g. right after login)
  Aegis.init({
    apiUrl: 'https://your-aegis-api.com',
    userId: 'user_abc123',
    token:  'Bearer eyJhbGci...',

    autoFingerprint: true,   // collect device fingerprint on init
    autoGPS:         false,  // set true to start GPS tracking immediately

    // Called every time a risk score is returned
    onRisk(result) {
      console.log('Risk level:', result.risk_level);   // low | medium | high | critical
      console.log('Score:',      result.unified_score); // 0.0 → 1.0

      if (result.risk_level === 'critical') {
        // Show step-up auth challenge, lock account, etc.
        showMFAChallenge();
      }
    },
  });

  // 2. Track a login event
  document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const t0 = Date.now();
    const ok = await doLogin(/* ... */);

    Aegis.trackLogin({
      success:      ok,
      method:       'password',
      duration_ms:  Date.now() - t0,
    });
  });

  // 3. Check a password for breaches (k-anonymity — only 5-char hash prefix sent)
  document.getElementById('passwordInput').addEventListener('blur', (e) => {
    Aegis.trackPassword(e.target.value);
  });

  // 4. Score a transaction
  async function handlePurchase(tx) {
    const result = await Aegis.trackTransaction({
      amount:          tx.amount,
      currency:        tx.currency,
      merchant:        tx.merchant,
      is_international: tx.crossBorder,
    });
    if (result?.risk_level === 'high') blockTransaction();
  }

  // 5. Request a score on-demand
  async function checkNow() {
    const result = await Aegis.scoreNow();
    console.log(result);
  }
</script>
</body>
</html>
```

---

## React hook

```tsx
// App.tsx
import React from 'react';
import { useAegis } from '@aegis/client-sdk/react';

function App() {
  const {
    isReady,
    riskScore,
    riskLevel,
    primaryThreats,
    trackLogin,
    trackPassword,
    pushGPS,
    scoreNow,
  } = useAegis({
    apiUrl: process.env.NEXT_PUBLIC_API_URL,
    userId: currentUser.id,
    token:  `Bearer ${currentUser.accessToken}`,
    autoFingerprint: true,
  });

  const handleLogin = async (credentials) => {
    const t0 = Date.now();
    await login(credentials);
    await trackLogin({ success: true, method: 'password', duration_ms: Date.now() - t0 });
  };

  return (
    <div>
      {riskLevel === 'critical' && <MFAChallenge />}

      <RiskBadge level={riskLevel} score={riskScore} threats={primaryThreats} />

      <LoginForm onSubmit={handleLogin} onPasswordBlur={trackPassword} />
    </div>
  );
}
```

### Inline risk badge example

```tsx
function RiskBadge({ level, score }: { level: string | null; score: number | null }) {
  const colors = { low: '#00ff88', medium: '#ffaa00', high: '#ff6600', critical: '#ff0040' };
  if (!level) return null;
  return (
    <span style={{
      background: colors[level] ?? '#888',
      borderRadius: 4,
      padding: '2px 8px',
      color: '#000',
      fontWeight: 700,
      fontSize: 12,
    }}>
      {level.toUpperCase()} — {Math.round((score ?? 0) * 100)}%
    </span>
  );
}
```

---

## GPS tracking

```js
// Manual — push a single reading
Aegis.pushGPS(51.505, -0.09);

// Automatic — set autoGPS: true in init()
// The SDK calls navigator.geolocation.watchPosition internally and
// pushes to /gps/score whenever the user moves.
Aegis.init({ ..., autoGPS: true });

// Stop GPS (and all other intervals)
Aegis.destroy();
```

---

## App-unlock / biometric event

```js
// After a successful biometric authentication
Aegis.trackAppUnlock({
  method:         'biometric',
  success:        true,
  failed_attempts: 0,
  is_rooted:      false,
  is_emulator:    false,
});
```

---

## API reference

| Method | Description |
|--------|-------------|
| `Aegis.init(config)` | Initialise SDK. **Must be called first.** |
| `Aegis.trackLogin(data)` | Score a login attempt. Posts to `/risk/unified`. |
| `Aegis.trackPassword(pw)` | HIBP k-anonymity breach check. Posts to `/breach/check/password`. |
| `Aegis.trackTransaction(tx)` | Fraud score a transaction. Posts to `/fraud/score`. |
| `Aegis.trackAppUnlock(data)` | Device/biometric risk event. Posts to `/device/score`. |
| `Aegis.pushGPS(lat, lon)` | Push GPS observation. Posts to `/gps/score`. |
| `Aegis.scoreNow(extra?)` | Full on-demand unified risk score. |
| `Aegis.flush()` | Flush queued events immediately. |
| `Aegis.destroy()` | Stop GPS + clear all intervals. |

### `Aegis.init()` config

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apiUrl` | `string` | `"http://localhost:8000"` | AEGIS API base URL |
| `userId` | `string` | **required** | Current user identifier |
| `token` | `string` | `null` | `Authorization` header value |
| `autoFingerprint` | `boolean` | `true` | Collect device fingerprint on init |
| `autoGPS` | `boolean` | `false` | Start GPS watchPosition on init |
| `onRisk` | `function` | `null` | Callback fired on every score response |
| `batchSize` | `number` | `10` | Events to queue before auto-flush |
| `flushInterval` | `number` | `30000` | Auto-flush interval (ms) |

---

## Privacy & security

- **Passwords are never sent.** Only the first 5 characters of the SHA-1 hash are transmitted (HIBP k-anonymity model).
- **GPS is opt-in.** Set `autoGPS: true` or call `pushGPS()` explicitly.
- **All requests are HTTPS.** Configure `apiUrl` with an `https://` endpoint in production.
- **Token management.** Pass a short-lived Bearer token; the SDK does not store credentials beyond the current session.

---

## Backend integration

The SDK talks directly to these AEGIS API endpoints:

| SDK method | Endpoint | Auth |
|------------|----------|------|
| `trackLogin` | `POST /risk/unified` | Bearer |
| `trackPassword` | `POST /breach/check/password` | Bearer |
| `trackTransaction` | `POST /fraud/score` | Bearer |
| `trackAppUnlock` | `POST /device/score` | Bearer |
| `pushGPS` | `POST /gps/score` | Bearer |
| `scoreNow` | `POST /risk/unified` | Bearer |

Start the backend:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"healthy","modules":{"login":true,"gps":true,"device":true,...}}
```

---

## Browser support

| Feature | Requirement |
|---------|-------------|
| `fetch` | Chrome 42+, Firefox 39+, Safari 10.1+ (XHR fallback provided) |
| `crypto.subtle` | Chrome 37+, Firefox 34+, Safari 11+ (password hashing) |
| `navigator.geolocation` | All modern browsers (GPS, opt-in) |
| WebGL fingerprint | Chrome, Firefox, Safari, Edge |

---

## Changelog

### v1.0.0
- Initial release
- UMD bundle (browser global / CommonJS / AMD)
- `useAegis()` React hook
- `withAegis()` HOC
- TypeScript declarations (`aegis.d.ts`)
- GPS trajectory accumulator
- HIBP k-anonymity via Web Crypto API
- Event queue with batch + interval flush
- XHR fallback for environments without `fetch`
