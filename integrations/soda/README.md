# Connecting SODA (or any app) to Apeilo

SODA is a **Vite + React** frontend (host port **5173**) and an **Express/Node**
backend (host port **5001**). These files wire an app into Apeilo so Apeilo
watches real user activity and calls back when it detects a threat.

```
  App frontend (5173)             Apeilo API (8000)              App backend (5001)
  ───────────────────            ──────────────────             ───────────────────
  sign-up / sign-in ─────────►   scores + stores (tenant)
  transactions / invoices ───►   fraud model
                                 risk high/critical? ──────────► POST /api/apeilo/webhook
                                                                  verify + notify user
```

Everything runs locally, **no AWS, no cost**.

## The files

| File in this folder | Copy into the app at | What it is |
|---|---|---|
| `frontend/apeiloClient.js` | `frontend/src/services/apeiloClient.js` | API client (sends the key) |
| `frontend/ApeiloContext.jsx` | `frontend/src/context/ApeiloContext.jsx` | Provider + `useApeilo()` + threat toast |
| `backend/apeiloWebhook.js` | `backend/routes/apeiloWebhook.js` | Receives + verifies threat notifications |

## Credentials

```
API key         apeilo_sk_00b7c3ac63d0827e359e88691db9f5b9a9f00125
Webhook secret  whsec_7d0e46a69a1e5205fbe33f77b8f2af7f4c482e8f8b451b5b
```

---

## What gets tracked (sign-up AND sign-in)

Both auth paths must be wired for full coverage. `<ApeiloProvider>` handles the
first three automatically the moment a real user appears; the last three need
one line each in your auth components.

| Signal | How | Sign-up | Sign-in |
|---|---|---|---|
| Profile created (name + email) | provider → `register` | auto | auto |
| Session scored (real local hour) | provider → `trackLogin` | auto | auto |
| New / unknown device (`is_new_comp`) | provider (first time this browser sees the account) | auto | auto |
| GPS location | `apeilo.requestLocation()` in the submit handler | **add** | **add** |
| Password breach (HIBP) | `apeilo.trackPassword(password, email)` after success | **add** | **add** |
| Failed attempts (10-min window) | `apeilo.trackFailedLogin(email)` in the catch | n/a | **add** |

> `requestLocation()` **must** be called inside the click/submit handler.
> Browsers only show the location prompt during a user gesture — calling it
> later (e.g. from an effect) is silently ignored.

> Failed-attempt tracking is sign-in only by design. Counting failed *sign-ups*
> against an email would let anyone inflate another account's risk just by
> typing their address into the registration form.

---

## FRONTEND

### 1. Copy the two files
```
frontend/src/services/apeiloClient.js   ← frontend/apeiloClient.js
frontend/src/context/ApeiloContext.jsx  ← frontend/ApeiloContext.jsx
```

### 2. Env vars — `frontend/.env` (and as Docker build args if you build in Docker)
```bash
VITE_APEILO_URL=http://localhost:8000
VITE_APEILO_API_KEY=apeilo_sk_00b7c3ac63d0827e359e88691db9f5b9a9f00125
```

### 3. Wrap the app — `src/App.jsx`
```jsx
import { ApeiloProvider } from './context/ApeiloContext';

<ApeiloProvider
  apiKey={import.meta.env.VITE_APEILO_API_KEY}
  apiUrl={import.meta.env.VITE_APEILO_URL}
  userId={user?.email || 'guest'}
  name={user?.fullName || user?.email || ''}
  email={user?.email || ''}
>
  {content}
</ApeiloProvider>
```
The provider watches `userId`. When it changes from `guest` to a real user —
which happens on **both** sign-up and sign-in — it registers the profile,
flags a first-time device, attaches GPS, and scores the session.

### 4. Sign-in component
```jsx
const apeilo = useApeilo();

const handleSubmit = async (e) => {
  e.preventDefault();
  apeilo.requestLocation().catch(() => {});      // ← inside the gesture

  try {
    await login(email, password);
    apeilo.trackPassword(password, email).catch(() => {});
  } catch (err) {
    setError(err.message);
    apeilo.trackFailedLogin(email).catch(() => {});   // brute-force signal
  }
};
```
For an OAuth/Google button, call `apeilo.requestLocation()` in its handler too
(no `trackPassword` — there's no password).

### 5. Sign-up component
```jsx
const apeilo = useApeilo();

const handleSubmit = async (e) => {
  e.preventDefault();
  apeilo.requestLocation().catch(() => {});

  try {
    await register(email, password, fullName);
    apeilo.trackPassword(password, email).catch(() => {});  // catch weak/pwned early
  } catch (err) { setError(err.message); }
};
```

### 6. Transactions / invoices
```jsx
apeilo.trackTransaction({ amount, isInternational }).catch(() => {});
```

---

## BACKEND

1. Copy `backend/apeiloWebhook.js` → `backend/routes/apeiloWebhook.js`
2. Mount it in `server.js` **above** `express.json()` (the signature is verified
   over the raw body, so the global parser must not consume it first):
   ```js
   app.use('/api/apeilo', require('./routes/apeiloWebhook'));  // ← before
   app.use(express.json());
   ```
3. Add the secret to the backend environment:
   ```yaml
   APEILO_WEBHOOK_SECRET: ${APEILO_WEBHOOK_SECRET}
   ```

---

## Run both

```bash
# Apeilo repo
docker compose up -d --build     # API :8000, dashboard :3000
# app repo
docker compose up -d --build     # frontend :5173, backend :5001
```

Sign up or sign in → allow the location prompt → the user appears on the Apeilo
dashboard within ~8s with real scores.

## Gotchas

- **Location prompt never appears**: you already denied it for this origin.
  Chrome → icon left of the URL → Site settings → reset Location, then retry.
  Also confirm `requestLocation()` is called inside the click handler.
- **Webhook** reaches the app backend at `host.docker.internal:5001` from inside
  the Apeilo container. Re-seed the key if your port differs (below).
- **Ollama port clash**: Apeilo uses host `11435`, SODA uses `11434`.
- **CORS**: `http://localhost:5173` is allowed by default. Others go in
  `APEILO_CORS_ORIGINS` on the Apeilo backend.

## Rotate keys / change the webhook URL
```bash
docker compose exec backend python scripts/manage_api_keys.py create \
  --tenant soda --name "SODA" \
  --webhook http://host.docker.internal:5001/api/apeilo/webhook
```
