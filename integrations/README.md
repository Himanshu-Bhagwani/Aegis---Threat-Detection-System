# Integrating your app with Apeilo

Drop-in files to connect **any** application to Apeilo. Apeilo scores your
users' sign-ins and transactions in real time and calls your app back over a
signed webhook when it detects a threat.

```
   Your frontend            Apeilo API                     Your backend
   ─────────────           ─────────────                  ─────────────
   track login/txn ──►  scores + stores (your tenant)
                        risk high/critical? ───────────►  POST /api/apeilo/webhook
                                                          verify + react
```

Everything runs locally with no cloud account required.

## The files

| File in this folder | Copy into your app | Purpose |
|---|---|---|
| `frontend/apeiloClient.js` | e.g. `src/services/apeiloClient.js` | Tiny, dependency-free API client |
| `frontend/ApeiloContext.jsx` | e.g. `src/context/ApeiloContext.jsx` | React provider + `useApeilo()` hook + threat toast |
| `backend/apeiloWebhook.js` | e.g. `backend/routes/apeiloWebhook.js` | Receives & verifies threat / lockdown webhooks |

> `apeiloClient.js` is framework-agnostic — use it directly from React, Vue,
> Svelte, or plain JS. `ApeiloContext.jsx` is a convenience wrapper for React;
> skip it and call `createApeilo()` yourself in any other stack.

## 1. Get a tenant API key

Each app is an isolated tenant. Create a key (and register a webhook URL) with:

```bash
docker compose exec backend python scripts/manage_api_keys.py create \
  --tenant myapp --name "My App" \
  --webhook http://host.docker.internal:4000/api/apeilo/webhook
```

It prints an **API key** and a **webhook secret** — put them in your app's
environment. Point `--webhook` at wherever your backend will receive callbacks.

---

## 2. Frontend

### a. Copy the client (and, for React, the provider)
```
src/services/apeiloClient.js   ← frontend/apeiloClient.js
src/context/ApeiloContext.jsx  ← frontend/ApeiloContext.jsx   (React only)
```

### b. Add the API key + URL to your frontend env
```
APEILO_URL=http://localhost:8000
APEILO_API_KEY=<your api key>
```
(Use whatever env mechanism your build uses — e.g. `VITE_…`, `NEXT_PUBLIC_…`.)

### c. Wrap your app (React)
```jsx
import { ApeiloProvider } from "./context/ApeiloContext";

<ApeiloProvider
  apiKey={APEILO_API_KEY}
  apiUrl={APEILO_URL}
  userId={user?.email || "guest"}
  name={user?.name || ""}
  email={user?.email || ""}
>
  {/* your app */}
</ApeiloProvider>
```
The provider watches `userId`. The moment it becomes a real signed-in user, it
registers the profile, requests GPS (with permission), and scores the sign-in.

### d. Track events
```js
const apeilo = useApeilo();               // or createApeilo({...}) outside React

await apeilo.trackLogin({ success: true, method: "password" });
await apeilo.trackFailedLogin(email);      // in your login catch block
await apeilo.trackTransaction({ amount, isInternational });
await apeilo.trackPassword(password, email);   // breach + strength check
await apeilo.pushGPS(lat, lng);
```

> **Location:** call `apeilo.requestLocation()` **inside the sign-in click
> handler**. Browsers only show the location prompt during a user gesture —
> calling it later is silently ignored.

---

## 3. Backend

1. Copy `backend/apeiloWebhook.js` → e.g. `backend/routes/apeiloWebhook.js`
2. Mount it **before** any global JSON body parser — the signature is verified
   over the raw request body, so the parser must not consume it first:
   ```js
   app.use("/api/apeilo", require("./routes/apeiloWebhook"));  // ← before
   app.use(express.json());
   ```
3. Add the webhook secret to your backend environment:
   ```
   APEILO_WEBHOOK_SECRET=<your webhook secret>
   ```
4. Fill in the `TODO` block in the webhook to react — notify the user, force
   re-auth, write an audit row, etc.

---

## 4. What gets tracked

| Signal | How |
|---|---|
| Profile created (name + email) | provider auto-registers on sign-in |
| Session scored (real local hour, new-device) | provider auto on sign-in |
| GPS location | `requestLocation()` in your sign-in handler |
| Failed attempts (10-min window) | `trackFailedLogin(email)` in the login catch |
| Password breach (HIBP) | `trackPassword(password, email)` after auth |
| Transactions / invoices → fraud | `trackTransaction({ amount })` on submit |

---

## 5. Forced sign-out (optional)

When an account is locked from the Apeilo dashboard, Apeilo can revoke every
active session. The provider clears the local token and fires a browser event:

```js
window.dispatchEvent(new CustomEvent("apeilo:session-revoked"));
```

Have your auth layer listen for it and drop to the login screen:

```js
window.addEventListener("apeilo:session-revoked", () => {
  /* clear auth state, redirect to /login */
});
```

For *instant* revocation of an active session, your backend needs to reject the
locked user's token on its next request (e.g. an in-memory revocation check in
your auth middleware) — the webhook payload tells you which user and for how
long.

---

## Notes

- **Tenant isolation:** every event is stored under `<tenant>#<userId>`, so your
  app never sees another tenant's data.
- **CORS:** add your frontend origin to `APEILO_CORS_ORIGINS` on the Apeilo
  backend if it isn't on the default localhost allow-list.
- **No key = default tenant:** requests without `X-Api-Key` fall back to
  Apeilo's built-in demo tenant. Set `APEILO_REQUIRE_API_KEY=true` to reject
  keyless requests.
