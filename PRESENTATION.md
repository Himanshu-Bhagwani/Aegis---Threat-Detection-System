# Apeilo — Threat Detection System
### Recruiter Presentation Script & Technical Brief

> Use this as a talking script. The **"Say this"** blocks are what you speak;
> the tables and notes are reference so you can answer follow-ups confidently.

---

## 0. One-liner

**Apeilo is a plug-in fraud & account-takeover detection service.** Any app
integrates it with an API key; Apeilo scores every login and transaction in
real time, and calls the app back when it detects a threat — like a Stripe
Radar, but for identity and account security.

---

## 1. The 60-second elevator pitch

> **Say this:**
> "Most apps only find out an account was compromised *after* money's gone or a
> user complains. I built Apeilo to catch it in the moment. It's a standalone
> detection **service** — an app plugs it in with a single API key, and from
> then on every sign-in and transaction gets scored by a blend of machine-
> learning models and behavioural rules. If something looks like an account
> takeover — a brute-force login, an impossible-travel sign-in, a payment far
> outside a user's normal pattern — Apeilo flags it, asks the user *'was this
> you?'*, and can lock the account down in the connected app instantly.
>
> To prove it's genuinely reusable and not a demo, I integrated it into a
> second, completely separate app I built — a corporate finance dashboard —
> and it detects real threats across the two without them sharing a database or
> a codebase. It runs end-to-end with a single `docker compose up`."

---

## 2. The problem

- Account takeover and payment fraud are detected **too late** — usually after
  the damage.
- Most teams **rebuild** the same detection logic in every product.
- Real fraud signals are **relative to the individual** — a ₹5-lakh payment is
  normal for one user and a red flag for another. Fixed thresholds miss this.

**Apeilo's thesis:** detection should be a *shared, pluggable service* that
learns each user's normal behaviour and scores deviations in real time.

---

## 3. What it does — the product

| Capability | What it means |
|---|---|
| **Login anomaly detection** | Brute-force bursts, off-hours sign-ins, new devices, and impossible travel — scored per user against their own baseline. |
| **Fraud transaction scoring** | Every transaction judged against that user's *own* spending history — deviation, velocity, and timing. Works in any currency. |
| **Breach intelligence** | Passwords checked against Have I Been Pwned via k-anonymity; email exposure and password-strength assessment. |
| **Impossible-travel detection** | Compares each sign-in's location & time to the previous one; flags journeys physically impossible to make. |
| **Unified risk score** | All signals fused into one 0–100% score per user, dominated by the login & fraud signals. |
| **Step-up verification** | On a suspicious event, the account owner is asked *"was this you?"* — a "no" becomes a confirmed incident. |
| **Instant account lockdown** | The monitor can block login for a chosen window and force-sign-out every active session in the connected app in seconds. |
| **Multi-tenant by design** | Each connected app is an isolated tenant — separate data namespace, its own API key and webhook. |

---

## 4. Architecture

```
   ┌─────────────────────┐         X-Api-Key          ┌──────────────────────────┐
   │  Connected app       │ ──── SDK / REST ─────────► │  Apeilo API (FastAPI)     │
   │  (e.g. SODA finance   │ ◄─── signed webhook ────── │  • detection modules      │
   │   dashboard)          │      (threat / lockdown)   │  • fusion engine          │
   └─────────────────────┘                             │  • WebSocket live feed    │
                                                        └───────────┬──────────────┘
   ┌─────────────────────┐   polls /profiles, /alerts             │
   │  Apeilo dashboard    │ ◄──────── WebSocket ──────────────────┤
   │  (Next.js)           │                                        │
   └─────────────────────┘                      ┌─────────────────┼─────────────────┐
                                                 ▼                 ▼                 ▼
                                         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                                         │  ML models   │  │  DynamoDB     │  │  HIBP API     │
                                         │ (sklearn/XGB/│  │ (events,      │  │ (k-anonymity  │
                                         │  TensorFlow) │  │  profiles,    │  │  breach check)│
                                         └──────────────┘  │  alerts, keys)│  └──────────────┘
                                                           └──────────────┘
```

> **Say this:** "Three moving parts. The connected app sends events. Apeilo's
> FastAPI backend scores them with the ML models, persists them per-tenant in
> DynamoDB, and pushes live updates to its own Next.js dashboard over a
> WebSocket. When risk crosses a threshold, it calls the connected app back
> with an HMAC-signed webhook."

---

## 5. Tech stack

**Apeilo — detection service**

| Layer | Technology |
|---|---|
| API backend | **FastAPI** + **Uvicorn** (async Python), Pydantic v2, WebSockets |
| ML / data | **scikit-learn**, **XGBoost**, **TensorFlow/Keras**, imbalanced-learn, pandas, numpy |
| Dashboard | **Next.js 14** (App Router), **React 18**, **TypeScript**, Recharts, **Three.js** (3D threat globe) |
| Storage | **AWS DynamoDB** — or a local `amazon/dynamodb-local` container for zero-cost dev |
| Auth & tenancy | API-key per tenant, local **JWT** (python-jose), optional **AWS Cognito**; HMAC-signed webhooks |
| Infra | **Docker** + **docker-compose** — whole stack up with one command |

**SODA — the demo app that consumes Apeilo** *(proves reusability)*

| Layer | Technology |
|---|---|
| Frontend | **Vite** + **React 19** |
| Backend | **Node.js / Express 5** |
| Database | **PostgreSQL** |
| Auth | JWT access/refresh tokens, **bcrypt**, **Google OAuth** |

---

## 6. The detection engine — the technical heart

> This is where you show depth. Pick 2–3 modules to go deep on.

### Login anomaly — ensemble ML + behavioural rules
- **Models:** an ensemble of **Isolation Forest** (unsupervised outliers),
  a **Gradient Boosting Machine**, and a **Keras autoencoder** (reconstruction
  error), trained on the **LANL authentication dataset** (a real-world corporate
  auth-log benchmark).
- **Rules layer:** brute-force (combined with *noisy-OR* so one strong signal
  isn't averaged away), off-hours, new-device, and impossible-travel heuristics.
- **Design decision worth mentioning:** the ML models were trained on enterprise
  auth features that don't map cleanly to consumer web logins, so a naïve blend
  scored *every* login as risky. I made the **rules the primary signal** for the
  web domain and kept ML as a bounded nudge — an example of not trusting a model
  blindly outside its training distribution.

### Fraud scoring — relative-to-the-user, not fixed thresholds
- **Models:** **XGBoost + Isolation Forest** ensemble with a rule-based fallback.
- **Key idea:** each transaction is scored against that user's **own median
  spend** (median resists a single outlier poisoning the baseline), on a
  **log-scaled deviation curve** — so a payment 100× someone's normal is
  critical regardless of the raw amount or currency.
- Adds **velocity** (transactions/hour vs their usual pace) and confirmation
  memory — once a user approves a large payment, similar amounts stop being
  flagged.

### Impossible-travel — physics as a signal
- Haversine distance between two consecutive sign-ins ÷ elapsed time = implied
  speed. Above ~1000 km/h, no legitimate journey explains it → flagged.
- Coordinates are stored rounded to 3 decimals (~110 m) — enough for the
  detection, deliberately **not** a precise movement history (privacy by design).

### Fusion engine
- Combines all module scores into one figure via configurable strategies
  (**weighted average / max-threat / Bayesian**), weighted so **login & fraud
  dominate**, with a floor so a single critical signal still surfaces.

---

## 7. Engineering decisions that show judgment

> **Say these when asked "what was hard?"**

- **Made it a true multi-tenant service, not a demo.** API-key auth,
  per-tenant data namespacing in DynamoDB, and signed webhooks — then proved it
  by integrating a *second, independent* app.
- **Ran real AWS services locally at zero cost.** Swapped AWS DynamoDB for the
  official local container behind the same boto3 code path — the app can't tell
  the difference, so it develops for free and deploys to real AWS unchanged.
- **Instant forced logout across a stateless-JWT app.** Access tokens are
  stateless, so I added an in-memory revocation registry checked in O(1) on each
  request, plus a client heartbeat — an admin clicks "lock", and the attacker is
  signed out within seconds, with no per-request database hit.
- **Privacy-conscious by default** — password never stored (only its breach
  assessment), coordinates rounded, explicit user consent before monitoring.
- **Webhook integrity** — every outbound call is HMAC-SHA256 signed so the
  receiver can verify it genuinely came from Apeilo.

---

## 8. Live demo — 3-minute walkthrough

1. **`docker compose up`** → whole stack boots (backend, dashboard, database).
2. **Sign in to SODA** (the finance app) → a live profile appears on the Apeilo
   dashboard within seconds, with a real login score.
3. **Fail the login a few times, then succeed** → the login-anomaly score climbs;
   Apeilo raises a *"was this you?"* step-up challenge on the monitoring side.
4. **Add a huge transaction in SODA** → it's scored against the user's history
   and flagged critical; the fraud panel shows the amount, timing, and how many
   times larger it is than normal.
5. **Click "No — this wasn't me"** on Apeilo → choose a lock duration → the SODA
   session is signed out automatically and login is blocked for that window.

> **Say this:** "Everything you're seeing is real detection on real events
> flowing between two separate applications — not scripted."

---

## 9. Anticipated recruiter questions

**Q: Is this just a wrapper around an API?**
A: No — it's a detection service with its own trained ML models, a fusion
engine, per-user behavioural baselines, and multi-tenant infrastructure. HIBP is
one external input among several.

**Q: How does it scale?**
A: Stateless FastAPI workers behind a load balancer; DynamoDB is already
horizontally scalable. The one in-memory piece (session revocation) would move
to Redis for multi-instance — I noted that boundary deliberately.

**Q: What would you do next?**
A: Retrain the login models on real consumer web-auth data (the current ones are
enterprise-domain), move revocation to Redis, add IP-geolocation to strengthen
impossible-travel, and add SSE for sub-second push logout.

**Q: What did you learn?**
A: That a model is only as good as its training distribution — the most
important calibration work was recognising where the ML was wrong for my domain
and letting well-designed rules lead. And that building it as a *reusable
service* forced much better boundaries than a monolith would have.

---

## 10. At a glance

- **~2 applications**, fully separate, integrated over a clean API + webhook contract
- **5 detection modules**, real trained ML artifacts (sklearn / XGBoost / TensorFlow)
- **Multi-tenant**, **Dockerised**, **one-command** run
- Languages: **Python**, **TypeScript/JavaScript**, **SQL**
- Frameworks: **FastAPI**, **Next.js**, **React**, **Express**
- Data: **DynamoDB**, **PostgreSQL**, **HIBP**
