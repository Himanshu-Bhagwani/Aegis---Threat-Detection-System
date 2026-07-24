/**
 * Apeilo client for SODA (Vite + React)
 * =====================================
 * A tiny, dependency-free wrapper around the Apeilo Threat Detection API.
 * Copy to:  frontend/src/services/apeiloClient.js
 *
 * Every request carries SODA's `X-Api-Key`, so all of SODA's events, user
 * profiles and alerts stay in SODA's own isolated namespace on Apeilo.
 */

function toLevel(v) {
  const s = String(v || "unknown");
  return ["minimal", "low", "medium", "high", "critical"].includes(s) ? s : "unknown";
}

// Pull a 0..1 score out of any of Apeilo's response shapes.
function extractScore(r) {
  const n =
    r?.unified_score ??
    r?.spoof_probability ??
    r?.anomaly_probability ??
    r?.fraud_probability ??
    r?.breach_probability ??
    r?.risk_score ??
    r?.device_risk_score ??
    0;
  return typeof n === "number" ? n : 0;
}

/* ── Failed-login bookkeeping ─────────────────────────────────────────────
   Browsers give us no server-side view of failed attempts, so we count them
   locally per email within a rolling 10-minute window — the same window the
   Apeilo login model expects as `failed_10min`.                           */

const FAIL_KEY = "apeilo_failed_logins";
const FAIL_WINDOW_MS = 10 * 60 * 1000;

function readFailures() {
  try { return JSON.parse(localStorage.getItem(FAIL_KEY) || "{}"); } catch { return {}; }
}
function writeFailures(obj) {
  try { localStorage.setItem(FAIL_KEY, JSON.stringify(obj)); } catch { /* ignore */ }
}
/** Timestamps of failures for `email` inside the rolling window. */
export function recentFailures(email) {
  if (!email) return [];
  const all = readFailures();
  return (all[email] || []).filter((ts) => Date.now() - ts < FAIL_WINDOW_MS);
}
function addFailure(email) {
  const all = readFailures();
  const list = recentFailures(email);
  list.push(Date.now());
  all[email] = list;
  writeFailures(all);
  return list.length;
}
function clearFailures(email) {
  const all = readFailures();
  delete all[email];
  writeFailures(all);
}

/* ── Known-device detection ──────────────────────────────────────────────
   The first time an account is seen in this browser it's a new device — the
   signal the login model reads as `is_new_comp`. Covers both a fresh sign-up
   and an existing account signing in somewhere new.                       */

const KNOWN_KEY = "apeilo_known_users";

function isKnownUser(uid) {
  try { return JSON.parse(localStorage.getItem(KNOWN_KEY) || "[]").includes(uid); }
  catch { return false; }
}
function markUserKnown(uid) {
  try {
    const list = JSON.parse(localStorage.getItem(KNOWN_KEY) || "[]");
    if (!list.includes(uid)) {
      list.push(uid);
      localStorage.setItem(KNOWN_KEY, JSON.stringify(list));
    }
  } catch { /* ignore */ }
}

/* ── Monitoring consent ──────────────────────────────────────────────────
   Asked once, at first sign-up / sign-in. Location is only ever requested
   after the user allows it.                                              */

export const CONSENT_KEY = "apeilo_monitoring_consent";

/** true = allowed, false = declined, null = not asked yet. */
export function hasConsent() {
  try {
    const v = localStorage.getItem(CONSENT_KEY);
    return v === null ? null : v === "true";
  } catch { return null; }
}
export function setConsent(allowed) {
  try { localStorage.setItem(CONSENT_KEY, allowed ? "true" : "false"); } catch { /* ignore */ }
}

/* ── Geolocation ─────────────────────────────────────────────────────────
   Browsers only reliably show the permission prompt for a call made during a
   user gesture, so SODA calls requestLocation() from the sign-in click. The
   result is cached here and reused by the login scoring that follows.     */

let _lastPosition = null;

function getPosition() {
  return new Promise((resolve) => {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      console.warn("[Apeilo] geolocation unavailable in this browser");
      return resolve(null);
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      (err) => {
        console.warn("[Apeilo] geolocation denied/unavailable:", err && err.message);
        resolve(null);
      },
      { timeout: 10000, maximumAge: 300000, enableHighAccuracy: false },
    );
  });
}

/**
 * @param {{ apiUrl?: string, apiKey: string, userId: string, debug?: boolean,
 *           onRisk?: (r: object) => void }} config
 */
export function createApeilo(config) {
  const apiUrl = (config.apiUrl || "http://localhost:8000").replace(/\/$/, "");

  const log = (...a) => config.debug && console.log("[Apeilo]", ...a);

  async function post(path, body) {
    const res = await fetch(apiUrl + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Api-Key": config.apiKey },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`Apeilo ${path} -> ${res.status} ${await res.text().catch(() => "")}`);
    return res.json();
  }

  async function get(path) {
    const res = await fetch(apiUrl + path, { headers: { "X-Api-Key": config.apiKey } });
    if (!res.ok) throw new Error(`Apeilo ${path} -> ${res.status}`);
    return res.json();
  }

  function normalise(event, raw, detail) {
    const result = {
      event,
      score: extractScore(raw),
      level: toLevel(raw?.risk_level),
      primaryThreats: raw?.primary_threats || [],
      recommendedActions: raw?.recommended_actions || [],
      detail: detail || {},   // context for the "was this you?" prompt
      raw,
    };
    if (typeof config.onRisk === "function") config.onRisk(result);
    return result;
  }

  return {
    /** Change the tracked user (e.g. after a fresh login). */
    setUser(userId) {
      config.userId = userId;
    },

    /**
     * Ask the browser for the user's location. Call this from a click handler
     * (e.g. the sign-in button) so the permission prompt actually appears.
     * The position is cached and reused by the login scoring.
     */
    async requestLocation() {
      const pos = await getPosition();
      if (pos) _lastPosition = pos;
      return pos;
    },

    /** The most recently captured position, if any. */
    getLastPosition() {
      return _lastPosition;
    },

    /**
     * Register/refresh a user's profile on Apeilo (name + email), so the user
     * appears on the dashboard from their very first sign-in.
     */
    async register(name, email, userId) {
      const raw = await post("/identity/register", {
        user_id: userId || config.userId,
        name: name || "",
        email: email || "",
      });
      log("registered", userId || config.userId);
      return raw;
    },

    /**
     * One-call session tracking — use for BOTH sign-up and sign-in.
     * Registers the profile, folds in recent failed attempts, flags a
     * first-time device, attaches GPS, then scores the session. Never throws —
     * tracking must not block your app.
     */
    async identify({ userId, name, email, newDevice } = {}) {
      try {
        if (userId) config.userId = userId;
        const uid = config.userId;
        await this.register(name, email, uid);

        const failedAttempts = recentFailures(uid).length;
        // A brand-new sign-up, or a known account on an unfamiliar browser.
        const firstTimeHere = !isKnownUser(uid);
        markUserKnown(uid);

        // Location only when the user has explicitly allowed monitoring.
        const gps = hasConsent() === true
          ? (_lastPosition || (await this.requestLocation()))
          : null;

        const res = await this.trackLogin({
          failedAttempts,
          newDevice: newDevice !== undefined ? newDevice : firstTimeHere,
          gps,
          userId: uid,
          success: true,   // a real sign-in — this is what shapes their baseline
        });
        clearFailures(uid); // a successful sign-in resets the counter

        // A success right after a brute-force burst can't be trusted on its own.
        // Signal the app to hold access until the owner confirms on Apeilo.
        const STEP_UP_THRESHOLD = 5;
        return {
          result: res,
          failedAttempts,
          stepUpRequired: failedAttempts >= STEP_UP_THRESHOLD,
        };
      } catch (e) {
        log("identify failed (non-fatal)", e?.message || e);
        return { result: null, failedAttempts: 0, stepUpRequired: false };
      }
    },

    /**
     * Poll the step-up status for a suspicious success.
     * Returns { status: 'pending' | 'confirmed' | 'denied' | 'none' }.
     */
    async getStepUpStatus(userId) {
      try {
        return await get(`/identity/${encodeURIComponent(userId || config.userId)}/stepup-status`);
      } catch {
        return { status: "pending" };   // keep waiting on a transient error
      }
    },

    /**
     * Record a FAILED sign-in attempt for `email` and score it. Repeated
     * failures push that account's login-anomaly score up on the dashboard.
     */
    async trackFailedLogin(email) {
      if (!email) return null;
      try {
        const count = addFailure(email);
        // Make sure the account exists on the dashboard even if login never succeeds
        // (this is exactly the brute-force case worth surfacing).
        await this.register("", email, email).catch(() => {});
        return await this.trackLogin({
          userId: email,
          failedAttempts: count,
          gps: _lastPosition,
          success: false,  // a failure must never make this hour look "normal"
        });
      } catch (e) {
        log("trackFailedLogin failed (non-fatal)", e?.message || e);
        return null;
      }
    },

    /** Score a login attempt. Uses the unified engine so it factors in history. */
    async trackLogin(info = {}) {
      const now = new Date();
      const body = {
        user_id: info.userId || config.userId,
        login_data: {
          hour_of_day: info.hourOfDay ?? now.getHours(),   // real local hour
          failed_10min: info.failedAttempts ?? 0,
          is_new_comp: info.newDevice ? 1 : 0,
          // Only successful sign-ins build the user's "normal hours" baseline.
          ...(info.success !== undefined ? { success: !!info.success } : {}),
        },
      };
      const gps = info.gps || _lastPosition;
      if (gps && typeof gps.lat === "number") {
        body.gps_data = { trajectory: [{ lat: gps.lat, lng: gps.lng, timestamp: Date.now() }] };
      }
      const raw = await post("/risk/unified", body);
      log("login", raw.unified_score, raw.risk_level);
      return normalise("login", raw, {
        hour: body.login_data.hour_of_day,
        failed_attempts: body.login_data.failed_10min,
        new_device: !!info.newDevice,
      });
    },

    /**
     * Score a financial / high-value transaction for fraud.
     * NOTE: the API expects the transaction fields under `payload`.
     */
    async trackTransaction(tx = {}) {
      const now = new Date();
      const raw = await post("/fraud/score", {
        user_id: tx.userId || config.userId,
        payload: {
          amount: Number(tx.amount) || 0,
          is_international: !!tx.isInternational,
          hour: tx.hour ?? now.getHours(),
          tx_count_1h: tx.txCount1h ?? 1,
          time_since_last_tx: tx.timeSinceLastTx ?? 3600,
          merchant_freq_user: tx.merchantFreq ?? 1,
          // NOTE: amount_ratio is deliberately omitted. The server derives how
          // far this amount deviates from THIS user's own history — sending a
          // hard-coded 1.0 here told it "perfectly normal" and suppressed the
          // strongest fraud signal we have.
          ...(tx.amountRatio !== undefined ? { amount_ratio: tx.amountRatio } : {}),
        },
      });
      log("transaction", raw.fraud_probability);
      return normalise("transaction", raw, {
        amount: Number(tx.amount) || 0,
        hour: tx.hour ?? now.getHours(),
        source: tx.source || "manual",
      });
    },

    /**
     * Score a batch of transactions (e.g. an uploaded PDF/CSV statement).
     * Returns the individual results plus the riskiest one, so the caller can
     * challenge the user about it.
     * @param {Array<{amount:number, hour?:number, isInternational?:boolean, date?:string}>} list
     */
    async trackTransactionBatch(list = [], opts = {}) {
      const items = (list || []).filter(t => Number(t.amount) > 0);
      if (!items.length) return { scored: [], riskiest: null };

      const scored = [];
      // Sequential: each score uses the running history, so a statement's own
      // outliers are judged against the rest of that user's spending.
      for (const t of items) {
        try {
          const r = await this.trackTransaction({
            amount: t.amount,
            hour: t.hour,
            isInternational: t.isInternational,
            source: opts.source || "statement_upload",
          });
          scored.push(r);
        } catch (e) {
          log("batch item failed (skipped)", e?.message || e);
        }
      }
      const riskiest = scored.reduce((a, b) => (b.score > (a?.score ?? -1) ? b : a), null);
      log("batch scored", scored.length, "riskiest", riskiest?.score);
      return { scored, riskiest };
    },

    /**
     * Report the user's answer to a "was this you?" challenge.
     * A denial becomes a confirmed incident on Apeilo (alert + webhook).
     */
    async verifyActivity({ activity = "transaction", confirmed, riskScore = 0, detail = {} } = {}) {
      try {
        return await post("/identity/verify-activity", {
          user_id: config.userId,
          activity,
          confirmed: !!confirmed,
          risk_score: riskScore,
          detail,
        });
      } catch (e) {
        log("verifyActivity failed (non-fatal)", e?.message || e);
        return null;
      }
    },

    /**
     * Breach-check a password (only a hash prefix leaves the browser boundary
     * on the server side — k-anonymity). Populates the user's breach risk.
     */
    async trackPassword(password, userId) {
      if (!password) return null;
      try {
        const raw = await post("/breach/check/password", {
          password,
          user_id: userId || config.userId,
        });
        log("password breach", raw.risk_level, raw.is_pwned);
        return normalise("password", raw);
      } catch (e) {
        log("trackPassword failed (non-fatal)", e?.message || e);
        return null;
      }
    },

    /** Push a GPS observation for spoof / impossible-travel detection. */
    async pushGPS(lat, lng) {
      const raw = await post("/gps/score", {
        user_id: config.userId,
        trajectory: [{ lat, lng, timestamp: Date.now() }],
      });
      log("gps", raw.spoof_probability);
      return normalise("gps", raw);
    },

    /** Run a full unified risk score on demand. */
    async scoreNow(extra = {}) {
      const now = new Date();
      const raw = await post("/risk/unified", {
        user_id: config.userId,
        login_data: { hour_of_day: now.getHours(), failed_10min: 0, is_new_comp: 0 },
        ...extra,
      });
      return normalise("unified", raw);
    },
  };
}
