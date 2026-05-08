/**
 * Apeilo API Client v2
 * TypeScript client for all backend endpoints.
 * Includes WebSocket manager for real-time event streaming.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE  = process.env.NEXT_PUBLIC_WS_URL  || "ws://localhost:8000";

// ─────────────────────────────────────────────
// Auth token helpers
// ─────────────────────────────────────────────
export function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("aegis_token");
}
export function storeToken(token: string) {
  localStorage.setItem("aegis_token", token);
}
export function clearToken() {
  localStorage.removeItem("aegis_token");
  localStorage.removeItem("aegis_user");
}
export function getAuthHeader(): Record<string, string> {
  const token = getStoredToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ─────────────────────────────────────────────
// Core fetch wrapper
// ─────────────────────────────────────────────
async function apiFetch<T = any>(
  path: string,
  options: RequestInit = {},
  auth = true,
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(auth ? getAuthHeader() : {}),
    ...(options.headers as Record<string, string> || {}),
  };
  const resp = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ═══════════════════════════════════════════════════════
// AUTH
// ═══════════════════════════════════════════════════════
export async function signUp(email: string, password: string, givenName = "", familyName = "") {
  return apiFetch("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, password, given_name: givenName, family_name: familyName }),
  }, false);
}

export async function signIn(email: string, password: string) {
  const result = await apiFetch<{
    success: boolean; access_token: string; id_token?: string;
    email?: string; user_sub?: string; mock_mode?: boolean;
  }>("/auth/signin", { method: "POST", body: JSON.stringify({ email, password }) }, false);
  if (result.access_token) {
    storeToken(result.access_token);
    localStorage.setItem("aegis_user", JSON.stringify({ email: result.email || email, sub: result.user_sub }));
  }
  return result;
}

export async function signOut(accessToken: string) {
  clearToken();
  return apiFetch("/auth/signout", { method: "POST", body: JSON.stringify({ access_token: accessToken }) });
}

// ═══════════════════════════════════════════════════════
// DETECTION ENDPOINTS
// ═══════════════════════════════════════════════════════

export interface GPSPoint {
  lat: number; lng: number; timestamp: number;
  speed?: number; heading?: number; accuracy?: number;
}
// Map frontend lat/lng → backend latitude/longitude field names
function toBackendGPSPoints(trajectory: GPSPoint[]) {
  return trajectory.map(p => ({
    latitude:   p.lat,
    longitude:  p.lng,
    timestamp:  p.timestamp,
    speed:      p.speed      ?? 0,
    heading:    p.heading    ?? 0,
    time_delta: 0,
  }));
}
export async function scoreGPS(trajectory: GPSPoint[]) {
  return apiFetch("/gps/score", {
    method: "POST",
    body: JSON.stringify({ trajectory: toBackendGPSPoints(trajectory) }),
  });
}

export interface LoginEventInput {
  user_deg?: number; comp_deg?: number;
  hour_of_day?: number; failed_10min?: number;
  is_new_comp?: number; is_new_user?: number;
  impossible_travel?: number; time_since_user_last?: number;
}
export async function scoreLogin(data: LoginEventInput) {
  return apiFetch("/login/score", { method: "POST", body: JSON.stringify(data) });
}

export async function scorePassword(password: string) {
  return apiFetch("/password/score", { method: "POST", body: JSON.stringify({ password }) });
}

export interface FraudInput {
  amount: number; is_international?: boolean; hour?: number;
  tx_count_1h?: number; time_since_last_tx?: number;
  amount_ratio?: number; merchant_freq_user?: number;
  device_changed?: number; distance_from_home?: number;
}
export async function scoreFraud(data: FraudInput) {
  // Router expects { payload: dict } wrapper
  return apiFetch("/fraud/score", { method: "POST", body: JSON.stringify({ payload: data }) });
}

export interface DeviceFingerprint {
  user_agent?: string; platform?: string; screen_resolution?: string;
  timezone?: string; language?: string; hardware_concurrency?: number;
  color_depth?: number; touch_support?: boolean; webgl_renderer?: string;
}
export async function scoreDevice(
  userId: string,
  fingerprint: DeviceFingerprint,
  opts: {
    failed_biometric?: boolean; failed_biometric_count?: number;
    app_unlock_attempted?: boolean; app_bundle?: string;
    unusual_time?: boolean; location_mismatch?: boolean;
    rooted_jailbroken?: boolean; emulator_detected?: boolean;
  } = {}
) {
  return apiFetch("/device/score", {
    method: "POST",
    body: JSON.stringify({ user_id: userId, fingerprint, ...opts }),
  });
}

export async function checkPasswordBreach(password: string, userId?: string) {
  return apiFetch("/breach/check/password", {
    method: "POST",
    body: JSON.stringify({ password, user_id: userId }),
  });
}

export async function checkEmailBreach(email: string, userId?: string) {
  return apiFetch("/breach/check/email", {
    method: "POST",
    body: JSON.stringify({ email, user_id: userId }),
  });
}

export interface UnifiedRiskInput {
  user_id?: string;
  event_id?: string;
  gps_data?: { spoof_probability?: number; confidence?: number; trajectory?: GPSPoint[] };
  login_data?: LoginEventInput & { anomaly_probability?: number };
  password?: string;
  password_score?: number;
  transaction_data?: { fraud_probability?: number; amount?: number; is_international?: boolean };
  fusion_strategy?: "weighted_average" | "max_threat" | "bayesian";
}
export interface UnifiedRiskResponse {
  unified_score: number;
  risk_level: string;
  gps_risk: number;
  login_risk: number;
  password_risk: number;
  fraud_risk: number;
  breach_risk?: number;
  device_risk?: number;
  primary_threats: string[];
  recommended_actions: string[];
  fusion_strategy: string;
}
export async function computeUnifiedRisk(data: UnifiedRiskInput) {
  return apiFetch("/risk/unified", { method: "POST", body: JSON.stringify(data) });
}

// ═══════════════════════════════════════════════════════
// IDENTITY & ALERTS
// ═══════════════════════════════════════════════════════
export async function getIdentityProfile(userId: string) {
  return apiFetch(`/identity/${userId}`);
}
export async function getUserEvents(userId: string, limit = 50, eventType?: string) {
  const q = eventType ? `?limit=${limit}&event_type=${eventType}` : `?limit=${limit}`;
  return apiFetch(`/identity/${userId}/events${q}`);
}
export async function getAlerts(limit = 20) {
  return apiFetch(`/alerts?limit=${limit}`);
}
export async function dismissAlert(alertId: string) {
  return apiFetch(`/alerts/${alertId}/dismiss`, { method: "POST" });
}

// ═══════════════════════════════════════════════════════
// HEALTH
// ═══════════════════════════════════════════════════════
export async function getHealth() {
  return apiFetch("/health", {}, false);
}

// ═══════════════════════════════════════════════════════
// WEBSOCKET MANAGER
// ═══════════════════════════════════════════════════════

export type WSEventType =
  | "connected" | "heartbeat" | "detection_event"
  | "risk_update" | "alert" | "system";

export interface WSEvent {
  type:           WSEventType;
  event_type?:    string;
  user_id?:       string;
  risk_score?:    number;
  risk_level?:    string;
  timestamp?:     string;
  details?:       Record<string, unknown>;
  event_id?:      string;
  unified_score?: number;
  primary_threats?: string[];
  message?:       string;
}

type WSCallback = (event: WSEvent) => void;

class WebSocketClient {
  private ws:          WebSocket | null = null;
  private callbacks:   Set<WSCallback>  = new Set();
  private reconnectMs: number           = 3000;
  private dead:        boolean          = false;

  connect(url: string = `${WS_BASE}/ws`) {
    if (typeof window === "undefined") return;
    if (this.ws?.readyState === WebSocket.OPEN) return;
    try {
      this.ws = new WebSocket(url);
      this.ws.onmessage = (e) => {
        try { const data = JSON.parse(e.data); this.callbacks.forEach(cb => cb(data)); }
        catch {}
      };
      this.ws.onclose = () => {
        if (!this.dead) setTimeout(() => this.connect(url), this.reconnectMs);
      };
      this.ws.onerror = () => this.ws?.close();
    } catch {}
  }

  subscribe(cb: WSCallback): () => void {
    this.callbacks.add(cb);
    return () => this.callbacks.delete(cb);
  }

  send(data: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  get status(): "connected" | "connecting" | "disconnected" {
    if (!this.ws) return "disconnected";
    if (this.ws.readyState === WebSocket.OPEN)       return "connected";
    if (this.ws.readyState === WebSocket.CONNECTING) return "connecting";
    return "disconnected";
  }

  destroy() {
    this.dead = true;
    this.ws?.close();
  }
}

export const wsClient = new WebSocketClient();

// Auto-connect when running in the browser
if (typeof window !== "undefined") {
  wsClient.connect(`${WS_BASE}/ws`);
}

// ═══════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════

/** Collect browser device fingerprint fields for /device/score */
export function collectBrowserFingerprint(): DeviceFingerprint {
  if (typeof window === "undefined") return {};
  return {
    user_agent:           typeof navigator !== "undefined" ? navigator.userAgent : undefined,
    platform:             typeof navigator !== "undefined" ? navigator.platform : undefined,
    language:             typeof navigator !== "undefined" ? navigator.language : undefined,
    hardware_concurrency: typeof navigator !== "undefined" ? navigator.hardwareConcurrency : undefined,
    color_depth:          typeof screen !== "undefined" ? screen.colorDepth : undefined,
    touch_support:        typeof navigator !== "undefined" ? navigator.maxTouchPoints > 0 : undefined,
    screen_resolution:    typeof screen !== "undefined" ? `${screen.width}x${screen.height}` : undefined,
    timezone:             typeof Intl !== "undefined" ? Intl.DateTimeFormat().resolvedOptions().timeZone : undefined,
    webgl_renderer: (() => {
      try {
        const c = document.createElement("canvas");
        const gl = c.getContext("webgl") || c.getContext("experimental-webgl");
        if (gl && "getExtension" in gl) {
          const dbg = (gl as WebGLRenderingContext).getExtension("WEBGL_debug_renderer_info");
          if (dbg) return (gl as WebGLRenderingContext).getParameter(dbg.UNMASKED_RENDERER_WEBGL) as string;
        }
      } catch { /* ignore */ }
      return undefined;
    })(),
  };
}

/** CSS colour token for a risk level string or numeric score 0-1 */
export function riskColor(level: string | number): string {
  if (typeof level === "number") {
    if (level >= 0.75) return "#ff0040";
    if (level >= 0.50) return "#ff6600";
    if (level >= 0.25) return "#ffaa00";
    return "#00ff88";
  }
  switch (level?.toLowerCase()) {
    case "critical": return "#ff0040";
    case "high":     return "#ff6600";
    case "medium":   return "#ffaa00";
    case "low":      return "#00ff88";
    default:         return "#888888";
  }
}

/** Human-readable label for a risk level string or numeric score */
export function riskLabel(level: string | number): string {
  if (typeof level === "number") {
    if (level >= 0.75) return "Critical";
    if (level >= 0.50) return "High";
    if (level >= 0.25) return "Medium";
    if (level >= 0.10) return "Low";
    return "Minimal";
  }
  const s = String(level ?? "");
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

/** Format a 0-1 score as a percentage string */
export function formatScore(score: number | null | undefined, decimals = 1): string {
  if (score == null) return "—";
  return `${(score * 100).toFixed(decimals)}%`;
}
