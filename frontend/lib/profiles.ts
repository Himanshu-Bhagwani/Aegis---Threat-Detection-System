/**
 * Apeilo Profile Store
 */

export interface ProfileMetrics {
  gps_spoof:      number;
  login_anomaly:  number;
  password_leak:  number;
  fraud_risk:     number;
  breach_risk:    number;
  unified_score:  number;
  risk_level:     string;
  last_updated:   string;
}

export interface Profile {
  id:          string;
  name:        string;
  email:       string;
  created_at:  string;
  metrics:     ProfileMetrics;
  notes:       string;
  is_demo:     boolean;
}

export interface AlertRecord {
  id:         string;
  profile_id: string;
  user_name:  string;
  metric:     string;
  value:      number;
  threshold:  number;
  severity:   "high" | "critical";
  message:    string;
  created_at: string;
  dismissed:  boolean;
}

export const THRESHOLDS: Record<string, number> = {
  gps_spoof:     0.60,
  login_anomaly: 0.65,
  password_leak: 0.60,
  fraud_risk:    0.60,
  breach_risk:   0.50,
  unified_score: 0.70,
};

export const METRIC_LABELS: Record<string, string> = {
  gps_spoof:     "GPS Spoofing",
  login_anomaly: "Login Anomaly",
  password_leak: "Password Leak",
  fraud_risk:    "Fraud Possibility",
  breach_risk:   "Breach Risk",
  unified_score: "Unified Risk",
};

export function scoreToLevel(s: number): string {
  if (s >= 0.75) return "critical";
  if (s >= 0.50) return "high";
  if (s >= 0.25) return "medium";
  if (s >= 0.10) return "low";
  return "minimal";
}

export function computeUnified(m: Pick<ProfileMetrics, "gps_spoof"|"login_anomaly"|"password_leak"|"fraud_risk"|"breach_risk">): number {
  // Weights match the backend fusion engine (src/fusion/risk_scoring.py)
  const W_GPS = 1.5, W_LOGIN = 2.0, W_PW = 1.0, W_FRAUD = 2.5, W_BREACH = 1.8;
  const total = W_GPS + W_LOGIN + W_PW + W_FRAUD + W_BREACH; // 8.8
  return (
    m.gps_spoof    * W_GPS   +
    m.login_anomaly * W_LOGIN +
    m.password_leak * W_PW   +
    m.fraud_risk    * W_FRAUD +
    m.breach_risk   * W_BREACH
  ) / total;
}

const NOW = new Date().toISOString();

const DEMO_PROFILES: Profile[] = [
  {
    id: "himanshu", name: "Himanshu", email: "himanshu@apeilo.local",
    created_at: NOW, is_demo: true,
    notes: "High-risk profile: data breach confirmed, GPS spoofing detected, multiple failed logins",
    metrics: {
      gps_spoof: 0.87, login_anomaly: 0.74, password_leak: 0.79,
      fraud_risk: 0.66, breach_risk: 0.85,
      unified_score: 0.782, risk_level: "critical", last_updated: NOW,
    },
  },
  {
    id: "anmol", name: "Anmol", email: "anmol@apeilo.local",
    created_at: NOW, is_demo: true,
    notes: "Medium-risk profile: password anomaly detected, occasional unusual login hours",
    metrics: {
      gps_spoof: 0.22, login_anomaly: 0.48, password_leak: 0.73,
      fraud_risk: 0.36, breach_risk: 0.31,
      unified_score: 0.42, risk_level: "high", last_updated: NOW,
    },
  },
  {
    id: "sanidhya", name: "Sanidhya", email: "sanidhya@apeilo.local",
    created_at: NOW, is_demo: true,
    notes: "Low-risk profile: normal behavior, strong passwords, no breaches detected",
    metrics: {
      gps_spoof: 0.04, login_anomaly: 0.07, password_leak: 0.11,
      fraud_risk: 0.05, breach_risk: 0.03,
      unified_score: 0.06, risk_level: "minimal", last_updated: NOW,
    },
  },
  {
    id: "rudra", name: "Rudra", email: "rudra@apeilo.local",
    created_at: NOW, is_demo: true,
    notes: "Low-medium risk profile: slightly elevated password and breach indicators",
    metrics: {
      gps_spoof: 0.25, login_anomaly: 0.30, password_leak: 0.35,
      fraud_risk: 0.28, breach_risk: 0.37,
      unified_score: 0.31, risk_level: "medium", last_updated: NOW,
    },
  },
];

const STORAGE_KEY = "apeilo_profiles_v3";

export function loadProfiles(): Profile[] {
  if (typeof window === "undefined") return DEMO_PROFILES;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEMO_PROFILES;
    const parsed: Profile[] = JSON.parse(raw);
    const custom = parsed.filter(p => !p.is_demo);
    return [...DEMO_PROFILES, ...custom];
  } catch {
    return DEMO_PROFILES;
  }
}

export function saveProfiles(profiles: Profile[]) {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles));
}

export function addProfile(profiles: Profile[], name: string, email: string): Profile[] {
  const now = new Date().toISOString();
  const newProfile: Profile = {
    id: `user_${Date.now()}`, name, email,
    created_at: now, is_demo: false, notes: "Custom profile",
    metrics: {
      gps_spoof: 0, login_anomaly: 0, password_leak: 0,
      fraud_risk: 0, breach_risk: 0,
      unified_score: 0, risk_level: "minimal", last_updated: now,
    },
  };
  const updated = [...profiles, newProfile];
  saveProfiles(updated);
  return updated;
}

export function updateProfileMetrics(
  profiles: Profile[],
  id: string,
  partial: Partial<Pick<ProfileMetrics, "gps_spoof"|"login_anomaly"|"password_leak"|"fraud_risk"|"breach_risk"|"unified_score">>
): Profile[] {
  const updated = profiles.map(p => {
    if (p.id !== id) return p;
    const m = { ...p.metrics, ...partial };
    // Use backend-provided unified_score when available; otherwise recompute locally with correct weights
    const unified = partial.unified_score != null ? partial.unified_score : computeUnified(m);
    return { ...p, metrics: { ...m, unified_score: unified, risk_level: scoreToLevel(unified), last_updated: new Date().toISOString() } };
  });
  saveProfiles(updated);
  return updated;
}

export function generateAlerts(profiles: Profile[]): AlertRecord[] {
  const alerts: AlertRecord[] = [];
  for (const p of profiles) {
    const m = p.metrics;
    const checks: { key: keyof typeof THRESHOLDS; value: number }[] = [
      { key: "gps_spoof",     value: m.gps_spoof },
      { key: "login_anomaly", value: m.login_anomaly },
      { key: "password_leak", value: m.password_leak },
      { key: "fraud_risk",    value: m.fraud_risk },
      { key: "breach_risk",   value: m.breach_risk },
      { key: "unified_score", value: m.unified_score },
    ];
    for (const { key, value } of checks) {
      if (value > THRESHOLDS[key]) {
        alerts.push({
          id: `${p.id}_${key}`, profile_id: p.id, user_name: p.name,
          metric: METRIC_LABELS[key], value, threshold: THRESHOLDS[key],
          severity: value >= 0.75 ? "critical" : "high",
          message: `${METRIC_LABELS[key]} score ${(value * 100).toFixed(0)}% exceeds threshold ${(THRESHOLDS[key] * 100).toFixed(0)}%`,
          created_at: m.last_updated, dismissed: false,
        });
      }
    }
  }
  return alerts.sort((a, b) => b.value - a.value);
}
