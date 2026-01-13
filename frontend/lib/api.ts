/**
 * Aegis Threat Detection API Client
 * 
 * Provides typed interfaces for all threat detection endpoints:
 * - GPS Spoofing Detection
 * - Login Anomaly Detection
 * - Password Risk Assessment
 * - Transaction Fraud Detection
 * - Unified Risk Scoring
 */

import { getAuthHeader } from "./auth";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

// ============================================================
// Types
// ============================================================

// Auth Types
type Credentials = {
  email: string;
  password: string;
};

// GPS Types
export interface GPSPoint {
  latitude: number;
  longitude: number;
  speed?: number;
  acceleration?: number;
  heading?: number;
  heading_change?: number;
  timestamp?: number;
  time_delta?: number;
}

export interface GPSSpoofResponse {
  spoof_probability: number;
  is_spoofed: boolean;
  confidence: number;
  model_scores: Record<string, number>;
  models_used: string[];
  user_id?: string;
  device_id?: string;
}

// Login Types
export interface LoginEvent {
  user_deg?: number;
  comp_deg?: number;
  time_since_user_last?: number;
  time_since_comp_last?: number;
  hour_of_day?: number;
  is_new_user?: number;
  is_new_comp?: number;
  failed_10min?: number;
  impossible_travel?: number;
  device_changed?: number;
  user_id?: string;
  session_id?: string;
  source_ip?: string;
}

export interface LoginAnomalyResponse {
  anomaly_probability: number;
  is_anomalous: boolean;
  risk_level: string;
  confidence: number;
  model_scores: Record<string, number>;
  rule_scores: Record<string, number>;
  models_used: string[];
  user_id?: string;
  session_id?: string;
}

// Password Types
export interface PasswordScoreResponse {
  breach_probability: number;
  strength_score: number;
  risk_level: string;
  features: Record<string, number>;
  recommendations: string[];
  entropy_bits: number;
}

// Fraud Types
export interface FraudScoreResponse {
  fraud_probability: number;
  method?: "ml_model" | "rule_based" | "fallback";
  confidence?: number;
}

// Unified Risk Types
export interface UnifiedRiskRequest {
  user_id?: string;
  event_id?: string;
  gps_data?: {
    trajectory?: GPSPoint[];
    spoof_probability?: number;
    confidence?: number;
  };
  login_data?: LoginEvent & {
    anomaly_probability?: number;
    confidence?: number;
  };
  password?: string;
  password_score?: number;
  transaction_data?: {
    fraud_probability?: number;
    payload?: Record<string, unknown>;
  };
  fusion_strategy?: 'weighted_average' | 'max_threat' | 'bayesian';
}

export interface ThreatSignal {
  category: string;
  probability: number;
  confidence: number;
  source_model: string;
  details: Record<string, unknown>;
  timestamp: string;
}

export interface UnifiedRiskResponse {
  unified_score: number;
  risk_level: 'minimal' | 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  gps_risk: number;
  login_risk: number;
  password_risk: number;
  fraud_risk: number;
  primary_threats: string[];
  threat_signals: ThreatSignal[];
  recommended_actions: string[];
  user_id?: string;
  event_id?: string;
  timestamp: string;
  models_used: string[];
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  components: Record<string, string>;
}

// ============================================================
// API Client
// ============================================================

interface RequestOptions extends RequestInit {
  requireAuth?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { requireAuth = true, ...fetchOptions } = options;
  
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(fetchOptions.headers || {})
  };

  // Add authentication token if available and required
  if (requireAuth) {
    const authHeader = getAuthHeader();
    if (authHeader) {
      headers["Authorization"] = authHeader;
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...fetchOptions,
    headers
  });

  const data = (await response.json()) as T;

  if (!response.ok) {
    // Handle 401 Unauthorized - token might be invalid
    if (response.status === 401) {
      // Clear invalid token
      if (typeof window !== "undefined") {
        localStorage.removeItem("aegis.idToken");
        localStorage.removeItem("ageis.idToken"); // Remove old typo key
      }
      throw new Error("Authentication failed. Please log in again.");
    }
    
    throw new Error(
      typeof data === "object" && data !== null && "error" in data
        ? String((data as { error: unknown }).error)
        : typeof data === "object" && data !== null && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : "Request failed"
    );
  }

  return data;
}

// ============================================================
// Auth Endpoints
// ============================================================

export async function login(credentials: Credentials) {
  return request<{ idToken: string; email: string; uid: string }>("/login", {
    method: "POST",
    body: JSON.stringify(credentials),
    requireAuth: false // Login endpoint doesn't require auth
  });
}

export async function signup(credentials: Credentials) {
  return request<{ uid: string; email: string }>("/signup", {
    method: "POST",
    body: JSON.stringify(credentials),
    requireAuth: false // Signup endpoint doesn't require auth
  });
}

// ============================================================
// GPS Spoofing Detection
// ============================================================

export async function scoreGPSTrajectory(
  trajectory: GPSPoint[],
  options?: { user_id?: string; device_id?: string; ensemble?: boolean }
): Promise<GPSSpoofResponse> {
  return request<GPSSpoofResponse>("/gps/score", {
    method: "POST",
    body: JSON.stringify({
      trajectory,
      ...options
    })
  });
}

export async function scoreGPSPoint(
  latitude: number,
  longitude: number,
  options?: {
    speed?: number;
    heading?: number;
    prev_latitude?: number;
    prev_longitude?: number;
  }
): Promise<{ spoof_probability: number; is_spoofed: boolean; confidence: number; warning: string }> {
  return request("/gps/score/point", {
    method: "POST",
    body: JSON.stringify({
      latitude,
      longitude,
      ...options
    })
  });
}

// ============================================================
// Login Anomaly Detection
// ============================================================

export async function scoreLoginEvent(event: LoginEvent): Promise<LoginAnomalyResponse> {
  return request<LoginAnomalyResponse>("/login/score", {
    method: "POST",
    body: JSON.stringify(event)
  });
}

export async function scoreLoginBatch(events: LoginEvent[]): Promise<{
  results: LoginAnomalyResponse[];
  total_events: number;
  high_risk_count: number;
  anomalous_count: number;
}> {
  return request("/login/score/batch", {
    method: "POST",
    body: JSON.stringify({ events })
  });
}

// ============================================================
// Password Risk Assessment
// ============================================================

export async function scorePassword(
  password: string,
  checkBreach = true
): Promise<PasswordScoreResponse> {
  return request<PasswordScoreResponse>("/password/score", {
    method: "POST",
    body: JSON.stringify({ password, check_breach: checkBreach })
  });
}

export async function analyzePassword(password: string): Promise<{
  strength_score: number;
  features: Record<string, number>;
  entropy_bits: number;
  character_diversity: number;
  unique_ratio: number;
  issues: Record<string, boolean>;
}> {
  return request("/password/analyze", {
    method: "POST",
    body: JSON.stringify({ password })
  });
}

// ============================================================
// Transaction Fraud Detection
// ============================================================

export async function scoreFraudTransaction(
  payload: Record<string, unknown>
): Promise<FraudScoreResponse> {
  return request<FraudScoreResponse>("/fraud/score", {
    method: "POST",
    body: JSON.stringify({ payload })
  });
}

// ============================================================
// Unified Risk Scoring
// ============================================================

export async function computeUnifiedRisk(
  request_data: UnifiedRiskRequest
): Promise<UnifiedRiskResponse> {
  return request<UnifiedRiskResponse>("/risk/unified", {
    method: "POST",
    body: JSON.stringify(request_data)
  });
}

export async function quickUnifiedRisk(
  scores: {
    gps_risk?: number;
    login_risk?: number;
    password_risk?: number;
    fraud_risk?: number;
  },
  user_id?: string,
  fusion_strategy: 'weighted_average' | 'max_threat' | 'bayesian' = 'weighted_average'
): Promise<{
  unified_score: number;
  risk_level: string;
  primary_threats: string[];
  recommended_actions: string[];
}> {
  const params = new URLSearchParams();
  if (scores.gps_risk !== undefined) params.append('gps_risk', String(scores.gps_risk));
  if (scores.login_risk !== undefined) params.append('login_risk', String(scores.login_risk));
  if (scores.password_risk !== undefined) params.append('password_risk', String(scores.password_risk));
  if (scores.fraud_risk !== undefined) params.append('fraud_risk', String(scores.fraud_risk));
  if (user_id) params.append('user_id', user_id);
  params.append('fusion_strategy', fusion_strategy);
  
  return request(`/risk/unified/quick?${params.toString()}`, {
    method: "POST"
  });
}

export async function getRiskThresholds(): Promise<{
  thresholds: Record<string, { min: number; max: number }>;
  component_weights: Record<string, number>;
  fusion_strategies_available: string[];
}> {
  return request("/risk/thresholds", { method: "GET" });
}

// ============================================================
// Health & Status
// ============================================================

export async function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health", { method: "GET" });
}

export async function getGPSHealth(): Promise<{ status: string; models_available: string[]; models_missing: string[] }> {
  return request("/gps/health", { method: "GET" });
}

export async function getLoginHealth(): Promise<{ status: string; models_available: string[]; rules_active: string[] }> {
  return request("/login/health", { method: "GET" });
}

export async function getPasswordHealth(): Promise<{ status: string; model_loaded: boolean }> {
  return request("/password/health", { method: "GET" });
}

export async function getRiskHealth(): Promise<{
  status: string;
  components: Record<string, boolean>;
  available_count: number;
  total_components: number;
}> {
  return request("/risk/health", { method: "GET" });
}

// ============================================================
// Utility Functions
// ============================================================

export function getRiskLevelColor(level: string): string {
  const colors: Record<string, string> = {
    minimal: '#22c55e',  // green
    low: '#84cc16',      // lime
    medium: '#eab308',   // yellow
    high: '#f97316',     // orange
    critical: '#ef4444'  // red
  };
  return colors[level] || '#6b7280';
}

export function getRiskLevelEmoji(level: string): string {
  const emojis: Record<string, string> = {
    minimal: '✅',
    low: '🟢',
    medium: '🟡',
    high: '🟠',
    critical: '🔴'
  };
  return emojis[level] || '⚪';
}

export function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}

export function getThreatCategoryLabel(category: string): string {
  const labels: Record<string, string> = {
    gps_spoofing: 'GPS Spoofing',
    login_anomaly: 'Login Anomaly',
    password_weakness: 'Weak Password',
    transaction_fraud: 'Transaction Fraud',
    account_takeover: 'Account Takeover',
    identity_theft: 'Identity Theft'
  };
  return labels[category] || category;
}
