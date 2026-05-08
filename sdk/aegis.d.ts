/**
 * AEGIS Client Tracking SDK — TypeScript declarations
 */

export interface AegisConfig {
  /** Base URL of the AEGIS API. Default: "http://localhost:8000" */
  apiUrl?: string;
  /** Unique identifier for the current user */
  userId: string;
  /** Bearer token for API authentication */
  token?: string;
  /** Automatically collect device fingerprint on init. Default: true */
  autoFingerprint?: boolean;
  /** Automatically enable GPS tracking on init. Default: false */
  autoGPS?: boolean;
  /** Callback fired whenever a risk score is returned */
  onRisk?: (result: RiskResult) => void;
  /** Number of events to queue before flushing. Default: 10 */
  batchSize?: number;
  /** Milliseconds between automatic queue flushes. Default: 30000 */
  flushInterval?: number;
}

export interface RiskResult {
  unified_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  primary_threats: string[];
  module_scores: Record<string, number | null>;
  timestamp: string;
}

export interface LoginData {
  success: boolean;
  method?: 'password' | 'biometric' | 'sso' | 'magic_link' | string;
  /** Milliseconds the login attempt took */
  duration_ms?: number;
  failed_attempts?: number;
  user_agent?: string;
  ip_address?: string;
}

export interface TransactionData {
  amount: number;
  currency?: string;
  merchant?: string;
  category?: string;
  is_international?: boolean;
  [key: string]: unknown;
}

export interface AppUnlockData {
  method?: 'pin' | 'biometric' | 'pattern' | string;
  success?: boolean;
  failed_attempts?: number;
  is_rooted?: boolean;
  is_emulator?: boolean;
}

export interface ScoreOptions {
  [key: string]: unknown;
}

export interface AegisSDK {
  /** Initialise the SDK. Must be called before any tracking methods. */
  init(config: AegisConfig): void;

  /**
   * Track a login attempt.
   * Posts to /risk/unified with login + GPS context.
   */
  trackLogin(data: LoginData): Promise<RiskResult | null>;

  /**
   * Check a password against HaveIBeenPwned (k-anonymity).
   * Posts to /breach/check/password — only a 5-char hash prefix is sent.
   */
  trackPassword(password: string): Promise<unknown>;

  /**
   * Score a financial transaction for fraud risk.
   * Posts to /fraud/score.
   */
  trackTransaction(data: TransactionData): Promise<unknown>;

  /**
   * Score an app-unlock event (biometrics, rooted device, etc.).
   * Posts to /device/score.
   */
  trackAppUnlock(data: AppUnlockData): Promise<unknown>;

  /**
   * Push a GPS observation.
   * Posts to /gps/score with accumulated trajectory.
   * @param lat  Latitude  (decimal degrees)
   * @param lon  Longitude (decimal degrees)
   */
  pushGPS(lat: number, lon: number): Promise<unknown>;

  /**
   * Request a full unified risk score right now.
   * @param extraData  Any additional payload fields to merge
   */
  scoreNow(extraData?: ScoreOptions): Promise<RiskResult | null>;

  /** Flush queued events immediately. */
  flush(): void;

  /** Stop GPS tracking and clear all intervals. */
  destroy(): void;
}

declare const Aegis: AegisSDK;
export default Aegis;
export = Aegis;
