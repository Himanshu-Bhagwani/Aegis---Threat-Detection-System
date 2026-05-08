/**
 * AEGIS React Hook
 * ================
 * Provides a `useAegis()` hook that wraps the vanilla SDK
 * and surfaces risk state into React component lifecycle.
 *
 * Usage:
 *   import { useAegis } from '@aegis/client-sdk/react';
 *
 *   function App() {
 *     const { riskLevel, riskScore, trackLogin, isReady } = useAegis({
 *       apiUrl: 'https://your-api.com',
 *       userId: currentUser.id,
 *       token: `Bearer ${currentUser.token}`,
 *     });
 *
 *     const handleLogin = async (creds) => {
 *       await login(creds);
 *       trackLogin({ success: true, method: 'password' });
 *     };
 *
 *     return <RiskBadge level={riskLevel} score={riskScore} />;
 *   }
 */

import { useEffect, useRef, useCallback, useState } from 'react';
// The SDK is UMD — import it as a CommonJS module
// eslint-disable-next-line @typescript-eslint/no-var-requires
const Aegis = require('./aegis.js') as import('./aegis').AegisSDK;

/* ─── Types ─────────────────────────────────────────────── */

export type RiskLevel = 'low' | 'medium' | 'high' | 'critical' | null;

export interface AegisHookConfig {
  apiUrl?: string;
  userId: string;
  token?: string;
  autoFingerprint?: boolean;
  autoGPS?: boolean;
}

export interface AegisHookResult {
  /** Whether the SDK has been initialised */
  isReady: boolean;
  /** Latest unified risk score (0–1), or null before first score */
  riskScore: number | null;
  /** Latest risk level string, or null before first score */
  riskLevel: RiskLevel;
  /** Primary threat categories from the last score */
  primaryThreats: string[];
  /** Per-module breakdown from the last score */
  moduleScores: Record<string, number | null>;
  /** ISO timestamp of the last score */
  lastScoreAt: string | null;

  /** Track a login event */
  trackLogin: (data: import('./aegis').LoginData) => Promise<import('./aegis').RiskResult | null>;
  /** Check a password against HIBP */
  trackPassword: (password: string) => Promise<unknown>;
  /** Score a transaction */
  trackTransaction: (data: import('./aegis').TransactionData) => Promise<unknown>;
  /** Track an app-unlock event */
  trackAppUnlock: (data: import('./aegis').AppUnlockData) => Promise<unknown>;
  /** Push a GPS reading */
  pushGPS: (lat: number, lon: number) => Promise<unknown>;
  /** Request a full score immediately */
  scoreNow: (extra?: import('./aegis').ScoreOptions) => Promise<import('./aegis').RiskResult | null>;
}

/* ─── Hook ──────────────────────────────────────────────── */

export function useAegis(config: AegisHookConfig): AegisHookResult {
  const [isReady, setIsReady]           = useState(false);
  const [riskScore, setRiskScore]       = useState<number | null>(null);
  const [riskLevel, setRiskLevel]       = useState<RiskLevel>(null);
  const [primaryThreats, setPrimaryThreats] = useState<string[]>([]);
  const [moduleScores, setModuleScores] = useState<Record<string, number | null>>({});
  const [lastScoreAt, setLastScoreAt]   = useState<string | null>(null);

  // Keep latest config in a ref so the onRisk callback always has it
  const cfgRef = useRef(config);
  cfgRef.current = config;

  useEffect(() => {
    Aegis.init({
      apiUrl:          config.apiUrl,
      userId:          config.userId,
      token:           config.token,
      autoFingerprint: config.autoFingerprint ?? true,
      autoGPS:         config.autoGPS ?? false,
      onRisk(result) {
        setRiskScore(result.unified_score);
        setRiskLevel(result.risk_level);
        setPrimaryThreats(result.primary_threats ?? []);
        setModuleScores(result.module_scores ?? {});
        setLastScoreAt(result.timestamp);
      },
    });
    setIsReady(true);

    return () => {
      Aegis.destroy();
    };
    // Re-init only when userId or apiUrl changes (not token — that updates inside onRisk via cfgRef)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.userId, config.apiUrl]);

  /* Stable callback wrappers ─────────────────────────────── */

  const trackLogin = useCallback(
    (data: import('./aegis').LoginData) => Aegis.trackLogin(data),
    []
  );

  const trackPassword = useCallback(
    (password: string) => Aegis.trackPassword(password),
    []
  );

  const trackTransaction = useCallback(
    (data: import('./aegis').TransactionData) => Aegis.trackTransaction(data),
    []
  );

  const trackAppUnlock = useCallback(
    (data: import('./aegis').AppUnlockData) => Aegis.trackAppUnlock(data),
    []
  );

  const pushGPS = useCallback(
    (lat: number, lon: number) => Aegis.pushGPS(lat, lon),
    []
  );

  const scoreNow = useCallback(
    (extra?: import('./aegis').ScoreOptions) => Aegis.scoreNow(extra),
    []
  );

  return {
    isReady,
    riskScore,
    riskLevel,
    primaryThreats,
    moduleScores,
    lastScoreAt,
    trackLogin,
    trackPassword,
    trackTransaction,
    trackAppUnlock,
    pushGPS,
    scoreNow,
  };
}

/* ─── Higher-order component ────────────────────────────── */

/**
 * withAegis(Component, config)
 * Injects `aegis` prop into any class or function component.
 *
 * Example:
 *   export default withAegis(MyPage, { userId: () => store.userId });
 */
export function withAegis<P extends { aegis?: AegisHookResult }>(
  Component: React.ComponentType<P>,
  configOrGetter: AegisHookConfig | (() => AegisHookConfig)
) {
  return function AegisWrapped(props: Omit<P, 'aegis'>) {
    const cfg = typeof configOrGetter === 'function' ? configOrGetter() : configOrGetter;
    const aegis = useAegis(cfg);
    return <Component {...(props as P)} aegis={aegis} />;
  };
}
