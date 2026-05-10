"use client";

import { useState } from "react";
import { scoreLogin, riskColor, formatScore, getWeightedUnifiedScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

const IconLock = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>
  </svg>
);
const IconPlay = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>
);
const IconLoader = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}>
    <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>
  </svg>
);
const IconCheck = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);
const IconChevronRight = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);

export default function LoginAnomalyPage() {
  const { selected, updateMetrics } = useProfiles();
  const [form, setForm] = useState({ hour_of_day: 14, failed_10min: 0, is_new_comp: 0, comp_deg: 50, user_deg: 5, time_since_user_last: 3600 });
  const [result,  setResult]  = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [updated, setUpdated] = useState(false);
  const set = (k: string, v: number) => setForm(p => ({ ...p, [k]: v }));

  async function run() {
    setLoading(true); setUpdated(false);
    try {
      const res = await scoreLogin(form);
      setResult(res);
      if (res?.anomaly_probability != null && selected) {
        const newLogin = Math.min(1, Math.max(0, selected.metrics.login_anomaly * 0.7 + res.anomaly_probability * 0.3));
        const m = selected.metrics;
        const unified = await getWeightedUnifiedScore(m.gps_spoof, newLogin, m.password_leak, m.fraud_risk, m.breach_risk, selected.id);
        updateMetrics(selected.id, { login_anomaly: newLogin, unified_score: unified });
        setUpdated(true);
      }
    } catch (e: any) { setResult({ error: e?.message }); }
    finally { setLoading(false); }
  }

  const SLIDERS = [
    { key: "hour_of_day",          label: "Hour of Day",            min: 0, max: 23,    step: 1,    fmt: (v: number) => `${v}:00` },
    { key: "failed_10min",         label: "Failed Attempts (10 min)",min: 0, max: 10,   step: 1,    fmt: (v: number) => String(v) },
    { key: "comp_deg",             label: "Computer Deviation",     min: 0, max: 360,   step: 1,    fmt: (v: number) => `${v}°` },
    { key: "time_since_user_last", label: "Hours Since Last Login", min: 0, max: 86400, step: 3600, fmt: (v: number) => `${(v/3600).toFixed(0)}h` },
  ];

  const resultScore = result?.anomaly_probability ?? 0;
  const resultColor = riskColor(resultScore);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{ color: "var(--accent-cyan)" }}><IconLock /></div>
            <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
              Login Anomaly Detection
            </h1>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            Behavioral baseline analysis for unusual login time, device, and failure patterns
          </p>
        </div>
        {selected && (
          <div style={{
            display: "flex", alignItems: "center", gap: 8, padding: "8px 16px",
            borderRadius: 20, background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)", fontSize: 12, flexShrink: 0,
          }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: riskColor(selected.metrics.login_anomaly), boxShadow: `0 0 8px ${riskColor(selected.metrics.login_anomaly)}` }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{selected.name}</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.login_anomaly), fontWeight: 800 }}>
              {(selected.metrics.login_anomaly * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16 }}>

        {/* Signals panel */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
            letterSpacing: "0.10em", marginBottom: 18, paddingBottom: 12,
            borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>Login Signals</div>

          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            {SLIDERS.map(s => {
              const val = (form as any)[s.key];
              const pct = ((val - s.min) / (s.max - s.min)) * 100;
              return (
                <div key={s.key}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "var(--text-secondary)", fontWeight: 500 }}>{s.label}</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent)", fontWeight: 700, fontSize: 12 }}>{s.fmt(val)}</span>
                  </div>
                  <input
                    type="range" min={s.min} max={s.max} step={s.step} value={val}
                    onChange={e => set(s.key, parseInt(e.target.value))}
                    style={{ width: "100%", accentColor: "var(--accent)" }}
                  />
                </div>
              );
            })}

            {/* Checkbox */}
            <label style={{
              display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
              padding: "10px 12px", borderRadius: 8,
              background: form.is_new_comp ? "rgba(61,127,255,0.08)" : "rgba(255,255,255,0.03)",
              border: `1px solid ${form.is_new_comp ? "rgba(61,127,255,0.25)" : "rgba(255,255,255,0.07)"}`,
              transition: "all 0.2s",
            }}>
              <input
                type="checkbox" checked={form.is_new_comp === 1}
                onChange={e => set("is_new_comp", e.target.checked ? 1 : 0)}
                style={{ accentColor: "var(--accent)", width: 14, height: 14 }}
              />
              <span style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>New / unknown device</span>
            </label>

            {/* Run button */}
            <button
              onClick={run} disabled={loading}
              style={{
                display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                padding: "12px", borderRadius: 9, fontSize: 13, fontWeight: 700,
                background: loading ? "rgba(61,127,255,0.3)" : "linear-gradient(135deg, #3d7fff, #2563eb)",
                color: "#fff", border: "none", cursor: loading ? "not-allowed" : "pointer",
                boxShadow: loading ? "none" : "0 0 20px rgba(61,127,255,0.3)",
                transition: "all 0.2s",
              }}
            >
              {loading ? <><IconLoader /> Scoring…</> : <><IconPlay /> Analyse Login</>}
            </button>
          </div>
        </div>

        {/* Result panel */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
            letterSpacing: "0.10em", marginBottom: 18, paddingBottom: 12,
            borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>Analysis Result</div>

          {!result && !loading && (
            <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-disabled)" }}>
              <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>🔍</div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Configure signals and run analysis</div>
              <div style={{ fontSize: 11, marginTop: 6 }}>Adjust the sliders and click Analyse Login</div>
            </div>
          )}

          {loading && (
            <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-muted)" }}>
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}><IconLoader /></div>
              <div style={{ fontSize: 13 }}>Running behavioral analysis…</div>
            </div>
          )}

          {result && !result.error && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14, animation: "fadeUp 0.4s var(--ease-out) both" }}>
              {/* Score */}
              <div style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
                <div style={{
                  fontSize: 64, fontWeight: 900, fontFamily: "var(--font-mono)",
                  color: resultColor, letterSpacing: "-0.04em", lineHeight: 1,
                  textShadow: `0 0 40px ${resultColor}60`,
                }}>
                  {formatScore(resultScore)}
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>Anomaly Probability</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                    confidence: <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>{formatScore(result.confidence ?? 0)}</span>
                  </div>
                </div>
              </div>

              {/* Score bar */}
              <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.07)", overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${resultScore * 100}%`, background: resultColor,
                  borderRadius: 3, transition: "width 0.8s var(--ease-out)",
                  boxShadow: `0 0 12px ${resultColor}60`,
                }} />
              </div>

              {/* Risk factors */}
              {result.risk_factors?.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>
                    Triggered Signals
                  </div>
                  {result.risk_factors.map((f: string, i: number) => (
                    <div key={i} style={{ display: "flex", gap: 8, fontSize: 12, color: "var(--text-secondary)", alignItems: "flex-start" }}>
                      <span style={{ color: "var(--risk-high)", flexShrink: 0, marginTop: 1 }}><IconChevronRight /></span>{f}
                    </div>
                  ))}
                </div>
              )}

              {updated && selected && (
                <div style={{
                  padding: "10px 16px", borderRadius: 10,
                  background: "rgba(61,127,255,0.08)", border: "1px solid rgba(61,127,255,0.25)",
                  fontSize: 12, color: "var(--accent)", display: "flex", alignItems: "center", gap: 8,
                }}>
                  <IconCheck />
                  Updated {selected.name}&apos;s login anomaly to{" "}
                  <strong style={{ fontFamily: "var(--font-mono)" }}>{(selected.metrics.login_anomaly * 100).toFixed(0)}%</strong>
                </div>
              )}

              {/* Security recommendations */}
              <div style={{
                padding: "16px 18px", borderRadius: 12,
                background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
              }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 12 }}>
                  Security Recommendations
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {(resultScore >= 0.5 ? [
                    { icon: "🔴", text: "Immediately challenge the user with multi-factor authentication before allowing access." },
                    { icon: "🚪", text: "Terminate all active sessions from unfamiliar devices and force re-authentication." },
                    { icon: "📧", text: "Send a security alert to the registered email and phone number right now." },
                    { icon: "🔒", text: "Lock the account temporarily after 3 more failed attempts and require identity verification to unlock." },
                    { icon: "🌍", text: "Block logins from the anomalous geography or device until the user confirms it was them." },
                  ] : [
                    { icon: "✅", text: "Enable login alerts for new devices, browsers, or geographic locations." },
                    { icon: "📱", text: "Set up a trusted device list and require approval for new devices before granting access." },
                    { icon: "⏱️", text: "Configure automatic session expiry after 15–30 minutes of inactivity on sensitive operations." },
                    { icon: "🛡️", text: "Enable passkey or hardware security key authentication to eliminate password-based login risks." },
                  ]).map((r, i) => (
                    <div key={i} style={{ display: "flex", gap: 10, fontSize: 12, color: "var(--text-secondary)", alignItems: "flex-start" }}>
                      <span style={{ flexShrink: 0, fontSize: 13 }}>{r.icon}</span>
                      <span style={{ lineHeight: 1.6 }}>{r.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {result?.error && (
            <div style={{
              padding: "14px 18px", borderRadius: 10,
              background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)",
              color: "var(--risk-critical)", fontSize: 13,
            }}>
              {result.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
