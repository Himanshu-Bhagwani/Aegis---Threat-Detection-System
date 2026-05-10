"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { scoreFraud, riskColor, formatScore, getWeightedUnifiedScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

const DynAmtRisk = dynamic(
  () => import("@/components/PageChart").then(m => ({ default: m.AmountRiskChart })),
  { ssr: false }
);
const DynSignal = dynamic(
  () => import("@/components/PageChart").then(m => ({ default: m.SignalContribChart })),
  { ssr: false }
);

const IconCreditCard = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>
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

export default function FraudPage() {
  const { selected, updateMetrics } = useProfiles();
  const [form, setForm] = useState({ amount: 500, hour: 14, tx_count_1h: 2, time_since_last_tx: 3600, amount_ratio: 1.0, is_international: false });
  const [result,  setResult]  = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [updated, setUpdated] = useState(false);

  async function run() {
    setLoading(true); setUpdated(false);
    try {
      const res = await scoreFraud({ ...form, merchant_freq_user: 3 });
      setResult(res);
      if (res?.fraud_probability != null && selected) {
        const newFraud = Math.min(1, Math.max(0, selected.metrics.fraud_risk * 0.7 + res.fraud_probability * 0.3));
        const m = selected.metrics;
        const unified = await getWeightedUnifiedScore(m.gps_spoof, m.login_anomaly, m.password_leak, newFraud, m.breach_risk, selected.id);
        updateMetrics(selected.id, { fraud_risk: newFraud, unified_score: unified });
        setUpdated(true);
      }
    } catch (e: any) { setResult({ error: e?.message }); }
    finally { setLoading(false); }
  }

  const resultScore = result?.fraud_probability ?? 0;
  const resultColor = riskColor(resultScore);

  const SLIDERS = [
    { key: "amount",       label: "Transaction Amount ($)", min: 0, max: 10000, step: 50,  fmt: (v: number) => `$${v.toLocaleString()}` },
    { key: "hour",         label: "Hour of Day",            min: 0, max: 23,    step: 1,   fmt: (v: number) => `${v}:00` },
    { key: "tx_count_1h",  label: "Transactions per Hour",  min: 0, max: 20,    step: 1,   fmt: (v: number) => String(v) },
    { key: "amount_ratio", label: "Amount Ratio vs Average", min: 0.1, max: 10, step: 0.1, fmt: (v: number) => `${v.toFixed(1)}×` },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{ color: "var(--risk-high)" }}><IconCreditCard /></div>
            <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
              Fraud Detection
            </h1>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            XGBoost + Isolation Forest ensemble scoring for transaction fraud signals
          </p>
        </div>
        {selected && (
          <div style={{
            display: "flex", alignItems: "center", gap: 8, padding: "8px 16px",
            borderRadius: 20, background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)", fontSize: 12, flexShrink: 0,
          }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: riskColor(selected.metrics.fraud_risk), boxShadow: `0 0 8px ${riskColor(selected.metrics.fraud_risk)}` }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{selected.name}</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.fraud_risk), fontWeight: 800 }}>
              {(selected.metrics.fraud_risk * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* ── Amount vs fraud risk reference ─────── */}
      <div className="panel" style={{ padding: "18px 22px" }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 14 }}>
          Transaction Amount vs Fraud Risk — Reference Curve
        </div>
        <DynAmtRisk currentAmount={form.amount} />
        <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 6 }}>
          White dot = current amount (${form.amount.toLocaleString()}) · Red line = 60% detection threshold
        </div>
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16 }}>

        {/* Signals */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
            letterSpacing: "0.10em", marginBottom: 18, paddingBottom: 12,
            borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>Transaction Signals</div>

          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            {SLIDERS.map(s => {
              const val = (form as any)[s.key];
              return (
                <div key={s.key}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontSize: 11, color: "var(--text-secondary)", fontWeight: 500 }}>{s.label}</span>
                    <span style={{ fontFamily: "var(--font-mono)", color: "var(--risk-high)", fontWeight: 700, fontSize: 12 }}>{s.fmt(val)}</span>
                  </div>
                  <input
                    type="range" min={s.min} max={s.max} step={s.step} value={val}
                    onChange={e => setForm(p => ({ ...p, [s.key]: parseFloat(e.target.value) }))}
                    style={{ width: "100%", accentColor: "var(--risk-high)" }}
                  />
                </div>
              );
            })}

            {/* International checkbox */}
            <label style={{
              display: "flex", alignItems: "center", gap: 10, cursor: "pointer",
              padding: "10px 12px", borderRadius: 8,
              background: form.is_international ? "rgba(249,115,22,0.08)" : "rgba(255,255,255,0.03)",
              border: `1px solid ${form.is_international ? "rgba(249,115,22,0.25)" : "rgba(255,255,255,0.07)"}`,
              transition: "all 0.2s",
            }}>
              <input
                type="checkbox" checked={form.is_international}
                onChange={e => setForm(p => ({ ...p, is_international: e.target.checked }))}
                style={{ accentColor: "var(--risk-high)", width: 14, height: 14 }}
              />
              <span style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>International transaction</span>
            </label>

            <button
              onClick={run} disabled={loading}
              style={{
                display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                padding: "12px", borderRadius: 9, fontSize: 13, fontWeight: 700,
                background: loading ? "rgba(249,115,22,0.3)" : "linear-gradient(135deg, #f97316, #ea580c)",
                color: "#fff", border: "none", cursor: loading ? "not-allowed" : "pointer",
                boxShadow: loading ? "none" : "0 0 20px rgba(249,115,22,0.3)",
                transition: "all 0.2s",
              }}
            >
              {loading ? <><IconLoader /> Scoring…</> : <><IconPlay /> Score Transaction</>}
            </button>
          </div>
        </div>

        {/* Result */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
            letterSpacing: "0.10em", marginBottom: 18, paddingBottom: 12,
            borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>Fraud Assessment</div>

          {!result && !loading && (
            <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-disabled)" }}>
              <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>💳</div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Configure transaction signals</div>
              <div style={{ fontSize: 11, marginTop: 6 }}>Adjust the sliders and click Score Transaction</div>
            </div>
          )}

          {loading && (
            <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-muted)" }}>
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}><IconLoader /></div>
              <div style={{ fontSize: 13 }}>Running ensemble scoring…</div>
            </div>
          )}

          {result && !result.error && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeUp 0.4s var(--ease-out) both" }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
                <div style={{
                  fontSize: 64, fontWeight: 900, fontFamily: "var(--font-mono)",
                  color: resultColor, letterSpacing: "-0.04em", lineHeight: 1,
                  textShadow: `0 0 40px ${resultColor}60`,
                }}>
                  {formatScore(resultScore)}
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>Fraud Probability</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                    method: <span style={{ color: "var(--text-secondary)" }}>{result.method ?? "ensemble"}</span>
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

              {/* Rule contribution chart */}
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                  Risk Factor Breakdown
                </div>
                <DynSignal factors={[
                  { name: "Transaction Amount",   score: Math.min(1, form.amount / 10000),                                   color: "#ef4444" },
                  { name: "International Txn",    score: form.is_international ? 0.70 : 0.02,                                color: "#f97316" },
                  { name: "Off-hours (10pm–5am)", score: (form.hour >= 22 || form.hour <= 5) ? 0.55 : 0.05,                 color: "#8b5cf6" },
                  { name: "Velocity (tx/hour)",   score: Math.min(1, form.tx_count_1h / 20),                                color: "#3d7fff" },
                  { name: "Amount Ratio vs Avg",  score: Math.min(1, Math.max(0, (form.amount_ratio - 1) / 9)),             color: "#f59e0b" },
                ]} />
              </div>

              {/* Risk factors text */}
              {result.risk_factors?.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>
                    Triggered Rules
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
                  Updated {selected.name}&apos;s fraud risk to{" "}
                  <strong style={{ fontFamily: "var(--font-mono)" }}>{(selected.metrics.fraud_risk * 100).toFixed(0)}%</strong>
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
                    { icon: "🛑", text: "Block this transaction immediately and hold it for manual review before any funds are moved." },
                    { icon: "📲", text: "Send a real-time SMS and email alert to the cardholder with a one-tap approval or decline link." },
                    { icon: "🔐", text: "Require OTP or biometric confirmation from the registered mobile device before re-attempting." },
                    { icon: "❄️", text: "Temporarily freeze card for international transactions and high-value purchases until verified." },
                    { icon: "📋", text: "Escalate to the fraud investigation team and log transaction metadata for audit trail." },
                  ] : [
                    { icon: "🔔", text: "Enable instant transaction notifications for all card activity, no matter the amount." },
                    { icon: "🌍", text: "Set a spending cap on international transactions and require prior approval for travel-mode activation." },
                    { icon: "📊", text: "Review recent transaction history for small test charges — fraudsters often probe with micro-transactions first." },
                    { icon: "🛡️", text: "Enroll in 3D Secure (Verified by Visa / Mastercard SecureCode) for additional authentication on online purchases." },
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
