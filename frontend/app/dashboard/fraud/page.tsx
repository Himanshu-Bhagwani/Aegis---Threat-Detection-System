"use client";
import { useState } from "react";
import { scoreFraud, riskColor, formatScore, getWeightedUnifiedScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>Fraud Possibility</h1>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>XGBoost + Isolation Forest ensemble scoring for transaction fraud signals</p>
        </div>
        {selected && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 20, background: "var(--bg-raised)", border: "1px solid var(--border-subtle)", fontSize: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: riskColor(selected.metrics.fraud_risk) }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{selected.name}</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.fraud_risk), fontWeight: 700 }}>
              Fraud {(selected.metrics.fraud_risk * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16 }}>
        <div className="panel" style={{ padding: "20px 22px" }}>
          <div className="panel-header">Transaction Signals</div>
          {[
            { key: "amount",       label: "Amount ($)",         min: 0, max: 10000, step: 50,  fmt: (v: number) => `$${v}` },
            { key: "hour",         label: "Hour of Day",        min: 0, max: 23,    step: 1,   fmt: (v: number) => `${v}:00` },
            { key: "tx_count_1h",  label: "Transactions / hr",  min: 0, max: 20,    step: 1,   fmt: (v: number) => String(v) },
            { key: "amount_ratio", label: "Amount Ratio",       min: 0.1, max: 10,  step: 0.1, fmt: (v: number) => `${v.toFixed(1)}x` },
          ].map(s => (
            <div key={s.key} style={{ marginTop: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                <span>{s.label}</span>
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{s.fmt((form as any)[s.key])}</span>
              </div>
              <input type="range" min={s.min} max={s.max} step={s.step} value={(form as any)[s.key]}
                onChange={e => setForm(p => ({ ...p, [s.key]: parseFloat(e.target.value) }))} style={{ width: "100%" }} />
            </div>
          ))}
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 14, fontSize: 12, color: "var(--text-secondary)", cursor: "pointer" }}>
            <input type="checkbox" checked={form.is_international} onChange={e => setForm(p => ({ ...p, is_international: e.target.checked }))} />
            International transaction
          </label>
          <button className="btn-primary" style={{ width: "100%", marginTop: 16 }} onClick={run} disabled={loading}>
            {loading ? "Scoring…" : "Score Transaction →"}
          </button>
        </div>

        <div className="panel" style={{ padding: "20px 22px" }}>
          {!result && <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-disabled)", fontSize: 13 }}>Configure transaction and score</div>}
          {result && !result.error && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ fontSize: 52, fontWeight: 900, fontFamily: "var(--font-mono)", color: riskColor(result.fraud_probability ?? 0), letterSpacing: "-0.04em", lineHeight: 1 }}>
                {formatScore(result.fraud_probability ?? 0)}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>fraud probability · method: {result.method ?? "ensemble"}</div>
              {result.risk_factors?.map((f: string, i: number) => (
                <div key={i} style={{ display: "flex", gap: 7, fontSize: 12, color: "var(--text-secondary)" }}>
                  <span style={{ color: "var(--risk-high)" }}>→</span>{f}
                </div>
              ))}
              {updated && selected && (
                <div style={{ padding: "8px 14px", borderRadius: 8, background: "rgba(61,127,255,0.1)", border: "1px solid rgba(61,127,255,0.3)", fontSize: 12, color: "var(--accent)", marginTop: 4 }}>
                  ✓ Updated {selected.name}'s fraud risk to <strong>{(selected.metrics.fraud_risk * 100).toFixed(0)}%</strong>
                </div>
              )}
            </div>
          )}
          {result?.error && <div style={{ color: "var(--risk-high)", fontSize: 13 }}>{result.error}</div>}
        </div>
      </div>
    </div>
  );
}
