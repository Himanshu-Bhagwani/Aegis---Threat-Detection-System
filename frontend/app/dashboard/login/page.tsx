"use client";
import { useState } from "react";
import { scoreLogin, riskColor, formatScore, getWeightedUnifiedScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

export default function LoginPage() {
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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>Login Anomaly Detection</h1>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>Behavioral baseline analysis for unusual login time, device, location, and failure patterns</p>
        </div>
        {selected && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 20, background: "var(--bg-raised)", border: "1px solid var(--border-subtle)", fontSize: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: riskColor(selected.metrics.login_anomaly) }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{selected.name}</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.login_anomaly), fontWeight: 700 }}>
              Login {(selected.metrics.login_anomaly * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16 }}>
        <div className="panel" style={{ padding: "20px 22px" }}>
          <div className="panel-header">Login Signals</div>
          {SLIDERS.map(s => (
            <div key={s.key} style={{ marginTop: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                <span>{s.label}</span>
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{s.fmt((form as any)[s.key])}</span>
              </div>
              <input type="range" min={s.min} max={s.max} step={s.step} value={(form as any)[s.key]}
                onChange={e => set(s.key, parseInt(e.target.value))} style={{ width: "100%" }} />
            </div>
          ))}
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 14, fontSize: 12, color: "var(--text-secondary)", cursor: "pointer" }}>
            <input type="checkbox" checked={form.is_new_comp === 1} onChange={e => set("is_new_comp", e.target.checked ? 1 : 0)} />
            New / unknown device
          </label>
          <button className="btn-primary" style={{ width: "100%", marginTop: 16 }} onClick={run} disabled={loading}>
            {loading ? "Scoring…" : "Analyse Login →"}
          </button>
        </div>

        <div className="panel" style={{ padding: "20px 22px" }}>
          {!result && <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-disabled)", fontSize: 13 }}>Configure signals and analyse</div>}
          {result && !result.error && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ fontSize: 52, fontWeight: 900, fontFamily: "var(--font-mono)", color: riskColor(result.anomaly_probability ?? 0), letterSpacing: "-0.04em", lineHeight: 1 }}>
                {formatScore(result.anomaly_probability ?? 0)}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>anomaly probability · confidence: {formatScore(result.confidence ?? 0)}</div>
              {result.risk_factors?.map((f: string, i: number) => (
                <div key={i} style={{ display: "flex", gap: 7, fontSize: 12, color: "var(--text-secondary)" }}>
                  <span style={{ color: "var(--risk-high)" }}>→</span>{f}
                </div>
              ))}
              {updated && selected && (
                <div style={{ padding: "8px 14px", borderRadius: 8, background: "rgba(61,127,255,0.1)", border: "1px solid rgba(61,127,255,0.3)", fontSize: 12, color: "var(--accent)", marginTop: 4 }}>
                  ✓ Updated {selected.name}'s login anomaly to <strong>{(selected.metrics.login_anomaly * 100).toFixed(0)}%</strong>
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
