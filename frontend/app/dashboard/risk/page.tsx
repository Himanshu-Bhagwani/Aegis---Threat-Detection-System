"use client";

import { useState } from "react";
import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor, formatScore, riskLabel } from "@/lib/api";
import { THRESHOLDS, updateProfileMetrics, computeUnified } from "@/lib/profiles";

const MODULE_KEYS = [
  { key: "gps_spoof",     label: "GPS Spoofing",              icon: "◉", desc: "Probability of GPS location spoofing" },
  { key: "login_anomaly", label: "Login Anomaly",             icon: "◐", desc: "Unusual login patterns or failed attempts" },
  { key: "password_leak", label: "Password Leak Possibility", icon: "◑", desc: "Password found in breach databases" },
  { key: "fraud_risk",    label: "Fraud Possibility",         icon: "◆", desc: "Transaction fraud probability" },
  { key: "breach_risk",   label: "Breach Risk",               icon: "◍", desc: "Data breach exposure likelihood" },
];

export default function RiskPage() {
  const { profiles, selectedId, setSelectedId, selected, updateMetrics } = useProfiles();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>Risk Analysis</h1>
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
          All identity profiles — click a profile to inspect or adjust module scores
        </p>
      </div>

      {/* Profile grid */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {profiles.map(p => {
          const expanded = expandedId === p.id;
          const c = riskColor(p.metrics.unified_score);

          return (
            <div key={p.id} className="panel" style={{ padding: 0, overflow: "hidden", borderColor: expanded ? "var(--accent)" : undefined }}>
              {/* Header row */}
              <div
                onClick={() => { setExpandedId(expanded ? null : p.id); setSelectedId(p.id); }}
                style={{ display: "flex", alignItems: "center", gap: 14, padding: "16px 20px", cursor: "pointer" }}
              >
                <div style={{ width: 38, height: 38, borderRadius: "50%", background: c + "22", border: `2px solid ${c}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, fontWeight: 700, color: c, flexShrink: 0 }}>
                  {p.name[0]}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>{p.name}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1 }}>{p.email}</div>
                </div>

                {/* Mini module bars */}
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  {MODULE_KEYS.map(({ key, icon }) => {
                    const val = (p.metrics as any)[key] ?? 0;
                    const col = riskColor(val);
                    return (
                      <div key={key} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
                        <span style={{ fontSize: 11, opacity: 0.6 }}>{icon}</span>
                        <div style={{ width: 28, height: 4, borderRadius: 2, background: "var(--bg-raised)", overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${val * 100}%`, background: col, borderRadius: 2 }} />
                        </div>
                        <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: col }}>{(val * 100).toFixed(0)}</span>
                      </div>
                    );
                  })}
                </div>

                <div style={{ textAlign: "right", flexShrink: 0, marginLeft: 16 }}>
                  <div style={{ fontSize: 22, fontWeight: 800, fontFamily: "var(--font-mono)", color: c, letterSpacing: "-0.02em" }}>
                    {(p.metrics.unified_score * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em", color: c, fontWeight: 600 }}>
                    {p.metrics.risk_level}
                  </div>
                </div>
                <span style={{ fontSize: 14, color: "var(--text-disabled)", marginLeft: 8 }}>{expanded ? "▲" : "▼"}</span>
              </div>

              {/* Expanded detail */}
              {expanded && (
                <div style={{ borderTop: "1px solid var(--border-subtle)", padding: "20px 20px 16px" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                    {/* Module sliders */}
                    <div>
                      <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 14, fontWeight: 600 }}>Adjust Metrics</div>
                      {MODULE_KEYS.map(({ key, label, desc }) => {
                        const val = (p.metrics as any)[key] ?? 0;
                        const col = riskColor(val);
                        const exceeded = val > THRESHOLDS[key];
                        return (
                          <div key={key} style={{ marginBottom: 14 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                              <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1 }}>{label}</span>
                              {exceeded && <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 8, background: "rgba(239,68,68,0.12)", color: "var(--risk-critical)", fontWeight: 600 }}>ALERT</span>}
                              <span style={{ fontSize: 12, fontWeight: 700, fontFamily: "var(--font-mono)", color: col }}>{(val * 100).toFixed(0)}%</span>
                            </div>
                            <input
                              type="range" min={0} max={1} step={0.01} value={val}
                              onChange={e => updateMetrics(p.id, { [key]: parseFloat(e.target.value) } as any)}
                              style={{ width: "100%", accentColor: col }}
                            />
                            <div style={{ fontSize: 10, color: "var(--text-disabled)" }}>{desc} · Threshold: {(THRESHOLDS[key] * 100).toFixed(0)}%</div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Right: score + notes */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                      <div className="panel" style={{ padding: "18px 20px", background: "var(--bg-raised)" }}>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>Unified Score</div>
                        <div style={{ fontSize: 46, fontWeight: 900, fontFamily: "var(--font-mono)", color: c, letterSpacing: "-0.03em", lineHeight: 1 }}>
                          {(p.metrics.unified_score * 100).toFixed(1)}%
                        </div>
                        <div style={{ height: 6, borderRadius: 3, background: "var(--bg-base)", overflow: "hidden", marginTop: 10 }}>
                          <div style={{ height: "100%", width: `${p.metrics.unified_score * 100}%`, background: c, borderRadius: 3, transition: "width 0.4s" }} />
                        </div>
                        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: c, fontWeight: 700, marginTop: 8 }}>{p.metrics.risk_level}</div>
                      </div>

                      <div className="panel" style={{ padding: "16px 18px", background: "var(--bg-raised)" }}>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Profile Notes</div>
                        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6 }}>{p.notes}</div>
                        <div style={{ marginTop: 10, fontSize: 11, color: "var(--text-disabled)" }}>
                          Last updated: {new Date(p.metrics.last_updated).toLocaleString()}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
