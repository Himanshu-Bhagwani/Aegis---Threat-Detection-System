"use client";

import { useState } from "react";
import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor, formatScore } from "@/lib/api";
import { THRESHOLDS } from "@/lib/profiles";

const MODULE_KEYS = [
  { key: "gps_spoof",     label: "GPS Spoofing",              desc: "Probability of GPS location spoofing", color: "var(--accent)" },
  { key: "login_anomaly", label: "Login Anomaly",             desc: "Unusual login patterns or failed attempts", color: "var(--accent-cyan)" },
  { key: "password_leak", label: "Password Leak Possibility", desc: "Password found in breach databases", color: "var(--accent-purple)" },
  { key: "fraud_risk",    label: "Fraud Possibility",         desc: "Transaction fraud probability", color: "var(--risk-high)" },
  { key: "breach_risk",   label: "Breach Risk",               desc: "Data breach exposure likelihood", color: "var(--risk-critical)" },
];

const IconChart = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);
const IconChevronDown = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9"/>
  </svg>
);
const IconChevronUp = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="18 15 12 9 6 15"/>
  </svg>
);

export default function RiskPage() {
  const { profiles, setSelectedId, updateMetrics } = useProfiles();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{ color: "var(--accent)" }}><IconChart /></div>
          <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>Risk Analysis</h1>
        </div>
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          All identity profiles — click a profile to inspect or adjust module scores
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {profiles.map(p => {
          const expanded = expandedId === p.id;
          const c = riskColor(p.metrics.unified_score);

          return (
            <div
              key={p.id}
              className="panel"
              style={{
                padding: 0, overflow: "hidden",
                borderColor: expanded ? "rgba(61,127,255,0.3)" : undefined,
                boxShadow: expanded ? "0 0 24px rgba(61,127,255,0.08)" : undefined,
                transition: "all 0.25s var(--ease-out)",
              }}
            >
              {/* Header row */}
              <div
                onClick={() => { setExpandedId(expanded ? null : p.id); setSelectedId(p.id); }}
                style={{
                  display: "flex", alignItems: "center", gap: 14,
                  padding: "18px 22px", cursor: "pointer",
                  transition: "background 0.2s",
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.02)"; }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                {/* Avatar */}
                <div style={{
                  width: 40, height: 40, borderRadius: "50%", flexShrink: 0,
                  background: `${c}18`, border: `2px solid ${c}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 16, fontWeight: 800, color: c,
                  boxShadow: `0 0 12px ${c}40`,
                }}>
                  {p.name[0]}
                </div>

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>{p.name}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{p.email}</div>
                </div>

                {/* Mini module bars */}
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
                  {MODULE_KEYS.map(({ key, color }) => {
                    const val = (p.metrics as any)[key] ?? 0;
                    const col = riskColor(val);
                    return (
                      <div key={key} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                        <div style={{ width: 4, height: 32, borderRadius: 2, background: "rgba(255,255,255,0.07)", overflow: "hidden", position: "relative" }}>
                          <div style={{
                            position: "absolute", bottom: 0, left: 0, right: 0,
                            height: `${val * 100}%`, background: col, borderRadius: 2,
                            boxShadow: `0 0 6px ${col}60`,
                            transition: "height 0.4s var(--ease-out)",
                          }} />
                        </div>
                        <span style={{ fontSize: 8, fontFamily: "var(--font-mono)", color: col, fontWeight: 700 }}>
                          {(val * 100).toFixed(0)}
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Unified score */}
                <div style={{ textAlign: "right", flexShrink: 0, marginLeft: 8 }}>
                  <div style={{
                    fontSize: 24, fontWeight: 900, fontFamily: "var(--font-mono)",
                    color: c, letterSpacing: "-0.03em",
                    textShadow: `0 0 16px ${c}60`,
                  }}>
                    {(p.metrics.unified_score * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: c, fontWeight: 700 }}>
                    {p.metrics.risk_level}
                  </div>
                </div>

                <div style={{ color: "var(--text-muted)", marginLeft: 6, flexShrink: 0 }}>
                  {expanded ? <IconChevronUp /> : <IconChevronDown />}
                </div>
              </div>

              {/* Expanded detail */}
              {expanded && (
                <div style={{
                  borderTop: "1px solid rgba(255,255,255,0.05)",
                  padding: "22px 22px 20px",
                  animation: "fadeDown 0.25s var(--ease-out) both",
                }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>
                    {/* Sliders */}
                    <div>
                      <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 16 }}>
                        Adjust Metrics
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                        {MODULE_KEYS.map(({ key, label, desc, color }) => {
                          const val = (p.metrics as any)[key] ?? 0;
                          const col = riskColor(val);
                          const exceeded = val > THRESHOLDS[key];
                          return (
                            <div key={key}>
                              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                <div style={{ width: 7, height: 7, borderRadius: "50%", background: col, flexShrink: 0 }} />
                                <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1, fontWeight: 500 }}>{label}</span>
                                {exceeded && (
                                  <span style={{
                                    fontSize: 9, padding: "2px 7px", borderRadius: 8,
                                    background: "rgba(239,68,68,0.12)", color: "var(--risk-critical)",
                                    fontWeight: 800, border: "1px solid rgba(239,68,68,0.25)",
                                    letterSpacing: "0.05em",
                                  }}>ALERT</span>
                                )}
                                <span style={{ fontSize: 12, fontWeight: 800, fontFamily: "var(--font-mono)", color: col }}>{(val * 100).toFixed(0)}%</span>
                              </div>
                              <input
                                type="range" min={0} max={1} step={0.01} value={val}
                                onChange={e => updateMetrics(p.id, { [key]: parseFloat(e.target.value) } as any)}
                                style={{ width: "100%", accentColor: col }}
                              />
                              <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 3 }}>
                                {desc} · threshold: {(THRESHOLDS[key] * 100).toFixed(0)}%
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Right side */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                      {/* Unified score */}
                      <div style={{
                        padding: "20px 22px", borderRadius: 12,
                        background: `${c}08`, border: `1px solid ${c}25`,
                      }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 12 }}>
                          Unified Score
                        </div>
                        <div style={{
                          fontSize: 52, fontWeight: 900, fontFamily: "var(--font-mono)",
                          color: c, letterSpacing: "-0.04em", lineHeight: 1,
                          textShadow: `0 0 30px ${c}60`,
                        }}>
                          {(p.metrics.unified_score * 100).toFixed(1)}%
                        </div>
                        <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.08)", overflow: "hidden", marginTop: 14 }}>
                          <div style={{
                            height: "100%", width: `${p.metrics.unified_score * 100}%`,
                            background: c, borderRadius: 3,
                            boxShadow: `0 0 12px ${c}60`,
                            transition: "width 0.5s var(--ease-out)",
                          }} />
                        </div>
                        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.10em", color: c, fontWeight: 800, marginTop: 10 }}>
                          {p.metrics.risk_level}
                        </div>
                      </div>

                      {/* Notes */}
                      <div style={{
                        padding: "16px 18px", borderRadius: 12,
                        background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.06)",
                      }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 8 }}>
                          Profile Notes
                        </div>
                        <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.7 }}>{p.notes}</div>
                        <div style={{ marginTop: 10, fontSize: 10, color: "var(--text-disabled)", fontFamily: "var(--font-mono)" }}>
                          Updated {new Date(p.metrics.last_updated).toLocaleString()}
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
