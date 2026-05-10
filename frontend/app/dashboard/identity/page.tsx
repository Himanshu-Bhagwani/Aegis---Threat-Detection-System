"use client";

import { useState } from "react";
import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor, formatScore } from "@/lib/api";
import { THRESHOLDS, METRIC_LABELS } from "@/lib/profiles";

const MODULE_KEYS = ["gps_spoof", "login_anomaly", "password_leak", "fraud_risk", "breach_risk"] as const;

const IconUsers = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
  </svg>
);
const IconSearch = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
);
const IconAlertTriangle = () => (
  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
);

export default function IdentityPage() {
  const { profiles, selectedId, setSelectedId, alerts } = useProfiles();
  const [search, setSearch] = useState("");

  const filtered = profiles.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    p.email.toLowerCase().includes(search.toLowerCase()) ||
    p.id.toLowerCase().includes(search.toLowerCase())
  );

  const viewing = profiles.find(p => p.id === selectedId) ?? profiles[0];
  const profileAlerts = alerts.filter(a => a.profile_id === viewing?.id);
  const viewingColor = viewing ? riskColor(viewing.metrics.unified_score) : "var(--accent)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{ color: "var(--accent-cyan)" }}><IconUsers /></div>
          <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
            Identity Management
          </h1>
        </div>
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          Search profiles and view full risk reports
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16, alignItems: "start" }}>

        {/* Search + list */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ position: "relative" }}>
            <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-disabled)", pointerEvents: "none" }}>
              <IconSearch />
            </div>
            <input
              style={{
                width: "100%", padding: "10px 12px 10px 34px", borderRadius: 9,
                background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                color: "var(--text-primary)", fontSize: 12, outline: "none",
                boxSizing: "border-box", transition: "border-color 0.2s",
              }}
              placeholder="Search by name, email or ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {filtered.map(p => {
              const c = riskColor(p.metrics.unified_score);
              const active = p.id === selectedId;
              return (
                <div
                  key={p.id}
                  onClick={() => setSelectedId(p.id)}
                  className="panel"
                  style={{
                    padding: "12px 14px", cursor: "pointer",
                    borderColor: active ? "rgba(6,182,212,0.4)" : undefined,
                    background: active ? "rgba(6,182,212,0.05)" : undefined,
                    transition: "all 0.2s",
                  }}
                  onMouseEnter={e => { if (!active) (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.03)"; }}
                  onMouseLeave={e => { if (!active) (e.currentTarget as HTMLElement).style.background = ""; }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
                      background: `${c}18`, border: `2px solid ${c}`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 13, fontWeight: 800, color: c,
                      boxShadow: active ? `0 0 10px ${c}40` : "none",
                    }}>
                      {p.name[0]}
                    </div>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.name}</div>
                      <div style={{ fontSize: 10, color: "var(--text-muted)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.email}</div>
                    </div>
                    <div style={{ fontSize: 12, fontWeight: 800, fontFamily: "var(--font-mono)", color: c, flexShrink: 0 }}>
                      {(p.metrics.unified_score * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              );
            })}
            {filtered.length === 0 && (
              <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center", padding: "24px 0" }}>
                No profiles match
              </div>
            )}
          </div>
        </div>

        {/* Detail panel */}
        {viewing && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

            {/* Profile header */}
            <div className="panel" style={{
              padding: "22px 24px",
              borderColor: `${viewingColor}25`,
              boxShadow: `0 0 30px ${viewingColor}06`,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div style={{
                  width: 52, height: 52, borderRadius: "50%", flexShrink: 0,
                  background: `${viewingColor}18`, border: `2px solid ${viewingColor}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 22, fontWeight: 900, color: viewingColor,
                  boxShadow: `0 0 20px ${viewingColor}40`,
                }}>
                  {viewing.name[0]}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 18, fontWeight: 900, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>{viewing.name}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3, fontFamily: "var(--font-mono)" }}>
                    {viewing.email} · {viewing.id}
                  </div>
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{
                    fontSize: 36, fontWeight: 900, fontFamily: "var(--font-mono)",
                    color: viewingColor, letterSpacing: "-0.03em", lineHeight: 1,
                    textShadow: `0 0 20px ${viewingColor}60`,
                  }}>
                    {(viewing.metrics.unified_score * 100).toFixed(1)}%
                  </div>
                  <div style={{
                    fontSize: 11, textTransform: "uppercase", letterSpacing: "0.10em",
                    color: viewingColor, fontWeight: 800, marginTop: 4,
                  }}>
                    {viewing.metrics.risk_level}
                  </div>
                </div>
              </div>

              {viewing.notes && (
                <div style={{
                  marginTop: 14, fontSize: 12, color: "var(--text-secondary)",
                  padding: "10px 14px", borderRadius: 8,
                  background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
                  lineHeight: 1.7,
                }}>
                  {viewing.notes}
                </div>
              )}
            </div>

            {/* Module breakdown */}
            <div className="panel" style={{ padding: "20px 22px" }}>
              <div style={{
                fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
                letterSpacing: "0.10em", marginBottom: 18, paddingBottom: 12,
                borderBottom: "1px solid rgba(255,255,255,0.05)",
              }}>Risk Breakdown</div>

              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                {MODULE_KEYS.map(key => {
                  const val      = (viewing.metrics as any)[key] ?? 0;
                  const c        = riskColor(val);
                  const exceeded = val > THRESHOLDS[key];
                  return (
                    <div key={key}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                        <div style={{ width: 7, height: 7, borderRadius: "50%", background: c, flexShrink: 0 }} />
                        <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1, fontWeight: 500 }}>
                          {METRIC_LABELS[key]}
                        </span>
                        {exceeded && (
                          <span style={{
                            fontSize: 9, padding: "2px 7px", borderRadius: 8,
                            background: "rgba(239,68,68,0.12)", color: "var(--risk-critical)",
                            fontWeight: 800, border: "1px solid rgba(239,68,68,0.25)",
                            display: "flex", alignItems: "center", gap: 4, letterSpacing: "0.05em",
                          }}>
                            <IconAlertTriangle /> ALERT
                          </span>
                        )}
                        <span style={{ fontSize: 13, fontWeight: 800, fontFamily: "var(--font-mono)", color: c }}>
                          {formatScore(val)}
                        </span>
                      </div>
                      <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,0.07)", overflow: "hidden", position: "relative" }}>
                        <div style={{
                          height: "100%", width: `${val * 100}%`, background: c, borderRadius: 3,
                          boxShadow: exceeded ? `0 0 8px ${c}80` : "none",
                          transition: "width 0.6s var(--ease-out)",
                        }} />
                        <div style={{
                          position: "absolute", top: 0, left: `${THRESHOLDS[key] * 100}%`,
                          width: 1, height: "100%", background: "rgba(255,255,255,0.3)",
                        }} />
                      </div>
                      <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 3 }}>
                        Threshold {(THRESHOLDS[key] * 100).toFixed(0)}%{exceeded ? ` · Exceeded by ${((val - THRESHOLDS[key]) * 100).toFixed(0)}%` : ""}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Active alerts */}
            {profileAlerts.length > 0 && (
              <div className="panel" style={{ padding: "20px 22px" }}>
                <div style={{
                  fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
                  letterSpacing: "0.10em", marginBottom: 14, paddingBottom: 12,
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                }}>
                  Active Alerts ({profileAlerts.length})
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {profileAlerts.map(a => (
                    <div key={a.id} style={{
                      display: "flex", alignItems: "center", gap: 10, padding: "10px 14px",
                      borderRadius: 8,
                      background: a.severity === "critical" ? "rgba(239,68,68,0.06)" : "rgba(249,115,22,0.06)",
                      border: `1px solid ${a.severity === "critical" ? "rgba(239,68,68,0.2)" : "rgba(249,115,22,0.2)"}`,
                    }}>
                      <div style={{
                        width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
                        background: a.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)",
                        boxShadow: `0 0 6px ${a.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)"}`,
                        animation: "pulse-green 2s ease-in-out infinite",
                      }} />
                      <div style={{ flex: 1, fontSize: 12, color: "var(--text-primary)" }}>{a.message}</div>
                      <div style={{ fontSize: 13, fontWeight: 800, fontFamily: "var(--font-mono)", color: riskColor(a.value) }}>
                        {(a.value * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Meta */}
            <div className="panel" style={{ padding: "16px 20px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 0 }}>
                {[
                  { label: "Profile ID",   val: viewing.id,                                            mono: true },
                  { label: "Type",         val: viewing.is_demo ? "Demo" : "Custom",                   mono: false },
                  { label: "Created",      val: new Date(viewing.created_at).toLocaleDateString(),      mono: false },
                  { label: "Last Updated", val: new Date(viewing.metrics.last_updated).toLocaleString(), mono: false },
                ].map(({ label, val, mono }, i, arr) => (
                  <div key={label} style={{ padding: "9px 0", borderBottom: i < arr.length - 1 && i < arr.length - 2 ? "1px solid rgba(255,255,255,0.04)" : "none" }}>
                    <div style={{ fontSize: 10, color: "var(--text-disabled)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
                    <div style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 600, marginTop: 3, fontFamily: mono ? "var(--font-mono)" : undefined }}>
                      {val}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
