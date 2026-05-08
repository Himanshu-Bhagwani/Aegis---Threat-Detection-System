"use client";

import { useState } from "react";
import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor, formatScore } from "@/lib/api";
import { THRESHOLDS, METRIC_LABELS } from "@/lib/profiles";

const MODULE_KEYS = ["gps_spoof", "login_anomaly", "password_leak", "fraud_risk", "breach_risk"] as const;

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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>Identity</h1>
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
          Search profiles and view full risk reports
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16, alignItems: "start" }}>
        {/* Search + list */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <input
            className="input"
            placeholder="Search by name, email or ID…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
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
                    borderColor: active ? "var(--accent)" : undefined,
                    background: active ? "var(--bg-raised)" : undefined,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 30, height: 30, borderRadius: "50%", background: c + "22", border: `2px solid ${c}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700, color: c, flexShrink: 0 }}>
                      {p.name[0]}
                    </div>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.name}</div>
                      <div style={{ fontSize: 10, color: "var(--text-muted)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.email}</div>
                    </div>
                    <div style={{ fontSize: 12, fontWeight: 700, fontFamily: "var(--font-mono)", color: c, flexShrink: 0 }}>
                      {(p.metrics.unified_score * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              );
            })}
            {filtered.length === 0 && (
              <div style={{ fontSize: 12, color: "var(--text-muted)", textAlign: "center", padding: "20px 0" }}>No profiles match</div>
            )}
          </div>
        </div>

        {/* Detail panel */}
        {viewing && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {/* Header */}
            <div className="panel" style={{ padding: "20px 22px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                <div style={{
                  width: 48, height: 48, borderRadius: "50%",
                  background: riskColor(viewing.metrics.unified_score) + "22",
                  border: `2px solid ${riskColor(viewing.metrics.unified_score)}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 20, fontWeight: 700, color: riskColor(viewing.metrics.unified_score), flexShrink: 0,
                }}>
                  {viewing.name[0]}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 18, fontWeight: 800, color: "var(--text-primary)" }}>{viewing.name}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{viewing.email} · {viewing.id}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 32, fontWeight: 900, fontFamily: "var(--font-mono)", color: riskColor(viewing.metrics.unified_score), letterSpacing: "-0.03em" }}>
                    {(viewing.metrics.unified_score * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.08em", color: riskColor(viewing.metrics.unified_score), fontWeight: 700, marginTop: 2 }}>
                    {viewing.metrics.risk_level}
                  </div>
                </div>
              </div>
              {viewing.notes && (
                <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-secondary)", padding: "10px 12px", borderRadius: 8, background: "var(--bg-raised)", lineHeight: 1.6 }}>
                  {viewing.notes}
                </div>
              )}
            </div>

            {/* Module breakdown */}
            <div className="panel" style={{ padding: "18px 20px" }}>
              <div className="panel-header" style={{ marginBottom: 16 }}>Risk Breakdown</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {MODULE_KEYS.map(key => {
                  const val      = (viewing.metrics as any)[key] ?? 0;
                  const c        = riskColor(val);
                  const exceeded = val > THRESHOLDS[key];
                  return (
                    <div key={key}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                        <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1 }}>{METRIC_LABELS[key]}</span>
                        {exceeded && <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 8, background: "rgba(239,68,68,0.12)", color: "var(--risk-critical)", fontWeight: 600 }}>⚠ ALERT</span>}
                        <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--font-mono)", color: c }}>{formatScore(val)}</span>
                      </div>
                      <div style={{ height: 6, borderRadius: 3, background: "var(--bg-raised)", overflow: "hidden", position: "relative" }}>
                        <div style={{ height: "100%", width: `${val * 100}%`, background: c, borderRadius: 3 }} />
                        <div style={{ position: "absolute", top: 0, left: `${THRESHOLDS[key] * 100}%`, width: 1, height: "100%", background: "rgba(255,255,255,0.25)" }} />
                      </div>
                      <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 2 }}>
                        Threshold: {(THRESHOLDS[key] * 100).toFixed(0)}% {exceeded ? `· Exceeded by ${((val - THRESHOLDS[key]) * 100).toFixed(0)}%` : ""}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Active alerts */}
            {profileAlerts.length > 0 && (
              <div className="panel" style={{ padding: "18px 20px" }}>
                <div className="panel-header" style={{ marginBottom: 12 }}>Active Alerts ({profileAlerts.length})</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {profileAlerts.map(a => (
                    <div key={a.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8, background: a.severity === "critical" ? "rgba(239,68,68,0.08)" : "rgba(255,102,0,0.08)", border: `1px solid ${a.severity === "critical" ? "rgba(239,68,68,0.2)" : "rgba(255,102,0,0.2)"}` }}>
                      <div style={{ width: 7, height: 7, borderRadius: "50%", background: a.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)", flexShrink: 0 }} />
                      <div style={{ flex: 1, fontSize: 12, color: "var(--text-primary)" }}>{a.message}</div>
                      <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--font-mono)", color: riskColor(a.value) }}>{(a.value * 100).toFixed(0)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Meta */}
            <div className="panel" style={{ padding: "14px 18px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 0 }}>
                {[
                  { label: "Profile ID",    val: viewing.id },
                  { label: "Type",          val: viewing.is_demo ? "Demo" : "Custom" },
                  { label: "Created",       val: new Date(viewing.created_at).toLocaleDateString() },
                  { label: "Last Updated",  val: new Date(viewing.metrics.last_updated).toLocaleString() },
                ].map(({ label, val }) => (
                  <div key={label} style={{ padding: "8px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                    <div style={{ fontSize: 10, color: "var(--text-disabled)", textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
                    <div style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 500, marginTop: 2, fontFamily: label === "Profile ID" ? "var(--font-mono)" : undefined }}>{val}</div>
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
