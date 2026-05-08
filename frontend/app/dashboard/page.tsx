"use client";

import dynamic from "next/dynamic";
import { useState, useEffect, useRef, useCallback } from "react";
import { wsClient, riskColor, riskLabel, formatScore, type WSEvent } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";
import { METRIC_LABELS, THRESHOLDS } from "@/lib/profiles";

const RiskGauge3D = dynamic(() => import("@/components/three/RiskGauge3D"), { ssr: false });
const ThreatGlobe = dynamic(() => import("@/components/three/ThreatGlobe"),  { ssr: false });

type RiskLevel = "minimal" | "low" | "medium" | "high" | "critical";

interface GlobePing { lat: number; lon: number; risk: RiskLevel; label: string; }

const DEMO_PINGS: GlobePing[] = [
  { lat:  40.7, lon: -74.0, risk: "high",     label: "New York"  },
  { lat:  51.5, lon:  -0.1, risk: "medium",   label: "London"    },
  { lat:  35.7, lon: 139.7, risk: "critical", label: "Tokyo"     },
  { lat: -33.9, lon: 151.2, risk: "low",      label: "Sydney"    },
  { lat:  48.9, lon:   2.4, risk: "minimal",  label: "Paris"     },
  { lat:  55.8, lon:  37.6, risk: "high",     label: "Moscow"    },
  { lat:  28.6, lon:  77.2, risk: "medium",   label: "Delhi"     },
  { lat: -23.5, lon: -46.6, risk: "low",      label: "São Paulo" },
];

const MODULE_KEYS = [
  { key: "gps_spoof",     label: "GPS Spoofing",            icon: "◉" },
  { key: "login_anomaly", label: "Login Anomaly",           icon: "◐" },
  { key: "password_leak", label: "Password Leak Possibility", icon: "◑" },
  { key: "fraud_risk",    label: "Fraud Possibility",       icon: "◆" },
  { key: "breach_risk",   label: "Breach Risk",             icon: "◍" },
];

function ScoreBar({ score }: { score: number }) {
  const color = riskColor(score);
  return (
    <div className="score-bar" style={{ marginTop: 6 }}>
      <div className="score-bar-fill" style={{ width: `${score * 100}%`, background: color }} />
    </div>
  );
}

export default function DashboardPage() {
  const { profiles, selectedId, setSelectedId, selected, alerts } = useProfiles();
  const [wsConnected, setWsConnected] = useState(false);
  const [globePings,  setGlobePings]  = useState<GlobePing[]>(DEMO_PINGS);
  const [liveCount,   setLiveCount]   = useState(0);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.subscribe((evt: WSEvent) => {
      if (evt.type === "connected") { setWsConnected(true); return; }
      if (evt.type === "detection_event") setLiveCount(n => n + 1);
    });
    return () => { unsub(); };
  }, []);

  const score     = selected?.metrics.unified_score ?? 0;
  const riskLvl   = selected?.metrics.risk_level ?? "minimal";
  const userAlerts = alerts.filter(a => a.profile_id === selectedId);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

      {/* ── Profile cards ───────────────────────────── */}
      <div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10, fontWeight: 600 }}>
          Identity Profiles
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
          {profiles.map(p => {
            const c      = riskColor(p.metrics.unified_score);
            const active = p.id === selectedId;
            return (
              <div
                key={p.id}
                onClick={() => setSelectedId(p.id)}
                className="panel"
                style={{
                  padding: "16px 18px", cursor: "pointer",
                  borderColor: active ? "var(--accent)" : undefined,
                  boxShadow:   active ? "0 0 0 1px var(--accent)" : undefined,
                  transition: "all 0.15s",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                  <div style={{
                    width: 34, height: 34, borderRadius: "50%",
                    background: c + "22", border: `2px solid ${c}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 14, fontWeight: 700, color: c, flexShrink: 0,
                  }}>
                    {p.name[0]}
                  </div>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.name}</div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.email}</div>
                  </div>
                  <div style={{ marginLeft: "auto", textAlign: "right", flexShrink: 0 }}>
                    <div style={{ fontSize: 16, fontWeight: 800, fontFamily: "var(--font-mono)", color: c }}>
                      {(p.metrics.unified_score * 100).toFixed(0)}%
                    </div>
                    <div style={{ fontSize: 10, textTransform: "capitalize", color: c, fontWeight: 600 }}>
                      {p.metrics.risk_level}
                    </div>
                  </div>
                </div>
                <div style={{ height: 4, borderRadius: 2, background: "var(--bg-raised)", overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${p.metrics.unified_score * 100}%`, background: c, borderRadius: 2 }} />
                </div>
                {p.notes && (
                  <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 8, lineHeight: 1.4, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                    {p.notes}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Stat bar ─────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {[
          { label: "Active Alerts",  value: userAlerts.length,                               icon: "◈", warn: userAlerts.length > 0, color: userAlerts.length > 0 ? "var(--risk-high)" : undefined },
          { label: "Breach Risk",    value: formatScore(selected?.metrics.breach_risk),       icon: "◍", warn: (selected?.metrics.breach_risk ?? 0) > THRESHOLDS.breach_risk },
          { label: "GPS Spoof",      value: formatScore(selected?.metrics.gps_spoof),         icon: "◉", warn: (selected?.metrics.gps_spoof ?? 0) > THRESHOLDS.gps_spoof },
          { label: "WS Status",      value: wsConnected ? "Live" : "Connecting",              icon: "●", color: wsConnected ? "var(--risk-minimal)" : "var(--text-disabled)" },
        ].map(stat => (
          <div key={stat.label} className="panel" style={{ padding: "14px 16px" }}>
            <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6, display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ color: stat.color ?? (stat.warn ? "var(--risk-high)" : "var(--accent)") }}>{stat.icon}</span>
              {stat.label}
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "var(--font-mono)", color: stat.color ?? (stat.warn ? "var(--risk-high)" : "var(--text-primary)"), letterSpacing: "-0.02em" }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* ── Main grid ─────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr 300px", gap: 16 }}>

        {/* Gauge */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "20px 0 16px" }}>
          <div className="panel-header" style={{ width: "100%", padding: "0 18px 12px" }}>Unified Risk Score</div>
          <RiskGauge3D score={score} size={210} />
          <div style={{ marginTop: 4, fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: riskColor(score) }}>
            {riskLvl}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", padding: "6px 18px 0" }}>
            {selected?.name ?? "No profile selected"}
          </div>
        </div>

        {/* Module scores for selected profile */}
        <div className="panel" style={{ padding: "18px 20px" }}>
          <div className="panel-header" style={{ marginBottom: 14 }}>Module Breakdown — {selected?.name}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {MODULE_KEYS.map(({ key, label, icon }) => {
              const val       = (selected?.metrics as any)?.[key] ?? 0;
              const color     = riskColor(val);
              const threshold = THRESHOLDS[key];
              const exceeded  = val > threshold;
              return (
                <div key={key}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                    <span style={{ fontSize: 13, opacity: 0.7 }}>{icon}</span>
                    <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1 }}>{label}</span>
                    {exceeded && (
                      <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 8, background: "rgba(239,68,68,0.12)", color: "var(--risk-critical)", fontWeight: 600 }}>
                        ALERT
                      </span>
                    )}
                    <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--font-mono)", color }}>{(val * 100).toFixed(0)}%</span>
                  </div>
                  <div style={{ height: 5, borderRadius: 3, background: "var(--bg-raised)", overflow: "hidden", position: "relative" }}>
                    <div style={{ height: "100%", width: `${val * 100}%`, background: color, borderRadius: 3, transition: "width 0.4s" }} />
                    <div style={{ position: "absolute", top: 0, left: `${threshold * 100}%`, width: 1, height: "100%", background: "rgba(255,255,255,0.3)" }} />
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 2 }}>
                    Threshold: {(threshold * 100).toFixed(0)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Globe */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "16px 0" }}>
          <div className="panel-header" style={{ width: "100%", padding: "0 16px 12px" }}>Threat Globe</div>
          <ThreatGlobe pings={globePings} width={268} height={268} />
          <div style={{ width: "100%", padding: "10px 16px 0" }}>
            {(["critical","high","medium","low","minimal"] as RiskLevel[]).map(lvl => (
              <div key={lvl} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: riskColor(lvl === "critical" ? 1 : lvl === "high" ? 0.75 : lvl === "medium" ? 0.5 : lvl === "low" ? 0.25 : 0.05), flexShrink: 0 }} />
                <span style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "capitalize" }}>{lvl}</span>
                <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-disabled)", fontFamily: "var(--font-mono)" }}>
                  {globePings.filter(p => p.risk === lvl).length}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Active alerts for this profile ───────────── */}
      {userAlerts.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10, fontWeight: 600 }}>
            Active Alerts — {selected?.name}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {userAlerts.map(a => (
              <div key={a.id} className="panel" style={{ padding: "12px 16px", display: "flex", alignItems: "center", gap: 14, borderColor: a.severity === "critical" ? "rgba(239,68,68,0.3)" : "rgba(255,102,0,0.3)" }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: a.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)", flexShrink: 0, boxShadow: `0 0 8px ${a.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)"}` }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>{a.message}</div>
                </div>
                <div style={{ fontSize: 13, fontWeight: 700, fontFamily: "var(--font-mono)", color: riskColor(a.value) }}>{(a.value * 100).toFixed(0)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
