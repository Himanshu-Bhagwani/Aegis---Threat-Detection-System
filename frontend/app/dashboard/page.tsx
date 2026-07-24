"use client";

import dynamic from "next/dynamic";
import { useState, useEffect, useRef } from "react";
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
const PING_POOL: Omit<GlobePing, "risk">[] = [
  { lat: 37.8, lon: -122.4, label: "San Francisco" },
  { lat: 41.9, lon:   12.5, label: "Rome"          },
  { lat: 39.9, lon:  116.4, label: "Beijing"       },
  { lat: 19.1, lon:  -99.1, label: "Mexico City"   },
  { lat: 52.5, lon:   13.4, label: "Berlin"        },
  { lat: -34.6, lon: -58.4, label: "Buenos Aires"  },
  { lat:  1.35, lon: 103.8, label: "Singapore"     },
  { lat: 55.7, lon:   37.6, label: "Moscow"        },
];

const MODULE_KEYS = [
  { key: "gps_spoof",     label: "GPS Spoofing",              color: "var(--accent)" },
  { key: "login_anomaly", label: "Login Anomaly",             color: "var(--accent-cyan)" },
  { key: "fraud_risk",    label: "Fraud Possibility",         color: "var(--risk-high)" },
  { key: "breach_risk",   label: "Breach Risk",               color: "var(--risk-critical)" },
];

const IconBell = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 01-3.46 0"/>
  </svg>
);
const IconShield = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const IconMapPin = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/>
  </svg>
);
const IconActivity = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);

export default function DashboardPage() {
  const { profiles, selectedId, setSelectedId, selected, alerts } = useProfiles();
  const [wsConnected, setWsConnected] = useState(false);
  const [globePings,  setGlobePings]  = useState<GlobePing[]>(DEMO_PINGS);
  const [liveCount,   setLiveCount]   = useState(0);
  const pingIdxRef = useRef(0);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.subscribe((evt: WSEvent) => {
      if (evt.type === "connected") { setWsConnected(true); return; }
      if (evt.type === "detection_event") {
        setLiveCount(n => n + 1);
        const riskLvl = (evt.risk_level as RiskLevel) || (
          (evt.risk_score ?? 0) >= 0.75 ? "critical" :
          (evt.risk_score ?? 0) >= 0.50 ? "high" :
          (evt.risk_score ?? 0) >= 0.25 ? "medium" : "low"
        );
        const loc = PING_POOL[pingIdxRef.current % PING_POOL.length];
        pingIdxRef.current += 1;
        const label = evt.event_type ? evt.event_type.replace("_", " ") : loc.label;
        setGlobePings(prev => [...prev.slice(-19), { ...loc, risk: riskLvl, label }]);
      }
    });
    return () => { unsub(); };
  }, []);

  const score   = selected?.metrics.unified_score ?? 0;
  const riskLvl = selected?.metrics.risk_level ?? "minimal";
  const userAlerts = alerts.filter(a => a.profile_id === selectedId);

  const scoreColor = riskColor(score);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── Page header ─────────────────────────── */}
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)", marginBottom: 4 }}>
          Threat Overview
        </h1>
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          Real-time identity risk dashboard · All detection modules active
        </p>
      </div>

      {/* ── Profile cards ───────────────────────── */}
      <div>
        <div style={{ fontSize: 10, color: "var(--text-disabled)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 12, fontWeight: 700 }}>
          Identity Profiles
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
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
                  borderColor: active ? "rgba(61,127,255,0.35)" : undefined,
                  boxShadow: active ? "0 0 20px rgba(61,127,255,0.1), 0 0 0 1px rgba(61,127,255,0.2)" : undefined,
                  transition: "all 0.2s var(--ease-out)",
                }}
                onMouseEnter={e => { if (!active) { (e.currentTarget as HTMLElement).style.transform = "translateY(-2px)"; } }}
                onMouseLeave={e => { if (!active) { (e.currentTarget as HTMLElement).style.transform = "translateY(0)"; } }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: "50%", flexShrink: 0,
                    background: `${c}18`,
                    border: `2px solid ${c}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 14, fontWeight: 800, color: c,
                    boxShadow: `0 0 12px ${c}40`,
                  }}>
                    {p.name[0]}
                  </div>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.name}</div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{p.email}</div>
                  </div>
                  <div style={{ textAlign: "right", flexShrink: 0 }}>
                    <div style={{ fontSize: 17, fontWeight: 900, fontFamily: "var(--font-mono)", color: c, letterSpacing: "-0.02em", textShadow: `0 0 12px ${c}60` }}>
                      {(p.metrics.unified_score * 100).toFixed(0)}%
                    </div>
                    <div style={{ fontSize: 9, textTransform: "uppercase", color: c, fontWeight: 700, letterSpacing: "0.06em" }}>
                      {p.metrics.risk_level}
                    </div>
                  </div>
                </div>
                {/* Score bar */}
                <div style={{ height: 3, borderRadius: 2, background: "rgba(255,255,255,0.07)", overflow: "hidden" }}>
                  <div style={{
                    height: "100%", width: `${p.metrics.unified_score * 100}%`,
                    background: c, borderRadius: 2,
                    boxShadow: `0 0 8px ${c}60`,
                    transition: "width 0.6s var(--ease-out)",
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Stat bar ────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {[
          {
            label: "Active Alerts", value: userAlerts.length, icon: <IconBell />,
            color: userAlerts.length > 0 ? "var(--risk-critical)" : "var(--text-muted)",
            glow: userAlerts.length > 0 ? "rgba(239,68,68,0.12)" : "transparent",
            border: userAlerts.length > 0 ? "rgba(239,68,68,0.2)" : "var(--glass-border)",
          },
          {
            label: "Breach Risk", value: formatScore(selected?.metrics.breach_risk), icon: <IconShield />,
            color: (selected?.metrics.breach_risk ?? 0) > THRESHOLDS.breach_risk ? "var(--risk-critical)" : "var(--text-primary)",
            glow: "transparent", border: "var(--glass-border)",
          },
          {
            label: "GPS Spoof", value: formatScore(selected?.metrics.gps_spoof), icon: <IconMapPin />,
            color: (selected?.metrics.gps_spoof ?? 0) > THRESHOLDS.gps_spoof ? "var(--risk-high)" : "var(--text-primary)",
            glow: "transparent", border: "var(--glass-border)",
          },
          {
            label: "WS Feed", value: wsConnected ? "Live" : "Connecting", icon: <IconActivity />,
            color: wsConnected ? "var(--risk-minimal)" : "var(--text-muted)",
            glow: wsConnected ? "rgba(0,255,136,0.06)" : "transparent",
            border: wsConnected ? "rgba(0,255,136,0.15)" : "var(--glass-border)",
          },
        ].map(stat => (
          <div key={stat.label} className="panel" style={{ padding: "16px 18px", background: stat.glow, borderColor: stat.border }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 6,
              fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase",
              letterSpacing: "0.08em", marginBottom: 10, fontWeight: 700,
            }}>
              <span style={{ color: stat.color }}>{stat.icon}</span>
              {stat.label}
            </div>
            <div style={{
              fontSize: 24, fontWeight: 900, fontFamily: "var(--font-mono)",
              color: stat.color, letterSpacing: "-0.02em",
              textShadow: stat.color !== "var(--text-muted)" && stat.color !== "var(--text-primary)" ? `0 0 16px ${stat.color}60` : "none",
            }}>
              {String(stat.value)}
            </div>
          </div>
        ))}
      </div>

      {/* ── Main 3-col grid ─────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "256px 1fr 280px", gap: 16 }}>

        {/* 3D Risk Gauge */}
        <div className="panel" style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          padding: "20px 0 18px",
          background: "rgba(255,255,255,0.025)",
          borderColor: score > 0.5 ? `${scoreColor}30` : "var(--glass-border)",
          boxShadow: score > 0.5 ? `0 0 30px ${scoreColor}12` : "none",
        }}>
          <div style={{
            width: "100%", padding: "0 18px 14px",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
            marginBottom: 8,
          }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em" }}>
              Unified Risk Score
            </div>
          </div>
          <RiskGauge3D score={score} size={210} />
          <div style={{
            marginTop: 8, fontSize: 12, fontWeight: 800, textTransform: "uppercase",
            letterSpacing: "0.1em", color: scoreColor,
            textShadow: `0 0 16px ${scoreColor}`,
          }}>
            {riskLvl}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", padding: "6px 18px 0" }}>
            {selected?.name ?? "No profile selected"}
          </div>
        </div>

        {/* Module Breakdown */}
        <div className="panel" style={{ padding: "20px 22px" }}>
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            marginBottom: 20, paddingBottom: 14, borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em" }}>
              Module Breakdown
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{selected?.name}</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            {MODULE_KEYS.map(({ key, label, color }) => {
              const val       = (selected?.metrics as any)?.[key] ?? 0;
              const c         = riskColor(val);
              const threshold = THRESHOLDS[key];
              const exceeded  = val > threshold;
              return (
                <div key={key}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: c, boxShadow: `0 0 8px ${c}`, flexShrink: 0 }} />
                    <span style={{ fontSize: 12, color: "var(--text-secondary)", flex: 1, fontWeight: 500 }}>{label}</span>
                    {exceeded && (
                      <span style={{
                        fontSize: 9, padding: "2px 7px", borderRadius: 8,
                        background: "rgba(239,68,68,0.12)", color: "var(--risk-critical)",
                        fontWeight: 800, letterSpacing: "0.05em",
                        border: "1px solid rgba(239,68,68,0.25)",
                        animation: "neon-pulse 1.5s ease-in-out infinite",
                      }}>ALERT</span>
                    )}
                    <span style={{ fontSize: 14, fontWeight: 900, fontFamily: "var(--font-mono)", color: c, letterSpacing: "-0.01em" }}>
                      {(val * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,0.06)", overflow: "hidden", position: "relative" }}>
                    <div style={{
                      height: "100%", width: `${val * 100}%`, background: c,
                      borderRadius: 3, transition: "width 0.6s var(--ease-out)",
                      boxShadow: `0 0 10px ${c}60`,
                    }} />
                    <div style={{ position: "absolute", top: 0, left: `${threshold * 100}%`, width: 1, height: "100%", background: "rgba(255,255,255,0.4)" }} />
                  </div>
                  <div style={{ fontSize: 9, color: "var(--text-disabled)", marginTop: 3, fontFamily: "var(--font-mono)" }}>
                    threshold: {(threshold * 100).toFixed(0)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Threat Globe */}
        <div className="panel" style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "16px 0 12px" }}>
          <div style={{
            width: "100%", padding: "0 16px 12px",
            borderBottom: "1px solid rgba(255,255,255,0.05)", marginBottom: 4,
          }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em" }}>
              Threat Globe
            </div>
          </div>
          <ThreatGlobe pings={globePings} width={248} height={248} />
          {/* Legend */}
          <div style={{ width: "100%", padding: "10px 16px 0" }}>
            {(["critical","high","medium","low","minimal"] as RiskLevel[]).map(lvl => {
              const c = riskColor(lvl === "critical" ? 1 : lvl === "high" ? 0.75 : lvl === "medium" ? 0.5 : lvl === "low" ? 0.25 : 0.05);
              const count = globePings.filter(p => p.risk === lvl).length;
              return (
                <div key={lvl} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                  <div style={{ width: 7, height: 7, borderRadius: "50%", background: c, boxShadow: `0 0 6px ${c}`, flexShrink: 0 }} />
                  <span style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "capitalize", flex: 1 }}>{lvl}</span>
                  <span style={{ fontSize: 10, color: "var(--text-disabled)", fontFamily: "var(--font-mono)", fontWeight: 700 }}>{count}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Active Alerts ───────────────────────── */}
      {userAlerts.length > 0 && (
        <div>
          <div style={{ fontSize: 10, color: "var(--text-disabled)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 12, fontWeight: 700 }}>
            Active Alerts — {selected?.name}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {userAlerts.map(a => (
              <div
                key={a.id}
                className={`alert-row ${a.severity}`}
              >
                <div style={{
                  width: 10, height: 10, borderRadius: "50%", flexShrink: 0,
                  background: a.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)",
                  boxShadow: `0 0 12px ${a.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)"}80`,
                  animation: "pulse-green 2s ease-in-out infinite",
                }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>{a.message}</div>
                </div>
                <div style={{
                  fontSize: 14, fontWeight: 900, fontFamily: "var(--font-mono)",
                  color: riskColor(a.value), letterSpacing: "-0.01em",
                }}>
                  {(a.value * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
