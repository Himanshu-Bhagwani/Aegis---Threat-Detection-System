"use client";

import { useState } from "react";
import { scoreGPS, riskColor, formatScore, getWeightedUnifiedScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

const now = Date.now();
const SCENARIOS = [
  {
    label: "Normal Movement",
    desc:  "Walking pace, coherent path",
    risk: "low",
    icon: "🚶",
    traj: [
      { lat: 37.7749, lng: -122.4194, timestamp: now - 120_000, speed: 1.2 },
      { lat: 37.7755, lng: -122.4200, timestamp: now -  60_000, speed: 1.5 },
      { lat: 37.7761, lng: -122.4206, timestamp: now,           speed: 1.4 },
    ],
  },
  {
    label: "Impossible Velocity",
    desc:  "NYC → London in 60 seconds",
    risk: "critical",
    icon: "⚡",
    traj: [
      { lat: 40.7128, lng: -74.0060, timestamp: now - 60_000, speed: 0 },
      { lat: 51.5074, lng:  -0.1278, timestamp: now,          speed: 0 },
    ],
  },
  {
    label: "Null Island Spoof",
    desc:  "Locked near 0°N 0°E (emulator default)",
    risk: "critical",
    icon: "🎯",
    traj: [
      { lat: 0.0001, lng: 0.0001, timestamp: now - 90_000, speed: 0 },
      { lat: 0.0002, lng: 0.0001, timestamp: now - 60_000, speed: 0 },
      { lat: 0.0001, lng: 0.0002, timestamp: now - 30_000, speed: 0 },
      { lat: 0.0002, lng: 0.0002, timestamp: now,          speed: 0 },
    ],
  },
  {
    label: "Static Replay",
    desc:  "Position locked for 3 minutes (replay attack)",
    risk: "high",
    icon: "🔄",
    traj: Array.from({ length: 10 }, (_, i) => ({
      lat: 37.7749, lng: -122.4194,
      timestamp: now - (9 - i) * 20_000,
      speed: 0,
    })),
  },
];

const IconMapPin = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/>
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

export default function GpsPage() {
  const { selected, updateMetrics } = useProfiles();
  const [result,  setResult]  = useState<any>(null);
  const [loading, setLoading] = useState<number | null>(null);
  const [updated, setUpdated] = useState(false);

  async function run(traj: any[], idx: number) {
    setLoading(idx); setUpdated(false);
    try {
      const res = await scoreGPS(traj, selected?.id);
      setResult(res);
      const spoof = res?.spoof_probability ?? res?.risk_score;
      if (spoof != null && selected) {
        const newGps = Math.min(1, Math.max(0, selected.metrics.gps_spoof * 0.7 + spoof * 0.3));
        const m = selected.metrics;
        const unified = await getWeightedUnifiedScore(newGps, m.login_anomaly, m.password_leak, m.fraud_risk, m.breach_risk, selected.id);
        updateMetrics(selected.id, { gps_spoof: newGps, unified_score: unified });
        setUpdated(true);
      }
    } catch (e: any) { setResult({ error: e?.message }); }
    finally { setLoading(null); }
  }

  const resultScore = result?.risk_score ?? 0;
  const resultColor = riskColor(resultScore);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{ color: "var(--accent)" }}><IconMapPin /></div>
            <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
              GPS Spoofing Detection
            </h1>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-muted)", maxWidth: 480 }}>
            Trajectory analysis for mock GPS, impossible velocity, and location spoofing signals
          </p>
        </div>
        {selected && (
          <div style={{
            display: "flex", alignItems: "center", gap: 8, padding: "8px 16px",
            borderRadius: 20, background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)", fontSize: 12, flexShrink: 0,
          }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: riskColor(selected.metrics.gps_spoof), boxShadow: `0 0 8px ${riskColor(selected.metrics.gps_spoof)}` }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{selected.name}</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.gps_spoof), fontWeight: 800 }}>
              {(selected.metrics.gps_spoof * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* Scenario cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {SCENARIOS.map((s, i) => {
          const isLoading = loading === i;
          const borderColor = s.risk === "critical" ? "rgba(239,68,68,0.2)"
            : s.risk === "high" ? "rgba(249,115,22,0.2)"
            : "var(--glass-border)";
          return (
            <div key={s.label} className="panel" style={{ padding: "20px 18px", borderColor }}>
              <div style={{ fontSize: 24, marginBottom: 10 }}>{s.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", marginBottom: 6 }}>{s.label}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 16, lineHeight: 1.6 }}>{s.desc}</div>
              <button
                onClick={() => run(s.traj, i)}
                disabled={isLoading || loading !== null}
                style={{
                  width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
                  padding: "10px", borderRadius: 8, fontSize: 12, fontWeight: 700,
                  background: isLoading ? "rgba(61,127,255,0.3)" : "linear-gradient(135deg, #3d7fff, #2563eb)",
                  color: "#fff", border: "none", cursor: isLoading || loading !== null ? "not-allowed" : "pointer",
                  boxShadow: isLoading ? "none" : "0 0 16px rgba(61,127,255,0.3)",
                  opacity: (loading !== null && !isLoading) ? 0.5 : 1,
                  transition: "all 0.2s",
                }}
              >
                {isLoading ? <><IconLoader /> Scoring…</> : <><IconPlay /> Run Test</>}
              </button>
            </div>
          );
        })}
      </div>

      {/* Result */}
      {result && !result.error && (
        <div className="panel" style={{
          padding: "24px 28px",
          borderColor: `${resultColor}30`,
          boxShadow: `0 0 30px ${resultColor}08`,
          animation: "fadeUp 0.4s var(--ease-out) both",
        }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 24, flexWrap: "wrap" }}>
            {/* Big score */}
            <div>
              <div style={{
                fontSize: 60, fontWeight: 900, fontFamily: "var(--font-mono)",
                color: resultColor, letterSpacing: "-0.04em", lineHeight: 1,
                textShadow: `0 0 30px ${resultColor}60`,
              }}>
                {formatScore(resultScore)}
              </div>
              <div style={{
                marginTop: 6, fontSize: 11, fontWeight: 700, textTransform: "uppercase",
                letterSpacing: "0.08em", color: resultColor,
              }}>
                {result.is_spoofed ? "SPOOFED" : "Clean trajectory"}
              </div>
            </div>

            {/* Meta */}
            <div style={{ flex: 1, minWidth: 200 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>
                GPS Spoof Risk Assessment
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>
                Confidence: <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)", fontWeight: 600 }}>{((result.confidence ?? 0) * 100).toFixed(0)}%</span>
                {" · "}
                Models: <span style={{ color: "var(--accent)" }}>{(result.models_used ?? []).join(", ") || "rule-based"}</span>
              </div>

              {/* Risk factors */}
              {result.risk_factors?.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  {result.risk_factors.map((f: string, i: number) => (
                    <div key={i} style={{ display: "flex", gap: 8, fontSize: 12, color: "var(--text-secondary)", alignItems: "flex-start" }}>
                      <IconChevronRight />
                      {f}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Model scores */}
            {result.model_scores && Object.keys(result.model_scores).length > 0 && (
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
                {Object.entries(result.model_scores as Record<string, number>)
                  .filter(([, v]) => v >= 0)
                  .map(([name, val]) => (
                    <div key={name} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: 16, fontWeight: 800, fontFamily: "var(--font-mono)", color: riskColor(val), letterSpacing: "-0.01em" }}>
                        {(val * 100).toFixed(0)}%
                      </div>
                      <div style={{ fontSize: 9, color: "var(--text-disabled)", textTransform: "uppercase", letterSpacing: "0.06em", marginTop: 3 }}>
                        {name.replace(/_/g, " ")}
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>

          {updated && selected && (
            <div style={{
              marginTop: 16, padding: "10px 16px", borderRadius: 10,
              background: "rgba(61,127,255,0.08)", border: "1px solid rgba(61,127,255,0.25)",
              fontSize: 12, color: "var(--accent)", display: "flex", alignItems: "center", gap: 8,
            }}>
              <IconCheck />
              Updated {selected.name}&apos;s GPS risk to{" "}
              <strong style={{ fontFamily: "var(--font-mono)" }}>{(selected.metrics.gps_spoof * 100).toFixed(0)}%</strong>
            </div>
          )}

          {/* Security recommendations */}
          <div style={{
            marginTop: 8, padding: "18px 20px", borderRadius: 12,
            background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
          }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 12 }}>
              Security Recommendations
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {(resultScore >= 0.5 ? [
                { icon: "🔴", text: "Cross-verify user location using IP geolocation as an independent source of truth." },
                { icon: "🔐", text: "Immediately trigger step-up authentication (MFA) for this session — do not trust location-dependent features." },
                { icon: "📵", text: "Notify the user to disable any VPN, mock-location apps, or GPS emulators on their device." },
                { icon: "🚫", text: "Suspend high-risk actions (payments, account changes) until location can be independently confirmed." },
                { icon: "📋", text: "Log this event and escalate to the security team for manual review." },
              ] : [
                { icon: "✅", text: "Enable GPS tampering alerts on this account to catch future anomalies early." },
                { icon: "📍", text: "Build a baseline location profile — flag logins from new regions automatically." },
                { icon: "🔔", text: "Configure real-time alerts for impossible velocity events (e.g., crossing continents in seconds)." },
                { icon: "🛡️", text: "Combine GPS signals with IP geolocation and device fingerprinting for layered location verification." },
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
  );
}
