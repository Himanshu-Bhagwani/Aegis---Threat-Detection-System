"use client";
import { useState } from "react";
import { scoreGPS, riskColor, formatScore, getWeightedUnifiedScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

const now = Date.now();
const SCENARIOS = [
  {
    label: "Normal movement",
    desc:  "Walking pace, coherent path",
    traj: [
      { lat: 37.7749, lng: -122.4194, timestamp: now - 120_000, speed: 1.2 },
      { lat: 37.7755, lng: -122.4200, timestamp: now -  60_000, speed: 1.5 },
      { lat: 37.7761, lng: -122.4206, timestamp: now,           speed: 1.4 },
    ],
  },
  {
    label: "Impossible velocity",
    desc:  "NYC → London in 60 seconds",
    traj: [
      { lat: 40.7128, lng: -74.0060, timestamp: now - 60_000, speed: 0 },
      { lat: 51.5074, lng:  -0.1278, timestamp: now,          speed: 0 },
    ],
  },
  {
    label: "Mock GPS / null island",
    desc:  "Coordinates locked near 0°N 0°E (common emulator default)",
    traj: [
      { lat: 0.0001, lng: 0.0001, timestamp: now - 90_000, speed: 0 },
      { lat: 0.0002, lng: 0.0001, timestamp: now - 60_000, speed: 0 },
      { lat: 0.0001, lng: 0.0002, timestamp: now - 30_000, speed: 0 },
      { lat: 0.0002, lng: 0.0002, timestamp: now,          speed: 0 },
    ],
  },
  {
    label: "Static spoof",
    desc:  "Position locked to one point for 3 minutes (replay attack)",
    traj: Array.from({ length: 10 }, (_, i) => ({
      lat: 37.7749, lng: -122.4194,
      timestamp: now - (9 - i) * 20_000,
      speed: 0,
    })),
  },
];

export default function GpsPage() {
  const { selected, updateMetrics } = useProfiles();
  const [result,  setResult]  = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [updated, setUpdated] = useState(false);

  async function run(traj: any[]) {
    setLoading(true); setUpdated(false);
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
    finally { setLoading(false); }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>GPS Spoofing Detection</h1>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>Trajectory analysis for mock GPS, impossible velocity, and location spoofing signals</p>
        </div>
        {selected && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 20, background: "var(--bg-raised)", border: "1px solid var(--border-subtle)", fontSize: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: riskColor(selected.metrics.gps_spoof) }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{selected.name}</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.gps_spoof), fontWeight: 700 }}>
              GPS {(selected.metrics.gps_spoof * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {SCENARIOS.map(s => (
          <div key={s.label} className="panel" style={{ padding: "16px 18px" }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{s.label}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 12 }}>{s.desc}</div>
            <button className="btn-primary" style={{ width: "100%", fontSize: 12 }} onClick={() => run(s.traj)} disabled={loading}>
              {loading ? "Scoring…" : "Run Test"}
            </button>
          </div>
        ))}
      </div>

      {result && !result.error && (
        <div className="panel" style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: 52, fontWeight: 900, fontFamily: "var(--font-mono)", color: riskColor(result.risk_score ?? 0), letterSpacing: "-0.04em", lineHeight: 1 }}>
              {formatScore(result.risk_score ?? 0)}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
                GPS Spoof Risk — {result.is_spoofed ? "SPOOFED" : "Clean"}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
                Confidence: {((result.confidence ?? 0) * 100).toFixed(0)}% · Models: {(result.models_used ?? []).join(", ") || "rule-based"}
              </div>
            </div>
            {updated && selected && (
              <div style={{ padding: "8px 14px", borderRadius: 8, background: "rgba(61,127,255,0.1)", border: "1px solid rgba(61,127,255,0.3)", fontSize: 12, color: "var(--accent)" }}>
                ✓ Updated {selected.name}'s GPS to <strong>{(selected.metrics.gps_spoof * 100).toFixed(0)}%</strong>
              </div>
            )}
          </div>

          {/* Risk factors */}
          {result.risk_factors?.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>Triggered Rules</div>
              {result.risk_factors.map((f: string, i: number) => (
                <div key={i} style={{ display: "flex", gap: 8, fontSize: 12, color: "var(--text-secondary)" }}>
                  <span style={{ color: "var(--risk-high)", flexShrink: 0 }}>→</span>{f}
                </div>
              ))}
            </div>
          )}

          {/* Per-model scores */}
          {result.model_scores && Object.keys(result.model_scores).length > 0 && (
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {Object.entries(result.model_scores as Record<string, number>)
                .filter(([, v]) => v >= 0)
                .map(([name, val]) => (
                  <div key={name} style={{ textAlign: "center", minWidth: 72 }}>
                    <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "var(--font-mono)", color: riskColor(val) }}>
                      {(val * 100).toFixed(0)}%
                    </div>
                    <div style={{ fontSize: 9, color: "var(--text-disabled)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      {name.replace(/_/g, " ")}
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
      {result?.error && <div style={{ color: "var(--risk-high)", fontSize: 13 }}>{result.error}</div>}
    </div>
  );
}
