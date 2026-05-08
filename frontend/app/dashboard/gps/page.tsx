"use client";
import { useState } from "react";
import { scoreGPS, riskColor, formatScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

const SCENARIOS = [
  { label: "Normal movement",     desc: "Walking pace, coherent path",         traj: [{ lat: 37.77, lng: -122.41, timestamp: Date.now() - 120000, speed: 1.5 }, { lat: 37.775, lng: -122.415, timestamp: Date.now() - 60000, speed: 2.0 }, { lat: 37.78, lng: -122.42, timestamp: Date.now(), speed: 1.8 }] },
  { label: "Impossible velocity", desc: "NYC→London in 60 seconds",            traj: [{ lat: 37.77, lng: -122.41, timestamp: Date.now() - 60000, speed: 300 }, { lat: 51.5, lng: -0.12, timestamp: Date.now(), speed: 250 }] },
  { label: "Mock GPS / spoof",    desc: "Null island coordinates, zero speed", traj: [{ lat: 0, lng: 0, timestamp: Date.now() - 30000, speed: 0 }, { lat: 0.001, lng: 0.001, timestamp: Date.now(), speed: 0.1 }] },
];

export default function GpsPage() {
  const { selected, updateMetrics } = useProfiles();
  const [result,  setResult]  = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [updated, setUpdated] = useState(false);

  async function run(traj: any[]) {
    setLoading(true); setUpdated(false);
    try {
      const res = await scoreGPS(traj);
      setResult(res);
      if (res?.risk_score != null && selected) {
        const newVal = selected.metrics.gps_spoof * 0.7 + res.risk_score * 0.3;
        updateMetrics(selected.id, { gps_spoof: Math.min(1, Math.max(0, newVal)) });
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

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
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
        <div className="panel" style={{ padding: "20px 24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <div style={{ fontSize: 52, fontWeight: 900, fontFamily: "var(--font-mono)", color: riskColor(result.risk_score ?? 0), letterSpacing: "-0.04em", lineHeight: 1 }}>
              {formatScore(result.risk_score ?? 0)}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>GPS Spoof Risk Score</div>
              {result.signals && (
                <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                  {Object.entries(result.signals).filter(([, v]) => v).map(([k]) => k).join(" · ") || "No anomalies detected"}
                </div>
              )}
            </div>
            {updated && selected && (
              <div style={{ padding: "8px 14px", borderRadius: 8, background: "rgba(61,127,255,0.1)", border: "1px solid rgba(61,127,255,0.3)", fontSize: 12, color: "var(--accent)" }}>
                ✓ Updated {selected.name}'s GPS profile to <strong>{(selected.metrics.gps_spoof * 100).toFixed(0)}%</strong>
              </div>
            )}
          </div>
        </div>
      )}
      {result?.error && <div style={{ color: "var(--risk-high)", fontSize: 13 }}>{result.error}</div>}
    </div>
  );
}
