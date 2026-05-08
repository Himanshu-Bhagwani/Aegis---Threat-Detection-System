"use client";

import { useState, useEffect } from "react";
import { scoreDevice, collectBrowserFingerprint, riskColor, formatScore } from "@/lib/api";

export default function DevicePage() {
  const [fp,      setFp]      = useState<any>(null);
  const [result,  setResult]  = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [userId,  setUserId]  = useState("demo_user_001");

  useEffect(() => {
    setFp(collectBrowserFingerprint());
  }, []);

  async function runScan() {
    if (!fp) return;
    setLoading(true);
    try {
      const res = await scoreDevice(userId, fp, {
        unusual_time:     new Date().getHours() < 6 || new Date().getHours() > 22,
        location_mismatch: false,
      });
      setResult(res);
    } catch (e: any) {
      setResult({ error: e?.message });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>Device Fingerprinting</h1>
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
          SHA-256 stable device IDs with weighted similarity scoring across 9 browser signals
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Fingerprint panel */}
        <div className="panel" style={{ padding: "20px 22px" }}>
          <div className="panel-header">Browser Fingerprint (live)</div>
          <div style={{ marginTop: 12 }}>
            <div style={{ marginBottom: 12, display: "flex", gap: 10 }}>
              <input
                className="input"
                style={{ flex: 1 }}
                placeholder="User ID"
                value={userId}
                onChange={e => setUserId(e.target.value)}
              />
              <button className="btn-primary" onClick={runScan} disabled={loading || !fp} style={{ padding: "0 20px" }}>
                {loading ? "Scoring…" : "Score Device"}
              </button>
            </div>
            {fp && Object.entries(fp).map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 11,
                padding: "5px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                <span style={{ color: "var(--text-muted)", textTransform: "capitalize" }}>
                  {k.replace(/_/g, " ")}
                </span>
                <span style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)",
                  maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Result panel */}
        <div className="panel" style={{ padding: "20px 22px" }}>
          <div className="panel-header">Risk Assessment</div>
          {!result && !loading && (
            <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-disabled)", fontSize: 13 }}>
              Click "Score Device" to analyse your current browser fingerprint
            </div>
          )}
          {loading && (
            <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
              Scoring…
            </div>
          )}
          {result && !result.error && (
            <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Risk Score</span>
                <span style={{ fontSize: 28, fontWeight: 800, fontFamily: "var(--font-mono)",
                  color: riskColor(result.risk_score ?? 0) }}>
                  {result.risk_score != null ? formatScore(result.risk_score) : "—"}
                </span>
              </div>
              <div style={{ height: 4, borderRadius: 2, background: "var(--bg-raised)", overflow: "hidden", marginBottom: 8 }}>
                <div style={{ height: "100%", borderRadius: 2, width: `${(result.risk_score ?? 0) * 100}%`,
                  background: riskColor(result.risk_score ?? 0), transition: "width 0.6s" }} />
              </div>
              {[
                { label: "Device ID",    val: result.device_id ? result.device_id.slice(0, 16) + "…" : "—" },
                { label: "Risk Level",   val: result.risk_level ?? "—" },
                { label: "Is New Device",val: result.signals?.is_new_device ? "Yes" : "No" },
                { label: "Match Score",  val: result.best_match_score != null ? `${(result.best_match_score * 100).toFixed(0)}%` : "—" },
              ].map(row => (
                <div key={row.label} style={{ display: "flex", justifyContent: "space-between",
                  fontSize: 12, padding: "7px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                  <span style={{ color: "var(--text-muted)" }}>{row.label}</span>
                  <span style={{ fontWeight: 600, fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>{row.val}</span>
                </div>
              ))}
              {result.recommended_actions?.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6,
                    textTransform: "uppercase", letterSpacing: "0.06em" }}>Actions</div>
                  {result.recommended_actions.map((a: string, i: number) => (
                    <div key={i} style={{ fontSize: 12, color: "var(--text-secondary)", padding: "3px 0",
                      display: "flex", gap: 7 }}>
                      <span style={{ color: "var(--accent)" }}>→</span>{a}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {result?.error && (
            <div style={{ marginTop: 12, fontSize: 12, color: "var(--risk-high)", padding: 10,
              background: "rgba(239,68,68,0.08)", borderRadius: 6 }}>
              {result.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
