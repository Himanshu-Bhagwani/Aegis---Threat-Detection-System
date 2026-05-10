"use client";

import { useState, useEffect } from "react";
import { scoreDevice, collectBrowserFingerprint, riskColor, formatScore } from "@/lib/api";

const IconMonitor = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
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
const IconChevronRight = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);
const IconUser = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/>
  </svg>
);
const IconCpu = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>
  </svg>
);

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
        unusual_time:      new Date().getHours() < 6 || new Date().getHours() > 22,
        location_mismatch: false,
      });
      setResult(res);
    } catch (e: any) {
      setResult({ error: e?.message });
    } finally {
      setLoading(false);
    }
  }

  const resultScore = result?.risk_score ?? 0;
  const resultColor = riskColor(resultScore);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{ color: "var(--accent-purple)" }}><IconMonitor /></div>
          <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
            Device Fingerprinting
          </h1>
        </div>
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          SHA-256 stable device IDs with weighted similarity scoring across 9 browser signals
        </p>
      </div>

      {/* User ID + trigger */}
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <div style={{ position: "relative", flex: 1, maxWidth: 360 }}>
          <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-disabled)", pointerEvents: "none" }}>
            <IconUser />
          </div>
          <input
            style={{
              width: "100%", padding: "10px 12px 10px 36px", borderRadius: 9,
              background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
              color: "var(--text-primary)", fontSize: 13, fontFamily: "var(--font-mono)",
              outline: "none", boxSizing: "border-box",
            }}
            placeholder="User ID"
            value={userId}
            onChange={e => setUserId(e.target.value)}
          />
        </div>
        <button
          onClick={runScan}
          disabled={loading || !fp}
          style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "10px 22px", borderRadius: 9, fontSize: 13, fontWeight: 700,
            background: loading ? "rgba(139,92,246,0.3)" : "linear-gradient(135deg, #8b5cf6, #7c3aed)",
            color: "#fff", border: "none", cursor: loading || !fp ? "not-allowed" : "pointer",
            boxShadow: loading ? "none" : "0 0 20px rgba(139,92,246,0.3)",
            transition: "all 0.2s", flexShrink: 0,
          }}
        >
          {loading ? <><IconLoader /> Scoring…</> : <><IconPlay /> Score Device</>}
        </button>
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Fingerprint panel */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{
            display: "flex", alignItems: "center", gap: 8, marginBottom: 18,
            paddingBottom: 12, borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>
            <div style={{ color: "var(--accent-purple)" }}><IconCpu /></div>
            <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em" }}>
              Browser Fingerprint (live)
            </span>
          </div>

          {fp ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
              {Object.entries(fp).map(([k, v], i) => (
                <div key={k} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "8px 0",
                  borderBottom: i < Object.keys(fp).length - 1 ? "1px solid rgba(255,255,255,0.04)" : "none",
                }}>
                  <span style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "capitalize", flexShrink: 0, marginRight: 12 }}>
                    {k.replace(/_/g, " ")}
                  </span>
                  <span style={{
                    fontSize: 11, color: "var(--accent-purple)", fontFamily: "var(--font-mono)",
                    maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    textAlign: "right",
                  }}>
                    {String(v)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-disabled)", fontSize: 12 }}>
              Collecting fingerprint…
            </div>
          )}
        </div>

        {/* Result panel */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{
            fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
            letterSpacing: "0.10em", marginBottom: 18, paddingBottom: 12,
            borderBottom: "1px solid rgba(255,255,255,0.05)",
          }}>Risk Assessment</div>

          {!result && !loading && (
            <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-disabled)" }}>
              <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>
                <IconMonitor />
              </div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Run fingerprint scan</div>
              <div style={{ fontSize: 11, marginTop: 6 }}>Enter a User ID and click Score Device</div>
            </div>
          )}

          {loading && (
            <div style={{ padding: "60px 0", textAlign: "center", color: "var(--text-muted)" }}>
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}><IconLoader /></div>
              <div style={{ fontSize: 13 }}>Analysing browser signals…</div>
            </div>
          )}

          {result && !result.error && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeUp 0.4s var(--ease-out) both" }}>
              {/* Big score */}
              <div style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
                <div style={{
                  fontSize: 64, fontWeight: 900, fontFamily: "var(--font-mono)",
                  color: resultColor, letterSpacing: "-0.04em", lineHeight: 1,
                  textShadow: `0 0 40px ${resultColor}60`,
                }}>
                  {formatScore(resultScore)}
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>Risk Score</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                    level: <span style={{ color: resultColor, fontWeight: 700 }}>{result.risk_level ?? "—"}</span>
                  </div>
                </div>
              </div>

              {/* Score bar */}
              <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.07)", overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${resultScore * 100}%`, background: resultColor,
                  borderRadius: 3, transition: "width 0.8s var(--ease-out)",
                  boxShadow: `0 0 12px ${resultColor}60`,
                }} />
              </div>

              {/* Detail rows */}
              <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                {[
                  { label: "Device ID",      val: result.device_id ? result.device_id.slice(0, 20) + "…" : "—", mono: true },
                  { label: "New Device",     val: result.signals?.is_new_device ? "Yes — first seen" : "No — known device", mono: false },
                  { label: "Match Score",    val: result.best_match_score != null ? `${(result.best_match_score * 100).toFixed(0)}%` : "—", mono: true },
                ].map((row, i, arr) => (
                  <div key={row.label} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "9px 0",
                    borderBottom: i < arr.length - 1 ? "1px solid rgba(255,255,255,0.04)" : "none",
                  }}>
                    <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{row.label}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", fontFamily: row.mono ? "var(--font-mono)" : undefined }}>
                      {row.val}
                    </span>
                  </div>
                ))}
              </div>

              {/* Recommended actions */}
              {result.recommended_actions?.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>
                    Recommended Actions
                  </div>
                  {result.recommended_actions.map((a: string, i: number) => (
                    <div key={i} style={{ display: "flex", gap: 8, fontSize: 12, color: "var(--text-secondary)", alignItems: "flex-start" }}>
                      <span style={{ color: "var(--accent-purple)", flexShrink: 0, marginTop: 1 }}><IconChevronRight /></span>{a}
                    </div>
                  ))}
                </div>
              )}
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
      </div>
    </div>
  );
}
