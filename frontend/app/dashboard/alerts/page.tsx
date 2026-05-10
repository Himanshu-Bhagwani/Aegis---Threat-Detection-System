"use client";

import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor, formatScore } from "@/lib/api";
import { THRESHOLDS } from "@/lib/profiles";

const IconBell = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 01-3.46 0"/>
  </svg>
);
const IconShieldOff = () => (
  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
  </svg>
);
const IconShieldCheck = () => (
  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>
  </svg>
);
const IconX = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

export default function AlertsPage() {
  const { alerts, dismissAlert, profiles } = useProfiles();

  const criticalCount = alerts.filter(a => a.severity === "critical").length;
  const highCount     = alerts.filter(a => a.severity === "high").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{ color: "var(--risk-critical)" }}><IconBell /></div>
            <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
              Security Alerts
            </h1>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            Metrics that exceeded configured thresholds across all identity profiles
          </p>
        </div>

        {alerts.length > 0 && (
          <div style={{ display: "flex", gap: 10, flexShrink: 0 }}>
            {criticalCount > 0 && (
              <div style={{
                padding: "7px 16px", borderRadius: 20,
                background: "rgba(239,68,68,0.10)", border: "1px solid rgba(239,68,68,0.30)",
                fontSize: 12, fontWeight: 800, color: "var(--risk-critical)",
                display: "flex", alignItems: "center", gap: 6,
              }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--risk-critical)", animation: "pulse-green 1.5s ease-in-out infinite" }} />
                {criticalCount} Critical
              </div>
            )}
            {highCount > 0 && (
              <div style={{
                padding: "7px 16px", borderRadius: 20,
                background: "rgba(249,115,22,0.10)", border: "1px solid rgba(249,115,22,0.30)",
                fontSize: 12, fontWeight: 800, color: "var(--risk-high)",
                display: "flex", alignItems: "center", gap: 6,
              }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--risk-high)" }} />
                {highCount} High
              </div>
            )}
          </div>
        )}
      </div>

      {/* Thresholds reference */}
      <div className="panel" style={{ padding: "18px 22px" }}>
        <div style={{
          fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase",
          letterSpacing: "0.10em", marginBottom: 14, paddingBottom: 10,
          borderBottom: "1px solid rgba(255,255,255,0.05)",
        }}>Alert Thresholds</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 14 }}>
          {Object.entries(THRESHOLDS).map(([key, val]) => {
            const c = riskColor(val);
            return (
              <div key={key} style={{
                padding: "10px 14px", borderRadius: 8,
                background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.05)",
                textAlign: "center",
              }}>
                <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "capitalize", marginBottom: 6, letterSpacing: "0.05em" }}>
                  {key.replace(/_/g, " ")}
                </div>
                <div style={{ fontSize: 18, fontWeight: 900, fontFamily: "var(--font-mono)", color: c, textShadow: `0 0 12px ${c}50` }}>
                  {(val * 100).toFixed(0)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Empty state */}
      {alerts.length === 0 && (
        <div className="panel" style={{
          padding: "64px 48px", textAlign: "center",
          borderColor: "rgba(0,255,136,0.15)", boxShadow: "0 0 40px rgba(0,255,136,0.04)",
        }}>
          <div style={{ display: "flex", justifyContent: "center", color: "var(--risk-minimal)", marginBottom: 16, opacity: 0.5 }}>
            <IconShieldCheck />
          </div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: 6 }}>
            All Clear
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 4 }}>No active alerts</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            All profile metrics are within configured thresholds
          </div>
        </div>
      )}

      {/* Alert list */}
      {alerts.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {alerts.map((alert, idx) => {
            const color   = riskColor(alert.value);
            const isCrit  = alert.severity === "critical";
            return (
              <div
                key={alert.id}
                className="panel"
                style={{
                  padding: "16px 20px",
                  display: "flex", alignItems: "center", gap: 14,
                  borderColor: isCrit ? "rgba(239,68,68,0.3)" : "rgba(249,115,22,0.25)",
                  boxShadow: isCrit ? "0 0 20px rgba(239,68,68,0.06)" : "0 0 20px rgba(249,115,22,0.04)",
                  animation: `fadeUp 0.3s var(--ease-out) ${idx * 0.05}s both`,
                }}
              >
                {/* Severity dot */}
                <div style={{
                  width: 9, height: 9, borderRadius: "50%", flexShrink: 0,
                  background: isCrit ? "var(--risk-critical)" : "var(--risk-high)",
                  boxShadow: `0 0 8px ${isCrit ? "var(--risk-critical)" : "var(--risk-high)"}`,
                  animation: "pulse-green 2s ease-in-out infinite",
                }} />

                {/* Avatar */}
                <div style={{
                  width: 34, height: 34, borderRadius: "50%", flexShrink: 0,
                  background: `${color}18`, border: `2px solid ${color}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 13, fontWeight: 800, color,
                  boxShadow: `0 0 10px ${color}30`,
                }}>
                  {alert.user_name[0]}
                </div>

                {/* Info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>{alert.user_name}</span>
                    <span style={{
                      fontSize: 9, padding: "2px 8px", borderRadius: 10, fontWeight: 800,
                      background: isCrit ? "rgba(239,68,68,0.12)" : "rgba(249,115,22,0.12)",
                      color: isCrit ? "var(--risk-critical)" : "var(--risk-high)",
                      textTransform: "uppercase", letterSpacing: "0.06em",
                      border: `1px solid ${isCrit ? "rgba(239,68,68,0.25)" : "rgba(249,115,22,0.25)"}`,
                    }}>
                      {alert.severity}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 3 }}>{alert.message}</div>
                </div>

                {/* Score + time */}
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{
                    fontFamily: "var(--font-mono)", fontSize: 20, fontWeight: 900, color,
                    textShadow: `0 0 16px ${color}60`,
                  }}>
                    {formatScore(alert.value)}
                  </div>
                  <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 2 }}>
                    {new Date(alert.created_at).toLocaleTimeString()}
                  </div>
                </div>

                {/* Dismiss button */}
                <button
                  onClick={() => dismissAlert(alert.id)}
                  style={{
                    display: "flex", alignItems: "center", gap: 6,
                    fontSize: 11, padding: "6px 12px", borderRadius: 7, flexShrink: 0,
                    background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                    color: "var(--text-muted)", cursor: "pointer", transition: "all 0.2s",
                    fontWeight: 600,
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLElement).style.background = "rgba(239,68,68,0.10)";
                    (e.currentTarget as HTMLElement).style.borderColor = "rgba(239,68,68,0.25)";
                    (e.currentTarget as HTMLElement).style.color = "var(--risk-critical)";
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)";
                    (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.08)";
                    (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
                  }}
                >
                  <IconX /> Dismiss
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
