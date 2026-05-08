"use client";

import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor, formatScore } from "@/lib/api";
import { THRESHOLDS } from "@/lib/profiles";

export default function AlertsPage() {
  const { alerts, dismissAlert, profiles } = useProfiles();

  const criticalCount = alerts.filter(a => a.severity === "critical").length;
  const highCount     = alerts.filter(a => a.severity === "high").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>Security Alerts</h1>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
            Metrics that exceeded configured thresholds across all identity profiles
          </p>
        </div>
        {alerts.length > 0 && (
          <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
            {criticalCount > 0 && (
              <div style={{ padding: "6px 14px", borderRadius: 20, background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)", fontSize: 12, fontWeight: 700, color: "var(--risk-critical)" }}>
                {criticalCount} Critical
              </div>
            )}
            {highCount > 0 && (
              <div style={{ padding: "6px 14px", borderRadius: 20, background: "rgba(255,102,0,0.12)", border: "1px solid rgba(255,102,0,0.3)", fontSize: 12, fontWeight: 700, color: "var(--risk-high)" }}>
                {highCount} High
              </div>
            )}
          </div>
        )}
      </div>

      {/* Thresholds reference */}
      <div className="panel" style={{ padding: "14px 18px" }}>
        <div className="panel-header" style={{ marginBottom: 10 }}>Alert Thresholds</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10 }}>
          {Object.entries(THRESHOLDS).map(([key, val]) => (
            <div key={key} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "capitalize", marginBottom: 4 }}>
                {key.replace(/_/g, " ")}
              </div>
              <div style={{ fontSize: 15, fontWeight: 700, fontFamily: "var(--font-mono)", color: riskColor(val) }}>
                {(val * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {alerts.length === 0 && (
        <div className="panel" style={{ padding: 48, textAlign: "center" }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>◉</div>
          <div style={{ fontSize: 14, color: "var(--text-secondary)" }}>No active alerts</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
            All profile metrics are within configured thresholds
          </div>
        </div>
      )}

      {alerts.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {alerts.map(alert => {
            const color = riskColor(alert.value);
            const profile = profiles.find(p => p.id === alert.profile_id);
            return (
              <div
                key={alert.id}
                className="panel"
                style={{
                  padding: "14px 18px", display: "flex", alignItems: "center", gap: 14,
                  borderColor: alert.severity === "critical" ? "rgba(239,68,68,0.35)" : "rgba(255,102,0,0.35)",
                }}
              >
                <div style={{
                  width: 10, height: 10, borderRadius: "50%", flexShrink: 0,
                  background: alert.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)",
                  boxShadow: `0 0 8px ${alert.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)"}80`,
                }} />

                {/* User avatar */}
                <div style={{
                  width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
                  background: color + "22", border: `1px solid ${color}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 13, fontWeight: 700, color,
                }}>
                  {alert.user_name[0]}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>{alert.user_name}</span>
                    <span style={{
                      fontSize: 10, padding: "1px 8px", borderRadius: 10, fontWeight: 700,
                      background: alert.severity === "critical" ? "rgba(239,68,68,0.12)" : "rgba(255,102,0,0.12)",
                      color: alert.severity === "critical" ? "var(--risk-critical)" : "var(--risk-high)",
                      textTransform: "uppercase", letterSpacing: "0.05em",
                    }}>
                      {alert.severity}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{alert.message}</div>
                </div>

                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 18, fontWeight: 700, color }}>{formatScore(alert.value)}</div>
                  <div style={{ fontSize: 10, color: "var(--text-disabled)", marginTop: 2 }}>
                    {new Date(alert.created_at).toLocaleTimeString()}
                  </div>
                </div>

                <button
                  onClick={() => dismissAlert(alert.id)}
                  style={{
                    fontSize: 11, padding: "5px 12px", borderRadius: 6, flexShrink: 0,
                    background: "transparent", border: "1px solid var(--border-subtle)",
                    color: "var(--text-muted)", cursor: "pointer",
                  }}
                >
                  Dismiss
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
