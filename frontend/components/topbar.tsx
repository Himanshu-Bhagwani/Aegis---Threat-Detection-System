"use client";

import { useEffect, useState } from "react";
import { wsClient, type WSEvent } from "@/lib/api";

const IconWifi = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 12.55a11 11 0 0114.08 0"/><path d="M1.42 9a16 16 0 0121.16 0"/><path d="M8.53 16.11a6 6 0 016.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>
  </svg>
);
const IconWifiOff = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0119 12.55"/><path d="M5 12.55a11 11 0 015.17-2.39"/><path d="M10.71 5.05A16 16 0 0122.56 9"/><path d="M1.42 9a15.91 15.91 0 014.7-2.88"/><path d="M8.53 16.11a6 6 0 016.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>
  </svg>
);
const IconUser = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/>
  </svg>
);
const IconAlertTriangle = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
);

export default function Topbar({ title }: { title?: string }) {
  const [wsConnected, setWsConnected] = useState(false);
  const [lastEvent,   setLastEvent]   = useState<WSEvent | null>(null);
  const [blink,       setBlink]       = useState(false);
  const [user,        setUser]        = useState<{ email?: string } | null>(null);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.subscribe((evt) => {
      if (evt.type === "connected") setWsConnected(true);
      if (evt.type === "detection_event") {
        setLastEvent(evt);
        setBlink(true);
        setTimeout(() => setBlink(false), 800);
      }
    });
    return () => { unsub(); };
  }, []);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("aegis_user");
      if (stored) setUser(JSON.parse(stored));
    } catch {}
  }, []);

  const riskBadgeColor = lastEvent?.risk_level === "critical" ? "var(--risk-critical)"
    : lastEvent?.risk_level === "high" ? "var(--risk-high)"
    : "var(--accent)";

  const riskBgColor = lastEvent?.risk_level === "critical" ? "rgba(239,68,68,0.12)"
    : lastEvent?.risk_level === "high" ? "rgba(249,115,22,0.12)"
    : "rgba(61,127,255,0.12)";

  return (
    <header className="app-topbar">
      {/* Title */}
      <div style={{ flex: 1 }}>
        {title && (
          <span style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
            {title}
          </span>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>

        {/* WebSocket status */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 14px", borderRadius: 20,
          background: "rgba(255,255,255,0.03)",
          border: `1px solid ${wsConnected ? "rgba(0,255,136,0.2)" : "rgba(255,255,255,0.06)"}`,
          fontSize: 12, color: wsConnected ? "var(--risk-minimal)" : "var(--text-muted)",
          transition: "all 0.3s",
        }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%",
            background: wsConnected ? "var(--risk-minimal)" : "var(--text-disabled)",
            boxShadow: wsConnected
              ? blink
                ? "0 0 14px var(--risk-minimal), 0 0 28px rgba(0,255,136,0.4)"
                : "0 0 8px var(--risk-minimal)"
              : "none",
            transition: "box-shadow 0.3s",
          }} />
          <span style={{ fontWeight: 600 }}>
            {wsConnected ? (blink ? "Event received" : "Live") : "Connecting…"}
          </span>
          {wsConnected ? <IconWifi /> : <IconWifiOff />}
        </div>

        {/* Last event badge */}
        {lastEvent?.risk_level && (
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "5px 12px", borderRadius: 20, fontSize: 11,
            background: riskBgColor,
            border: `1px solid ${riskBadgeColor}40`,
            color: riskBadgeColor, fontWeight: 700,
            letterSpacing: "0.05em", textTransform: "uppercase",
            animation: "fadeIn 0.2s ease both",
          }}>
            <IconAlertTriangle />
            {lastEvent.risk_level}
          </div>
        )}

        {/* Divider */}
        <div style={{ width: 1, height: 24, background: "rgba(255,255,255,0.06)" }} />

        {/* User pill */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 14px 6px 6px", borderRadius: 24,
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.07)",
          fontSize: 12, color: "var(--text-secondary)",
          cursor: "default",
        }}>
          <div style={{
            width: 26, height: 26, borderRadius: "50%",
            background: "linear-gradient(135deg, #3d7fff, #2563eb)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 11, fontWeight: 800, color: "#fff",
            boxShadow: "0 0 10px rgba(61,127,255,0.4)",
            flexShrink: 0,
          }}>
            {user?.email?.[0]?.toUpperCase() || "A"}
          </div>
          <span style={{ fontWeight: 500, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {user?.email || "demo@apeilo.com"}
          </span>
        </div>
      </div>
    </header>
  );
}
