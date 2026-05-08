"use client";

import { useEffect, useState } from "react";
import { wsClient, type WSEvent } from "@/lib/api";

export default function Topbar({ title }: { title?: string }) {
  const [wsConnected, setWsConnected] = useState(false);
  const [lastEvent, setLastEvent]     = useState<WSEvent | null>(null);
  const [blink, setBlink]             = useState(false);
  const [user, setUser]               = useState<{ email?: string } | null>(null);

  useEffect(() => {
    wsClient.connect();
    const unsub = wsClient.subscribe((evt) => {
      if (evt.type === "connected")       setWsConnected(true);
      if (evt.type === "detection_event") {
        setLastEvent(evt);
        setBlink(true);
        setTimeout(() => setBlink(false), 600);
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

  return (
    <header className="app-topbar">
      <div style={{ flex: 1 }}>
        {title && (
          <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>
            {title}
          </span>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
        {/* Live feed indicator */}
        <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12, color: "var(--text-secondary)" }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%",
            background: wsConnected ? "var(--risk-minimal)" : "var(--text-disabled)",
            boxShadow: wsConnected && blink ? "0 0 10px var(--risk-minimal)" : "none",
            transition: "box-shadow 0.3s",
          }} />
          {wsConnected ? "Live" : "Connecting…"}
          {lastEvent?.risk_level && (
            <span style={{
              padding: "1px 7px", borderRadius: 10, fontSize: 10,
              background: `rgba(${lastEvent.risk_level === "critical" ? "239,68,68" : "61,127,255"},0.12)`,
              color: lastEvent.risk_level === "critical" ? "var(--risk-critical)" : "var(--accent)",
              fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase",
            }}>
              {lastEvent.risk_level}
            </span>
          )}
        </div>

        {/* User pill */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "5px 12px", borderRadius: 20,
          background: "var(--bg-raised)", border: "1px solid var(--border-subtle)",
          fontSize: 12, color: "var(--text-secondary)",
        }}>
          <div style={{
            width: 22, height: 22, borderRadius: "50%",
            background: "var(--accent)", display: "flex",
            alignItems: "center", justifyContent: "center",
            fontSize: 11, fontWeight: 700, color: "#fff",
          }}>
            {user?.email?.[0]?.toUpperCase() || "A"}
          </div>
          {user?.email || "demo@apeilo.com"}
        </div>
      </div>
    </header>
  );
}
