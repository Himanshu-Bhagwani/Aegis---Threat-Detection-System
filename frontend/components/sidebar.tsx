"use client";

import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { clearToken } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor } from "@/lib/api";

const NAV_DETECTION = [
  { href: "/dashboard",        icon: "⬡", label: "Overview"              },
  { href: "/dashboard/risk",   icon: "◈", label: "Risk Analysis"         },
  { href: "/dashboard/gps",    icon: "◉", label: "GPS Tracking"          },
  { href: "/dashboard/login",  icon: "◐", label: "Login Events"          },
  { href: "/dashboard/fraud",  icon: "◆", label: "Fraud Possibility"     },
  { href: "/dashboard/breach", icon: "◑", label: "Breach Check"          },
];
const NAV_MANAGEMENT = [
  { href: "/dashboard/identity", icon: "◎", label: "Identity"  },
  { href: "/dashboard/alerts",   icon: "◈", label: "Alerts"    },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router   = useRouter();
  const { profiles, selectedId, setSelectedId, alerts, addNewProfile } = useProfiles();
  const [showAdd, setShowAdd]   = useState(false);
  const [newName, setNewName]   = useState("");
  const [newEmail, setNewEmail] = useState("");

  function handleSignOut() {
    const token = typeof window !== "undefined" ? localStorage.getItem("aegis_token") : null;
    clearToken();
    if (token) {
      fetch("/auth/signout", { method: "POST", body: JSON.stringify({ access_token: token }) }).catch(() => {});
    }
    router.push("/login");
  }

  function handleAdd() {
    if (!newName.trim()) return;
    addNewProfile(newName.trim(), newEmail.trim() || `${newName.toLowerCase().replace(/\s+/g,".")  }@apeilo.local`);
    setNewName(""); setNewEmail(""); setShowAdd(false);
  }

  const selectedProfile = profiles.find(p => p.id === selectedId);

  function NavLink({ href, icon, label }: { href: string; icon: string; label: string }) {
    const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
    return (
      <a href={href} style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "8px 10px", borderRadius: 7, fontSize: 13,
        fontWeight: active ? 600 : 400,
        color: active ? "var(--text-primary)" : "var(--text-secondary)",
        background: active ? "var(--bg-raised)" : "transparent",
        marginBottom: 2, transition: "all 0.12s", textDecoration: "none",
      }}
        onMouseEnter={e => { if (!active) (e.currentTarget as HTMLElement).style.background = "var(--bg-hover)"; }}
        onMouseLeave={e => { if (!active) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
      >
        <span style={{ fontSize: 14, opacity: active ? 1 : 0.6 }}>{icon}</span>
        {label}
        {active && <span style={{ marginLeft: "auto", width: 4, height: 4, borderRadius: "50%", background: "var(--accent)", display: "inline-block" }} />}
      </a>
    );
  }

  return (
    <aside className="app-sidebar" style={{ padding: 0 }}>
      {/* Logo */}
      <div style={{ padding: "20px 18px 16px", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 32, height: 32, borderRadius: 8, background: "var(--accent)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, fontWeight: 700, color: "#fff", flexShrink: 0 }}>Æ</div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>Apeilo</div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.08em", textTransform: "uppercase" }}>Threat Detection</div>
        </div>
        {alerts.length > 0 && (
          <div style={{ marginLeft: "auto", width: 18, height: 18, borderRadius: "50%", background: "var(--risk-critical)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: "#fff" }}>
            {alerts.length}
          </div>
        )}
      </div>

      {/* Profile selector */}
      <div style={{ padding: "12px 10px 8px", borderBottom: "1px solid var(--border-subtle)" }}>
        <div style={{ fontSize: 10, color: "var(--text-disabled)", letterSpacing: "0.10em", textTransform: "uppercase", padding: "0 2px 6px", fontWeight: 600 }}>Identity Profile</div>
        <select
          value={selectedId}
          onChange={e => setSelectedId(e.target.value)}
          style={{
            width: "100%", padding: "7px 10px", borderRadius: 7, fontSize: 12,
            background: "var(--bg-raised)", border: "1px solid var(--border-subtle)",
            color: "var(--text-primary)", cursor: "pointer", outline: "none",
          }}
        >
          {profiles.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        {selectedProfile && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6, padding: "0 2px" }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: riskColor(selectedProfile.metrics.unified_score), flexShrink: 0 }} />
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              Unified: <span style={{ color: riskColor(selectedProfile.metrics.unified_score), fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                {(selectedProfile.metrics.unified_score * 100).toFixed(0)}%
              </span>
            </span>
            <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-muted)", textTransform: "capitalize" }}>
              {selectedProfile.metrics.risk_level}
            </span>
          </div>
        )}
        <button
          onClick={() => setShowAdd(v => !v)}
          style={{ marginTop: 6, width: "100%", fontSize: 11, padding: "5px", borderRadius: 6, background: "transparent", border: "1px dashed var(--border-subtle)", color: "var(--text-muted)", cursor: "pointer" }}
        >
          + Add profile
        </button>
        {showAdd && (
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
            <input
              placeholder="Name"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              style={{ padding: "6px 8px", borderRadius: 6, fontSize: 12, background: "var(--bg-raised)", border: "1px solid var(--border-subtle)", color: "var(--text-primary)", outline: "none" }}
            />
            <input
              placeholder="Email (optional)"
              value={newEmail}
              onChange={e => setNewEmail(e.target.value)}
              style={{ padding: "6px 8px", borderRadius: 6, fontSize: 12, background: "var(--bg-raised)", border: "1px solid var(--border-subtle)", color: "var(--text-primary)", outline: "none" }}
            />
            <div style={{ display: "flex", gap: 6 }}>
              <button onClick={handleAdd} className="btn-primary" style={{ flex: 1, fontSize: 11, padding: "5px" }}>Create</button>
              <button onClick={() => setShowAdd(false)} style={{ flex: 1, fontSize: 11, padding: "5px", borderRadius: 6, background: "transparent", border: "1px solid var(--border-subtle)", color: "var(--text-muted)", cursor: "pointer" }}>Cancel</button>
            </div>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav style={{ padding: "12px 8px", flex: 1, overflow: "auto" }}>
        <div style={{ fontSize: 10, color: "var(--text-disabled)", letterSpacing: "0.10em", textTransform: "uppercase", padding: "0 10px 8px", fontWeight: 600 }}>Detection</div>
        {NAV_DETECTION.map(n => <NavLink key={n.href} {...n} />)}
        <div style={{ fontSize: 10, color: "var(--text-disabled)", letterSpacing: "0.10em", textTransform: "uppercase", padding: "14px 10px 8px", fontWeight: 600 }}>Management</div>
        {NAV_MANAGEMENT.map(n => <NavLink key={n.href} {...n} />)}
      </nav>

      {/* Footer */}
      <div style={{ padding: "12px 8px", borderTop: "1px solid var(--border-subtle)" }}>
        <button onClick={handleSignOut} style={{ width: "100%", display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", borderRadius: 7, fontSize: 13, color: "var(--text-muted)", background: "transparent", border: "none", cursor: "pointer", transition: "all 0.12s" }}
          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = "var(--bg-hover)"; (e.currentTarget as HTMLElement).style.color = "var(--risk-critical)"; }}
          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "transparent"; (e.currentTarget as HTMLElement).style.color = "var(--text-muted)"; }}
        >
          <span style={{ fontSize: 14 }}>↩</span> Sign out
        </button>
      </div>
    </aside>
  );
}
