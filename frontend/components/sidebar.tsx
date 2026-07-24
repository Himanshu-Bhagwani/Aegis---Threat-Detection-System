"use client";

import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { clearToken } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";
import { riskColor } from "@/lib/api";

/* ── SVG Icons ─────────────────────────────────────────── */
const IconGrid = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
  </svg>
);
const IconChart = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);
const IconMapPin = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/>
  </svg>
);
const IconLock = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>
  </svg>
);
const IconCreditCard = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>
  </svg>
);
const IconShield = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const IconUsers = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
  </svg>
);
const IconBell = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 01-3.46 0"/>
  </svg>
);
const IconSearch = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
);
const IconLogOut = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>
  </svg>
);
const IconPlus = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);
const IconChevronDown = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9"/>
  </svg>
);

const NAV_DETECTION = [
  { href: "/dashboard",        icon: <IconGrid />,       label: "Overview" },
  { href: "/dashboard/risk",   icon: <IconChart />,      label: "Risk Analysis" },
  { href: "/dashboard/gps",    icon: <IconMapPin />,     label: "GPS Tracking" },
  { href: "/dashboard/login",  icon: <IconLock />,       label: "Login Events" },
  { href: "/dashboard/fraud",  icon: <IconCreditCard />, label: "Fraud Detection" },
  { href: "/dashboard/breach", icon: <IconShield />,     label: "Breach Check" },
];
const NAV_MANAGEMENT = [
  { href: "/dashboard/identity", icon: <IconUsers />,  label: "Identity" },
  { href: "/dashboard/alerts",   icon: <IconBell />,   label: "Alerts" },
];

const NAV_AI = [
  { href: "/dashboard/query", icon: <IconSearch />, label: "AI Query" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router   = useRouter();
  const { profiles, selectedId, setSelectedId, alerts, addNewProfile, deleteProfile } = useProfiles();
  const [showAdd, setShowAdd]   = useState(false);
  const [newName, setNewName]   = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);

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
    addNewProfile(newName.trim(), newEmail.trim() || `${newName.toLowerCase().replace(/\s+/g, ".")}@apeilo.local`);
    setNewName(""); setNewEmail(""); setShowAdd(false);
  }

  const selectedProfile = profiles.find(p => p.id === selectedId);

  function NavLink({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) {
    const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
    return (
      <a href={href} className={`nav-item ${active ? "active" : ""}`}>
        <span style={{ flexShrink: 0, opacity: active ? 1 : 0.55 }}>{icon}</span>
        <span style={{ flex: 1 }}>{label}</span>
        {label === "Alerts" && alerts.length > 0 && (
          <span style={{
            minWidth: 18, height: 18, borderRadius: 9,
            background: "var(--risk-critical)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 10, fontWeight: 800, color: "#fff",
            boxShadow: "0 0 8px rgba(239,68,68,0.5)",
            animation: "pulse-blue 2s infinite",
            flexShrink: 0,
          }}>
            {alerts.length > 9 ? "9+" : alerts.length}
          </span>
        )}
      </a>
    );
  }

  return (
    <aside className="app-sidebar" style={{ padding: 0 }}>

      {/* ── Logo ──────────────────────────────────── */}
      <div style={{
        padding: "20px 18px 16px",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <div style={{
          width: 34, height: 34, borderRadius: 9, flexShrink: 0,
          background: "linear-gradient(135deg, #3d7fff, #2563eb)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16, fontWeight: 800, color: "#fff",
          boxShadow: "0 0 16px rgba(61,127,255,0.45)",
          animation: "pulse-blue 3s ease-in-out infinite",
        }}>Æ</div>
        <div>
          <div style={{ fontWeight: 800, fontSize: 14, letterSpacing: "-0.02em", color: "var(--text-primary)" }}>Apeilo</div>
          <div style={{ fontSize: 9, color: "var(--accent)", letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 700 }}>Threat Detection</div>
        </div>
        {alerts.length > 0 && (
          <div style={{
            marginLeft: "auto", minWidth: 20, height: 20, borderRadius: 10,
            background: "var(--risk-critical)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 10, fontWeight: 800, color: "#fff",
            boxShadow: "0 0 10px rgba(239,68,68,0.6)",
          }}>
            {alerts.length}
          </div>
        )}
      </div>

      {/* ── Profile selector ──────────────────────── */}
      <div style={{ padding: "12px 12px 10px", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <div style={{
          fontSize: 9, color: "var(--text-disabled)", letterSpacing: "0.12em",
          textTransform: "uppercase", padding: "0 4px 8px", fontWeight: 700,
        }}>Identity Profile</div>

        <div style={{ position: "relative" }}>
          <select
            value={selectedId}
            onChange={e => setSelectedId(e.target.value)}
            style={{
              width: "100%", padding: "8px 32px 8px 12px",
              borderRadius: 9, fontSize: 12, fontWeight: 500,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "var(--text-primary)", cursor: "pointer", outline: "none",
              appearance: "none", WebkitAppearance: "none",
            }}
          >
            {profiles.map(p => (
              <option key={p.id} value={p.id} style={{ background: "#0a0f1c" }}>{p.name}</option>
            ))}
          </select>
          <div style={{
            position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)",
            color: "var(--text-muted)", pointerEvents: "none",
          }}>
            <IconChevronDown />
          </div>
        </div>

        {selectedProfile && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 8, padding: "0 4px" }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: riskColor(selectedProfile.metrics.unified_score),
              boxShadow: `0 0 8px ${riskColor(selectedProfile.metrics.unified_score)}`,
              flexShrink: 0,
            }} />
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
              Unified:{" "}
              <span style={{ color: riskColor(selectedProfile.metrics.unified_score), fontWeight: 700, fontFamily: "var(--font-mono)" }}>
                {(selectedProfile.metrics.unified_score * 100).toFixed(0)}%
              </span>
            </span>
            <span style={{ marginLeft: "auto", fontSize: 10, color: riskColor(selectedProfile.metrics.unified_score), textTransform: "capitalize", fontWeight: 600 }}>
              {selectedProfile.metrics.risk_level}
            </span>
          </div>
        )}

        {/* Delete the selected profile (two-step to avoid accidents) */}
        {selectedProfile && (
          confirmDelete ? (
            <div style={{
              marginTop: 8, padding: "8px 10px", borderRadius: 8,
              background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.25)",
            }}>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", marginBottom: 7, lineHeight: 1.5 }}>
                Delete <b style={{ color: "var(--text-primary)" }}>{selectedProfile.name}</b>?
                {!selectedProfile.is_demo && " Its live profile will be removed from the backend."}
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  onClick={async () => { await deleteProfile(selectedProfile.id); setConfirmDelete(false); }}
                  style={{
                    flex: 1, fontSize: 10, fontWeight: 700, padding: "6px", borderRadius: 6,
                    background: "rgba(239,68,68,0.85)", border: "none", color: "#fff", cursor: "pointer",
                  }}
                >
                  Delete
                </button>
                <button
                  onClick={() => setConfirmDelete(false)}
                  style={{
                    flex: 1, fontSize: 10, fontWeight: 700, padding: "6px", borderRadius: 6,
                    background: "transparent", border: "1px solid rgba(255,255,255,0.12)",
                    color: "var(--text-muted)", cursor: "pointer",
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setConfirmDelete(true)}
              title={`Delete ${selectedProfile.name}`}
              style={{
                marginTop: 8, width: "100%", fontSize: 11, fontWeight: 600, padding: "7px",
                borderRadius: 8, background: "transparent",
                border: "1px solid rgba(239,68,68,0.18)",
                color: "rgba(239,68,68,0.85)", cursor: "pointer",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
              }}
            >
              ✕ Delete profile
            </button>
          )
        )}

        {/* Add profile button */}
        <button
          onClick={() => setShowAdd(v => !v)}
          style={{
            marginTop: 8, width: "100%", fontSize: 11, fontWeight: 600, padding: "7px",
            borderRadius: 8, background: "transparent",
            border: "1px dashed rgba(255,255,255,0.1)",
            color: "var(--text-muted)", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            transition: "all 0.2s",
          }}
          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "rgba(61,127,255,0.3)"; (e.currentTarget as HTMLElement).style.color = "var(--accent)"; }}
          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.1)"; (e.currentTarget as HTMLElement).style.color = "var(--text-muted)"; }}
        >
          <IconPlus /> Add profile
        </button>

        {showAdd && (
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6, animation: "fadeDown 0.2s var(--ease-out) both" }}>
            <input
              placeholder="Full name"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleAdd()}
              style={{
                padding: "7px 10px", borderRadius: 7, fontSize: 12,
                background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                color: "var(--text-primary)", outline: "none",
              }}
            />
            <input
              placeholder="Email (optional)"
              value={newEmail}
              onChange={e => setNewEmail(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleAdd()}
              style={{
                padding: "7px 10px", borderRadius: 7, fontSize: 12,
                background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                color: "var(--text-primary)", outline: "none",
              }}
            />
            <div style={{ display: "flex", gap: 6 }}>
              <button
                onClick={handleAdd}
                style={{
                  flex: 1, fontSize: 11, fontWeight: 700, padding: "7px",
                  borderRadius: 7, border: "none",
                  background: "linear-gradient(135deg, #3d7fff, #2563eb)",
                  color: "#fff", cursor: "pointer",
                  boxShadow: "0 0 12px rgba(61,127,255,0.3)",
                }}
              >Create</button>
              <button
                onClick={() => setShowAdd(false)}
                style={{
                  flex: 1, fontSize: 11, fontWeight: 600, padding: "7px",
                  borderRadius: 7, border: "1px solid rgba(255,255,255,0.08)",
                  background: "transparent", color: "var(--text-muted)", cursor: "pointer",
                }}
              >Cancel</button>
            </div>
          </div>
        )}
      </div>

      {/* ── Navigation ────────────────────────────── */}
      <nav style={{ padding: "12px 8px", flex: 1, overflow: "auto" }}>
        <div className="section-tag">Detection</div>
        {NAV_DETECTION.map(n => <NavLink key={n.href} {...n} />)}

        <div className="section-tag" style={{ marginTop: 12 }}>Management</div>
        {NAV_MANAGEMENT.map(n => <NavLink key={n.href} {...n} />)}

        <div className="section-tag" style={{ marginTop: 12 }}>AI</div>
        {NAV_AI.map(n => (
          <a key={n.href} href={n.href} className={`nav-item ${pathname === n.href ? "active" : ""}`}>
            <span style={{ flexShrink: 0, opacity: pathname === n.href ? 1 : 0.55, color: "var(--accent-purple)" }}>{n.icon}</span>
            <span style={{ flex: 1 }}>{n.label}</span>
            <span style={{
              fontSize: 9, padding: "2px 6px", borderRadius: 5,
              background: "rgba(139,92,246,0.15)", color: "var(--accent-purple)",
              border: "1px solid rgba(139,92,246,0.3)", fontWeight: 800,
              letterSpacing: "0.06em",
            }}>LLM</span>
          </a>
        ))}
      </nav>

      {/* ── Footer ────────────────────────────────── */}
      <div style={{ padding: "10px 8px 14px", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
        <button
          onClick={handleSignOut}
          style={{
            width: "100%", display: "flex", alignItems: "center", gap: 10,
            padding: "9px 12px", borderRadius: 9, fontSize: 13, fontWeight: 500,
            color: "var(--text-muted)", background: "transparent",
            border: "none", cursor: "pointer", transition: "all 0.18s",
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLElement).style.background = "rgba(239,68,68,0.08)";
            (e.currentTarget as HTMLElement).style.color = "var(--risk-critical)";
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLElement).style.background = "transparent";
            (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
          }}
        >
          <IconLogOut />
          <span>Sign out</span>
        </button>
      </div>
    </aside>
  );
}
