"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { checkPasswordBreach, checkEmailBreach, riskColor, formatScore, getWeightedUnifiedScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

const DynMiniBar = dynamic(
  () => import("@/components/PageChart").then(m => ({ default: m.MiniModuleBar })),
  { ssr: false }
);
const DynPwStrength = dynamic(
  () => import("@/components/PageChart").then(m => ({ default: m.PasswordStrengthChart })),
  { ssr: false }
);

const IconShield = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const IconKey = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 11-7.778 7.778 5.5 5.5 0 017.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>
  </svg>
);
const IconMail = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/>
  </svg>
);
const IconLoader = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ animation: "spin 1s linear infinite" }}>
    <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>
  </svg>
);
const IconSearch = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
);
const IconChevronRight = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);

export default function BreachPage() {
  const { selected, profiles, updateMetrics } = useProfiles();
  const [password, setPassword] = useState("");
  const [email,    setEmail]    = useState("");
  const [pwResult, setPwResult] = useState<any>(null);
  const [emResult, setEmResult] = useState<any>(null);
  const [loading,  setLoading]  = useState<"pw"|"em"|null>(null);
  const [updated,  setUpdated]  = useState<string | null>(null);
  const [autoEmail, setAutoEmail] = useState<any>(null);
  const [autoLoading, setAutoLoading] = useState(false);

  // Automatically assess the selected profile: run the email check for their
  // address and surface the stored detail from their last password check.
  const pwMeta = selected?.password_meta ?? null;

  useEffect(() => {
    let cancelled = false;
    async function auto() {
      if (!selected?.email) { setAutoEmail(null); return; }
      setAutoLoading(true);
      try {
        const res = await checkEmailBreach(selected.email, selected.id);
        if (!cancelled) setAutoEmail(res);
      } catch { if (!cancelled) setAutoEmail(null); }
      finally { if (!cancelled) setAutoLoading(false); }
    }
    auto();
    return () => { cancelled = true; };
  }, [selected?.id, selected?.email]); // eslint-disable-line react-hooks/exhaustive-deps

  async function checkPw() {
    if (!password.trim()) return;
    setLoading("pw"); setUpdated(null);
    try {
      // Pass the selected profile so the check attributes to that user
      // instead of spawning an "anonymous" profile.
      const res = await checkPasswordBreach(password, selected?.id);
      setPwResult(res);
      if (res?.breach_probability != null && selected) {
        const newBreach = Math.min(1, Math.max(0, selected.metrics.breach_risk * 0.7 + res.breach_probability * 0.3));
        const m = selected.metrics;
        const unified = await getWeightedUnifiedScore(m.gps_spoof, m.login_anomaly, m.password_leak, m.fraud_risk, newBreach, selected.id);
        updateMetrics(selected.id, { breach_risk: newBreach, unified_score: unified });
        setUpdated(`Updated ${selected.name}'s breach risk to ${(newBreach * 100).toFixed(0)}%`);
      }
    } catch (e: any) { setPwResult({ error: e?.message }); }
    finally { setLoading(null); }
  }

  async function checkEm() {
    if (!email.trim()) return;
    setLoading("em"); setUpdated(null);
    try {
      const res = await checkEmailBreach(email, selected?.id);
      setEmResult(res);
      if (selected) {
        // Use the backend's combined score (confirmed breaches + structural
        // exposure) rather than inferring purely from a breach count that is
        // always 0 when no HIBP key is configured.
        const inferredRisk = res?.risk_score ?? Math.min(1, (res?.breach_count ?? 0) * 0.15);
        const newBreach = Math.min(1, Math.max(0, selected.metrics.breach_risk * 0.7 + inferredRisk * 0.3));
        const m = selected.metrics;
        const unified = await getWeightedUnifiedScore(m.gps_spoof, m.login_anomaly, m.password_leak, m.fraud_risk, newBreach, selected.id);
        updateMetrics(selected.id, { breach_risk: newBreach, unified_score: unified });
        setUpdated(`Updated ${selected.name}'s breach risk to ${(newBreach * 100).toFixed(0)}%`);
      }
    } catch (e: any) { setEmResult({ error: e?.message }); }
    finally { setLoading(null); }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
            <div style={{ color: "var(--risk-critical)" }}><IconShield /></div>
            <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
              Breach Intelligence
            </h1>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            k-anonymity HIBP checks — only 5 chars of your SHA-1 hash ever leave the device
          </p>
        </div>
        {selected && (
          <div style={{
            display: "flex", alignItems: "center", gap: 8, padding: "8px 16px",
            borderRadius: 20, background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)", fontSize: 12, flexShrink: 0,
          }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: riskColor(selected.metrics.breach_risk), boxShadow: `0 0 8px ${riskColor(selected.metrics.breach_risk)}` }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{selected.name}</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.breach_risk), fontWeight: 800 }}>
              {(selected.metrics.breach_risk * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {/* Update success */}
      {updated && (
        <div style={{
          padding: "12px 18px", borderRadius: 10,
          background: "rgba(61,127,255,0.08)", border: "1px solid rgba(61,127,255,0.25)",
          fontSize: 12, color: "var(--accent)", display: "flex", alignItems: "center", gap: 8,
          animation: "fadeDown 0.3s var(--ease-out) both",
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
          {updated}
        </div>
      )}

      {/* ── Breach risk across all profiles ────── */}
      {profiles.length > 0 && (
        <div className="panel" style={{ padding: "18px 22px" }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 14 }}>
            Breach Exposure — All Profiles
          </div>
          <DynMiniBar data={profiles.map(p => ({
            module: p.name.split(" ")[0],
            score:  Math.round(p.metrics.breach_risk * 100),
            color:  p.metrics.breach_risk >= 0.75 ? "#ef4444" : p.metrics.breach_risk >= 0.50 ? "#f97316" : p.metrics.breach_risk >= 0.25 ? "#f59e0b" : "#00ff88",
          }))} />
        </div>
      )}

      {/* ── Automatic assessment for the selected profile ───────── */}
      {selected && (
        <div className="panel" style={{ padding: "18px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em" }}>
              Automatic Assessment — {selected.name}
            </div>
            <span style={{ fontSize: 11, color: "var(--text-disabled)", fontFamily: "var(--font-mono)" }}>
              {selected.email || "no email on profile"}
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>

            {/* Password posture from the last recorded check */}
            <div style={{ padding: "14px 16px", borderRadius: 10, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "var(--accent-purple)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                Password Posture
              </div>
              {pwMeta ? (
                <>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
                    <span style={{ fontSize: 26, fontWeight: 800, fontFamily: "var(--font-mono)", color: riskColor(1 - (pwMeta.strength_score ?? 0)) }}>
                      {Math.round((pwMeta.strength_score ?? 0) * 100)}%
                    </span>
                    <span style={{ fontSize: 11, color: "var(--text-muted)" }}>strength</span>
                    {pwMeta.is_pwned && (
                      <span style={{ marginLeft: "auto", fontSize: 9, padding: "3px 8px", borderRadius: 5, fontWeight: 800, color: "#fff", background: "#b91c1c" }}>
                        PWNED ×{(pwMeta.pwned_count ?? 0).toLocaleString()}
                      </span>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
                    {[
                      { ok: (pwMeta.length ?? 0) >= 12, good: `${pwMeta.length} chars`, bad: `Only ${pwMeta.length} chars` },
                      { ok: (pwMeta.char_types ?? 0) >= 4, good: "All character types", bad: `${pwMeta.char_types}/4 character types` },
                      { ok: (pwMeta.entropy_bits ?? 0) >= 60, good: `${Math.round(pwMeta.entropy_bits)} bits entropy`, bad: `Low entropy (${Math.round(pwMeta.entropy_bits)} bits)` },
                      { ok: !pwMeta.is_pwned, good: "Not in breach corpora", bad: "Found in data breaches" },
                    ].map((t, i) => (
                      <span key={i} style={{
                        fontSize: 9.5, padding: "3px 8px", borderRadius: 5, fontWeight: 700,
                        color: t.ok ? "var(--risk-minimal)" : "#ef4444",
                        background: t.ok ? "rgba(0,255,136,0.08)" : "rgba(239,68,68,0.1)",
                        border: `1px solid ${t.ok ? "rgba(0,255,136,0.22)" : "rgba(239,68,68,0.28)"}`,
                      }}>
                        {t.ok ? "✓ " : "✕ "}{t.ok ? t.good : t.bad}
                      </span>
                    ))}
                  </div>
                  {pwMeta.recommendations?.length > 0 && (
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      {pwMeta.recommendations.slice(0, 3).map((r: string, i: number) => (
                        <div key={i} style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5 }}>• {r}</div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.6 }}>
                  No password check recorded yet. Passwords are never stored, so this fills in the
                  first time {selected.name} signs in to the connected app — or check one manually below.
                </div>
              )}
            </div>

            {/* Email exposure, checked automatically */}
            <div style={{ padding: "14px 16px", borderRadius: 10, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#f97316", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                Email Exposure
              </div>
              {autoLoading ? (
                <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>Checking…</div>
              ) : autoEmail && !autoEmail.error ? (
                <>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 10 }}>
                    <span style={{ fontSize: 26, fontWeight: 800, fontFamily: "var(--font-mono)", color: riskColor(autoEmail.risk_score ?? 0) }}>
                      {Math.round((autoEmail.risk_score ?? 0) * 100)}%
                    </span>
                    <span style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "capitalize" }}>
                      {autoEmail.risk_level ?? "unknown"} exposure
                    </span>
                    {autoEmail.breach_count > 0 && (
                      <span style={{ marginLeft: "auto", fontSize: 9, padding: "3px 8px", borderRadius: 5, fontWeight: 800, color: "#fff", background: "#b91c1c" }}>
                        {autoEmail.breach_count} BREACHES
                      </span>
                    )}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {(autoEmail.assessment?.signals ?? []).slice(0, 3).map((s: string, i: number) => (
                      <div key={i} style={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.5 }}>• {s}</div>
                    ))}
                  </div>
                  {autoEmail.note && (
                    <div style={{ marginTop: 9, fontSize: 10, color: autoEmail.verified ? "var(--risk-minimal)" : "#f59e0b" }}>
                      {autoEmail.verified ? "Verified" : "Not verified"} — {autoEmail.note}
                    </div>
                  )}
                </>
              ) : (
                <div style={{ fontSize: 11.5, color: "var(--text-muted)" }}>
                  No email on this profile to assess.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Manual checks */}
      <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em", marginTop: 4 }}>
        Manual Check
      </div>

      {/* Two-column check panels */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Password check */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18, paddingBottom: 14, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
            <span style={{ color: "var(--accent-purple)" }}><IconKey /></span>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em" }}>
              Password Breach Check
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            <input
              type="text" className="input" style={{ flex: 1 }}
              placeholder="Enter password to check…"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={e => e.key === "Enter" && checkPw()}
            />
            <button
              onClick={checkPw} disabled={loading === "pw" || !password.trim()}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "0 16px", borderRadius: 8, fontSize: 12, fontWeight: 700,
                background: loading === "pw" ? "rgba(139,92,246,0.3)" : "linear-gradient(135deg, #8b5cf6, #7c3aed)",
                color: "#fff", border: "none", cursor: loading === "pw" || !password.trim() ? "not-allowed" : "pointer",
                boxShadow: "0 0 16px rgba(139,92,246,0.3)", whiteSpace: "nowrap",
                flexShrink: 0, transition: "all 0.2s",
              }}
            >
              {loading === "pw" ? <IconLoader /> : <IconSearch />}
              {loading === "pw" ? "Checking…" : "Check"}
            </button>
          </div>

          {pwResult && !pwResult.error && (
            <div style={{ display: "flex", flexDirection: "column", gap: 1, animation: "fadeUp 0.4s var(--ease-out) both" }}>
              {[
                { label: "Breach Probability", val: pwResult.breach_probability != null ? formatScore(pwResult.breach_probability) : "—", color: (pwResult.breach_probability ?? 0) >= 0.5 ? "var(--risk-critical)" : "var(--risk-minimal)" },
                { label: "Strength Score",     val: pwResult.strength_score != null ? formatScore(pwResult.strength_score) : "—" },
                { label: "Entropy",            val: pwResult.entropy_bits != null ? `${pwResult.entropy_bits.toFixed(1)} bits` : "—" },
                { label: "Risk Level",         val: pwResult.risk_level ?? "—" },
                { label: "Pwned Count",        val: pwResult.pwned_count != null ? pwResult.pwned_count.toLocaleString() : "—", color: pwResult.pwned_count > 0 ? "var(--risk-critical)" : "var(--risk-minimal)" },
              ].map(row => (
                <div key={row.label} style={{
                  display: "flex", justifyContent: "space-between",
                  fontSize: 12, padding: "9px 0", borderBottom: "1px solid rgba(255,255,255,0.04)",
                }}>
                  <span style={{ color: "var(--text-muted)" }}>{row.label}</span>
                  <span style={{ fontWeight: 700, fontFamily: "var(--font-mono)", color: row.color ?? "var(--text-primary)" }}>{row.val}</span>
                </div>
              ))}

              {/* Password strength chart */}
              {pwResult.entropy_bits != null && (
                <div style={{ marginTop: 14, padding: "14px 16px", borderRadius: 10, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                    Strength Breakdown
                  </div>
                  <DynPwStrength data={{
                    entropy:      pwResult.entropy_bits ?? 0,
                    length_score: Math.min(1, (pwResult.length ?? 0) / 20),
                    diversity:    pwResult.char_diversity ?? (pwResult.strength_score ?? 0.5),
                    uniqueness:   pwResult.pwned_count === 0 ? 0.95 : 0.05,
                  }} />
                </div>
              )}

              {/* Recommendations */}
              <div style={{ marginTop: 14, padding: "14px 16px", borderRadius: 10, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                  Security Recommendations
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                  {(pwResult.pwned_count > 0 ? [
                    { icon: "🔴", text: "Change this password immediately on every account where it's used — do not reuse it anywhere." },
                    { icon: "🔐", text: "Enable two-factor authentication (2FA) on all accounts that used this password." },
                    { icon: "🗝️", text: "Switch to a password manager (Bitwarden, 1Password) to generate and store unique strong passwords." },
                    { icon: "📧", text: "Check all accounts linked to your email for unauthorized activity or unfamiliar sessions." },
                    { icon: "📊", text: "Monitor your credit report if the breach involved financial or identity data." },
                  ] : [
                    { icon: "✅", text: "Your password wasn't found in known breaches — keep it unique per account." },
                    { icon: "🔄", text: "Rotate passwords for sensitive accounts every 90 days as a best practice." },
                    { icon: "🛡️", text: "Enable 2FA on high-value accounts (email, banking, social) even without a known breach." },
                    { icon: "🗝️", text: "Use a passphrase (4+ random words) for memorable yet strong passwords on primary accounts." },
                  ]).map((r, i) => (
                    <div key={i} style={{ display: "flex", gap: 9, fontSize: 12, color: "var(--text-secondary)", alignItems: "flex-start" }}>
                      <span style={{ flexShrink: 0, fontSize: 13 }}>{r.icon}</span>
                      <span style={{ lineHeight: 1.6 }}>{r.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          {pwResult?.error && (
            <div style={{ padding: "12px 14px", borderRadius: 8, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", color: "var(--risk-critical)", fontSize: 12 }}>
              {pwResult.error}
            </div>
          )}
        </div>

        {/* Email check */}
        <div className="panel" style={{ padding: "22px 24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18, paddingBottom: 14, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
            <span style={{ color: "var(--risk-high)" }}><IconMail /></span>
            <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.10em" }}>
              Email Breach Check
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            <input
              type="email" className="input" style={{ flex: 1 }}
              placeholder="user@example.com"
              value={email}
              onChange={e => setEmail(e.target.value)}
              onKeyDown={e => e.key === "Enter" && checkEm()}
            />
            <button
              onClick={checkEm} disabled={loading === "em" || !email.trim()}
              style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "0 16px", borderRadius: 8, fontSize: 12, fontWeight: 700,
                background: loading === "em" ? "rgba(249,115,22,0.3)" : "linear-gradient(135deg, #f97316, #ea580c)",
                color: "#fff", border: "none", cursor: loading === "em" || !email.trim() ? "not-allowed" : "pointer",
                boxShadow: "0 0 16px rgba(249,115,22,0.3)", whiteSpace: "nowrap",
                flexShrink: 0, transition: "all 0.2s",
              }}
            >
              {loading === "em" ? <IconLoader /> : <IconSearch />}
              {loading === "em" ? "Checking…" : "Check"}
            </button>
          </div>

          {emResult && !emResult.error && (
            <div style={{ display: "flex", flexDirection: "column", gap: 1, animation: "fadeUp 0.4s var(--ease-out) both" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "9px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <span style={{ color: "var(--text-muted)" }}>Breach Count</span>
                <span style={{ fontWeight: 700, fontFamily: "var(--font-mono)", color: emResult.breach_count > 0 ? "var(--risk-critical)" : "var(--risk-minimal)", fontSize: 16 }}>
                  {emResult.breach_count ?? 0}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "9px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <span style={{ color: "var(--text-muted)" }}>Exposure Risk</span>
                <span style={{ fontWeight: 700, fontFamily: "var(--font-mono)", color: riskColor(emResult.risk_score ?? 0), fontSize: 16 }}>
                  {emResult.risk_score != null ? `${(emResult.risk_score * 100).toFixed(0)}%` : "—"}
                </span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "9px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                <span style={{ color: "var(--text-muted)" }}>Risk Level</span>
                <span style={{ fontWeight: 700, color: riskColor(emResult.risk_score ?? 0), textTransform: "capitalize" }}>
                  {emResult.risk_level ?? "unknown"}
                </span>
              </div>

              {/* Be explicit about whether this was verified against real breach data */}
              {emResult.note && (
                <div style={{
                  marginTop: 10, padding: "9px 12px", borderRadius: 8, fontSize: 11, lineHeight: 1.6,
                  background: emResult.verified ? "rgba(0,255,136,0.05)" : "rgba(245,158,11,0.06)",
                  border: `1px solid ${emResult.verified ? "rgba(0,255,136,0.18)" : "rgba(245,158,11,0.22)"}`,
                  color: "var(--text-secondary)",
                }}>
                  <b style={{ color: emResult.verified ? "var(--risk-minimal)" : "#f59e0b" }}>
                    {emResult.verified ? "Verified" : "Not verified"}
                  </b>{" — "}{emResult.note}
                </div>
              )}

              {/* Why this address scored the way it did */}
              {emResult.assessment?.signals?.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
                    Exposure Signals
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {emResult.assessment.signals.map((s: string, i: number) => (
                      <div key={i} style={{
                        display: "flex", gap: 8, alignItems: "flex-start",
                        padding: "8px 11px", borderRadius: 8, fontSize: 11.5, lineHeight: 1.55,
                        background: "rgba(255,255,255,0.02)",
                        border: "1px solid rgba(255,255,255,0.05)",
                        color: "var(--text-secondary)",
                      }}>
                        <span style={{ flexShrink: 0 }}>•</span><span>{s}</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 10 }}>
                    {[
                      emResult.assessment.is_disposable && { label: "Disposable domain", color: "#ef4444" },
                      emResult.assessment.is_freemail && { label: "Free provider", color: "#f59e0b" },
                      emResult.assessment.is_role_account && { label: "Role mailbox", color: "#f59e0b" },
                      !emResult.assessment.is_disposable && !emResult.assessment.is_freemail && { label: "Custom domain", color: "#00ff88" },
                    ].filter(Boolean).map((t: any, i: number) => (
                      <span key={i} style={{
                        fontSize: 9, padding: "3px 9px", borderRadius: 6, fontWeight: 800,
                        textTransform: "uppercase", letterSpacing: "0.07em",
                        color: t.color, background: `${t.color}14`, border: `1px solid ${t.color}3a`,
                      }}>{t.label}</span>
                    ))}
                  </div>
                </div>
              )}
              {emResult.breaches?.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
                    Breached Services
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {emResult.breaches.map((b: any, i: number) => (
                      <div key={i} style={{
                        padding: "9px 12px", borderRadius: 8,
                        background: "rgba(255,255,255,0.03)", border: "1px solid rgba(239,68,68,0.15)",
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                      }}>
                        <strong style={{ fontSize: 12, color: "var(--text-primary)" }}>{b.Name ?? b.name}</strong>
                        {b.BreachDate && (
                          <span style={{ fontSize: 10, color: "var(--text-disabled)", fontFamily: "var(--font-mono)" }}>{b.BreachDate}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Email recommendations */}
              <div style={{ marginTop: 14, padding: "14px 16px", borderRadius: 10, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                  Security Recommendations
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                  {((emResult.breach_count ?? 0) > 0 ? [
                    { icon: "🔴", text: "Change passwords on every service where this email was registered — treat all as compromised." },
                    { icon: "🔐", text: "Enable two-factor authentication on email and all linked accounts immediately." },
                    { icon: "👁️", text: "Sign up for breach monitoring (haveibeenpwned.com alerts) to get notified of future exposures." },
                    { icon: "🏦", text: "If financial data was in any breach, contact your bank and monitor for unauthorized transactions." },
                    { icon: "🛂", text: "Consider creating a new email alias for sensitive services to reduce future exposure surface." },
                  ] : [
                    { icon: "✅", text: "Email not found in known breaches — continue using strong unique passwords per service." },
                    { icon: "🔔", text: "Set up breach monitoring alerts so you're notified the moment this email appears in a new breach." },
                    { icon: "🔐", text: "Enable 2FA on your email account as it is the master key to all other accounts." },
                    { icon: "🧹", text: "Audit third-party app permissions granted to this email and revoke unused ones." },
                  ]).map((r, i) => (
                    <div key={i} style={{ display: "flex", gap: 9, fontSize: 12, color: "var(--text-secondary)", alignItems: "flex-start" }}>
                      <span style={{ flexShrink: 0, fontSize: 13 }}>{r.icon}</span>
                      <span style={{ lineHeight: 1.6 }}>{r.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          {emResult?.error && (
            <div style={{ padding: "12px 14px", borderRadius: 8, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", color: "var(--risk-critical)", fontSize: 12 }}>
              {emResult.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
