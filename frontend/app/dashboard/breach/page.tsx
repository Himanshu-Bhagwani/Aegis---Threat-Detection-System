"use client";
import { useState } from "react";
import { checkPasswordBreach, checkEmailBreach, riskColor, formatScore } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

export default function BreachPage() {
  const { selected, updateMetrics } = useProfiles();
  const [password, setPassword] = useState("");
  const [email,    setEmail]    = useState("");
  const [pwResult, setPwResult] = useState<any>(null);
  const [emResult, setEmResult] = useState<any>(null);
  const [loading,  setLoading]  = useState<"pw"|"em"|null>(null);
  const [updated,  setUpdated]  = useState<string | null>(null);

  async function checkPw() {
    if (!password.trim()) return;
    setLoading("pw"); setUpdated(null);
    try {
      const res = await checkPasswordBreach(password);
      setPwResult(res);
      if (res?.breach_probability != null && selected) {
        const newVal = selected.metrics.breach_risk * 0.7 + res.breach_probability * 0.3;
        updateMetrics(selected.id, { breach_risk: Math.min(1, Math.max(0, newVal)) });
        setUpdated(`Updated ${selected.name}'s breach risk to ${(Math.min(1, Math.max(0, newVal)) * 100).toFixed(0)}%`);
      }
    } catch (e: any) { setPwResult({ error: e?.message }); }
    finally { setLoading(null); }
  }

  async function checkEm() {
    if (!email.trim()) return;
    setLoading("em"); setUpdated(null);
    try {
      const res = await checkEmailBreach(email);
      setEmResult(res);
      // derive a breach probability from breach_count
      if (selected) {
        const count = res?.breach_count ?? 0;
        const inferredRisk = Math.min(1, count * 0.15);
        const newVal = selected.metrics.breach_risk * 0.7 + inferredRisk * 0.3;
        updateMetrics(selected.id, { breach_risk: Math.min(1, Math.max(0, newVal)) });
        setUpdated(`Updated ${selected.name}'s breach risk to ${(Math.min(1, Math.max(0, newVal)) * 100).toFixed(0)}%`);
      }
    } catch (e: any) { setEmResult({ error: e?.message }); }
    finally { setLoading(null); }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>Breach Intelligence</h1>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>k-anonymity HIBP checks — only 5 chars of your SHA-1 hash leave the device</p>
        </div>
        {selected && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 20, background: "var(--bg-raised)", border: "1px solid var(--border-subtle)", fontSize: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: riskColor(selected.metrics.breach_risk) }} />
            <span style={{ color: "var(--text-muted)" }}>Analyzing:</span>
            <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{selected.name}</span>
            <span style={{ color: "var(--text-muted)" }}>·</span>
            <span style={{ fontFamily: "var(--font-mono)", color: riskColor(selected.metrics.breach_risk), fontWeight: 700 }}>
              Breach {(selected.metrics.breach_risk * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      {updated && (
        <div style={{ padding: "10px 16px", borderRadius: 8, background: "rgba(61,127,255,0.1)", border: "1px solid rgba(61,127,255,0.3)", fontSize: 12, color: "var(--accent)" }}>
          ✓ {updated}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Password */}
        <div className="panel" style={{ padding: "20px 22px" }}>
          <div className="panel-header">Password Breach Check</div>
          <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
            <input type="text" className="input" placeholder="Enter password to check…" value={password}
              onChange={e => setPassword(e.target.value)} onKeyDown={e => e.key === "Enter" && checkPw()} />
            <button className="btn-primary" onClick={checkPw} disabled={loading === "pw" || !password.trim()}>
              {loading === "pw" ? "Checking…" : "Check Password →"}
            </button>
          </div>
          {pwResult && !pwResult.error && (
            <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                { label: "Breach Probability", val: pwResult.breach_probability != null ? formatScore(pwResult.breach_probability) : "—", color: (pwResult.breach_probability ?? 0) >= 0.5 ? "var(--risk-critical)" : "var(--risk-minimal)" },
                { label: "Strength Score",     val: pwResult.strength_score     != null ? formatScore(pwResult.strength_score)     : "—" },
                { label: "Entropy",            val: pwResult.entropy_bits       != null ? `${pwResult.entropy_bits.toFixed(1)} bits` : "—" },
                { label: "Risk Level",         val: pwResult.risk_level ?? "—" },
                { label: "Pwned Count",        val: pwResult.pwned_count != null ? pwResult.pwned_count.toLocaleString() : "—", color: pwResult.pwned_count > 0 ? "var(--risk-critical)" : "var(--risk-minimal)" },
              ].map(row => (
                <div key={row.label} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "7px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                  <span style={{ color: "var(--text-muted)" }}>{row.label}</span>
                  <span style={{ fontWeight: 600, fontFamily: "var(--font-mono)", color: row.color ?? "var(--text-primary)" }}>{row.val}</span>
                </div>
              ))}
              {pwResult.recommendations?.map((r: string, i: number) => (
                <div key={i} style={{ fontSize: 12, color: "var(--text-secondary)", display: "flex", gap: 7 }}>
                  <span style={{ color: "var(--accent)" }}>→</span>{r}
                </div>
              ))}
            </div>
          )}
          {pwResult?.error && <div style={{ marginTop: 12, fontSize: 12, color: "var(--risk-high)" }}>{pwResult.error}</div>}
        </div>

        {/* Email */}
        <div className="panel" style={{ padding: "20px 22px" }}>
          <div className="panel-header">Email Breach Check</div>
          <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
            <input type="email" className="input" placeholder="user@example.com" value={email}
              onChange={e => setEmail(e.target.value)} onKeyDown={e => e.key === "Enter" && checkEm()} />
            <button className="btn-primary" onClick={checkEm} disabled={loading === "em" || !email.trim()}>
              {loading === "em" ? "Checking…" : "Check Email →"}
            </button>
          </div>
          {emResult && !emResult.error && (
            <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "7px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                <span style={{ color: "var(--text-muted)" }}>Breach Count</span>
                <span style={{ fontWeight: 600, fontFamily: "var(--font-mono)", color: emResult.breach_count > 0 ? "var(--risk-critical)" : "var(--risk-minimal)" }}>{emResult.breach_count ?? 0}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "7px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                <span style={{ color: "var(--text-muted)" }}>Risk Level</span>
                <span style={{ fontWeight: 600, color: riskColor(emResult.breach_count > 3 ? 0.8 : emResult.breach_count > 0 ? 0.5 : 0.1) }}>
                  {emResult.risk_level ?? (emResult.breach_count > 0 ? "compromised" : "clean")}
                </span>
              </div>
              {emResult.breaches?.map((b: any, i: number) => (
                <div key={i} style={{ fontSize: 12, padding: "7px 10px", borderRadius: 6, background: "var(--bg-raised)", color: "var(--text-secondary)" }}>
                  <strong style={{ color: "var(--text-primary)" }}>{b.Name ?? b.name}</strong>
                  {b.BreachDate && <span style={{ color: "var(--text-muted)", marginLeft: 8 }}>{b.BreachDate}</span>}
                </div>
              ))}
            </div>
          )}
          {emResult?.error && <div style={{ marginTop: 12, fontSize: 12, color: "var(--risk-high)" }}>{emResult.error}</div>}
        </div>
      </div>
    </div>
  );
}
