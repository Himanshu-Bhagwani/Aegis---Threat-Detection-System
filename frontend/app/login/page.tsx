"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const IconMail = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/>
  </svg>
);
const IconKey = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 11-7.778 7.778 5.5 5.5 0 017.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>
  </svg>
);
const IconArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
  </svg>
);
const IconShield = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const IconZap = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
);

export default function LoginPage() {
  const router = useRouter();
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API}/auth/signin`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail ?? "Sign-in failed"); return; }
      const token = data.access_token ?? data.token;
      if (token) {
        localStorage.setItem("aegis_token", token);
        localStorage.setItem("aegis_user",  JSON.stringify({ email, user_sub: data.user_sub ?? email }));
      }
      router.push("/dashboard");
    } catch {
      setError("Could not reach the server. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  function useMockLogin() {
    localStorage.setItem("aegis_token", "mock-dev-token-001");
    localStorage.setItem("aegis_user",  JSON.stringify({ email: "demo@apeilo.com", user_sub: "mock-user-001" }));
    router.push("/dashboard");
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--bg-base)",
      display: "flex",
      overflow: "hidden",
      position: "relative",
    }}>

      {/* Background elements */}
      <div style={{ position: "absolute", inset: 0, zIndex: 0, pointerEvents: "none" }}>
        {/* Ambient glow */}
        <div style={{
          position: "absolute", width: 800, height: 800, borderRadius: "50%",
          top: "50%", left: "50%", transform: "translate(-50%,-50%)",
          background: "radial-gradient(circle, rgba(61,127,255,0.07) 0%, transparent 65%)",
        }} />
        <div style={{
          position: "absolute", width: 400, height: 400, borderRadius: "50%",
          bottom: "0%", right: "10%",
          background: "radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%)",
        }} />
        {/* Grid pattern */}
        <div style={{
          position: "absolute", inset: 0, opacity: 0.4,
          backgroundImage: `linear-gradient(rgba(61,127,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(61,127,255,0.04) 1px, transparent 1px)`,
          backgroundSize: "50px 50px",
        }} />
        {/* Decorative floating elements */}
        <div style={{
          position: "absolute", top: "15%", left: "8%",
          width: 60, height: 60, borderRadius: "50%",
          border: "1px solid rgba(61,127,255,0.15)",
          animation: "float 4s ease-in-out infinite",
        }} />
        <div style={{
          position: "absolute", bottom: "20%", left: "12%",
          width: 30, height: 30, borderRadius: 8,
          border: "1px solid rgba(139,92,246,0.2)",
          animation: "float 3s ease-in-out infinite 1s",
          transform: "rotate(45deg)",
        }} />
        <div style={{
          position: "absolute", top: "65%", right: "8%",
          width: 50, height: 50, borderRadius: "50%",
          border: "1px solid rgba(6,182,212,0.15)",
          animation: "float 5s ease-in-out infinite 0.5s",
        }} />
      </div>

      {/* Left decorative panel (desktop) */}
      <div style={{
        flex: 1, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "60px 60px", position: "relative", zIndex: 1,
        borderRight: "1px solid rgba(255,255,255,0.05)",
      }} className="hide-on-mobile">
        <div style={{ maxWidth: 420 }}>
          {/* Logo large */}
          <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 48 }}>
            <div style={{
              width: 52, height: 52, borderRadius: 14,
              background: "linear-gradient(135deg, #3d7fff, #2563eb)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 24, fontWeight: 900, color: "#fff",
              boxShadow: "0 0 32px rgba(61,127,255,0.5)",
            }}>Æ</div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 900, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>Apeilo</div>
              <div style={{ fontSize: 11, color: "var(--accent)", letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 700 }}>Threat Detection System</div>
            </div>
          </div>

          <h2 style={{ fontSize: 28, fontWeight: 900, letterSpacing: "-0.03em", color: "var(--text-primary)", marginBottom: 14, lineHeight: 1.2 }}>
            AI-powered identity<br />threat intelligence.
          </h2>
          <p style={{ fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.8, marginBottom: 40 }}>
            Multi-layer detection across GPS spoofing, login anomalies, breach exposure, device fingerprinting, and transaction fraud.
          </p>

          {/* Feature pills */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {[
              { icon: <IconShield />, label: "5 detection modules", color: "var(--accent)" },
              { icon: <IconZap />, label: "<50ms detection latency", color: "var(--accent-cyan)" },
              { icon: <IconKey />, label: "k-anonymity breach checks", color: "var(--accent-purple)" },
            ].map(item => (
              <div key={item.label} style={{
                display: "flex", alignItems: "center", gap: 12,
                padding: "10px 16px", borderRadius: 10,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.06)",
              }}>
                <span style={{ color: item.color }}>{item.icon}</span>
                <span style={{ fontSize: 13, color: "var(--text-secondary)", fontWeight: 500 }}>{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right: Login form */}
      <div style={{
        width: "100%", maxWidth: 480,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "60px 48px", position: "relative", zIndex: 1,
      }}>
        <div style={{ width: "100%", animation: "fadeUp 0.6s var(--ease-out) both" }}>

          {/* Mobile logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 40 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: "linear-gradient(135deg, #3d7fff, #2563eb)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 17, fontWeight: 800, color: "#fff",
              boxShadow: "0 0 16px rgba(61,127,255,0.4)",
            }}>Æ</div>
            <div>
              <div style={{ fontWeight: 800, fontSize: 15, color: "var(--text-primary)" }}>Apeilo</div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Threat Detection</div>
            </div>
          </div>

          <h1 style={{ fontSize: 28, fontWeight: 900, letterSpacing: "-0.03em", color: "var(--text-primary)", marginBottom: 6 }}>
            Welcome back
          </h1>
          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 32 }}>
            Sign in to your threat intelligence dashboard.
          </p>

          {/* Form card */}
          <div style={{
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 16,
            padding: "28px",
            backdropFilter: "blur(20px)",
          }}>
            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <label htmlFor="email">Email address</label>
                <div style={{ position: "relative" }}>
                  <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}>
                    <IconMail />
                  </div>
                  <input
                    id="email"
                    type="email"
                    className="input"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="you@company.com"
                    required
                    style={{ paddingLeft: 40 }}
                  />
                </div>
              </div>

              <div>
                <label htmlFor="password">Password</label>
                <div style={{ position: "relative" }}>
                  <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}>
                    <IconKey />
                  </div>
                  <input
                    id="password"
                    type="password"
                    className="input"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="••••••••"
                    required
                    style={{ paddingLeft: 40 }}
                  />
                </div>
              </div>

              {error && (
                <div style={{
                  padding: "11px 16px", borderRadius: 10, fontSize: 13,
                  background: "rgba(239,68,68,0.08)",
                  border: "1px solid rgba(239,68,68,0.22)",
                  color: "var(--risk-critical)",
                  display: "flex", alignItems: "center", gap: 8,
                }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  padding: "13px", borderRadius: 10, fontSize: 14, fontWeight: 700,
                  background: loading ? "rgba(61,127,255,0.4)" : "linear-gradient(135deg, #3d7fff, #2563eb)",
                  color: "#fff", border: "none", cursor: loading ? "not-allowed" : "pointer",
                  boxShadow: loading ? "none" : "0 0 20px rgba(61,127,255,0.35)",
                  transition: "all 0.2s",
                }}
              >
                {loading ? (
                  <>
                    <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                    Signing in…
                  </>
                ) : (
                  <> Sign in <IconArrowRight /> </>
                )}
              </button>
            </form>

            {/* Divider */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "20px 0", color: "var(--text-disabled)", fontSize: 11 }}>
              <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }} />
              OR
              <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }} />
            </div>

            {/* Mock login */}
            <button
              onClick={useMockLogin}
              style={{
                width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                padding: "11px", borderRadius: 10, fontSize: 13, fontWeight: 600,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "var(--text-secondary)", cursor: "pointer",
                transition: "all 0.2s",
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.05)";
                (e.currentTarget as HTMLElement).style.color = "var(--text-primary)";
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.03)";
                (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)";
              }}
            >
              <IconZap />
              Continue with mock token (dev mode)
            </button>
          </div>

          <p style={{ textAlign: "center", marginTop: 24, fontSize: 13, color: "var(--text-secondary)" }}>
            No account?{" "}
            <Link href="/signup" style={{ color: "var(--accent)", fontWeight: 700, textDecoration: "none" }}>
              Sign up free
            </Link>
          </p>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          .hide-on-mobile { display: none !important; }
        }
      `}</style>
    </div>
  );
}
