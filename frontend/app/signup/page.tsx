"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const PASSWORD_RULES = [
  { label: "8+ characters",    test: (p: string) => p.length >= 8 },
  { label: "Uppercase letter", test: (p: string) => /[A-Z]/.test(p) },
  { label: "Lowercase letter", test: (p: string) => /[a-z]/.test(p) },
  { label: "Number",           test: (p: string) => /\d/.test(p) },
];

const IconUser = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/>
  </svg>
);
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
const IconCheckCircle = () => (
  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
  </svg>
);

export default function SignupPage() {
  const router = useRouter();
  const [email,   setEmail]   = useState("");
  const [password, setPassword] = useState("");
  const [name,    setName]    = useState("");
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const pwStrength = PASSWORD_RULES.filter(r => r.test(password)).length;

  const strengthColor = pwStrength <= 1 ? "var(--risk-critical)"
    : pwStrength <= 2 ? "var(--risk-high)"
    : pwStrength <= 3 ? "var(--risk-medium)"
    : "var(--risk-minimal)";

  const strengthLabel = pwStrength <= 1 ? "Weak"
    : pwStrength <= 2 ? "Fair"
    : pwStrength <= 3 ? "Good"
    : "Strong";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (pwStrength < 4) { setError("Password does not meet all requirements."); return; }
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/auth/signup`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ email, password, given_name: name }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail ?? "Sign-up failed"); return; }
      const token = data.access_token ?? data.token;
      if (token) {
        localStorage.setItem("aegis_token", token);
        localStorage.setItem("aegis_user",  JSON.stringify({ email, user_sub: data.user_sub ?? email }));
        router.push("/dashboard");
      } else {
        setSuccess(true);
      }
    } catch {
      setError("Could not reach the server. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  if (success) {
    return (
      <div style={{
        minHeight: "100vh", background: "var(--bg-base)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
        position: "relative",
      }}>
        <div style={{
          position: "absolute", inset: 0, zIndex: 0, pointerEvents: "none",
          background: "radial-gradient(ellipse 80% 60% at 50% 50%, rgba(0,255,136,0.06) 0%, transparent 70%)",
        }} />
        <div style={{
          textAlign: "center", maxWidth: 420, animation: "fadeUp 0.6s var(--ease-out) both",
          position: "relative", zIndex: 1,
          background: "rgba(255,255,255,0.025)",
          border: "1px solid rgba(0,255,136,0.2)",
          borderRadius: 20, padding: "48px 40px",
          backdropFilter: "blur(20px)",
          boxShadow: "0 0 40px rgba(0,255,136,0.08)",
        }}>
          <div style={{ color: "var(--risk-minimal)", marginBottom: 20, display: "flex", justifyContent: "center" }}>
            <IconCheckCircle />
          </div>
          <h2 style={{ fontSize: 24, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)", marginBottom: 12 }}>
            Check your email
          </h2>
          <p style={{ fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.75, marginBottom: 32 }}>
            We sent a verification link to{" "}
            <strong style={{ color: "var(--text-primary)" }}>{email}</strong>.
            Click it to activate your account.
          </p>
          <Link href="/login" style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "13px 32px", borderRadius: 10, fontSize: 14, fontWeight: 700,
            background: "linear-gradient(135deg, #3d7fff, #2563eb)",
            color: "#fff", textDecoration: "none",
            boxShadow: "0 0 20px rgba(61,127,255,0.35)",
          }}>
            Go to Sign In <IconArrowRight />
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: "100vh", background: "var(--bg-base)",
      display: "flex", overflow: "hidden", position: "relative",
    }}>
      {/* Background */}
      <div style={{ position: "absolute", inset: 0, zIndex: 0, pointerEvents: "none" }}>
        <div style={{
          position: "absolute", width: 800, height: 800, borderRadius: "50%",
          top: "50%", left: "50%", transform: "translate(-50%,-50%)",
          background: "radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 65%)",
        }} />
        <div style={{
          position: "absolute", inset: 0, opacity: 0.4,
          backgroundImage: `linear-gradient(rgba(139,92,246,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.04) 1px, transparent 1px)`,
          backgroundSize: "50px 50px",
        }} />
        <div style={{
          position: "absolute", top: "10%", right: "8%",
          width: 70, height: 70, borderRadius: "50%",
          border: "1px solid rgba(139,92,246,0.15)",
          animation: "float 4s ease-in-out infinite",
        }} />
        <div style={{
          position: "absolute", bottom: "15%", left: "6%",
          width: 40, height: 40,
          border: "1px solid rgba(61,127,255,0.15)",
          borderRadius: 10, transform: "rotate(30deg)",
          animation: "float 3.5s ease-in-out infinite 0.8s",
        }} />
      </div>

      {/* Left panel */}
      <div style={{
        flex: 1, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "60px", position: "relative", zIndex: 1,
        borderRight: "1px solid rgba(255,255,255,0.05)",
      }} className="hide-on-mobile">
        <div style={{ maxWidth: 400 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 48 }}>
            <div style={{
              width: 48, height: 48, borderRadius: 13,
              background: "linear-gradient(135deg, #3d7fff, #2563eb)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 22, fontWeight: 900, color: "#fff",
              boxShadow: "0 0 28px rgba(61,127,255,0.5)",
            }}>Æ</div>
            <div>
              <div style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.03em", color: "var(--text-primary)" }}>Apeilo</div>
              <div style={{ fontSize: 10, color: "var(--accent-purple)", letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 700 }}>Threat Detection</div>
            </div>
          </div>
          <h2 style={{ fontSize: 26, fontWeight: 900, letterSpacing: "-0.03em", color: "var(--text-primary)", marginBottom: 14, lineHeight: 1.25 }}>
            Join the future of<br />identity protection.
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.8, marginBottom: 36 }}>
            Get instant access to all five detection modules — GPS spoofing, login anomaly, breach intelligence, device fingerprinting, and fraud scoring.
          </p>
          {/* Steps */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {[
              { n: "01", t: "Create account", d: "Takes less than 60 seconds" },
              { n: "02", t: "Explore the dashboard", d: "Pre-loaded with demo profiles" },
              { n: "03", t: "Connect your backend", d: "REST API + WebSocket support" },
            ].map(s => (
              <div key={s.n} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                <div style={{
                  width: 32, height: 32, borderRadius: 8, flexShrink: 0,
                  background: "rgba(61,127,255,0.10)", border: "1px solid rgba(61,127,255,0.2)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 11, fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--accent)",
                }}>{s.n}</div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>{s.t}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{s.d}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right: Signup form */}
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
            Create account
          </h1>
          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 32 }}>
            Start detecting threats across your identity surface.
          </p>

          <div style={{
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 16, padding: "28px",
            backdropFilter: "blur(20px)",
          }}>
            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <label htmlFor="name">Full Name</label>
                <div style={{ position: "relative" }}>
                  <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}>
                    <IconUser />
                  </div>
                  <input id="name" type="text" className="input" value={name} onChange={e => setName(e.target.value)} placeholder="Jane Smith" style={{ paddingLeft: 40 }} />
                </div>
              </div>

              <div>
                <label htmlFor="email">Email address</label>
                <div style={{ position: "relative" }}>
                  <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}>
                    <IconMail />
                  </div>
                  <input id="email" type="email" className="input" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@company.com" required style={{ paddingLeft: 40 }} />
                </div>
              </div>

              <div>
                <label htmlFor="password">Password</label>
                <div style={{ position: "relative" }}>
                  <div style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", pointerEvents: "none" }}>
                    <IconKey />
                  </div>
                  <input id="password" type="password" className="input" value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" required style={{ paddingLeft: 40 }} />
                </div>

                {password && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
                      {[0,1,2,3].map(i => (
                        <div key={i} style={{
                          flex: 1, height: 3, borderRadius: 2,
                          background: i < pwStrength ? strengthColor : "rgba(255,255,255,0.08)",
                          transition: "background 0.3s",
                          boxShadow: i < pwStrength ? `0 0 8px ${strengthColor}60` : "none",
                        }} />
                      ))}
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Password strength</span>
                      <span style={{ fontSize: 11, fontWeight: 700, color: strengthColor }}>{strengthLabel}</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px" }}>
                      {PASSWORD_RULES.map(r => (
                        <div key={r.label} style={{
                          fontSize: 11, display: "flex", alignItems: "center", gap: 6,
                          color: r.test(password) ? "var(--risk-minimal)" : "var(--text-disabled)",
                          transition: "color 0.2s",
                        }}>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                            {r.test(password)
                              ? <polyline points="20 6 9 17 4 12" />
                              : <><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></>
                            }
                          </svg>
                          {r.label}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div style={{
                  padding: "11px 16px", borderRadius: 10, fontSize: 13,
                  background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.22)",
                  color: "var(--risk-critical)", display: "flex", alignItems: "center", gap: 8,
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
                  padding: "13px", borderRadius: 10, fontSize: 14, fontWeight: 700, marginTop: 4,
                  background: loading ? "rgba(61,127,255,0.4)" : "linear-gradient(135deg, #3d7fff, #2563eb)",
                  color: "#fff", border: "none", cursor: loading ? "not-allowed" : "pointer",
                  boxShadow: loading ? "none" : "0 0 20px rgba(61,127,255,0.35)",
                  transition: "all 0.2s",
                }}
              >
                {loading ? (
                  <><span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Creating account…</>
                ) : (
                  <>Create Account <IconArrowRight /></>
                )}
              </button>
            </form>
          </div>

          <p style={{ textAlign: "center", marginTop: 24, fontSize: 13, color: "var(--text-secondary)" }}>
            Already have an account?{" "}
            <Link href="/login" style={{ color: "var(--accent)", fontWeight: 700, textDecoration: "none" }}>
              Sign in
            </Link>
          </p>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) { .hide-on-mobile { display: none !important; } }
      `}</style>
    </div>
  );
}
