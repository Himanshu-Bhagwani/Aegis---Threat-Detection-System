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

export default function SignupPage() {
  const router = useRouter();
  const [email,     setEmail]     = useState("");
  const [password,  setPassword]  = useState("");
  const [name,      setName]      = useState("");
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState<string | null>(null);
  const [success,   setSuccess]   = useState(false);

  const pwStrength = PASSWORD_RULES.filter(r => r.test(password)).length;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (pwStrength < 4) { setError("Password does not meet all requirements."); return; }

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API}/auth/signup`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ email, password, given_name: name }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail ?? "Sign-up failed");
        return;
      }

      // If a token is returned directly, log in immediately
      const token = data.access_token ?? data.token;
      if (token) {
        localStorage.setItem("aegis_token", token);
        localStorage.setItem("aegis_user",  JSON.stringify({ email, user_sub: data.user_sub ?? email }));
        router.push("/dashboard");
      } else {
        setSuccess(true); // Cognito email verification flow
      }
    } catch {
      setError("Could not reach the server. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  if (success) {
    return (
      <div style={{ minHeight: "100vh", background: "var(--bg-base)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
        <div style={{ textAlign: "center", maxWidth: 400, animation: "fadeUp 0.5s ease" }}>
          <div style={{ fontSize: 48, marginBottom: 20 }}>◉</div>
          <h2 style={{ fontSize: 24, fontWeight: 800, color: "var(--text-primary)", marginBottom: 10 }}>
            Check your email
          </h2>
          <p style={{ fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 28 }}>
            We sent a verification link to <strong style={{ color: "var(--text-primary)" }}>{email}</strong>.
            Click it to activate your account and access the dashboard.
          </p>
          <Link href="/login" className="btn-primary" style={{ fontSize: 14, padding: "12px 32px" }}>
            Go to Sign In →
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: "100vh", background: "var(--bg-base)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 24, position: "relative", overflow: "hidden",
    }}>
      {/* Background glow */}
      <div style={{
        position: "absolute", width: 600, height: 600, borderRadius: "50%",
        top: "50%", left: "50%", transform: "translate(-50%, -50%)",
        background: "radial-gradient(circle, rgba(61,127,255,0.06) 0%, transparent 70%)",
        pointerEvents: "none",
      }} />

      <div style={{ width: "100%", maxWidth: 400, position: "relative", animation: "fadeUp 0.5s ease" }}>

        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 36 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "var(--accent)", display: "flex",
            alignItems: "center", justifyContent: "center",
            fontSize: 18, fontWeight: 800, color: "#fff",
          }}>Æ</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15 }}>Apeilo</div>
            <div style={{ fontSize: 10, color: "var(--text-disabled)", textTransform: "uppercase",
              letterSpacing: "0.08em" }}>Threat Detection</div>
          </div>
        </div>

        <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.025em",
          color: "var(--text-primary)", marginBottom: 6 }}>
          Create account
        </h1>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 28 }}>
          Start detecting threats across your identity surface.
        </p>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <label style={{ display: "block", fontSize: 11, color: "var(--text-muted)",
              textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
              Full Name
            </label>
            <input
              type="text"
              className="input"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="Jane Smith"
              style={{ width: "100%" }}
            />
          </div>

          <div>
            <label style={{ display: "block", fontSize: 11, color: "var(--text-muted)",
              textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
              Email
            </label>
            <input
              type="email"
              className="input"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@company.com"
              required
              style={{ width: "100%" }}
            />
          </div>

          <div>
            <label style={{ display: "block", fontSize: 11, color: "var(--text-muted)",
              textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
              Password
            </label>
            <input
              type="password"
              className="input"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              style={{ width: "100%" }}
            />
            {/* Password strength meter */}
            {password && (
              <div style={{ marginTop: 8 }}>
                <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
                  {[0,1,2,3].map(i => (
                    <div key={i} style={{
                      flex: 1, height: 3, borderRadius: 2,
                      background: i < pwStrength
                        ? pwStrength <= 1 ? "var(--risk-critical)"
                          : pwStrength <= 2 ? "var(--risk-high)"
                          : pwStrength <= 3 ? "var(--risk-medium)"
                          : "var(--risk-minimal)"
                        : "var(--bg-raised)",
                      transition: "background 0.2s",
                    }} />
                  ))}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3px 12px" }}>
                  {PASSWORD_RULES.map(r => (
                    <div key={r.label} style={{
                      fontSize: 11, display: "flex", alignItems: "center", gap: 5,
                      color: r.test(password) ? "var(--risk-minimal)" : "var(--text-disabled)",
                    }}>
                      <span>{r.test(password) ? "✓" : "○"}</span>
                      {r.label}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {error && (
            <div style={{
              padding: "10px 14px", borderRadius: 8, fontSize: 13,
              background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.20)",
              color: "var(--risk-critical)",
            }}>
              {error}
            </div>
          )}

          <button
            type="submit"
            className="btn-primary"
            disabled={loading}
            style={{ marginTop: 4, fontSize: 14, padding: "12px 0" }}
          >
            {loading ? "Creating account…" : "Create Account →"}
          </button>
        </form>

        <p style={{ textAlign: "center", marginTop: 24, fontSize: 13, color: "var(--text-secondary)" }}>
          Already have an account?{" "}
          <Link href="/login" style={{ color: "var(--accent)", textDecoration: "none", fontWeight: 600 }}>
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
