"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

      if (!res.ok) {
        setError(data.detail ?? "Sign-in failed");
        return;
      }

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
      minHeight: "100vh", background: "var(--bg-base)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 24, position: "relative", overflow: "hidden",
    }}>
      {/* Background glow */}
      <div style={{
        position: "absolute", width: 600, height: 600,
        borderRadius: "50%", top: "50%", left: "50%",
        transform: "translate(-50%, -50%)",
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
          Sign in
        </h1>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 28 }}>
          Access your threat intelligence dashboard.
        </p>

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
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
            {loading ? "Signing in…" : "Sign in →"}
          </button>
        </form>

        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          margin: "20px 0", color: "var(--text-disabled)", fontSize: 12,
        }}>
          <div style={{ flex: 1, height: 1, background: "var(--border-subtle)" }} />
          or
          <div style={{ flex: 1, height: 1, background: "var(--border-subtle)" }} />
        </div>

        <button
          onClick={useMockLogin}
          className="btn-ghost"
          style={{ width: "100%", fontSize: 13, padding: "10px 0" }}
        >
          Continue with mock token (dev mode)
        </button>

        <p style={{ textAlign: "center", marginTop: 24, fontSize: 13, color: "var(--text-secondary)" }}>
          No account?{" "}
          <Link href="/signup" style={{ color: "var(--accent)", textDecoration: "none", fontWeight: 600 }}>
            Sign up free
          </Link>
        </p>
      </div>
    </div>
  );
}
