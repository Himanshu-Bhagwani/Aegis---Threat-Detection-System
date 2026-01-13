"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { login } from "@/lib/api";
import { setAuthToken } from "@/lib/auth";

const STORY_LINES = [
  "Live breach telemetry. Automatically triaged.",
  "Machine learning signals that evolve with every identity.",
  "Privacy hardened through zero-knowledge monitoring."
];

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isNarrativeActive, setIsNarrativeActive] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSuccessMessage(null);
    setIsLoading(true);

    try {
      const response = await login({ email, password });
      setAuthToken(response.idToken);
      setSuccessMessage("Securely signed in. Redirecting...");
      setTimeout(() => router.push("/dashboard"), 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to log in");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-panel">
        <span className="auth-brand">Aegis - Fraud &amp; Identity Protection</span>
        <div>
          <h1>Welcome back, Guardian</h1>
          <p className="auth-subtitle">
            Authenticate to review live identity signals and close incidents with confidence.
          </p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">
            Email
            <input
              className="auth-input"
              type="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@aegis-secure.com"
              autoComplete="email"
            />
          </label>

          <label className="auth-label">
            Password
            <input
              className="auth-input"
              type="password"
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your passphrase"
              autoComplete="current-password"
            />
          </label>

          {error && <p className="auth-error">{error}</p>}
          {successMessage && <p className="auth-success">{successMessage}</p>}

          <button
            className="auth-submit"
            type="submit"
            disabled={isLoading}
            onMouseEnter={() => setIsNarrativeActive(true)}
            onMouseLeave={() => setIsNarrativeActive(false)}
            onFocus={() => setIsNarrativeActive(true)}
            onBlur={() => setIsNarrativeActive(false)}
          >
            {isLoading ? "Securing session..." : "Sign in"}
          </button>
        </form>

        <p className="auth-footer">
          New to Aegis?{" "}
          <a href="/signup" className="auth-link">
            Create your shield
          </a>
        </p>
      </section>

      <aside
        className={`auth-hero auth-hero--network${isNarrativeActive ? " is-active" : ""}`}
      >
        <div className="network-stage" aria-hidden="true">
          <span className="network-core" />
          <span className="network-ring ring-a" />
          <span className="network-ring ring-b" />
          <span className="network-ring ring-c" />
          <span className="network-node node-1" />
          <span className="network-node node-2" />
          <span className="network-node node-3" />
          <span className="network-node node-4" />
        </div>

        <div className="auth-hero-content auth-hero-content--glass">
          <span className="auth-hero-kicker">Adaptive Defense</span>
          <h2>Your identity intelligence hub</h2>
          <p>
            Layered anomaly detection, breach telemetry, and{" "}
            <span className="auth-highlight">real-time AWS insights</span> keep your
            organisation ahead of sophisticated threat actors.
          </p>
          <span className="auth-hero-cta">Explore fraud analytics briefing</span>
          <span className="auth-hero-footnote">
            Backed by AWS Observability + ML Defense Stack
          </span>
        </div>

        <div className="story-lines" aria-hidden="true">
          {STORY_LINES.map((line, index) => (
            <span
              key={line}
              className="story-line"
              style={{ transitionDelay: `${120 + index * 160}ms` }}
            >
              {line}
            </span>
          ))}
        </div>
      </aside>
    </main>
  );
}
