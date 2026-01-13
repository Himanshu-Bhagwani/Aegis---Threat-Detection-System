"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { signup } from "@/lib/api";

const STORY_LINES = [
  "Deploy breach telemetry across every identity perimeter.",
  "Adaptive models harden with each new account you onboard.",
  "Privacy-first architecture, verified on AWS infrastructure."
];

export default function SignupPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [isNarrativeActive, setIsNarrativeActive] = useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSuccessMessage(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setIsLoading(true);

    try {
      await signup({ email, password });
      setSuccessMessage("Identity node activated. Redirecting to login...");
      setTimeout(() => router.push("/login"), 1800);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to sign up");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-panel">
        <span className="auth-brand">Aegis - Fraud &amp; Identity Protection</span>
        <div>
          <h1>Deploy your shield</h1>
          <p className="auth-subtitle">
            Create an account to activate breach telemetry, adaptive anomaly detection, and
            privacy-preserving monitoring across your identity surface.
          </p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">
            Work email
            <input
              className="auth-input"
              type="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="security@aegis-secure.com"
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
              placeholder="Minimum 6 characters"
              autoComplete="new-password"
              minLength={6}
            />
          </label>

          <label className="auth-label">
            Confirm password
            <input
              className="auth-input"
              type="password"
              required
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              placeholder="Re-enter your password"
              autoComplete="new-password"
              minLength={6}
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
            {isLoading ? "Provisioning..." : "Create account"}
          </button>
        </form>

        <p className="auth-footer">
          Already orchestrating defenses?{" "}
          <a href="/login" className="auth-link">
            Return to sign in
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
          <span className="auth-hero-kicker">Cloud-Native Vigilance</span>
          <h2>Build trust with every signal</h2>
          <p>
            Onboard identities into the <span className="auth-highlight">Aegis fraud intelligence network</span> and
            deliver continuous protection backed by AWS observability and adaptive ML.
          </p>
          <span className="auth-hero-cta">Preview the research playbook</span>
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
