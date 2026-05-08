"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useRef } from "react";

const ParticleHero = dynamic(() => import("@/components/three/ParticleHero"), { ssr: false });

/* ─── Feature data ───────────────────────────────────── */
const FEATURES = [
  {
    icon: "◉",
    title: "GPS Spoofing Detection",
    desc: "Machine-learning models detect real-time location fraud. Flags mock GPS apps, emulator signals, and impossible velocity jumps.",
    color: "var(--accent)",
  },
  {
    icon: "◐",
    title: "Login Anomaly Engine",
    desc: "Behavioral baseline per user. Detects unusual login hours, new device access, rapid credential cycling, and brute-force patterns.",
    color: "var(--risk-medium)",
  },
  {
    icon: "◑",
    title: "Breach Intelligence",
    desc: "k-anonymity HIBP lookups — only 5 chars of your SHA-1 hash ever leave the device. Scores password exposure and entropy in real time.",
    color: "var(--risk-high)",
  },
  {
    icon: "◆",
    title: "Fraud Transaction Scoring",
    desc: "XGBoost + Isolation Forest ensemble. Scores each transaction across amount, geography, merchant velocity, and time-of-day signals.",
    color: "var(--risk-critical)",
  },
  {
    icon: "◍",
    title: "Device Fingerprinting",
    desc: "SHA-256 stable device IDs. Weighted similarity scoring across platform, screen, timezone, WebGL renderer, and audio context fingerprint.",
    color: "var(--risk-low)",
  },
  {
    icon: "⬡",
    title: "Unified Fusion Engine",
    desc: "Weighted, max-threat, and Bayesian fusion strategies combine all signals into a single actionable risk score per identity event.",
    color: "var(--accent)",
  },
];

const STEPS = [
  { num: "01", title: "Event Arrives",   desc: "Login, transaction, GPS ping, or app-unlock fires from your client SDK or REST call." },
  { num: "02", title: "Parallel Scoring",desc: "All five detection modules score simultaneously — no sequential bottleneck." },
  { num: "03", title: "Fusion",          desc: "Scores are fused using your chosen strategy and weighted by confidence levels." },
  { num: "04", title: "Action",          desc: "Risk result returned in <50 ms. SNS fires if critical. Dashboard updates via WebSocket." },
];

const STATS = [
  { value: "<50ms",  label: "P99 Detection Latency" },
  { value: "5",      label: "Detection Modules"      },
  { value: "100%",   label: "AWS-Native Stack"        },
  { value: "k-anon", label: "Privacy Architecture"   },
];

/* ─── Scroll-reveal hook ─────────────────────────────── */
function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll(".reveal");
    const io  = new IntersectionObserver(
      entries => entries.forEach(e => e.isIntersecting && e.target.classList.add("revealed")),
      { threshold: 0.12 },
    );
    els.forEach(el => io.observe(el));
    return () => io.disconnect();
  }, []);
}

/* ─── Component ──────────────────────────────────────── */
export default function LandingPage() {
  useReveal();

  return (
    <div style={{ background: "var(--bg-base)", minHeight: "100vh", overflowX: "hidden" }}>

      {/* ── Nav bar ───────────────────────────────── */}
      <nav style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
        display: "flex", alignItems: "center", padding: "0 40px",
        height: 60,
        background: "rgba(8,9,13,0.85)",
        backdropFilter: "blur(12px)",
        borderBottom: "1px solid var(--border-subtle)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: "var(--accent)", display: "flex",
            alignItems: "center", justifyContent: "center",
            fontSize: 15, fontWeight: 700, color: "#fff",
          }}>Æ</div>
          <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>Apeilo</span>
          <span style={{ fontSize: 10, color: "var(--text-disabled)", letterSpacing: "0.08em",
            textTransform: "uppercase", marginLeft: 2 }}>Threat Detection</span>
        </div>
        <div style={{ display: "flex", gap: 28, fontSize: 13, color: "var(--text-secondary)" }}>
          <a href="#features"  style={{ color: "inherit", textDecoration: "none" }}>Features</a>
          <a href="#how"       style={{ color: "inherit", textDecoration: "none" }}>How It Works</a>
          <Link href="/login"  style={{ color: "inherit", textDecoration: "none" }}>Sign In</Link>
          <Link href="/signup" className="btn-primary" style={{ fontSize: 13, padding: "6px 16px" }}>
            Get Started
          </Link>
        </div>
      </nav>

      {/* ── Hero ─────────────────────────────────── */}
      <section style={{
        position: "relative", minHeight: "100vh",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        overflow: "hidden",
      }}>
        {/* Three.js particle canvas */}
        <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
          <ParticleHero />
        </div>

        {/* Hero text */}
        <div style={{
          position: "relative", zIndex: 1, textAlign: "center",
          maxWidth: 720, padding: "0 24px",
          animation: "fadeUp 0.9s ease both",
        }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "5px 14px", borderRadius: 20,
            background: "rgba(61,127,255,0.10)",
            border: "1px solid rgba(61,127,255,0.25)",
            fontSize: 11, color: "var(--accent)",
            letterSpacing: "0.08em", textTransform: "uppercase",
            fontWeight: 600, marginBottom: 24,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", display: "inline-block" }} />
            Real-time identity threat detection
          </div>

          <h1 style={{
            fontSize: "clamp(2.4rem, 6vw, 4rem)",
            fontWeight: 800,
            lineHeight: 1.08,
            letterSpacing: "-0.03em",
            color: "var(--text-primary)",
            marginBottom: 22,
          }}>
            Know when someone isn't<br />
            <span style={{
              background: "linear-gradient(90deg, var(--accent), #6db3ff)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}>who they say they are.</span>
          </h1>

          <p style={{
            fontSize: "clamp(0.95rem, 2vw, 1.15rem)",
            color: "var(--text-secondary)",
            maxWidth: 560, margin: "0 auto 36px",
            lineHeight: 1.7,
          }}>
            Apeilo fuses GPS spoofing, login anomalies, breach intelligence, device fingerprints,
            and transaction fraud into a single risk score — in under 50 ms.
          </p>

          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <Link href="/signup" className="btn-primary" style={{ fontSize: 14, padding: "12px 28px" }}>
              Start Free →
            </Link>
            <Link href="/dashboard" className="btn-ghost" style={{ fontSize: 14, padding: "12px 28px" }}>
              View Demo Dashboard
            </Link>
          </div>
        </div>

        {/* Scroll indicator */}
        <div style={{
          position: "absolute", bottom: 32, left: "50%", transform: "translateX(-50%)",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
          color: "var(--text-disabled)", fontSize: 10, letterSpacing: "0.1em",
          textTransform: "uppercase", animation: "fadeIn 1.5s 1s both",
        }}>
          <div style={{
            width: 1, height: 36,
            background: "linear-gradient(to bottom, transparent, var(--border))",
          }} />
          scroll
        </div>
      </section>

      {/* ── Stats bar ─────────────────────────────── */}
      <section style={{
        background: "var(--bg-surface)",
        borderTop:  "1px solid var(--border-subtle)",
        borderBottom: "1px solid var(--border-subtle)",
        padding: "32px 40px",
      }}>
        <div style={{
          maxWidth: 900, margin: "0 auto",
          display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20,
        }}>
          {STATS.map(s => (
            <div key={s.label} style={{ textAlign: "center" }} className="reveal">
              <div style={{
                fontSize: "clamp(1.6rem, 3vw, 2.2rem)", fontWeight: 800,
                fontFamily: "var(--font-mono)", color: "var(--accent)",
                letterSpacing: "-0.02em",
              }}>{s.value}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4,
                letterSpacing: "0.04em", textTransform: "uppercase" }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ──────────────────────────────── */}
      <section id="features" style={{ padding: "100px 40px", maxWidth: 1100, margin: "0 auto" }}>
        <div className="reveal" style={{ textAlign: "center", marginBottom: 60 }}>
          <div style={{ fontSize: 11, color: "var(--accent)", textTransform: "uppercase",
            letterSpacing: "0.1em", fontWeight: 600, marginBottom: 14 }}>Detection Stack</div>
          <h2 style={{ fontSize: "clamp(1.8rem, 4vw, 2.8rem)", fontWeight: 800,
            letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
            Every threat vector, covered.
          </h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
          {FEATURES.map((f, i) => (
            <div key={f.title} className="panel reveal" style={{
              padding: "28px 24px",
              animationDelay: `${i * 0.08}s`,
              transition: "transform 0.2s, border-color 0.2s",
              cursor: "default",
            }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.transform = "translateY(-3px)";
                (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
                (e.currentTarget as HTMLElement).style.borderColor = "var(--border-subtle)"; }}
            >
              <div style={{
                width: 40, height: 40, borderRadius: 10,
                background: `${f.color}14`,
                border: `1px solid ${f.color}30`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 20, color: f.color, marginBottom: 16,
              }}>
                {f.icon}
              </div>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8, color: "var(--text-primary)" }}>
                {f.title}
              </h3>
              <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65, margin: 0 }}>
                {f.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── How it works ──────────────────────────── */}
      <section id="how" style={{
        background: "var(--bg-surface)",
        borderTop: "1px solid var(--border-subtle)",
        borderBottom: "1px solid var(--border-subtle)",
        padding: "100px 40px",
      }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div className="reveal" style={{ textAlign: "center", marginBottom: 60 }}>
            <div style={{ fontSize: 11, color: "var(--accent)", textTransform: "uppercase",
              letterSpacing: "0.1em", fontWeight: 600, marginBottom: 14 }}>Architecture</div>
            <h2 style={{ fontSize: "clamp(1.8rem, 4vw, 2.8rem)", fontWeight: 800,
              letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
              From event to decision in milliseconds.
            </h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
            {STEPS.map((step, i) => (
              <div key={step.num} className="reveal" style={{ animationDelay: `${i * 0.1}s` }}>
                <div style={{
                  fontSize: 11, color: "var(--accent)", fontFamily: "var(--font-mono)",
                  fontWeight: 700, letterSpacing: "0.08em", marginBottom: 12,
                }}>
                  {step.num}
                  {i < STEPS.length - 1 && (
                    <span style={{ color: "var(--border)", marginLeft: 8 }}>────</span>
                  )}
                </div>
                <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8 }}>
                  {step.title}
                </div>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65 }}>
                  {step.desc}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────── */}
      <section style={{ padding: "100px 40px", textAlign: "center" }}>
        <div className="reveal" style={{ maxWidth: 560, margin: "0 auto" }}>
          <h2 style={{ fontSize: "clamp(1.8rem, 4vw, 2.8rem)", fontWeight: 800,
            letterSpacing: "-0.025em", color: "var(--text-primary)", marginBottom: 16 }}>
            Start detecting threats today.
          </h2>
          <p style={{ fontSize: 14, color: "var(--text-secondary)", marginBottom: 32, lineHeight: 1.7 }}>
            No credit card required. Runs fully local in mock mode.
            Connect AWS to go production in minutes.
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
            <Link href="/signup" className="btn-primary" style={{ fontSize: 14, padding: "12px 32px" }}>
              Create Account →
            </Link>
            <Link href="/dashboard" className="btn-ghost" style={{ fontSize: 14, padding: "12px 32px" }}>
              Try the Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────── */}
      <footer style={{
        borderTop: "1px solid var(--border-subtle)",
        padding: "28px 40px",
        display: "flex", alignItems: "center",
        color: "var(--text-disabled)", fontSize: 12,
      }}>
        <span>© {new Date().getFullYear()} Apeilo Threat Detection</span>
        <span style={{ marginLeft: "auto" }}>
          Built with FastAPI · Next.js · Three.js · AWS
        </span>
      </footer>

      {/* ── Scroll-reveal styles ─────────────────── */}
      <style>{`
        .reveal {
          opacity: 0;
          transform: translateY(24px);
          transition: opacity 0.6s ease, transform 0.6s ease;
        }
        .reveal.revealed {
          opacity: 1;
          transform: translateY(0);
        }
      `}</style>
    </div>
  );
}
