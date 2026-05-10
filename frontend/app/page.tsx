"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useRef } from "react";

const ParticleHero = dynamic(() => import("@/components/three/ParticleHero"), { ssr: false });

/* ── Icons ───────────────────────────────────────────────── */
const IconShield = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  </svg>
);
const IconMapPin = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/><circle cx="12" cy="10" r="3"/>
  </svg>
);
const IconLock = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>
  </svg>
);
const IconCreditCard = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>
  </svg>
);
const IconMonitor = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
  </svg>
);
const IconZap = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
);
const IconArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
  </svg>
);
const IconActivity = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);

/* ── Feature data ────────────────────────────────────────── */
const FEATURES = [
  { icon: <IconMapPin />, title: "GPS Spoofing Detection", desc: "ML models detect real-time location fraud — flags mock GPS apps, emulator signals, and impossible velocity jumps in under 50ms.", color: "var(--accent)", glow: "rgba(61,127,255,0.15)" },
  { icon: <IconLock />, title: "Login Anomaly Engine", desc: "Behavioral baseline per user. Detects unusual login hours, new device access, rapid credential cycling, and brute-force patterns.", color: "var(--risk-medium)", glow: "rgba(245,158,11,0.12)" },
  { icon: <IconShield />, title: "Breach Intelligence", desc: "k-anonymity HIBP lookups — only 5 chars of your SHA-1 hash ever leave the device. Scores exposure and entropy in real time.", color: "var(--risk-high)", glow: "rgba(249,115,22,0.12)" },
  { icon: <IconCreditCard />, title: "Fraud Transaction Scoring", desc: "XGBoost + Isolation Forest ensemble. Scores each transaction across amount, geography, merchant velocity, and time-of-day signals.", color: "var(--risk-critical)", glow: "rgba(239,68,68,0.12)" },
  { icon: <IconMonitor />, title: "Device Fingerprinting", desc: "SHA-256 stable device IDs. Weighted similarity scoring across platform, screen, timezone, WebGL renderer, and audio context.", color: "var(--risk-low)", glow: "rgba(132,204,22,0.12)" },
  { icon: <IconZap />, title: "Unified Fusion Engine", desc: "Weighted, max-threat, and Bayesian fusion strategies combine all signals into a single actionable risk score per identity event.", color: "var(--accent-cyan)", glow: "rgba(6,182,212,0.12)" },
];

const STEPS = [
  { num: "01", title: "Event Arrives", desc: "Login, transaction, GPS ping, or app-unlock fires from your client SDK or REST call." },
  { num: "02", title: "Parallel Scoring", desc: "All five detection modules score simultaneously — no sequential bottleneck." },
  { num: "03", title: "Fusion", desc: "Scores fused using your chosen strategy and weighted by confidence levels." },
  { num: "04", title: "Action", desc: "Risk result returned in <50ms. SNS fires if critical. Dashboard updates via WebSocket." },
];

const STATS = [
  { value: "<50ms", label: "P99 Detection Latency" },
  { value: "5", label: "Detection Modules" },
  { value: "100%", label: "AWS-Native Stack" },
  { value: "k-anon", label: "Privacy Architecture" },
];

/* ── Scroll-reveal hook ──────────────────────────────────── */
function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll(".reveal");
    const io = new IntersectionObserver(
      entries => entries.forEach(e => e.isIntersecting && e.target.classList.add("revealed")),
      { threshold: 0.1 },
    );
    els.forEach(el => io.observe(el));
    return () => io.disconnect();
  }, []);
}

/* ── Component ───────────────────────────────────────────── */
export default function LandingPage() {
  useReveal();

  return (
    <div style={{ background: "var(--bg-base)", minHeight: "100vh", overflowX: "hidden" }}>

      {/* ── Navbar ─────────────────────────────────── */}
      <nav style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
        display: "flex", alignItems: "center", padding: "0 32px",
        height: 64,
        background: "rgba(2,6,23,0.8)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 9,
            background: "linear-gradient(135deg, #3d7fff, #2563eb)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 800, color: "#fff",
            boxShadow: "0 0 16px rgba(61,127,255,0.5)",
          }}>Æ</div>
          <div>
            <span style={{ fontWeight: 800, fontSize: 15, letterSpacing: "-0.02em", color: "var(--text-primary)" }}>Apeilo</span>
            <span style={{ fontSize: 9, color: "var(--accent)", letterSpacing: "0.1em", textTransform: "uppercase", marginLeft: 6, fontWeight: 700 }}>· TDS</span>
          </div>
        </div>

        {/* Nav links */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {[
            { href: "#features", label: "Features" },
            { href: "#how", label: "How It Works" },
          ].map(link => (
            <a key={link.href} href={link.href} style={{
              padding: "6px 14px", borderRadius: 8, fontSize: 13,
              color: "var(--text-secondary)", textDecoration: "none",
              transition: "all 0.2s",
            }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = "var(--text-primary)"; (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.05)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)"; (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >{link.label}</a>
          ))}
          <div style={{ width: 1, height: 20, background: "rgba(255,255,255,0.08)", margin: "0 6px" }} />
          <Link href="/login" style={{
            padding: "6px 14px", borderRadius: 8, fontSize: 13,
            color: "var(--text-secondary)", textDecoration: "none",
          }}>Sign In</Link>
          <Link href="/signup" style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            padding: "8px 18px", borderRadius: 8, fontSize: 13, fontWeight: 600,
            background: "linear-gradient(135deg, #3d7fff, #2563eb)",
            color: "#fff", textDecoration: "none",
            boxShadow: "0 0 16px rgba(61,127,255,0.35)",
            transition: "all 0.2s",
          }}>
            Get Started <IconArrowRight />
          </Link>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────── */}
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

        {/* Ambient glow orbs */}
        <div style={{
          position: "absolute", width: 600, height: 600, borderRadius: "50%",
          top: "30%", left: "20%", transform: "translate(-50%,-50%)",
          background: "radial-gradient(circle, rgba(61,127,255,0.07) 0%, transparent 70%)",
          pointerEvents: "none",
        }} />
        <div style={{
          position: "absolute", width: 500, height: 500, borderRadius: "50%",
          top: "60%", right: "10%",
          background: "radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%)",
          pointerEvents: "none",
        }} />

        {/* Hero content */}
        <div style={{
          position: "relative", zIndex: 1, textAlign: "center",
          maxWidth: 760, padding: "0 24px",
          animation: "fadeUp 1s var(--ease-out) both",
        }}>
          {/* Tag */}
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "6px 16px", borderRadius: 20,
            background: "rgba(61,127,255,0.10)",
            border: "1px solid rgba(61,127,255,0.25)",
            fontSize: 11, color: "var(--accent)",
            letterSpacing: "0.08em", textTransform: "uppercase",
            fontWeight: 700, marginBottom: 28,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%", background: "var(--accent)",
              display: "inline-block", boxShadow: "0 0 8px var(--accent)",
              animation: "pulse-green 2s infinite",
            }} />
            Real-time Identity Threat Detection
          </div>

          {/* Headline */}
          <h1 style={{
            fontSize: "clamp(2.6rem, 6.5vw, 4.5rem)",
            fontWeight: 900,
            lineHeight: 1.06,
            letterSpacing: "-0.035em",
            color: "var(--text-primary)",
            marginBottom: 24,
          }}>
            Know when someone isn&apos;t<br />
            <span style={{
              background: "linear-gradient(90deg, #3d7fff 0%, #06b6d4 50%, #8b5cf6 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              backgroundSize: "200% 100%",
              animation: "gradient-shift 4s ease infinite",
            }}>
              who they say they are.
            </span>
          </h1>

          {/* Subhead */}
          <p style={{
            fontSize: "clamp(1rem, 2.2vw, 1.2rem)",
            color: "var(--text-secondary)",
            maxWidth: 580, margin: "0 auto 40px",
            lineHeight: 1.75,
          }}>
            Apeilo fuses GPS spoofing, login anomalies, breach intelligence, device fingerprints,
            and transaction fraud into a single risk score — in under 50ms.
          </p>

          {/* CTAs */}
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <Link href="/signup" style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              padding: "14px 32px", borderRadius: 10, fontSize: 14, fontWeight: 700,
              background: "linear-gradient(135deg, #3d7fff, #2563eb)",
              color: "#fff", textDecoration: "none",
              boxShadow: "0 0 24px rgba(61,127,255,0.4), 0 8px 24px rgba(37,99,235,0.3)",
              transition: "all 0.2s",
            }}>
              Start Free <IconArrowRight />
            </Link>
            <Link href="/dashboard" style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              padding: "14px 32px", borderRadius: 10, fontSize: 14, fontWeight: 600,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "var(--text-secondary)", textDecoration: "none",
              backdropFilter: "blur(10px)",
              transition: "all 0.2s",
            }}>
              View Demo Dashboard
            </Link>
          </div>
        </div>

        {/* Scroll indicator */}
        <div style={{
          position: "absolute", bottom: 36, left: "50%", transform: "translateX(-50%)",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
          color: "var(--text-disabled)", fontSize: 9, letterSpacing: "0.15em",
          textTransform: "uppercase", animation: "fadeIn 2s 1.5s both",
        }}>
          <div style={{
            width: 1, height: 48,
            background: "linear-gradient(to bottom, transparent, var(--accent))",
            animation: "float 2s ease-in-out infinite",
          }} />
          scroll
        </div>
      </section>

      {/* ── Stats ──────────────────────────────────── */}
      <section style={{
        background: "rgba(7,13,26,0.8)",
        borderTop: "1px solid rgba(255,255,255,0.05)",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        padding: "40px 40px",
        backdropFilter: "blur(20px)",
      }}>
        <div style={{ maxWidth: 900, margin: "0 auto", display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 24 }}>
          {STATS.map((s, i) => (
            <div key={s.label} className="reveal" style={{ textAlign: "center", animationDelay: `${i * 0.1}s` }}>
              <div style={{
                fontSize: "clamp(1.8rem, 3.5vw, 2.6rem)", fontWeight: 900,
                fontFamily: "var(--font-mono)", color: "var(--accent)",
                letterSpacing: "-0.02em", marginBottom: 6,
                textShadow: "0 0 30px rgba(61,127,255,0.5)",
              }}>{s.value}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.06em", textTransform: "uppercase", fontWeight: 600 }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ───────────────────────────────── */}
      <section id="features" style={{ padding: "100px 40px", maxWidth: 1140, margin: "0 auto" }}>
        <div className="reveal" style={{ textAlign: "center", marginBottom: 64 }}>
          <div className="chip chip-blue" style={{ marginBottom: 16 }}>
            <IconActivity />
            Detection Stack
          </div>
          <h2 style={{
            fontSize: "clamp(2rem, 4.5vw, 3rem)", fontWeight: 900,
            letterSpacing: "-0.03em", color: "var(--text-primary)",
          }}>
            Every threat vector,{" "}
            <span className="gradient-text-blue">covered.</span>
          </h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
          {FEATURES.map((f, i) => (
            <div
              key={f.title}
              className="panel reveal"
              style={{
                padding: "28px 26px",
                animationDelay: `${i * 0.08}s`,
                cursor: "default",
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLElement;
                el.style.transform = "translateY(-4px)";
                el.style.borderColor = `${f.color}40`;
                el.style.boxShadow = `0 12px 40px rgba(0,0,0,0.4), 0 0 30px ${f.glow}`;
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLElement;
                el.style.transform = "translateY(0)";
                el.style.borderColor = "";
                el.style.boxShadow = "";
              }}
            >
              <div style={{
                width: 44, height: 44, borderRadius: 12,
                background: f.glow,
                border: `1px solid ${f.color}30`,
                display: "flex", alignItems: "center", justifyContent: "center",
                color: f.color, marginBottom: 18,
                boxShadow: `0 0 16px ${f.glow}`,
              }}>
                {f.icon}
              </div>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 10, color: "var(--text-primary)" }}>
                {f.title}
              </h3>
              <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7, margin: 0 }}>
                {f.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── How it works ───────────────────────────── */}
      <section id="how" style={{
        background: "rgba(7,13,26,0.6)",
        borderTop: "1px solid rgba(255,255,255,0.05)",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        padding: "100px 40px",
        backdropFilter: "blur(20px)",
      }}>
        <div style={{ maxWidth: 960, margin: "0 auto" }}>
          <div className="reveal" style={{ textAlign: "center", marginBottom: 64 }}>
            <div className="chip chip-cyan" style={{ marginBottom: 16 }}>
              Architecture
            </div>
            <h2 style={{
              fontSize: "clamp(2rem, 4.5vw, 3rem)", fontWeight: 900,
              letterSpacing: "-0.03em", color: "var(--text-primary)",
            }}>
              From event to decision{" "}
              <span style={{ color: "var(--accent-cyan)" }}>in milliseconds.</span>
            </h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 24 }}>
            {STEPS.map((step, i) => (
              <div key={step.num} className="reveal" style={{ animationDelay: `${i * 0.12}s`, position: "relative" }}>
                {i < STEPS.length - 1 && (
                  <div style={{
                    position: "absolute", top: 22, left: "calc(100% + 4px)",
                    width: "calc(100% - 8px)", height: 1,
                    background: "linear-gradient(90deg, rgba(61,127,255,0.5), rgba(61,127,255,0.1))",
                  }} />
                )}
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "center",
                  width: 44, height: 44, borderRadius: 12,
                  background: "rgba(61,127,255,0.12)", border: "1px solid rgba(61,127,255,0.25)",
                  fontSize: 13, fontWeight: 800, fontFamily: "var(--font-mono)",
                  color: "var(--accent)", marginBottom: 16,
                  boxShadow: "0 0 16px rgba(61,127,255,0.15)",
                }}>
                  {step.num}
                </div>
                <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8 }}>
                  {step.title}
                </div>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7 }}>
                  {step.desc}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ────────────────────────────────────── */}
      <section style={{ padding: "100px 40px", textAlign: "center", position: "relative", overflow: "hidden" }}>
        <div style={{
          position: "absolute", inset: 0, zIndex: 0,
          background: "radial-gradient(ellipse 80% 60% at 50% 50%, rgba(61,127,255,0.07) 0%, transparent 70%)",
        }} />
        <div className="reveal" style={{ maxWidth: 600, margin: "0 auto", position: "relative", zIndex: 1 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8, marginBottom: 20,
            padding: "6px 16px", borderRadius: 20,
            background: "rgba(0,255,136,0.08)", border: "1px solid rgba(0,255,136,0.2)",
            fontSize: 11, color: "var(--risk-minimal)", fontWeight: 700, letterSpacing: "0.06em",
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--risk-minimal)", display: "inline-block", boxShadow: "0 0 8px var(--risk-minimal)" }} />
            No credit card required
          </div>
          <h2 style={{
            fontSize: "clamp(2rem, 4.5vw, 3rem)", fontWeight: 900,
            letterSpacing: "-0.03em", color: "var(--text-primary)", marginBottom: 16,
          }}>
            Start detecting threats today.
          </h2>
          <p style={{ fontSize: 15, color: "var(--text-secondary)", marginBottom: 36, lineHeight: 1.75 }}>
            Runs fully local in mock mode. Connect AWS to go production in minutes.
          </p>
          <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap" }}>
            <Link href="/signup" style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              padding: "14px 36px", borderRadius: 10, fontSize: 14, fontWeight: 700,
              background: "linear-gradient(135deg, #3d7fff, #2563eb)",
              color: "#fff", textDecoration: "none",
              boxShadow: "0 0 28px rgba(61,127,255,0.45), 0 8px 28px rgba(37,99,235,0.3)",
            }}>
              Create Account <IconArrowRight />
            </Link>
            <Link href="/dashboard" style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              padding: "14px 36px", borderRadius: 10, fontSize: 14, fontWeight: 600,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "var(--text-secondary)", textDecoration: "none",
              backdropFilter: "blur(10px)",
            }}>
              Try the Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────── */}
      <footer style={{
        borderTop: "1px solid rgba(255,255,255,0.05)",
        padding: "24px 40px",
        display: "flex", alignItems: "center",
        color: "var(--text-disabled)", fontSize: 12,
        background: "rgba(7,13,26,0.6)",
        backdropFilter: "blur(10px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 22, height: 22, borderRadius: 6,
            background: "linear-gradient(135deg, #3d7fff, #2563eb)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 11, fontWeight: 800, color: "#fff",
          }}>Æ</div>
          <span>© {new Date().getFullYear()} Apeilo Threat Detection</span>
        </div>
        <span style={{ marginLeft: "auto" }}>
          Built with FastAPI · Next.js · Three.js · AWS
        </span>
      </footer>
    </div>
  );
}
