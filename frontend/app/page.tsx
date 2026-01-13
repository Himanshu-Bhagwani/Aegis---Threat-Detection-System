import Link from "next/link";

import { AppShell } from "@/components/app-shell";

const heroCTA = [
  { label: "Get Started", href: "/signup", variant: "primary" as const },
  { label: "Learn How It Works", href: "#how-it-works", variant: "ghost" as const }
];

const featureList = [
  {
    title: "Real-Time Threat Monitoring",
    description:
      "Detect login anomalies, data breaches, and GPS spoofing attempts within seconds.",
    icon: "🛰️"
  },
  {
    title: "AI-Powered Fraud Detection",
    description:
      "Isolation Forest and Autoencoder models running on SageMaker to flag emerging threats.",
    icon: "🤖"
  },
  {
    title: "Privacy-Preserving Scans",
    description:
      "Hashed lookups with Bloom filters keep sensitive identity data protected end-to-end.",
    icon: "🔐"
  },
  {
    title: "Multi-Channel Alerts",
    description:
      "Instant notifications across email, SMS, and dashboard powered by AWS SNS and Pinpoint.",
    icon: "📡"
  },
  {
    title: "AWS Cloud Security Backbone",
    description:
      "Encryption with AWS KMS, observability with CloudWatch, and resilience across the stack.",
    icon: "☁️"
  }
];

const workflow = [
  {
    title: "User Onboarding",
    summary: "Secure enrollment with AWS Cognito MFA and contextual risk checks."
  },
  {
    title: "Data Monitoring",
    summary: "Identity telemetry streams into Aegis for breach and anomaly screening."
  },
  {
    title: "AI Detection",
    summary: "ML engines evaluate patterns for spoofing, account takeovers, and fraud."
  },
  {
    title: "Alerts & Recommendations",
    summary: "Targeted guidance helps resolve incidents faster via the Aegis dashboard."
  }
];

const differentiators = [
  {
    problem: "Complex manual fraud detection",
    solution: "Automated, ML-driven detection pipelines tailored to identity risk."
  },
  {
    problem: "Slow response to threats",
    solution: "Real-time Lambda + SageMaker inference with proactive auto-remediation."
  },
  {
    problem: "Privacy risk during breach checks",
    solution: "Secure hash-based verification—raw data never leaves the user boundary."
  },
  {
    problem: "Lack of cloud security",
    solution: "AWS-native encryption, logging, and monitoring baked into every layer."
  }
];

const dashboardHighlights = [
  {
    title: "Threat Intelligence Feed",
    detail: "Track alerts by severity with context-rich timelines and recommended actions."
  },
  {
    title: "Anomaly Analytics",
    detail: "Visualize login anomalies, velocity risks, and GPS deviations in real time."
  },
  {
    title: "Global Exposure Map",
    detail: "Pinpoint suspicious access by geography and device fingerprint."
  },
  {
    title: "Privacy Health Report",
    detail: "Monitor breach status, hashed leak checks, and remediation playbooks."
  }
];

export default function Home() {
  const landingContent = (
    <main className="landing">
      <section className="landing-hero" id="top">
        <div className="landing-hero-content">
          <span className="landing-eyebrow">Identity Intelligence Platform</span>
          <h1>Aegis: Intelligent Identity Protection for the Modern World</h1>
          <p>
            Protect your digital identity with real-time threat detection,
            privacy-preserving monitoring, and AI-driven fraud alerts — powered by AWS.
          </p>
          <div className="hero-cta-row">
            {heroCTA.map((cta) => (
              <Link
                key={cta.label}
                href={cta.href as any}
                className={`hero-cta hero-cta-${cta.variant}`}
              >
                {cta.label}
              </Link>
            ))}
          </div>
        </div>

        <div className="landing-hero-visual" aria-hidden="true">
          <div className="visual-node visual-detected">
            <span>Threat detected</span>
          </div>
          <div className="visual-connector" />
          <div className="visual-node visual-alert">
            <span>Alert dispatched</span>
          </div>
          <div className="visual-connector" />
          <div className="visual-node visual-secured">
            <span>Identity secured</span>
          </div>
        </div>
      </section>

      <section className="landing-section" id="features">
        <header>
          <h2>Stay ahead with intelligent defense</h2>
          <p>
            Aegis unifies streaming telemetry, AI, and AWS security services to safeguard
            every digital identity edge.
          </p>
        </header>
        <div className="feature-grid">
          {featureList.map((feature) => (
            <article key={feature.title} className="feature-card">
              <span className="feature-icon">{feature.icon}</span>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-section landing-section-alt" id="how-it-works">
        <header>
          <h2>How Aegis orchestrates protection</h2>
          <p>
            A pipeline built for security teams and researchers—transparent, powerful, and
            grounded in cloud-native best practices.
          </p>
        </header>
        <div className="workflow">
          {workflow.map((step, index) => (
            <div key={step.title} className="workflow-step">
              <span className="workflow-index">{index + 1}</span>
              <div>
                <h3>{step.title}</h3>
                <p>{step.summary}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="workflow-diagram" aria-hidden="true">
          <span>User</span>
          <span>Detection</span>
          <span>ML Engine</span>
          <span>Alert</span>
          <span>Dashboard</span>
        </div>
      </section>

      <section className="landing-section" id="why-aegis">
        <header>
          <h2>Why Aegis delivers uncompromised protection</h2>
          <p>
            We confront the toughest fraud and identity challenges with automation,
            observability, and privacy-first design.
          </p>
        </header>
        <div className="differentiator-grid">
          {differentiators.map((item) => (
            <div key={item.problem} className="differentiator-card">
              <h3>Problem</h3>
              <p>{item.problem}</p>
              <h4>Our Solution</h4>
              <p>{item.solution}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="landing-section landing-section-alt" id="dashboard">
        <header>
          <h2>The Aegis command center</h2>
          <p>
            Preview the dashboard experience that turns streaming identity signals into
            actionable intelligence.
          </p>
        </header>
        <div className="dashboard-grid">
          {dashboardHighlights.map((item) => (
            <article key={item.title} className="dashboard-card">
              <h3>{item.title}</h3>
              <p>{item.detail}</p>
              <div className="dashboard-placeholder" />
            </article>
          ))}
        </div>
      </section>

      <section className="landing-cta">
        <div className="landing-cta-card">
          <h2>Ready to secure every identity moment?</h2>
          <p>
            Join the early access program to partner on research, shape the roadmap, and
            unlock full-fidelity monitoring across your organization.
          </p>
          <div className="cta-actions">
            <Link href="/signup" className="hero-cta hero-cta-primary">
              Sign Up for Early Access
            </Link>
            <Link href="/dashboard" className="hero-cta hero-cta-ghost">
              Access your dashboard
            </Link>
          </div>
        </div>
      </section>
    </main>
  );

  return <AppShell>{landingContent}</AppShell>;
}
