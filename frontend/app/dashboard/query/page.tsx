"use client";

import dynamic from "next/dynamic";
import { useState, useRef, useEffect, useCallback } from "react";
import { queryNL, type NLQueryResponse } from "@/lib/api";
import { useProfiles } from "@/contexts/ProfileContext";

// Dynamic import required — Recharts uses DOM APIs not available in SSR
const QueryChart = dynamic(() => import("@/components/QueryChart"), {
  ssr:     false,
  loading: () => (
    <div style={{ height: 300, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: 12 }}>
      Loading chart…
    </div>
  ),
});

// ── Types ─────────────────────────────────────────────────

interface Message {
  id:        string;
  role:      "user" | "assistant";
  content:   string;
  response?: NLQueryResponse;
  loading?:  boolean;
  error?:    string;
  ts:        Date;
}

// ── Suggested queries ─────────────────────────────────────

const SUGGESTED = [
  "Which users have the highest fraud risk?",
  "Show me a breakdown of all detection module scores",
  "Who has critical risk scores right now?",
  "Compare login anomaly risk across all users",
  "What is Himanshu's risk profile?",
  "Show breach exposure for all users",
  "Which users are above the risk threshold?",
  "Give me an overview of the system's current threat status",
];

// ── Assistant bubble ──────────────────────────────────────

function AssistantBubble({ msg }: { msg: Message }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 24 }}>
      <div style={{ maxWidth: "90%", width: "100%" }}>

        {/* Header row */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8, flexShrink: 0,
            background: "linear-gradient(135deg, rgba(61,127,255,0.2), rgba(139,92,246,0.2))",
            border: "1px solid rgba(61,127,255,0.3)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 13, fontWeight: 800, color: "var(--accent)",
          }}>Æ</div>
          <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 600 }}>Aegis AI</span>
          {msg.response && (
            <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-disabled)", fontFamily: "var(--font-mono)" }}>
              {msg.response.query_time_ms}ms · {msg.response.ollama_model}
            </span>
          )}
        </div>

        {/* Loading state */}
        {msg.loading && (
          <div style={{
            padding: "16px 20px", borderRadius: "4px 16px 16px 16px",
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.07)",
            display: "flex", alignItems: "center", gap: 10,
          }}>
            <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
              {[0, 1, 2].map(i => (
                <div key={i} style={{
                  width: 7, height: 7, borderRadius: "50%",
                  background: "var(--accent)",
                  animation: `nlBounce 1.1s ease-in-out ${i * 0.18}s infinite`,
                }} />
              ))}
            </div>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Thinking with Ollama…</span>
          </div>
        )}

        {/* Error state */}
        {msg.error && (
          <div style={{
            padding: "14px 18px", borderRadius: "4px 16px 16px 16px",
            background: "rgba(239,68,68,0.07)",
            border: "1px solid rgba(239,68,68,0.2)",
            fontSize: 12, color: "var(--risk-critical)", lineHeight: 1.6,
          }}>
            <strong>Error:</strong> {msg.error}
            <div style={{ marginTop: 4, color: "var(--text-muted)", fontSize: 11 }}>
              Make sure the backend is running on port 8000 and Ollama is available.
            </div>
          </div>
        )}

        {/* Welcome message (no response object) */}
        {!msg.loading && !msg.error && !msg.response && msg.content && (
          <div style={{
            padding: "14px 18px", borderRadius: "4px 16px 16px 16px",
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.07)",
            fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7,
          }}>
            {msg.content}
          </div>
        )}

        {/* Full response with chart */}
        {msg.response && (
          <div style={{
            borderRadius: "4px 16px 16px 16px",
            background: "rgba(255,255,255,0.025)",
            border: "1px solid rgba(255,255,255,0.07)",
            overflow: "hidden",
          }}>
            {/* Answer text + badges */}
            <div style={{ padding: "16px 20px 12px" }}>
              <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 12 }}>
                {msg.response.answer}
              </p>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                <span style={{
                  fontSize: 9, padding: "3px 8px", borderRadius: 5,
                  background: "rgba(61,127,255,0.12)", color: "var(--accent)",
                  border: "1px solid rgba(61,127,255,0.25)", fontWeight: 800,
                  textTransform: "uppercase", letterSpacing: "0.07em",
                }}>
                  {msg.response.intent.replace(/_/g, " ")}
                </span>
                <span style={{
                  fontSize: 9, padding: "3px 8px", borderRadius: 5,
                  background: "rgba(139,92,246,0.12)", color: "var(--accent-purple)",
                  border: "1px solid rgba(139,92,246,0.25)", fontWeight: 800,
                  textTransform: "uppercase", letterSpacing: "0.07em",
                }}>
                  {msg.response.chart_type}
                </span>
                <span style={{
                  fontSize: 9, padding: "3px 8px", borderRadius: 5,
                  background: "rgba(0,255,136,0.08)", color: "var(--risk-minimal)",
                  border: "1px solid rgba(0,255,136,0.2)", fontWeight: 800,
                  textTransform: "uppercase", letterSpacing: "0.07em",
                }}>
                  live data
                </span>
              </div>
            </div>

            {/* Chart area */}
            <div style={{ padding: "4px 20px 20px" }}>
              <QueryChart response={msg.response} />
            </div>

            {/* Secondary table (shown when bar/line chart also has table_data) */}
            {msg.response.chart_type !== "table" &&
              msg.response.table_data?.length &&
              msg.response.table_headers?.length ? (
              <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "14px 20px 18px" }}>
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700, marginBottom: 10 }}>
                  Detail Table
                </div>
                <QueryChart response={{ ...msg.response, chart_type: "table" }} />
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────

export default function QueryPage() {
  const { profiles } = useProfiles();

  const [messages, setMessages] = useState<Message[]>([{
    id:      "welcome",
    role:    "assistant",
    content: "Ask me anything about the users in this system. I have access to real-time risk scores for all loaded profiles and can generate charts, tables, and comparisons. Try a suggestion below.",
    ts:      new Date(),
  }]);
  const [input,   setInput]   = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Build Ollama conversation history from past turns
  const buildHistory = useCallback(() =>
    messages
      .filter(m => m.role !== "assistant" || !!m.response)
      .slice(-6)
      .map(m => ({
        role:    m.role as "user" | "assistant",
        content: m.role === "assistant"
          ? (m.response?.answer ?? m.content)
          : m.content,
      })),
    [messages],
  );

  // Serialize current profiles into a compact payload for the backend
  const profilesPayload = useCallback(() =>
    profiles.map(p => ({
      id:          p.id,
      name:        p.name,
      email:       p.email,
      risk_level:  p.metrics.risk_level,
      unified_score: p.metrics.unified_score,
      gps_spoof:     p.metrics.gps_spoof,
      login_anomaly: p.metrics.login_anomaly,
      password_leak: p.metrics.password_leak,
      fraud_risk:    p.metrics.fraud_risk,
      breach_risk:   p.metrics.breach_risk,
    })),
    [profiles],
  );

  const sendQuery = useCallback(async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed || loading) return;

    const uid = `u-${Date.now()}`;
    const aid = `a-${Date.now()}`;

    setMessages(prev => [
      ...prev,
      { id: uid, role: "user",      content: trimmed, ts: new Date() },
      { id: aid, role: "assistant", content: "",       ts: new Date(), loading: true },
    ]);
    setInput("");
    setLoading(true);

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const result = await queryNL({
        query:    trimmed,
        history:  buildHistory(),
        profiles: profilesPayload(),
      });
      setMessages(prev => prev.map(m =>
        m.id === aid ? { ...m, content: result.answer, response: result, loading: false } : m,
      ));
    } catch (err: any) {
      setMessages(prev => prev.map(m =>
        m.id === aid ? { ...m, loading: false, error: err.message || "Request failed" } : m,
      ));
    } finally {
      setLoading(false);
    }
  }, [loading, buildHistory, profilesPayload]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery(input);
    }
  };

  const showSuggestions = messages.length <= 2;

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "calc(100vh - var(--nav-h) - 48px)",
    }}>

      {/* ── Header ───────────────────────────── */}
      <div style={{ marginBottom: 20, flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <h1 style={{ fontSize: 22, fontWeight: 900, letterSpacing: "-0.025em", color: "var(--text-primary)" }}>
            AI Query
          </h1>
          <span style={{
            fontSize: 9, padding: "3px 9px", borderRadius: 6,
            background: "rgba(139,92,246,0.12)", color: "var(--accent-purple)",
            border: "1px solid rgba(139,92,246,0.25)", fontWeight: 800,
            letterSpacing: "0.09em", textTransform: "uppercase",
          }}>Ollama · llama3.2</span>
          <span style={{
            fontSize: 9, padding: "3px 9px", borderRadius: 6,
            background: "rgba(0,255,136,0.08)", color: "var(--risk-minimal)",
            border: "1px solid rgba(0,255,136,0.2)", fontWeight: 800,
            letterSpacing: "0.09em", textTransform: "uppercase",
          }}>{profiles.length} profiles loaded</span>
        </div>
        <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
          Natural language interface to your live threat data · Runs entirely on-device
        </p>
      </div>

      {/* ── Chat messages ─────────────────────── */}
      <div style={{ flex: 1, overflowY: "auto", paddingRight: 4, scrollbarWidth: "thin" }}>
        {messages.map(msg => (
          msg.role === "user" ? (
            <div key={msg.id} style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
              <div style={{
                maxWidth: "70%", padding: "10px 16px",
                borderRadius: "16px 16px 4px 16px",
                background: "linear-gradient(135deg, #3d7fff, #2563eb)",
                color: "#fff", fontSize: 13, lineHeight: 1.55, fontWeight: 500,
                boxShadow: "0 4px 20px rgba(61,127,255,0.3)",
              }}>
                {msg.content}
              </div>
            </div>
          ) : (
            <AssistantBubble key={msg.id} msg={msg} />
          )
        ))}
        <div ref={bottomRef} />
      </div>

      {/* ── Suggestions ───────────────────────── */}
      {showSuggestions && (
        <div style={{ flexShrink: 0, paddingBottom: 12 }}>
          <div style={{ fontSize: 10, color: "var(--text-disabled)", textTransform: "uppercase", letterSpacing: "0.09em", fontWeight: 700, marginBottom: 8 }}>
            Try asking
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {SUGGESTED.map(s => (
              <button
                key={s}
                onClick={() => sendQuery(s)}
                disabled={loading}
                style={{
                  padding: "6px 12px", borderRadius: 8, fontSize: 11, fontWeight: 500,
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  color: "var(--text-secondary)", cursor: "pointer",
                  transition: "all 0.18s", whiteSpace: "nowrap",
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(61,127,255,0.35)";
                  (e.currentTarget as HTMLElement).style.color = "var(--accent)";
                  (e.currentTarget as HTMLElement).style.background = "rgba(61,127,255,0.07)";
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.08)";
                  (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)";
                  (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.03)";
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Input ─────────────────────────────── */}
      <div
        style={{
          flexShrink: 0,
          padding: "10px 14px",
          borderRadius: 14,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.09)",
          display: "flex", gap: 10, alignItems: "flex-end",
          transition: "border-color 0.2s",
          marginTop: 8,
        }}
        onFocusCapture={e  => (e.currentTarget as HTMLElement).style.borderColor = "rgba(61,127,255,0.4)"}
        onBlurCapture={e   => (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.09)"}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about risk scores, users, threats… (Enter to send)"
          rows={1}
          disabled={loading}
          style={{
            flex: 1, resize: "none", background: "transparent",
            border: "none", outline: "none",
            color: "var(--text-primary)", fontSize: 13, lineHeight: 1.55,
            maxHeight: 110, overflowY: "auto", fontFamily: "inherit",
          }}
          onInput={e => {
            const el = e.currentTarget;
            el.style.height = "auto";
            el.style.height = `${Math.min(el.scrollHeight, 110)}px`;
          }}
        />
        <button
          onClick={() => sendQuery(input)}
          disabled={loading || !input.trim()}
          style={{
            flexShrink: 0, width: 36, height: 36, borderRadius: 9, border: "none",
            background: loading || !input.trim()
              ? "rgba(61,127,255,0.15)"
              : "linear-gradient(135deg, #3d7fff, #2563eb)",
            color: loading || !input.trim() ? "rgba(255,255,255,0.25)" : "#fff",
            cursor: loading || !input.trim() ? "not-allowed" : "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            transition: "all 0.2s",
            boxShadow: !loading && input.trim() ? "0 0 20px rgba(61,127,255,0.45)" : "none",
          }}
        >
          {loading ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ animation: "nlSpin 0.9s linear infinite" }}>
              <path d="M21 12a9 9 0 11-6.219-8.56" strokeLinecap="round"/>
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>

      <style>{`
        @keyframes nlBounce {
          0%, 100% { transform: translateY(0); opacity: 0.35; }
          50%       { transform: translateY(-5px); opacity: 1; }
        }
        @keyframes nlSpin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
