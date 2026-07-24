"use client";

/**
 * ChallengeCenter — "was this you?" prompts, answered inside Apeilo.
 *
 * The connected app (SODA) only shows a passive alert toast; verification
 * happens here, where the account is actually monitored. Polls for queued
 * challenges and shows one at a time. Answering "No" escalates to a confirmed
 * incident and offers to lock the account down in the connected app.
 */

import { useCallback, useEffect, useState } from "react";
import { getChallenges, verifyActivity, lockdownAccount, Challenge, riskColor } from "@/lib/api";

const overlay: React.CSSProperties = {
  position: "fixed", inset: 0, zIndex: 10000,
  background: "rgba(3,7,18,0.72)", backdropFilter: "blur(3px)",
  display: "flex", alignItems: "center", justifyContent: "center", padding: 20,
};
const card: React.CSSProperties = {
  width: "100%", maxWidth: 470, background: "var(--bg-panel, #0f172a)",
  color: "var(--text-primary)", border: "1px solid rgba(239,68,68,0.35)",
  borderRadius: 16, padding: "24px 26px", boxShadow: "0 24px 60px rgba(0,0,0,0.6)",
};
const btnBase: React.CSSProperties = {
  flex: 1, padding: "11px 16px", borderRadius: 9, cursor: "pointer",
  fontSize: 13, fontWeight: 700,
};

function hourLabel(h: any): string | null {
  if (h === undefined || h === null) return null;
  const n = Number(h);
  if (Number.isNaN(n)) return null;
  return `${n % 12 || 12}:00 ${n < 12 ? "AM" : "PM"}`;
}

const inr = (n: number) =>
  n >= 1e7 ? `₹${(n / 1e7).toFixed(2)}Cr`
  : n >= 1e5 ? `₹${(n / 1e5).toFixed(2)}L`
  : `₹${n.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;

export default function ChallengeCenter() {
  const [queue,   setQueue]   = useState<Challenge[]>([]);
  const [current, setCurrent] = useState<Challenge | null>(null);
  const [denied,  setDenied]  = useState<Challenge | null>(null);
  const [busy,    setBusy]    = useState(false);
  const [lockMsg, setLockMsg] = useState<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const { challenges } = await getChallenges();
      setQueue(challenges || []);
    } catch { /* backend unreachable — try again next tick */ }
  }, []);

  useEffect(() => {
    poll();
    const t = setInterval(poll, 7000);
    return () => clearInterval(t);
  }, [poll]);

  // Show one at a time; don't interrupt an open dialog.
  useEffect(() => {
    if (!current && !denied && queue.length > 0) setCurrent(queue[0]);
  }, [queue, current, denied]);

  async function answer(confirmed: boolean) {
    if (!current) return;
    setBusy(true);
    try {
      await verifyActivity({
        user_id:    current.user_id,
        activity:   current.activity,
        confirmed,
        risk_score: current.risk_score,
        detail:     current.detail,
        alert_id:   current.alert_id,
      });
      setQueue(q => q.filter(x => x.alert_id !== current.alert_id));
      if (confirmed) setCurrent(null);
      else { setDenied(current); setCurrent(null); }   // offer lockdown next
    } catch {
      setCurrent(null);
    } finally {
      setBusy(false);
    }
  }

  async function lockdown(minutes: number) {
    if (!denied) return;
    setBusy(true);
    try {
      const res: any = await lockdownAccount(denied.user_id, minutes);
      setLockMsg(
        res?.webhook_delivered
          ? `Login access blocked for ${minutes} minutes in the connected app.`
          : `Lock recorded (${minutes} min), but the app's webhook could not be reached — secure the account manually.`,
      );
    } catch {
      setLockMsg("Could not reach the connected app — secure the account manually.");
    } finally {
      setBusy(false);
    }
  }

  // ── Verification prompt ──
  if (current) {
    const d      = current.detail || {};
    const isTxn  = current.activity === "transaction";
    const amount = typeof d.amount === "number" ? d.amount : null;
    const soft   = current.tone === "confirm";

    const facts: [string, string][] = [];
    facts.push(["Account", current.user_id]);
    if (isTxn) {
      if (amount !== null) facts.push(["Amount", inr(amount)]);
      if (d.amount_ratio) facts.push(["vs their normal", `${Number(d.amount_ratio).toLocaleString("en-IN", { maximumFractionDigits: 1 })}× larger`]);
      if (d.source) facts.push(["Source", d.source === "statement_upload" ? "Uploaded statement" : "Added in app"]);
    } else {
      if (d.failed_attempts) facts.push(["Failed attempts", `${d.failed_attempts} in 10 minutes`]);
    }
    if (hourLabel(d.hour)) facts.push(["Time", `around ${hourLabel(d.hour)}`]);
    if (current.timestamp) {
      facts.push(["Detected", new Date(current.timestamp).toLocaleString(undefined,
        { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })]);
    }

    return (
      <div style={overlay} role="dialog" aria-modal="true">
        <div style={card}>
          <div style={{ fontSize: 17, fontWeight: 800, marginBottom: 8, color: "#fecaca" }}>
            {soft ? "Confirming — was this you?" : "⚠ Was this you?"}
          </div>
          <p style={{ fontSize: 13, lineHeight: 1.65, color: "var(--text-secondary)", marginBottom: 12 }}>
            {isTxn
              ? <>A transaction{amount !== null ? <> of <b style={{ color: "var(--text-primary)" }}>{inr(amount)}</b></> : ""} on this account
                  {soft ? " is larger than usual." : " is well outside this account's normal pattern."}</>
              : current.activity === "login_stepup"
              ? <>A sign-in <b style={{ color: "var(--text-primary)" }}>succeeded right after {d.failed_attempts || "several"} failed attempts</b>.
                  Access is on hold until you confirm it was you.</>
              : current.activity === "login_failed"
              ? <><b style={{ color: "var(--text-primary)" }}>{d.failed_attempts || "Several"} failed sign-in attempts</b> were made on this account in a short window.</>
              : <>A sign-in on this account happened at an unusual time or place.</>}
          </p>

          <div style={{
            marginBottom: 12, padding: "10px 12px", borderRadius: 8,
            background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.10)",
          }}>
            {facts.map(([k, v], i) => (
              <div key={i} style={{
                display: "flex", justifyContent: "space-between", gap: 12,
                fontSize: 12.5, padding: "3px 0", color: "var(--text-secondary)",
              }}>
                <span style={{ color: "var(--text-muted)" }}>{k}</span>
                <span style={{ fontWeight: 700, color: "var(--text-primary)", textAlign: "right" }}>{v}</span>
              </div>
            ))}
          </div>

          <div style={{
            fontSize: 12, color: "var(--text-secondary)", marginBottom: 18,
            padding: "9px 12px", borderRadius: 8, background: "rgba(239,68,68,0.08)",
            border: "1px solid rgba(239,68,68,0.2)",
          }}>
            Risk score{" "}
            <b style={{ color: riskColor(current.risk_score) }}>
              {(current.risk_score * 100).toFixed(0)}%
            </b>. Answering “no” raises a confirmed security incident.
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button
              disabled={busy}
              onClick={() => answer(true)}
              style={{ ...btnBase, background: "transparent", border: "1px solid rgba(255,255,255,0.14)", color: "var(--text-secondary)" }}
            >
              Yes, that was me
            </button>
            <button
              disabled={busy}
              onClick={() => answer(false)}
              style={{ ...btnBase, background: "#b91c1c", border: "none", color: "#fff" }}
            >
              No — this wasn&apos;t me
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Follow-up after a denial: secure the account ──
  if (denied) {
    return (
      <div style={overlay} role="dialog" aria-modal="true">
        <div style={card}>
          <div style={{ fontSize: 17, fontWeight: 800, marginBottom: 8, color: "#fecaca" }}>
            Secure this account now
          </div>
          <p style={{ fontSize: 13, lineHeight: 1.65, color: "var(--text-secondary)", marginBottom: 14 }}>
            <b style={{ color: "var(--text-primary)" }}>{denied.user_id}</b> reported
            activity they didn&apos;t perform. Block login access to this account in
            the connected app while it&apos;s investigated. Existing sessions are
            ended too.
          </p>

          {lockMsg ? (
            <>
              <div style={{
                fontSize: 12.5, marginBottom: 14, padding: "10px 12px", borderRadius: 8,
                background: "rgba(0,255,136,0.06)", border: "1px solid rgba(0,255,136,0.2)",
                color: "var(--text-secondary)",
              }}>
                {lockMsg}
              </div>
              <button
                onClick={() => { setDenied(null); setLockMsg(null); }}
                style={{ ...btnBase, width: "100%", background: "transparent", border: "1px solid rgba(255,255,255,0.14)", color: "var(--text-muted)" }}
              >
                Done
              </button>
            </>
          ) : (
            <>
              <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700, marginBottom: 8 }}>
                Block login for
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
                {[10, 15, 30, 60].map((m) => (
                  <button
                    key={m}
                    disabled={busy}
                    onClick={() => lockdown(m)}
                    style={{ ...btnBase, background: "#b91c1c", border: "none", color: "#fff" }}
                  >
                    {m} minutes
                  </button>
                ))}
              </div>
              <button
                disabled={busy}
                onClick={() => { setDenied(null); setLockMsg(null); }}
                style={{ ...btnBase, width: "100%", background: "transparent", border: "1px solid rgba(255,255,255,0.14)", color: "var(--text-muted)" }}
              >
                Skip — don&apos;t lock
              </button>
            </>
          )}
        </div>
      </div>
    );
  }

  return null;
}
