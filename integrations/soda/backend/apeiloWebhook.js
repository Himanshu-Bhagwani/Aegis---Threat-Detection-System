/**
 * Apeilo webhook receiver for SODA (Express / Node)
 * =================================================
 * Copy to:  backend/routes/apeiloWebhook.js
 *
 * Apeilo POSTs here whenever a tracked user crosses the threat threshold. The
 * request is signed with SODA's webhook secret; we verify it before trusting
 * the payload, then react (notify the user, force re-auth, write an audit row).
 *
 * IMPORTANT — raw body:
 *   The signature is an HMAC over the EXACT request bytes, so this route uses
 *   its own express.raw() parser. Mount it BEFORE any global express.json()
 *   in server.js (see README), otherwise the body is already consumed and the
 *   signature check will fail.
 *
 * Env (backend): APEILO_WEBHOOK_SECRET=whsec_...
 */

const express = require("express");
const crypto = require("crypto");

const router = express.Router();

function verifySignature(rawBuf, signatureHeader, secret) {
  if (!signatureHeader || !secret) return false;
  const expected =
    "sha256=" + crypto.createHmac("sha256", secret).update(rawBuf).digest("hex");
  const a = Buffer.from(signatureHeader);
  const b = Buffer.from(expected);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

// express.raw gives us req.body as a Buffer — required for a correct HMAC.
router.post("/webhook", express.raw({ type: "*/*" }), (req, res) => {
  const secret = process.env.APEILO_WEBHOOK_SECRET || "";
  const rawBuf = Buffer.isBuffer(req.body) ? req.body : Buffer.from("");
  const signature = req.get("x-apeilo-signature");

  if (!verifySignature(rawBuf, signature, secret)) {
    return res.status(401).json({ error: "invalid signature" });
  }

  let threat;
  try {
    threat = JSON.parse(rawBuf.toString("utf8"));
  } catch {
    return res.status(400).json({ error: "invalid json" });
  }

  // ── React to the threat ─────────────────────────────
  console.warn(
    `[Apeilo] ${String(threat.risk_level).toUpperCase()} threat for user ${threat.user_id}: ` +
      `${(threat.primary_threats || []).join(", ")} (score ${threat.risk_score})`,
  );

  // TODO for SODA — do something useful, e.g.:
  //   await notifyUser(threat.user_id, threat);          // email / in-app banner
  //   if (threat.risk_level === "critical") await forceReauth(threat.user_id);
  //   await db.query("INSERT INTO security_events ...", [threat.user_id, ...]);
  //   io.emit("security_alert", threat);                  // socket.io live ops view

  res.json({ received: true });
});

module.exports = router;
