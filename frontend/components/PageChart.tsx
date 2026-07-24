"use client";
/**
 * PageChart — shared Recharts visualisation component used across all dashboard pages.
 * Imported with dynamic({ ssr: false }) in every page that uses it.
 */

import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

// ── Design tokens ─────────────────────────────────────────
const C = {
  gps:      "#3d7fff",
  login:    "#8b5cf6",
  password: "#06b6d4",
  fraud:    "#ef4444",
  breach:   "#f97316",
  unified:  "#f59e0b",
  green:    "#00ff88",
  grid:     "rgba(255,255,255,0.05)",
};

const RISK_COLOR = (v: number) => {
  if (v >= 75) return "#ef4444";
  if (v >= 50) return "#f97316";
  if (v >= 25) return "#f59e0b";
  if (v >= 10) return "#84cc16";
  return "#00ff88";
};

const TT = {
  contentStyle: {
    background:   "#101928",
    border:       "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10,
    fontSize:     12,
    color:        "#e8edf8",
  },
  cursor: { fill: "rgba(61,127,255,0.05)" },
};

// ── Section label ─────────────────────────────────────────
export function ChartLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, color: "var(--text-muted)",
      textTransform: "uppercase", letterSpacing: "0.10em", marginBottom: 14,
    }}>
      {children}
    </div>
  );
}

// ── 1. Profile comparison — grouped bar ───────────────────
export type ProfileRow = {
  name: string;
  // No Password series — password exposure is reported under Breach.
  GPS: number; Login: number; Fraud: number; Breach: number;
};

export function ProfileCompareChart({ data }: { data: ProfileRow[] }) {
  const rows = data.slice(0, 8);
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 4, right: 8, left: 0, bottom: 4 }} barSize={10} barCategoryGap="25%">
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
          <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
          <Tooltip {...TT} formatter={(v: number) => [`${v.toFixed(1)}%`]} />
          <Legend wrapperStyle={{ fontSize: 10, color: "var(--text-muted)" }} />
          <Bar dataKey="GPS"      fill={C.gps}      radius={[3,3,0,0]} />
          <Bar dataKey="Login"    fill={C.login}    radius={[3,3,0,0]} />
          <Bar dataKey="Fraud"    fill={C.fraud}    radius={[3,3,0,0]} />
          <Bar dataKey="Breach"   fill={C.breach}   radius={[3,3,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 2. Radar chart — single profile modules ───────────────
export type RadarRow = { module: string; value: number; fullMark: number };

export function ModuleRadarChart({ data }: { data: RadarRow[] }) {
  return (
    <div style={{ width: "100%", height: 240 }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 24 }}>
          <PolarGrid stroke="rgba(255,255,255,0.08)" />
          <PolarAngleAxis dataKey="module" tick={{ fill: "var(--text-muted)", fontSize: 10 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar dataKey="value" stroke={C.gps} fill={C.gps} fillOpacity={0.18} strokeWidth={2} dot={{ r: 3, fill: C.gps }} />
          <Tooltip {...TT} formatter={(v: number) => [`${v.toFixed(1)}%`]} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 3. Hour-of-day risk distribution bar ──────────────────
const HOUR_BASE = [
  0.72, 0.81, 0.90, 0.95, 0.88, 0.60, // 0-5 (night → dangerous)
  0.28, 0.18, 0.14, 0.12, 0.13, 0.15, // 6-11 (morning → safe)
  0.16, 0.17, 0.18, 0.19, 0.21, 0.24, // 12-17 (business hours)
  0.30, 0.35, 0.42, 0.52, 0.63, 0.70, // 18-23 (evening → rising)
];
export const HOUR_DATA = HOUR_BASE.map((v, h) => ({
  hour: `${String(h).padStart(2,"0")}:00`,
  risk: Math.round(v * 100),
}));

// ── Real transaction amounts (INR) ────────────────────────
export interface TxnPoint {
  timestamp: string; amount: number; risk: number; is_risky: boolean; hour: number | null;
}

function inrShort(n: number): string {
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)}Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)}L`;
  if (n >= 1e3) return `₹${(n / 1e3).toFixed(1)}k`;
  return `₹${Math.round(n)}`;
}

function TxnTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: "rgba(10,14,25,0.96)", border: "1px solid rgba(255,255,255,0.12)",
      borderRadius: 8, padding: "9px 12px", fontSize: 11, color: "var(--text-secondary)",
      boxShadow: "0 8px 24px rgba(0,0,0,0.45)",
    }}>
      <div style={{ fontWeight: 800, color: "var(--text-primary)", marginBottom: 4, fontFamily: "var(--font-mono)" }}>
        {d.full}
      </div>
      <div>Fraud risk: <b style={{ color: RISK_COLOR(d.risk) }}>{d.risk}%</b></div>
      <div style={{ marginTop: 3, color: "var(--text-muted)" }}>{d.when}</div>
      {d.is_risky && <div style={{ marginTop: 3, color: "#ef4444" }}>Flagged — well above this user's normal spend</div>}
    </div>
  );
}

/** Actual transaction amounts (INR) over time, coloured by fraud risk. */
export function TransactionAmountChart({ txns }: { txns: TxnPoint[] }) {
  // Oldest → newest so the timeline reads left to right.
  const data = [...txns].reverse().map((t, i) => {
    const d = t.timestamp ? new Date(t.timestamp) : null;
    return {
      idx: String(i + 1),
      amount: t.amount,
      risk: Math.round(t.risk * 100),
      is_risky: t.is_risky,
      full: `₹${t.amount.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`,
      when: d ? d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : `${t.hour}:00`,
    };
  });
  if (!data.length) return null;

  return (
    <div style={{ width: "100%", height: 200 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 6, right: 8, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
          <XAxis dataKey="when" tick={{ fill: "var(--text-muted)", fontSize: 9 }} axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={24} />
          <YAxis
            tick={{ fill: "var(--text-muted)", fontSize: 9 }}
            axisLine={false} tickLine={false} width={58}
            tickFormatter={(v: number) => inrShort(v)}
          />
          <Tooltip content={<TxnTooltip />} cursor={{ fill: "rgba(61,127,255,0.06)" }} />
          <Bar dataKey="amount" radius={[3, 3, 0, 0]} maxBarSize={44}>
            {data.map((d, i) => (
              <Cell key={i} fill={RISK_COLOR(d.risk)} fillOpacity={d.is_risky ? 1 : 0.72} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export interface HourBucket {
  hour: number; label: string; count: number; failed_count?: number;
  baseline: number; risk: number; familiarity: number; is_usual: boolean;
}

/** Tooltip that explains the adapted risk in terms of the user's own history. */
function HourTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: "rgba(10,14,25,0.96)", border: "1px solid rgba(255,255,255,0.12)",
      borderRadius: 8, padding: "9px 12px", fontSize: 11, color: "var(--text-secondary)",
      boxShadow: "0 8px 24px rgba(0,0,0,0.45)", maxWidth: 250,
    }}>
      <div style={{ fontWeight: 800, color: "var(--text-primary)", marginBottom: 4 }}>{d.hour}</div>
      <div>Anomaly Risk: <b style={{ color: d.count > 0 ? "#3d7fff" : RISK_COLOR(d.risk) }}>{d.risk}%</b></div>
      <div style={{ marginTop: 4, color: "var(--text-muted)" }}>
        {d.count > 0
          ? `You have logged in at this hour ${d.count} time${d.count === 1 ? "" : "s"}`
          : "No successful logins at this hour"}
      </div>
      {d.failed > 0 && (
        <div style={{ marginTop: 3, color: "#ef4444" }}>
          {d.failed} failed attempt{d.failed === 1 ? "" : "s"} — does not lower risk
        </div>
      )}
      {d.count > 0 && d.risk < d.baseline ? (
        <div style={{ marginTop: 4, color: "#3d7fff", fontSize: 10 }}>
          ↓ reduced from {d.baseline}% — normal for you
        </div>
      ) : d.count === 0 ? (
        <div style={{ marginTop: 4, color: "var(--text-disabled)", fontSize: 10 }}>
          Generic baseline — not based on your history
        </div>
      ) : null}
    </div>
  );
}

export function HourRiskChart({ highlightHour, hours }: { highlightHour?: number; hours?: HourBucket[] }) {
  // Use the user's personalised curve when available, else the generic prior.
  const data = hours?.length
    ? hours.map(h => ({
        hour: h.label,
        risk: Math.round(h.risk * 100),
        baseline: Math.round(h.baseline * 100),
        count: h.count,
        failed: h.failed_count ?? 0,
        usual: h.is_usual,
      }))
    : HOUR_DATA.map(d => ({ ...d, baseline: d.risk, count: 0, failed: 0, usual: false }));

  return (
    <div style={{ width: "100%", height: 180 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 4 }} barSize={10}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
          <XAxis dataKey="hour" tick={{ fill: "var(--text-muted)", fontSize: 9 }} interval={3} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} width={30} />
          <Tooltip content={<HourTooltip />} cursor={{ fill: "rgba(61,127,255,0.06)" }} />
          {highlightHour !== undefined && (
            <ReferenceLine x={`${String(highlightHour).padStart(2,"0")}:00`} stroke="#3d7fff" strokeDasharray="4 3" label={{ value: "Now", fill: "#3d7fff", fontSize: 9 }} />
          )}
          <Bar dataKey="risk" radius={[3,3,0,0]} maxBarSize={14}>
            {data.map((d, i) => (
              // Blue = hours you actually sign in at (personalised). Everything
              // else keeps the risk palette, so a low *generic* baseline is
              // never mistaken for "this is normal for you".
              <Cell
                key={i}
                fill={d.count > 0 ? "#3d7fff" : RISK_COLOR(d.risk)}
                fillOpacity={highlightHour !== undefined && i === highlightHour ? 1 : 0.75}
                stroke={d.usual ? "#00ff88" : undefined}
                strokeWidth={d.usual ? 1.5 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 4. Model scores bar chart ─────────────────────────────
export function ModelScoresChart({ scores }: { scores: Record<string, number> }) {
  const data = Object.entries(scores)
    .filter(([, v]) => v >= 0)
    .map(([k, v]) => ({ model: k.replace(/_/g, " "), score: Math.round(v * 100) }));
  if (!data.length) return null;
  return (
    <div style={{ width: "100%", height: 180 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
          <YAxis type="category" dataKey="model" tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} width={110} />
          <Tooltip {...TT} formatter={(v: number) => [`${v}%`]} cursor={{ fill: "rgba(61,127,255,0.06)" }} />
          <ReferenceLine x={50} stroke="rgba(239,68,68,0.4)" strokeDasharray="4 3" />
          <Bar dataKey="score" radius={[0,3,3,0]} maxBarSize={18}>
            {data.map((d, i) => <Cell key={i} fill={RISK_COLOR(d.score)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 5. Amount vs fraud risk reference line ────────────────
const AMT_DATA = [
  { amount: "$0",   risk: 5 }, { amount: "$200",  risk: 8 },
  { amount: "$500",  risk: 14 }, { amount: "$1k",   risk: 28 },
  { amount: "$2k",  risk: 40 }, { amount: "$5k",   risk: 62 },
  { amount: "$10k", risk: 85 }, { amount: "$20k",  risk: 96 },
];

export function AmountRiskChart({ currentAmount }: { currentAmount?: number }) {
  const nearestIdx = currentAmount
    ? AMT_DATA.reduce((best, d, i) => {
        const vals = [0,200,500,1000,2000,5000,10000,20000];
        return Math.abs(vals[i] - currentAmount) < Math.abs(vals[best] - currentAmount) ? i : best;
      }, 0)
    : -1;

  return (
    <div style={{ width: "100%", height: 180 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={AMT_DATA} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} />
          <XAxis dataKey="amount" tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} width={32} />
          <Tooltip {...TT} formatter={(v: number) => [`${v}%`, "Fraud Risk"]} />
          <ReferenceLine y={60} stroke="rgba(239,68,68,0.4)" strokeDasharray="4 3" label={{ value: "Threshold 60%", fill: "rgba(239,68,68,0.6)", fontSize: 9, position: "right" }} />
          <Line type="monotone" dataKey="risk" stroke={C.fraud} strokeWidth={2.5} dot={(props: any) => {
            const { cx, cy, index } = props;
            return <circle key={index} cx={cx} cy={cy} r={index === nearestIdx ? 6 : 3} fill={index === nearestIdx ? "#fff" : C.fraud} stroke={C.fraud} strokeWidth={2} />;
          }} activeDot={{ r: 5, fill: "#fff", stroke: C.fraud, strokeWidth: 2 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 6. Risk factor horizontal bar (signal contributions) ──
export function SignalContribChart({ factors }: { factors: { name: string; score: number; color: string }[] }) {
  const data = factors.map(f => ({ name: f.name, score: Math.round(f.score * 100) }));
  return (
    <div style={{ width: "100%", height: Math.max(140, data.length * 36 + 20) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
          <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} width={130} />
          <Tooltip {...TT} formatter={(v: number) => [`${v}%`]} cursor={{ fill: "rgba(61,127,255,0.06)" }} />
          <Bar dataKey="score" radius={[0,3,3,0]} maxBarSize={16}>
            {data.map((d, i) => <Cell key={i} fill={factors[i]?.color || RISK_COLOR(d.score)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 7. Alert severity donut ───────────────────────────────
export function AlertSeverityChart({ critical, high }: { critical: number; high: number }) {
  const data = [
    { name: "Critical", value: critical },
    { name: "High",     value: high },
    { name: "Clear",    value: Math.max(0, 10 - critical - high) },
  ].filter(d => d.value > 0);
  const COLS = ["#ef4444", "#f97316", "#00ff88"];
  return (
    <div style={{ width: "100%", height: 200 }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={4}
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            labelLine={{ stroke: "rgba(255,255,255,0.15)", strokeWidth: 1 }}
          >
            {data.map((_, i) => <Cell key={i} fill={COLS[i % COLS.length]} />)}
          </Pie>
          <Tooltip contentStyle={TT.contentStyle} formatter={(v: number) => [v, "alerts"]} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 8. Score vs threshold grouped bar ─────────────────────
export type ThreshRow = { name: string; current: number; threshold: number };

export function ThresholdChart({ data }: { data: ThreshRow[] }) {
  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 24 }} barSize={14}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
          <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} angle={-25} textAnchor="end" interval={0} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} width={30} />
          <Tooltip {...TT} formatter={(v: number) => [`${v.toFixed(1)}%`]} />
          <Legend wrapperStyle={{ fontSize: 10, color: "var(--text-muted)", paddingTop: 4 }} />
          <Bar dataKey="current"   name="Current Score" radius={[3,3,0,0]}>
            {data.map((d, i) => <Cell key={i} fill={RISK_COLOR(d.current)} />)}
          </Bar>
          <Bar dataKey="threshold" name="Threshold"    fill="rgba(255,255,255,0.12)" radius={[3,3,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 9. Password strength breakdown ────────────────────────
export type PwStrength = { entropy: number; length_score: number; diversity: number; uniqueness: number };

export function PasswordStrengthChart({ data }: { data: PwStrength }) {
  const bars = [
    { name: "Entropy",    score: Math.min(100, Math.round((data.entropy / 80) * 100)),          color: C.password },
    { name: "Length",     score: Math.round(data.length_score * 100),                            color: C.login },
    { name: "Diversity",  score: Math.round(data.diversity * 100),                               color: C.gps },
    { name: "Uniqueness", score: Math.round(data.uniqueness * 100),                              color: C.unified },
  ];
  return (
    <div style={{ width: "100%", height: 180 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
          <YAxis type="category" dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} width={80} />
          <Tooltip {...TT} formatter={(v: number) => [`${v}%`]} cursor={{ fill: "rgba(61,127,255,0.06)" }} />
          <ReferenceLine x={60} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 3" />
          <Bar dataKey="score" radius={[0,3,3,0]} maxBarSize={18}>
            {bars.map((b, i) => <Cell key={i} fill={b.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 10. Single-profile module bar ─────────────────────────
export type MiniBar = { module: string; score: number; color: string };

export function MiniModuleBar({ data }: { data: MiniBar[] }) {
  return (
    <div style={{ width: "100%", height: 160 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 4 }} barSize={20}>
          <CartesianGrid strokeDasharray="3 3" stroke={C.grid} vertical={false} />
          <XAxis dataKey="module" tick={{ fill: "var(--text-muted)", fontSize: 9 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} tick={{ fill: "var(--text-muted)", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} width={28} />
          <Tooltip {...TT} formatter={(v: number) => [`${v.toFixed(1)}%`]} />
          <Bar dataKey="score" radius={[4,4,0,0]}>
            {data.map((d, i) => <Cell key={i} fill={d.color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
