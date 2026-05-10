"use client";

/**
 * QueryChart — Recharts visualisation renderer for AI Query results.
 * Extracted into its own file so the dashboard page can import it with
 * `dynamic(..., { ssr: false })`, which is required in Next.js App Router
 * because Recharts relies on DOM measurements unavailable during SSR.
 */

import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import type { NLQueryResponse } from "@/lib/api";

const COLORS = [
  "#3d7fff","#8b5cf6","#06b6d4","#ef4444",
  "#f97316","#f59e0b","#00ff88","#84cc16","#ec4899",
];

const RISK_PALETTE: Record<string, string> = {
  minimal: "#00ff88",
  low:     "#84cc16",
  medium:  "#f59e0b",
  high:    "#f97316",
  critical:"#ef4444",
};

function riskCellColor(val: string): string {
  const v = val?.toLowerCase();
  if (v === "critical" || v === "yes") return "#ef4444";
  if (v === "high")    return "#f97316";
  if (v === "medium")  return "#f59e0b";
  if (v === "low")     return "#84cc16";
  return "var(--text-primary)";
}

// ── Tooltip style shared by all charts ──────────────────
const tooltipStyle = {
  contentStyle: {
    background:   "var(--bg-overlay)",
    border:       "1px solid var(--glass-border)",
    borderRadius: 10,
    fontSize:     12,
    color:        "var(--text-primary)",
  },
  cursor: { fill: "rgba(61,127,255,0.06)" },
};

// ── Bar chart ────────────────────────────────────────────
function BarViz({ chart_data, chart_config }: { chart_data: any[]; chart_config: any }) {
  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chart_data} margin={{ top: 8, right: 16, left: 0, bottom: 70 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey={chart_config.x_key}
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            angle={-35}
            textAnchor="end"
            interval={0}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            domain={chart_config.y_domain || ["auto", "auto"]}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${v}${chart_config.y_label?.includes("%") ? "%" : ""}`}
          />
          <Tooltip {...tooltipStyle} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "var(--text-muted)", paddingTop: 8 }}
          />
          {(chart_config.bars as any[]).map((b: any) => (
            <Bar key={b.key} dataKey={b.key} name={b.name} fill={b.color || "#3d7fff"} radius={[4, 4, 0, 0]} maxBarSize={48}>
              {chart_data.map((entry: any, i: number) => (
                <Cell
                  key={`cell-${i}`}
                  fill={
                    entry.fill ||
                    RISK_PALETTE[entry.risk_level?.toLowerCase()] ||
                    b.color ||
                    COLORS[i % COLORS.length]
                  }
                />
              ))}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Line chart ───────────────────────────────────────────
function LineViz({ chart_data, chart_config }: { chart_data: any[]; chart_config: any }) {
  const tickInterval = Math.max(1, Math.ceil(chart_data.length / 10));
  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chart_data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey={chart_config.x_key}
            tick={{ fill: "var(--text-muted)", fontSize: 10 }}
            interval={tickInterval - 1}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            domain={chart_config.y_domain || ["auto", "auto"]}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip {...tooltipStyle} cursor={{ stroke: "rgba(255,255,255,0.1)" }} />
          <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-muted)" }} />
          {(chart_config.lines as any[]).map((l: any) => (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              stroke={l.color || "#3d7fff"}
              strokeWidth={l.dashed ? 1.5 : 2}
              strokeDasharray={l.dashed ? "6 4" : undefined}
              dot={false}
              activeDot={{ r: 5, strokeWidth: 0, fill: l.color || "#3d7fff" }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Pie chart ────────────────────────────────────────────
function PieViz({ chart_data, chart_config }: { chart_data: any[]; chart_config: any }) {
  const colors = chart_config?.colors || COLORS;
  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chart_data}
            dataKey={chart_config?.value_key || "value"}
            nameKey={chart_config?.name_key || "name"}
            cx="50%" cy="50%"
            outerRadius={110}
            innerRadius={40}
            paddingAngle={3}
            label={({ name, percent }: { name: string; percent: number }) =>
              `${name} ${(percent * 100).toFixed(0)}%`
            }
            labelLine={{ stroke: "rgba(255,255,255,0.15)", strokeWidth: 1 }}
          >
            {chart_data.map((_: any, i: number) => (
              <Cell key={`pie-${i}`} fill={colors[i % colors.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={tooltipStyle.contentStyle}
            formatter={(value: number, name: string) => [`${value}`, name]}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-muted)" }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Metric cards ─────────────────────────────────────────
function MetricViz({ chart_data }: { chart_data: any[] }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 12 }}>
      {chart_data.map((m: any, i: number) => (
        <div key={i} style={{
          padding: "18px 20px", borderRadius: 12,
          background: `${COLORS[i % COLORS.length]}10`,
          border: `1px solid ${COLORS[i % COLORS.length]}25`,
        }}>
          <div style={{
            fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase",
            letterSpacing: "0.08em", fontWeight: 700, marginBottom: 10,
          }}>
            {m.label}
          </div>
          <div style={{
            fontSize: 30, fontWeight: 900, fontFamily: "var(--font-mono)",
            color: COLORS[i % COLORS.length], letterSpacing: "-0.03em",
          }}>
            {typeof m.value === "number" ? m.value.toLocaleString() : m.value}
            {m.unit && (
              <span style={{ fontSize: 14, marginLeft: 5, color: "var(--text-muted)", fontWeight: 500 }}>
                {m.unit}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Data table ───────────────────────────────────────────
function TableViz({ table_data, table_headers }: { table_data: Record<string, string>[]; table_headers: string[] }) {
  return (
    <div style={{ overflowX: "auto", borderRadius: 10, border: "1px solid rgba(255,255,255,0.07)" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            {table_headers.map(h => (
              <th key={h} style={{
                padding: "10px 14px", textAlign: "left",
                fontSize: 10, fontWeight: 700, letterSpacing: "0.08em",
                textTransform: "uppercase", color: "var(--text-muted)",
                borderBottom: "1px solid rgba(255,255,255,0.07)",
                background: "rgba(255,255,255,0.03)",
                whiteSpace: "nowrap",
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table_data.map((row, i) => (
            <tr
              key={i}
              style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", transition: "background 0.15s" }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.025)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              {table_headers.map(h => (
                <td key={h} style={{
                  padding: "9px 14px",
                  color: ["Risk Level", "VPN", "Emulator", "Rooted"].includes(h)
                    ? riskCellColor(row[h])
                    : "var(--text-secondary)",
                  fontFamily: ["Risk Score", "Device ID", "Score", "Unified Score"].includes(h)
                    ? "var(--font-mono)"
                    : "inherit",
                  fontWeight: h.includes("Score") ? 700 : 400,
                  whiteSpace: "nowrap",
                }}>{row[h]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main export ──────────────────────────────────────────
export default function QueryChart({ response }: { response: NLQueryResponse }) {
  const { chart_type, chart_data, chart_config, table_data, table_headers } = response;

  if (chart_type === "bar" && chart_config?.bars && chart_data?.length)
    return <BarViz chart_data={chart_data} chart_config={chart_config} />;

  if (chart_type === "line" && chart_config?.lines && chart_data?.length)
    return <LineViz chart_data={chart_data} chart_config={chart_config} />;

  if (chart_type === "pie" && chart_data?.length)
    return <PieViz chart_data={chart_data} chart_config={chart_config} />;

  if (chart_type === "metric" && chart_data?.length)
    return <MetricViz chart_data={chart_data} />;

  if (chart_type === "table" && table_data?.length && table_headers?.length)
    return <TableViz table_data={table_data} table_headers={table_headers} />;

  return (
    <div style={{
      padding: "20px", textAlign: "center",
      color: "var(--text-muted)", fontSize: 12,
      border: "1px dashed rgba(255,255,255,0.07)", borderRadius: 10,
    }}>
      No visualisation data returned for this query.
    </div>
  );
}
