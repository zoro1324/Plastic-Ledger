import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "react-router-dom";
import {
  BarChart3,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  ArrowRight,
  Activity,
  TrendingUp,
  MapPin,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts";

/* ── Types ─────────────────────────────────────────────────── */
interface PipelineRun {
  id: string;
  status: string;
  bbox: string;
  target_date: string;
  cloud_cover: number;
  backtrack_days: number;
  created_at: string;
  completed_at: string | null;
  summary: Record<string, unknown> | null;
  error_message: string | null;
}

const API_BASE = "http://127.0.0.1:8000/api";

/* ── Status helpers ────────────────────────────────────────── */
const statusConfig: Record<string, { color: string; Icon: typeof CheckCircle2 }> = {
  COMPLETED: { color: "text-emerald-400", Icon: CheckCircle2 },
  FAILED: { color: "text-red-400", Icon: XCircle },
  RUNNING: { color: "text-blue-400", Icon: Loader2 },
  PENDING: { color: "text-yellow-400", Icon: Clock },
};

function StatusChip({ status }: { status: string }) {
  const cfg = statusConfig[status] ?? { color: "text-muted-foreground", Icon: Clock };
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium ${cfg.color}`}>
      <cfg.Icon className={`w-3.5 h-3.5 ${status === "RUNNING" ? "animate-spin" : ""}`} />
      {status}
    </span>
  );
}

/* ── Stat Card ─────────────────────────────────────────────── */
function StatCard({
  label,
  value,
  icon: Icon,
  accent = "text-primary",
  delay = 0,
}: {
  label: string;
  value: string | number;
  icon: typeof Activity;
  accent?: string;
  delay?: number;
}) {
  return (
    <motion.div
      className="glass rounded-xl p-5 flex items-center gap-4"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
    >
      <div className={`w-10 h-10 rounded-lg bg-muted flex items-center justify-center ${accent}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-2xl font-heading font-bold text-foreground">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </motion.div>
  );
}

/* ── Pie chart colors ──────────────────────────────────────── */
const PIE_COLORS = ["#34d399", "#f87171", "#60a5fa", "#fbbf24"];

/* ── Main Component ────────────────────────────────────────── */
export default function Analytics() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/pipeline/runs/`);
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data: PipelineRun[] = await res.json();
      setRuns(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to fetch runs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  /* ── Derived stats ── */
  const totalRuns = runs.length;
  const completed = runs.filter((r) => r.status === "COMPLETED").length;
  const failed = runs.filter((r) => r.status === "FAILED").length;
  const running = runs.filter((r) => r.status === "RUNNING" || r.status === "PENDING").length;
  const successRate = totalRuns > 0 ? Math.round((completed / totalRuns) * 100) : 0;

  /* status distribution for pie chart */
  const statusDistribution = [
    { name: "Completed", value: completed },
    { name: "Failed", value: failed },
    { name: "Running", value: running },
    { name: "Pending", value: runs.filter((r) => r.status === "PENDING").length },
  ].filter((d) => d.value > 0);

  /* runs per day for area chart */
  const runsPerDay: Record<string, number> = {};
  runs.forEach((r) => {
    const day = r.created_at?.split("T")[0];
    if (day) runsPerDay[day] = (runsPerDay[day] || 0) + 1;
  });
  const runsPerDayData = Object.entries(runsPerDay)
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-14)
    .map(([date, count]) => ({ date: date.slice(5), runs: count }));

  /* cloud cover distribution for bar chart */
  const cloudBuckets = [
    { range: "0-20%", count: 0 },
    { range: "21-40%", count: 0 },
    { range: "41-60%", count: 0 },
    { range: "61-80%", count: 0 },
    { range: "81-100%", count: 0 },
  ];
  runs.forEach((r) => {
    const idx = Math.min(Math.floor(r.cloud_cover / 20), 4);
    cloudBuckets[idx].count++;
  });

  /* ── Render ── */
  return (
    <main className="bg-background min-h-screen pt-14">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header */}
        <motion.div
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div>
            <h1 className="font-heading font-bold text-2xl text-foreground flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-primary" />
              Analytics
            </h1>
            <p className="text-xs text-muted-foreground mt-1">Pipeline run history and performance metrics</p>
          </div>
          <div className="flex gap-2">
            <motion.button
              onClick={fetchRuns}
              disabled={loading}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg glass text-xs font-medium text-foreground hover:border-primary/40 transition-colors"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </motion.button>
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:shadow-[0_0_20px_hsl(166,72%,51%,0.3)] transition-shadow"
            >
              New Run <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </motion.div>

        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard label="Total Runs" value={totalRuns} icon={Activity} delay={0.05} />
          <StatCard label="Completed" value={completed} icon={CheckCircle2} accent="text-emerald-400" delay={0.1} />
          <StatCard label="Failed" value={failed} icon={XCircle} accent="text-red-400" delay={0.15} />
          <StatCard label="Success Rate" value={`${successRate}%`} icon={TrendingUp} accent="text-blue-400" delay={0.2} />
        </div>

        {/* Charts row */}
        {totalRuns > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Runs over time */}
            <motion.div
              className="lg:col-span-2 glass rounded-xl p-5"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 }}
            >
              <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-4">
                Runs Over Time (Last 14 Days)
              </h2>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={runsPerDayData}>
                  <defs>
                    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(166 72% 51%)" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="hsl(166 72% 51%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 25% 12%)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
                  <YAxis tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(222 47% 5%)",
                      border: "1px solid hsl(215 25% 12%)",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                  />
                  <Area type="monotone" dataKey="runs" stroke="hsl(166 72% 51%)" fill="url(#areaGrad)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Status Distribution pie */}
            <motion.div
              className="glass rounded-xl p-5 flex flex-col items-center"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-4 self-start">
                Status Distribution
              </h2>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie
                    data={statusDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={70}
                    paddingAngle={4}
                    dataKey="value"
                    strokeWidth={0}
                  >
                    {statusDistribution.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "hsl(222 47% 5%)",
                      border: "1px solid hsl(215 25% 12%)",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-3 mt-2">
                {statusDistribution.map((d, i) => (
                  <span key={d.name} className="flex items-center gap-1 text-[10px] text-muted-foreground">
                    <span className="w-2 h-2 rounded-full" style={{ background: PIE_COLORS[i % PIE_COLORS.length] }} />
                    {d.name}
                  </span>
                ))}
              </div>
            </motion.div>
          </div>
        )}

        {/* Cloud Cover Chart */}
        {totalRuns > 0 && (
          <motion.div
            className="glass rounded-xl p-5"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
          >
            <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider mb-4">
              Cloud Cover Distribution
            </h2>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={cloudBuckets}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 25% 12%)" />
                <XAxis dataKey="range" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
                <YAxis tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "hsl(222 47% 5%)",
                    border: "1px solid hsl(215 25% 12%)",
                    borderRadius: 8,
                    fontSize: 11,
                  }}
                />
                <Bar dataKey="count" fill="hsl(217 91% 60%)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        )}

        {/* Error banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="rounded-lg bg-destructive/10 border border-destructive/30 p-4 text-xs text-destructive flex items-center gap-2"
            >
              <XCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading state */}
        {loading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-6 h-6 text-primary animate-spin" />
          </div>
        )}

        {/* Runs table */}
        {!loading && runs.length > 0 && (
          <motion.div
            className="glass rounded-xl overflow-hidden"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className="px-5 py-4 border-b border-border/40">
              <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
                Recent Pipeline Runs
              </h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border/30">
                    <th className="text-left px-5 py-3 text-muted-foreground font-medium">Status</th>
                    <th className="text-left px-5 py-3 text-muted-foreground font-medium">BBox</th>
                    <th className="text-left px-5 py-3 text-muted-foreground font-medium">Target Date</th>
                    <th className="text-left px-5 py-3 text-muted-foreground font-medium">Cloud</th>
                    <th className="text-left px-5 py-3 text-muted-foreground font-medium">Created</th>
                    <th className="text-left px-5 py-3 text-muted-foreground font-medium">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => {
                    const created = new Date(run.created_at);
                    const completed = run.completed_at ? new Date(run.completed_at) : null;
                    const duration = completed
                      ? `${Math.round((completed.getTime() - created.getTime()) / 1000)}s`
                      : "—";
                    return (
                      <tr key={run.id} className="border-b border-border/20 hover:bg-muted/30 transition-colors">
                        <td className="px-5 py-3">
                          <StatusChip status={run.status} />
                        </td>
                        <td className="px-5 py-3 font-mono text-foreground/70 max-w-[180px] truncate flex items-center gap-1">
                          <MapPin className="w-3 h-3 text-primary flex-shrink-0" />
                          {run.bbox}
                        </td>
                        <td className="px-5 py-3 text-foreground/70">{run.target_date}</td>
                        <td className="px-5 py-3 text-foreground/70">{run.cloud_cover}%</td>
                        <td className="px-5 py-3 text-foreground/70">
                          {created.toLocaleDateString()} {created.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </td>
                        <td className="px-5 py-3 text-foreground/70 font-mono">{duration}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}

        {/* Empty state */}
        {!loading && runs.length === 0 && !error && (
          <motion.div
            className="flex flex-col items-center justify-center py-20 space-y-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center">
              <BarChart3 className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">No pipeline runs yet</p>
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:shadow-[0_0_20px_hsl(166,72%,51%,0.3)] transition-shadow"
            >
              Launch your first run <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </motion.div>
        )}
      </div>
    </main>
  );
}
