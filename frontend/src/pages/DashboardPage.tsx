import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { loadRunSummary } from "@/services/dataService";
import { RunSummary, PIPELINE_STAGES, POLYMER_COLORS } from "@/types";
import {
  Crosshair,
  FlaskConical,
  ShieldAlert,
  Clock,
  Cloud,
  Undo2,
  CheckCircle2,
  XCircle,
  SkipForward,
  ExternalLink,
} from "lucide-react";

function KpiCard({ label, value, icon: Icon, color }: { label: string; value: string; icon: any; color: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-5 flex items-center gap-4"
    >
      <div className={`w-11 h-11 rounded-xl flex items-center justify-center ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-2xl font-heading font-bold text-foreground">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </motion.div>
  );
}

const DashboardPage: React.FC = () => {
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRunSummary().then((d) => { setSummary(d); setLoading(false); });
  }, []);

  if (loading || !summary) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const pc = summary.outputs.polymer_counts;
  const totalDetections = Object.values(pc).reduce((a, b) => a + b, 0);
  const plasticCount = pc["Marine Debris (Plastic)"] || 0;
  const fpRate = (((totalDetections - plasticCount) / totalDetections) * 100).toFixed(1);

  const pieData = Object.entries(pc)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({
      name: name.replace("False Positive ", "FP "),
      value,
      color: POLYMER_COLORS[name] || "#6B7280",
    }));

  // Mock run list (only run_001 exists)
  const runs = [
    {
      id: "run_001",
      status: "Completed",
      region: "Honduras Coast",
      date: summary.target_date,
      detections: totalDetections,
      plastic: plasticCount,
    },
  ];

  return (
    <div className="min-h-screen bg-background pt-14">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 py-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          <KpiCard label="Total Detections" value={totalDetections.toString()} icon={Crosshair} color="bg-secondary/20 text-secondary" />
          <KpiCard label="Confirmed Plastic Clusters" value={plasticCount.toString()} icon={FlaskConical} color="bg-destructive/20 text-destructive" />
          <KpiCard label="False Positive Rate" value={`${fpRate}%`} icon={ShieldAlert} color="bg-yellow-500/20 text-yellow-400" />
          <KpiCard label="Processing Time" value={`${summary.elapsed_seconds}s`} icon={Clock} color="bg-primary/20 text-primary" />
          <KpiCard label="Cloud Cover" value="5.41%" icon={Cloud} color="bg-blue-400/20 text-blue-400" />
          <KpiCard label="Backtrack Days" value="7" icon={Undo2} color="bg-purple-400/20 text-purple-400" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Run List */}
          <div className="lg:col-span-2">
            <div className="glass-card overflow-hidden">
              <div className="px-5 py-4 border-b border-border/30 flex items-center justify-between">
                <h2 className="font-heading font-semibold text-lg">Run List</h2>
                <span className="text-xs text-muted-foreground">{runs.length} run(s)</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/20 text-muted-foreground">
                      <th className="text-left px-5 py-3 font-medium">Run ID</th>
                      <th className="text-left px-5 py-3 font-medium">Status</th>
                      <th className="text-left px-5 py-3 font-medium">Region</th>
                      <th className="text-left px-5 py-3 font-medium">Target Date</th>
                      <th className="text-left px-5 py-3 font-medium">Detections</th>
                      <th className="text-left px-5 py-3 font-medium">Plastic</th>
                      <th className="text-left px-5 py-3 font-medium"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr key={run.id} className="border-b border-border/10 hover:bg-muted/20 transition-colors">
                        <td className="px-5 py-3 font-mono text-primary">{run.id}</td>
                        <td className="px-5 py-3">
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/15 text-emerald-400 rounded-full text-xs font-medium">
                            <CheckCircle2 className="w-3 h-3" />
                            {run.status}
                          </span>
                        </td>
                        <td className="px-5 py-3 text-muted-foreground">{run.region}</td>
                        <td className="px-5 py-3 text-muted-foreground">{run.date}</td>
                        <td className="px-5 py-3 font-semibold">{run.detections}</td>
                        <td className="px-5 py-3 font-semibold text-destructive">{run.plastic}</td>
                        <td className="px-5 py-3">
                          <Link to="/run_details" className="text-primary hover:text-primary/80 transition-colors">
                            <ExternalLink className="w-4 h-4" />
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pipeline Stage Status */}
            <div className="glass-card p-5 mt-6">
              <h3 className="font-heading font-semibold mb-4">Run Stage Status</h3>
              <div className="flex flex-wrap gap-3">
                {PIPELINE_STAGES.map((stage) => {
                  const completed = summary.stages_completed.includes(stage.id);
                  const failed = summary.stages_failed.includes(stage.id);
                  const skipped = summary.stages_skipped.includes(stage.id);
                  return (
                    <div
                      key={stage.id}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium border ${
                        completed
                          ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                          : failed
                          ? "bg-destructive/10 border-destructive/30 text-destructive"
                          : skipped
                          ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-400"
                          : "bg-muted/20 border-border/30 text-muted-foreground"
                      }`}
                    >
                      {completed ? (
                        <CheckCircle2 className="w-3.5 h-3.5" />
                      ) : failed ? (
                        <XCircle className="w-3.5 h-3.5" />
                      ) : (
                        <SkipForward className="w-3.5 h-3.5" />
                      )}
                      <span>{stage.id}. {stage.name}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Polymer Distribution */}
          <div className="glass-card p-5">
            <h3 className="font-heading font-semibold mb-4">Polymer Distribution</h3>
            <div className="h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={90}
                    dataKey="value"
                    stroke="none"
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "hsl(220 30% 8%)",
                      border: "1px solid hsl(215 20% 16%)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-1.5 mt-2 max-h-[200px] overflow-y-auto">
              {pieData.map((entry) => (
                <div key={entry.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
                    <span className="text-muted-foreground truncate max-w-[180px]">{entry.name}</span>
                  </div>
                  <span className="font-medium text-foreground">{entry.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
