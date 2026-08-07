import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { MapContainer, TileLayer, CircleMarker, Rectangle, Polyline, Popup } from "react-leaflet";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { loadAttribution, loadBacktrackSummary, loadRunMetadata } from "@/services/dataService";
import { AttributionEntry, BacktrackEntry, RunMetadata, SOURCE_ICONS } from "@/types";
import {
  GitBranch,
  ChevronDown,
  ChevronUp,
  Database,
  Cpu,
  Wind,
} from "lucide-react";

const SCORE_COLORS = {
  fishing: "#3B82F6",
  industrial: "#F59E0B",
  shipping: "#8B5CF6",
  river: "#10B981",
};

const AttributionPage: React.FC = () => {
  const [attribution, setAttribution] = useState<AttributionEntry[]>([]);
  const [backtrack, setBacktrack] = useState<BacktrackEntry[]>([]);
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    Promise.all([loadAttribution(), loadBacktrackSummary(), loadRunMetadata()]).then(
      ([attr, bt, meta]) => {
        setAttribution(attr);
        setBacktrack(bt);
        setMetadata(meta);
        setLoading(false);
      }
    );
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Score breakdown data per cluster
  const scoreData = attribution.map((a) => ({
    name: `#${a.debris_cluster_id}`,
    fishing: parseFloat((a.fishing_score * 100).toFixed(1)),
    industrial: parseFloat((a.industrial_score * 100).toFixed(1)),
    shipping: parseFloat((a.shipping_score * 100).toFixed(1)),
    river: parseFloat((a.river_score * 100).toFixed(1)),
  }));

  const confColor = (c: string) =>
    c === "high" ? "text-emerald-400 bg-emerald-500/15" : c === "medium" ? "text-yellow-400 bg-yellow-500/15" : "text-red-400 bg-red-500/15";

  return (
    <div className="min-h-screen bg-background pt-14">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 py-6">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-primary" />
            Source Attribution
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Lagrangian particle backtracking shows where debris likely originated from.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Attribution Table */}
          <div className="glass-card overflow-hidden">
            <div className="px-5 py-4 border-b border-border/30">
              <h2 className="font-heading font-semibold">Attribution Results</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/20 text-muted-foreground">
                    <th className="text-left px-4 py-3 font-medium">Cluster</th>
                    <th className="text-left px-4 py-3 font-medium">Source</th>
                    <th className="text-left px-4 py-3 font-medium">Location</th>
                    <th className="text-left px-4 py-3 font-medium">Country</th>
                    <th className="text-left px-4 py-3 font-medium">Score</th>
                    <th className="text-left px-4 py-3 font-medium">Conf.</th>
                    <th className="text-left px-4 py-3 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {attribution.map((a) => (
                    <React.Fragment key={a.debris_cluster_id}>
                      <tr className="border-b border-border/10 hover:bg-muted/20 transition-colors">
                        <td className="px-4 py-3 font-mono text-primary">#{a.debris_cluster_id}</td>
                        <td className="px-4 py-3">
                          <span className="flex items-center gap-1.5">
                            <span>{SOURCE_ICONS[a.source_type] || "❓"}</span>
                            <span className="capitalize">{a.source_type}</span>
                          </span>
                        </td>
                        <td className="px-4 py-3 text-muted-foreground text-xs">{a.location_name}</td>
                        <td className="px-4 py-3 text-muted-foreground text-xs">{a.country}</td>
                        <td className="px-4 py-3 font-semibold">{(a.attribution_score * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${confColor(a.confidence)}`}>
                            {a.confidence}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <button onClick={() => setExpandedId(expandedId === a.debris_cluster_id ? null : a.debris_cluster_id)}>
                            {expandedId === a.debris_cluster_id ? (
                              <ChevronUp className="w-4 h-4 text-muted-foreground" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-muted-foreground" />
                            )}
                          </button>
                        </td>
                      </tr>
                      {expandedId === a.debris_cluster_id && (
                        <tr>
                          <td colSpan={7} className="px-4 py-3 bg-muted/10">
                            <p className="text-sm text-muted-foreground leading-relaxed">{a.explanation}</p>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Backtrack Map */}
          <div className="glass-card overflow-hidden">
            <div className="px-5 py-4 border-b border-border/30">
              <h2 className="font-heading font-semibold">Backtrack Map</h2>
            </div>
            <div className="h-[380px]">
              <MapContainer
                center={[16.28, -88.5]}
                zoom={9}
                className="w-full h-full"
                style={{ background: "#0A1628" }}
                zoomControl={false}
              >
                <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
                {/* Cluster markers (red) */}
                {backtrack.map((bt) => (
                  <CircleMarker
                    key={bt.cluster_id}
                    center={[bt.source_centroid[1], bt.source_centroid[0]]}
                    radius={7}
                    pathOptions={{ color: "#EF4444", fillColor: "#EF4444", fillOpacity: 0.8, weight: 2 }}
                  >
                    <Popup>
                      <div className="text-xs">
                        <strong>Cluster #{bt.cluster_id}</strong><br />
                        Particles: {bt.n_particles}<br />
                        Days: {bt.days_to_source}
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
                {/* Source bboxes (orange) */}
                {backtrack.map((bt) => (
                  <Rectangle
                    key={`bbox-${bt.cluster_id}`}
                    bounds={[
                      [bt.source_bbox[1], bt.source_bbox[0]],
                      [bt.source_bbox[3], bt.source_bbox[2]],
                    ]}
                    pathOptions={{ color: "#F59E0B", weight: 1.5, fillColor: "#F59E0B", fillOpacity: 0.08, dashArray: "4 4" }}
                  />
                ))}
                {/* Connection lines */}
                {attribution.map((a) => {
                  const bt = backtrack.find((b) => b.cluster_id === a.debris_cluster_id);
                  if (!bt) return null;
                  return (
                    <Polyline
                      key={`line-${a.debris_cluster_id}`}
                      positions={[
                        [bt.source_centroid[1], bt.source_centroid[0]],
                        [a.source_centroid[1], a.source_centroid[0]],
                      ]}
                      pathOptions={{ color: "#00C9A7", weight: 1.5, dashArray: "6 4", opacity: 0.6 }}
                    />
                  );
                })}
              </MapContainer>
            </div>
          </div>
        </div>

        {/* Score Breakdown + Config */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          {/* Score breakdown chart */}
          <div className="glass-card p-5">
            <h3 className="font-heading font-semibold mb-4">Source Score Breakdown</h3>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scoreData} layout="vertical" barGap={2}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(215 20% 16%)" />
                  <XAxis type="number" domain={[0, 50]} tick={{ fill: "#6B7280", fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: "#9CA3AF", fontSize: 12 }} width={50} />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(220 30% 8%)",
                      border: "1px solid hsl(215 20% 16%)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="fishing" stackId="a" fill={SCORE_COLORS.fishing} name="Fishing" />
                  <Bar dataKey="industrial" stackId="a" fill={SCORE_COLORS.industrial} name="Industrial" />
                  <Bar dataKey="shipping" stackId="a" fill={SCORE_COLORS.shipping} name="Shipping" />
                  <Bar dataKey="river" stackId="a" fill={SCORE_COLORS.river} name="River" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="flex gap-4 mt-3 justify-center">
              {Object.entries(SCORE_COLORS).map(([key, color]) => (
                <div key={key} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
                  <span className="capitalize">{key}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Backtrack config */}
          {metadata && (
            <div className="glass-card p-5">
              <h3 className="font-heading font-semibold mb-4">Backtrack Configuration</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {[
                  { icon: Database, label: "Ocean Data", value: metadata.cmems_product.split("_").slice(-2).join("_") },
                  { icon: Wind, label: "Wind Data", value: "ERA5 (u10, v10)" },
                  { icon: Cpu, label: "Integrator", value: metadata.integrator },
                  { icon: Cpu, label: "Time Step", value: `${metadata.time_step_hours}h` },
                  { icon: GitBranch, label: "Particles", value: `${metadata.n_particles} per cluster` },
                  { icon: GitBranch, label: "Duration", value: `${metadata.bt_days} days` },
                  { icon: Cpu, label: "Diffusion (Kh)", value: metadata.horizontal_diffusion_kh.toString() },
                ].map((item) => {
                  const Icon = item.icon;
                  return (
                    <div key={item.label} className="flex items-start gap-3 p-3 bg-muted/20 rounded-lg">
                      <Icon className="w-4 h-4 text-primary mt-0.5" />
                      <div>
                        <div className="text-xs text-muted-foreground">{item.label}</div>
                        <div className="text-sm font-medium">{item.value}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-4">
                <div className="text-xs text-muted-foreground mb-2">Kernels</div>
                <div className="flex flex-wrap gap-2">
                  {metadata.kernels.map((k) => (
                    <span key={k} className="px-2.5 py-1 bg-primary/10 text-primary rounded-md text-xs font-medium">
                      {k}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AttributionPage;
