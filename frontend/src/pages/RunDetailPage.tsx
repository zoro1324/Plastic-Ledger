import React, { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import {
  loadRunSummary,
  loadFinalReport,
  loadAttribution,
  loadRunMetadata,
  loadDebrisSummaryCsv,
} from "@/services/dataService";
import {
  RunSummary,
  DetectionFeatureCollection,
  AttributionEntry,
  RunMetadata,
  DebrisSummaryRow,
  PIPELINE_STAGES,
  POLYMER_COLORS,
  PipelineRun
} from "@/types";
import { getPipelineRun } from "@/lib/api";
import {
  ClipboardList,
  CheckCircle2,
  XCircle,
  SkipForward,
  Clock,
  MapPin,
  Calendar,
  Cloud,
  Cpu,
  Layers,
  ExternalLink,
  Map,
  GitBranch,
  BarChart3,
  FileText,
  Database,
} from "lucide-react";

function FitBounds({ data }: { data: DetectionFeatureCollection | null }) {
  const map = useMap();
  useEffect(() => {
    if (data && data.features.length > 0) {
      const layer = L.geoJSON(data as any);
      const bounds = layer.getBounds();
      if (bounds.isValid()) map.flyToBounds(bounds, { padding: [30, 30], duration: 1 });
    }
  }, [data, map]);
  return null;
}

const tooltipStyle = {
  background: "hsl(220 30% 8%)",
  border: "1px solid hsl(215 20% 16%)",
  borderRadius: "8px",
  fontSize: "12px",
};

const RunDetailPage: React.FC = () => {
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const runId = searchParams.get("id") || "";

  const [run, setRun] = useState<PipelineRun | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [detections, setDetections] = useState<DetectionFeatureCollection | null>(null);
  const [attribution, setAttribution] = useState<AttributionEntry[]>([]);
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [csv, setCsv] = useState<DebrisSummaryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    if (!runId) return;
    let interval: ReturnType<typeof setInterval>;

    const loadStatusAndData = async () => {
      try {
        const runData = await getPipelineRun(runId);
        setRun(runData);

        if (runData.status === "COMPLETED") {
          const [sum, det, attr, meta, csvData] = await Promise.all([
            loadRunSummary(runId),
            loadFinalReport(runId),
            loadAttribution(runId),
            loadRunMetadata(runId),
            loadDebrisSummaryCsv(runId),
          ]);
          setSummary(sum);
          setDetections(det);
          setAttribution(attr);
          setMetadata(meta);
          setCsv(csvData);
          setLoading(false);
          clearInterval(interval);
        } else if (runData.status === "FAILED") {
          setLoading(false);
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Error loading run data:", err);
      }
    };

    loadStatusAndData();
    interval = setInterval(loadStatusAndData, 5000);

    return () => clearInterval(interval);
  }, [runId]);

  if (loading || !run) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-muted-foreground">Loading Run Data...</p>
        </div>
      </div>
    );
  }

  if (run.status !== "COMPLETED") {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="glass-card p-8 max-w-md w-full text-center">
          <h2 className="text-xl font-heading font-bold mb-2">Run {run.status}</h2>
          <p className="text-muted-foreground text-sm mb-6">
            {run.status === "FAILED" ? run.error_message : "The pipeline is still processing this run."}
          </p>
          <Link to="/dashboard" className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-semibold inline-block">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (!summary) return null;

  const pc = summary.outputs.polymer_counts;
  const totalDetections = Object.values(pc).reduce((a, b) => a + b, 0);
  const plasticCount = pc["Marine Debris (Plastic)"] || 0;
  const fpRate = (((totalDetections - plasticCount) / totalDetections) * 100).toFixed(1);
  const sceneId = summary.outputs.raw_scenes?.[0] || "N/A";

  const pieData = Object.entries(pc)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({
      name: name.replace("False Positive ", "FP "),
      value,
      color: POLYMER_COLORS[name] || "#6B7280",
    }));

  const tabs = [
    { key: "overview", label: "Overview", icon: ClipboardList },
    { key: "map", label: "Detection Map", icon: Map },
    { key: "attribution", label: "Attribution", icon: GitBranch },
    { key: "analytics", label: "Analytics", icon: BarChart3 },
    { key: "reports", label: "Reports", icon: FileText },
    { key: "raw", label: "Raw Data", icon: Database },
  ];

  return (
    <div className="min-h-screen bg-background pt-14">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6 py-6">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div>
              <h1 className="font-heading text-2xl font-bold flex items-center gap-2">
                <ClipboardList className="w-6 h-6 text-primary" />
                Run Detail — <span className="text-primary">{runId.substring(0, 8)}</span>
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                Complete overview of pipeline execution and results.
              </p>
            </div>
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/15 text-emerald-400 rounded-full text-sm font-medium self-start">
              <CheckCircle2 className="w-4 h-4" />
              Completed
            </span>
          </div>

          {/* Run info cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
            {[
              { icon: MapPin, label: "Bbox", value: run.bbox },
              { icon: Calendar, label: "Target Date", value: run.target_date },
              { icon: Clock, label: "Duration", value: `${summary.elapsed_seconds}s` },
              { icon: Cloud, label: "Cloud Cover", value: `${run.cloud_cover}%` },
              { icon: Cpu, label: "CRS", value: "EPSG:32616" },
              { icon: Layers, label: "Resolution", value: "10 m" },
            ].map((item) => {
              const Icon = item.icon;
              return (
                <div key={item.label} className="glass-card p-3 flex items-center gap-3">
                  <Icon className="w-4 h-4 text-primary flex-shrink-0" />
                  <div className="min-w-0">
                    <div className="text-xs text-muted-foreground">{item.label}</div>
                    <div className="text-sm font-medium truncate">{item.value}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pipeline stages */}
          <div className="glass-card p-4 mb-6">
            <h3 className="text-sm font-heading font-semibold mb-3">Pipeline Stages</h3>
            <div className="flex flex-wrap items-center gap-2">
              {PIPELINE_STAGES.map((stage, i) => {
                const completed = summary.stages_completed.includes(stage.id);
                const failed = summary.stages_failed.includes(stage.id);
                return (
                  <React.Fragment key={stage.id}>
                    <div
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium ${
                        completed
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : failed
                          ? "bg-red-500/10 text-red-400 border border-red-500/20"
                          : "bg-muted/30 text-muted-foreground border border-border/20"
                      }`}
                    >
                      {completed ? <CheckCircle2 className="w-3.5 h-3.5" /> : failed ? <XCircle className="w-3.5 h-3.5" /> : <SkipForward className="w-3.5 h-3.5" />}
                      {stage.short}
                    </div>
                    {i < PIPELINE_STAGES.length - 1 && (
                      <div className={`w-6 h-px ${completed ? "bg-emerald-500/40" : "bg-border/40"}`} />
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        </motion.div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 overflow-x-auto scrollbar-hide border-b border-border/20 pb-px">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium whitespace-nowrap transition-all border-b-2 -mb-px ${
                  activeTab === tab.key
                    ? "text-primary border-primary"
                    : "text-muted-foreground border-transparent hover:text-foreground hover:border-border/50"
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <motion.div key={activeTab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          {/* Overview */}
          {activeTab === "overview" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Processing Summary */}
              <div className="glass-card p-5">
                <h3 className="font-heading font-semibold mb-4">Processing Summary</h3>
                <div className="space-y-3">
                  {[
                    { label: "Total Detections", value: totalDetections.toString() },
                    { label: "Confirmed Plastic Clusters", value: plasticCount.toString() },
                    { label: "False Positive Rate", value: `${fpRate}%` },
                    { label: "CRS", value: "EPSG:32616" },
                    { label: "Avg Confidence", value: csv.length > 0 ? (csv.reduce((a, r) => a + r.confidence, 0) / csv.length).toFixed(3) : "N/A" },
                    { label: "Pipeline Time", value: `${summary.elapsed_seconds}s` },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{item.label}</span>
                      <span className="font-semibold">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Polymer chart */}
              <div className="glass-card p-5">
                <h3 className="font-heading font-semibold mb-4">Polymer Distribution</h3>
                <div className="h-[220px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={80} dataKey="value" stroke="none">
                        {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                      </Pie>
                      <Tooltip contentStyle={tooltipStyle} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="grid grid-cols-2 gap-1 mt-2 text-xs">
                  {pieData.slice(0, 6).map((e) => (
                    <div key={e.name} className="flex items-center gap-1.5">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
                      <span className="text-muted-foreground truncate">{e.name}</span>
                      <span className="ml-auto font-medium">{e.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Scene Info */}
              <div className="glass-card p-5">
                <h3 className="font-heading font-semibold mb-4">Scene Information</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-muted-foreground text-xs block mb-0.5">Scene ID</span>
                    <span className="font-mono text-xs break-all">{sceneId}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <span className="text-muted-foreground text-xs block mb-0.5">Date</span>
                      <span className="font-medium">{summary.target_date}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground text-xs block mb-0.5">Cloud Cover</span>
                      <span className="font-medium">5.41%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground text-xs block mb-0.5">CRS</span>
                      <span className="font-medium">EPSG:32616</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground text-xs block mb-0.5">Resolution</span>
                      <span className="font-medium">10 m</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick links */}
              <div className="glass-card p-5">
                <h3 className="font-heading font-semibold mb-4">Quick Links</h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Detection Map", to: `/detection?id=${runId}`, icon: Map, color: "text-blue-400" },
                    { label: "Attribution", to: `/attribution?id=${runId}`, icon: GitBranch, color: "text-emerald-400" },
                    { label: "Analytics", to: `/analytics?id=${runId}`, icon: BarChart3, color: "text-purple-400" },
                    { label: "Reports", to: `/reports?id=${runId}`, icon: FileText, color: "text-yellow-400" },
                  ].map((link) => {
                    const Icon = link.icon;
                    return (
                      <Link
                        key={link.label}
                        to={link.to}
                        className="flex items-center gap-3 p-3 bg-muted/20 rounded-lg hover:bg-muted/40 transition-colors group"
                      >
                        <Icon className={`w-5 h-5 ${link.color}`} />
                        <span className="text-sm font-medium group-hover:text-foreground">{link.label}</span>
                        <ExternalLink className="w-3.5 h-3.5 ml-auto text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </Link>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Map tab */}
          {activeTab === "map" && (
            <div className="glass-card overflow-hidden" style={{ height: "500px" }}>
              <MapContainer center={[16.1, -88.4]} zoom={10} className="w-full h-full" style={{ background: "#0A1628" }} zoomControl={false}>
                <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" />
                {detections && (
                  <GeoJSON
                    data={detections as any}
                    style={(f: any) => ({
                      color: POLYMER_COLORS[f.properties.polymer_type] || "#6B7280",
                      weight: 1.5,
                      fillColor: POLYMER_COLORS[f.properties.polymer_type] || "#6B7280",
                      fillOpacity: 0.3,
                    })}
                  />
                )}
                <FitBounds data={detections} />
              </MapContainer>
            </div>
          )}

          {/* Attribution tab */}
          {activeTab === "attribution" && (
            <div className="glass-card overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/20 text-muted-foreground">
                      <th className="text-left px-4 py-3 font-medium">Cluster</th>
                      <th className="text-left px-4 py-3 font-medium">Source Type</th>
                      <th className="text-left px-4 py-3 font-medium">Location</th>
                      <th className="text-left px-4 py-3 font-medium">Country</th>
                      <th className="text-left px-4 py-3 font-medium">Score</th>
                      <th className="text-left px-4 py-3 font-medium">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {attribution.map((a) => (
                      <tr key={a.debris_cluster_id} className="border-b border-border/10 hover:bg-muted/20">
                        <td className="px-4 py-3 font-mono text-primary">#{a.debris_cluster_id}</td>
                        <td className="px-4 py-3 capitalize">{a.source_type}</td>
                        <td className="px-4 py-3 text-muted-foreground">{a.location_name}</td>
                        <td className="px-4 py-3 text-muted-foreground">{a.country}</td>
                        <td className="px-4 py-3 font-semibold">{(a.attribution_score * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${
                            a.confidence === "high" ? "bg-emerald-500/15 text-emerald-400" :
                            a.confidence === "medium" ? "bg-yellow-500/15 text-yellow-400" :
                            "bg-red-500/15 text-red-400"
                          }`}>{a.confidence}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Analytics tab */}
          {activeTab === "analytics" && (
            <div className="text-center py-12">
              <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground mb-4">View detailed analytics on the dedicated page.</p>
              <Link to={`/analytics?id=${runId}`} className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-semibold">
                Open Analytics <ExternalLink className="w-4 h-4" />
              </Link>
            </div>
          )}

          {/* Reports tab */}
          {activeTab === "reports" && (
            <div className="text-center py-12">
              <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground mb-4">Download and preview all report files.</p>
              <Link to={`/reports?id=${runId}`} className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-semibold">
                Open Reports <ExternalLink className="w-4 h-4" />
              </Link>
            </div>
          )}

          {/* Raw Data tab */}
          {activeTab === "raw" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass-card p-5">
                <h3 className="font-heading font-semibold mb-3">run_summary.json</h3>
                <pre className="text-xs font-mono text-muted-foreground bg-muted/20 rounded-lg p-4 overflow-auto max-h-[400px]">
                  {JSON.stringify(summary, null, 2)}
                </pre>
              </div>
              {metadata && (
                <div className="glass-card p-5">
                  <h3 className="font-heading font-semibold mb-3">run_metadata.json (attribution)</h3>
                  <pre className="text-xs font-mono text-muted-foreground bg-muted/20 rounded-lg p-4 overflow-auto max-h-[400px]">
                    {JSON.stringify(metadata, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default RunDetailPage;
