import { useState, useEffect, useCallback, useRef } from "react";
import L from "leaflet";
import "leaflet-draw";
import { motion, AnimatePresence } from "framer-motion";
import {
  Map as MapIcon,
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  Calendar,
  Cloud,
  RotateCcw,
  Trash2,
  Info,
} from "lucide-react";

/* ── Types ─────────────────────────────────────────────────── */
interface BBox {
  lon_min: number;
  lat_min: number;
  lon_max: number;
  lat_max: number;
}

type RunStatus = "IDLE" | "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";

interface PipelineRunResponse {
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

/* ── Helpers ───────────────────────────────────────────────── */
function formatBBox(b: BBox) {
  return `${b.lon_min.toFixed(4)}, ${b.lat_min.toFixed(4)}, ${b.lon_max.toFixed(4)}, ${b.lat_max.toFixed(4)}`;
}

function todayISO() {
  return new Date().toISOString().split("T")[0];
}

/* ── StatusBadge ───────────────────────────────────────────── */
function StatusBadge({ status }: { status: RunStatus }) {
  const cfg: Record<RunStatus, { bg: string; text: string; Icon: typeof CheckCircle2 }> = {
    IDLE: { bg: "bg-muted", text: "Draw a region", Icon: Info },
    PENDING: { bg: "bg-yellow-500/20", text: "Queued", Icon: Loader2 },
    RUNNING: { bg: "bg-blue-500/20", text: "Running…", Icon: Loader2 },
    COMPLETED: { bg: "bg-emerald-500/20", text: "Completed", Icon: CheckCircle2 },
    FAILED: { bg: "bg-red-500/20", text: "Failed", Icon: XCircle },
  };
  const { bg, text, Icon } = cfg[status];
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${bg} text-foreground`}>
      <Icon className={`w-3.5 h-3.5 ${status === "RUNNING" || status === "PENDING" ? "animate-spin" : ""}`} />
      {text}
    </span>
  );
}

/* ── Main Component ────────────────────────────────────────── */
export default function Dashboard() {
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<L.Map | null>(null);
  const drawnItemsRef = useRef<L.FeatureGroup | null>(null);

  const [bbox, setBBox] = useState<BBox | null>(null);
  const [targetDate, setTargetDate] = useState(todayISO());
  const [cloudCover, setCloudCover] = useState(20);
  const [backtrackDays, setBacktrackDays] = useState(30);
  const [runStatus, setRunStatus] = useState<RunStatus>("IDLE");
  const [runId, setRunId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);

  /* ── Initialise Leaflet map (plain API — no react-leaflet) ── */
  useEffect(() => {
    if (!mapRef.current || leafletMapRef.current) return;

    const map = L.map(mapRef.current, {
      center: [20, 0],
      zoom: 3,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    drawnItemsRef.current = drawnItems;

    const drawControl = new (L.Control as any).Draw({
      position: "topleft",
      draw: {
        rectangle: {
          shapeOptions: {
            color: "#34d399",
            weight: 3,
            opacity: 1,
            fillColor: "#34d399",
            fillOpacity: 0.25,
          },
        },
        polygon: false,
        polyline: false,
        circle: false,
        circlemarker: false,
        marker: false,
      },
      edit: {
        featureGroup: drawnItems,
        remove: true,
      },
    });
    map.addControl(drawControl);

    map.on(L.Draw.Event.CREATED, (e: any) => {
      drawnItems.clearLayers();
      const layer = e.layer as L.Rectangle;
      drawnItems.addLayer(layer);
      const bounds = layer.getBounds();
      setBBox({
        lon_min: bounds.getWest(),
        lat_min: bounds.getSouth(),
        lon_max: bounds.getEast(),
        lat_max: bounds.getNorth(),
      });
    });

    map.on(L.Draw.Event.EDITED, (e: any) => {
      const layers = e.layers;
      layers.eachLayer((layer: any) => {
        if (layer instanceof L.Rectangle) {
          const bounds = layer.getBounds();
          setBBox({
            lon_min: bounds.getWest(),
            lat_min: bounds.getSouth(),
            lon_max: bounds.getEast(),
            lat_max: bounds.getNorth(),
          });
        }
      });
    });

    map.on(L.Draw.Event.DELETED, () => {
      setBBox(null);
    });

    leafletMapRef.current = map;

    // Ensure map resizes correctly
    const resizeObserver = new ResizeObserver(() => {
      map.invalidateSize();
    });
    resizeObserver.observe(mapRef.current);

    // Minor delay to catch first render paints
    setTimeout(() => map.invalidateSize(), 200);

    return () => {
      resizeObserver.disconnect();
      map.remove();
      leafletMapRef.current = null;
      drawnItemsRef.current = null;
    };
  }, []);

  /* ── Clear drawn bbox ── */
  const clearSelection = useCallback(() => {
    drawnItemsRef.current?.clearLayers();
    setBBox(null);
    setRunStatus("IDLE");
    setRunId(null);
    setErrorMsg(null);
    setSummary(null);
  }, []);

  /* ── Polling ── */
  const pollRun = useCallback((id: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/pipeline/runs/${id}/`);
        const data: PipelineRunResponse = await res.json();
        if (data.status === "COMPLETED") {
          setRunStatus("COMPLETED");
          setSummary(data.summary);
          clearInterval(interval);
        } else if (data.status === "FAILED") {
          setRunStatus("FAILED");
          setErrorMsg(data.error_message || "Pipeline failed");
          clearInterval(interval);
        } else {
          setRunStatus(data.status as RunStatus);
        }
      } catch {
        /* network blip — keep polling */
      }
    }, 3000);
  }, []);

  /* ── Submit ── */
  const runPipeline = useCallback(async () => {
    if (!bbox) return;
    setRunStatus("PENDING");
    setErrorMsg(null);
    setSummary(null);
    try {
      const res = await fetch(`${API_BASE}/pipeline/runs/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bbox: `${bbox.lon_min},${bbox.lat_min},${bbox.lon_max},${bbox.lat_max}`,
          target_date: targetDate,
          cloud_cover: cloudCover,
          backtrack_days: backtrackDays,
        }),
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const data: PipelineRunResponse = await res.json();
      setRunId(data.id);
      setRunStatus("RUNNING");
      pollRun(data.id);
    } catch (err: unknown) {
      setRunStatus("FAILED");
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
    }
  }, [bbox, targetDate, cloudCover, backtrackDays, pollRun]);

  /* ── Render ── */
  return (
    <main className="bg-background min-h-screen pt-14">
      <div className="flex flex-col lg:flex-row h-[calc(100vh-3.5rem)]">
        {/* ─── Map ─── */}
        <div className="flex-1 relative w-full h-full min-h-[500px]">
          <div
            ref={mapRef}
            className="absolute inset-0 z-0"
            style={{ background: "hsl(222, 47%, 2%)" }}
          />

          {/* Floating hint */}
          {!bbox && (
            <motion.div
              className="absolute bottom-6 left-1/2 -translate-x-1/2 z-[400] glass px-4 py-2 rounded-full text-xs text-muted-foreground font-mono flex items-center gap-2"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <MapIcon className="w-3.5 h-3.5 text-primary" />
              Click the rectangle tool on the map to draw a bounding box
            </motion.div>
          )}
        </div>

        {/* ─── Side Panel ─── */}
        <motion.aside
          className="w-full lg:w-[380px] border-l border-border/40 bg-card/50 backdrop-blur-md overflow-y-auto"
          initial={{ x: 40, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="p-6 space-y-6">
            {/* Header */}
            <div>
              <h1 className="font-heading font-bold text-xl text-foreground">Pipeline Control</h1>
              <p className="text-xs text-muted-foreground mt-1">
                Draw a bounding box on the map, configure parameters, and launch the pipeline.
              </p>
            </div>

            <div className="h-px bg-border/40" />

            {/* Status */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground font-mono uppercase tracking-wider">Status</span>
              <StatusBadge status={runStatus} />
            </div>

            {/* BBox display */}
            <AnimatePresence mode="wait">
              {bbox && (
                <motion.div
                  key="bbox-card"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="glass rounded-lg p-4 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-primary uppercase tracking-wider">Selected Region</span>
                    <button
                      onClick={clearSelection}
                      className="text-muted-foreground hover:text-destructive transition-colors"
                      title="Clear selection"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <p className="text-xs font-mono text-foreground/80 break-all leading-relaxed">
                    {formatBBox(bbox)}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="h-px bg-border/40" />

            {/* Parameters */}
            <div className="space-y-4">
              <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">Parameters</h2>

              {/* Target Date */}
              <label className="block space-y-1.5">
                <span className="flex items-center gap-1.5 text-xs text-foreground/80">
                  <Calendar className="w-3.5 h-3.5 text-primary" /> Target Date
                </span>
                <input
                  type="date"
                  value={targetDate}
                  onChange={(e) => setTargetDate(e.target.value)}
                  className="w-full rounded-md bg-input border border-border/60 px-3 py-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </label>

              {/* Cloud Cover */}
              <label className="block space-y-1.5">
                <span className="flex items-center gap-1.5 text-xs text-foreground/80">
                  <Cloud className="w-3.5 h-3.5 text-primary" /> Max Cloud Cover
                  <span className="ml-auto font-mono text-primary">{cloudCover}%</span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={cloudCover}
                  onChange={(e) => setCloudCover(Number(e.target.value))}
                  className="w-full h-1.5 rounded-full appearance-none bg-muted accent-primary cursor-pointer"
                />
              </label>

              {/* Backtrack Days */}
              <label className="block space-y-1.5">
                <span className="flex items-center gap-1.5 text-xs text-foreground/80">
                  <RotateCcw className="w-3.5 h-3.5 text-primary" /> Backtrack Days
                  <span className="ml-auto font-mono text-primary">{backtrackDays}d</span>
                </span>
                <input
                  type="range"
                  min={1}
                  max={90}
                  value={backtrackDays}
                  onChange={(e) => setBacktrackDays(Number(e.target.value))}
                  className="w-full h-1.5 rounded-full appearance-none bg-muted accent-primary cursor-pointer"
                />
              </label>
            </div>

            <div className="h-px bg-border/40" />

            {/* Run button */}
            <motion.button
              onClick={runPipeline}
              disabled={!bbox || runStatus === "RUNNING" || runStatus === "PENDING"}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-primary text-primary-foreground font-heading font-semibold text-sm
                         hover:shadow-[0_0_30px_hsl(166,72%,51%,0.35)] disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300"
              whileHover={{ scale: bbox ? 1.02 : 1 }}
              whileTap={{ scale: bbox ? 0.98 : 1 }}
            >
              {runStatus === "RUNNING" || runStatus === "PENDING" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Running Pipeline…
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" /> Run Pipeline
                </>
              )}
            </motion.button>

            {/* Run ID */}
            {runId && (
              <p className="text-[10px] font-mono text-muted-foreground text-center break-all">
                Run ID: {runId}
              </p>
            )}

            {/* Error banner */}
            <AnimatePresence>
              {errorMsg && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="rounded-lg bg-destructive/10 border border-destructive/30 p-3 text-xs text-destructive"
                >
                  {errorMsg}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Summary */}
            <AnimatePresence>
              {summary && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="glass rounded-lg p-4 space-y-2"
                >
                  <span className="text-xs font-mono text-primary uppercase tracking-wider">Pipeline Summary</span>
                  <pre className="text-[10px] text-foreground/70 font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                    {JSON.stringify(summary, null, 2)}
                  </pre>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.aside>
      </div>
    </main>
  );
}
