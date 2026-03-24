import { useState, useEffect, useCallback, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { motion, AnimatePresence } from "framer-motion";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

import {
  ZoomIn,
  ZoomOut,
  MousePointerSquareDashed,
  Pencil,
  Check,
  X,
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  Calendar,
  Cloud,
  RotateCcw,
  Trash2,
  Info
} from "lucide-react";

/* ── Types ─────────────────────────────────────────────────── */
interface BBox {
  lon_min: number;
  lat_min: number;
  lon_max: number;
  lat_max: number;
}

type RunStatus = "IDLE" | "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
type InteractionState = "IDLE" | "DRAWING" | "EDITING";

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
function StatusBadge({ status, state }: { status: RunStatus; state: InteractionState }) {
  if (state === "DRAWING") {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-primary/20 text-primary">
        <MousePointerSquareDashed className="w-3.5 h-3.5 animate-pulse" />
        Drawing: Click & Drag
      </span>
    );
  }
  if (state === "EDITING") {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-500">
        <Pencil className="w-3.5 h-3.5 animate-pulse" />
        Editing Region (Hit Enter)
      </span>
    );
  }

  const cfg: Record<RunStatus, { bg: string; text: string; Icon: typeof CheckCircle2 }> = {
    IDLE: { bg: "bg-muted", text: "Ready", Icon: Info },
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

  // Interaction & Drawing State
  const [interactionState, setInteractionState] = useState<InteractionState>("IDLE");
  const tempBoundsRef = useRef<L.LatLngBounds | null>(null);
  
  const drawingRectRef = useRef<L.Rectangle | null>(null);
  const cornerMarkersRef = useRef<L.Marker[]>([]);

  // Pipeline state
  const [bbox, setBBox] = useState<BBox | null>(null);
  const [targetDate, setTargetDate] = useState(todayISO());
  const [cloudCover, setCloudCover] = useState(20);
  const [backtrackDays, setBacktrackDays] = useState(30);
  const [runStatus, setRunStatus] = useState<RunStatus>("IDLE");
  const [runId, setRunId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);

  /* ── Initialise Leaflet map (vanilla API) ── */
  useEffect(() => {
    if (!mapRef.current || leafletMapRef.current) return;

    // We disable the default zoom control because we supply our custom nav bar overlay
    const map = L.map(mapRef.current, {
      center: [20, 0],
      zoom: 3,
      zoomControl: false,
      zoomAnimation: true,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    leafletMapRef.current = map;

    const resizeObserver = new ResizeObserver(() => {
      map.invalidateSize();
    });
    resizeObserver.observe(mapRef.current);

    setTimeout(() => map.invalidateSize(), 200);

    return () => {
      resizeObserver.disconnect();
      map.remove();
      leafletMapRef.current = null;
    };
  }, []);

  /* ── Native UI Architecture Hooks ── */
  const enterEditMode = useCallback(() => {
    if (bbox) {
      tempBoundsRef.current = L.latLngBounds(
        [bbox.lat_min, bbox.lon_min],
        [bbox.lat_max, bbox.lon_max]
      );
    }
    setInteractionState("EDITING");
  }, [bbox]);

  const confirmEdit = useCallback(() => {
    if (tempBoundsRef.current) {
      const b = tempBoundsRef.current;
      setBBox({
        lon_min: b.getWest(),
        lat_min: b.getSouth(),
        lon_max: b.getEast(),
        lat_max: b.getNorth(),
      });
    }
    setInteractionState("IDLE");
  }, []);

  const cancelEdit = useCallback(() => {
    // If user presses Escape, revert. If we have bbox, it goes back. If not, it clears it.
    tempBoundsRef.current = null;
    setInteractionState("IDLE");
  }, []);

  const clearSelection = useCallback(() => {
    setBBox(null);
    tempBoundsRef.current = null;
    setInteractionState("IDLE");
    setRunStatus("IDLE");
    setRunId(null);
    setErrorMsg(null);
    setSummary(null);
  }, []);

  /* ── Keyboard Listeners (Enter / Escape) ── */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (interactionState === "DRAWING" || interactionState === "EDITING") {
        if (e.key === "Escape") {
          cancelEdit();
        } else if (e.key === "Enter" && interactionState === "EDITING") {
          confirmEdit();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [interactionState, cancelEdit, confirmEdit]);

  /* ── Native Drawing / Editing Engine ── */
  useEffect(() => {
    const map = leafletMapRef.current;
    if (!map) return;

    // Reset visual layers on any state boundary change
    if (drawingRectRef.current) map.removeLayer(drawingRectRef.current);
    cornerMarkersRef.current.forEach(m => map.removeLayer(m));
    cornerMarkersRef.current = [];
    drawingRectRef.current = null;

    if (interactionState === "IDLE") {
      map.dragging.enable();
      L.DomUtil.removeClass(map.getContainer(), 'crosshair-cursor-map');

      if (bbox) {
        // Draw the locked selection
        const bounds = L.latLngBounds([bbox.lat_min, bbox.lon_min], [bbox.lat_max, bbox.lon_max]);
        drawingRectRef.current = L.rectangle(bounds, {
          color: "#34d399",
          weight: 3,
          fillColor: "#34d399",
          fillOpacity: 0.15,
          interactive: false
        }).addTo(map);
      }
      return;
    }

    if (interactionState === "DRAWING") {
      map.dragging.disable();
      L.DomUtil.addClass(map.getContainer(), 'crosshair-cursor-map');

      let startPoint: L.LatLng | null = null;
      let tempRect: L.Rectangle | null = null;

      const onMouseDown = (e: L.LeafletMouseEvent) => {
        startPoint = e.latlng;
        tempRect = L.rectangle(L.latLngBounds(e.latlng, e.latlng), {
          color: "#eab308", // Yellow to indicate unconfirmed active work
          weight: 3,
          fillColor: "#eab308",
          fillOpacity: 0.2,
          interactive: false,
          dashArray: "6, 6"
        }).addTo(map);
        drawingRectRef.current = tempRect;
      };

      const onMouseMove = (e: L.LeafletMouseEvent) => {
        if (!startPoint || !tempRect) return;
        tempRect.setBounds(L.latLngBounds(startPoint, e.latlng));
      };

      const onMouseUp = () => {
        if (!startPoint || !tempRect) return;
        tempBoundsRef.current = tempRect.getBounds();
        startPoint = null;
        setInteractionState("EDITING"); // Switch cleanly into Edit to let handles render
      };

      map.on('mousedown', onMouseDown);
      map.on('mousemove', onMouseMove);
      map.on('mouseup', onMouseUp);

      return () => {
        map.off('mousedown', onMouseDown);
        map.off('mousemove', onMouseMove);
        map.off('mouseup', onMouseUp);
      };
    }

    if (interactionState === "EDITING") {
      map.dragging.enable(); // Allow map dragging to shift focus while editing
      L.DomUtil.removeClass(map.getContainer(), 'crosshair-cursor-map');

      if (!tempBoundsRef.current) return;

      const rectBounds = tempBoundsRef.current;
      const rect = L.rectangle(rectBounds, {
        color: "#eab308",
        weight: 3,
        fillColor: "#eab308",
        fillOpacity: 0.2,
        interactive: false,
        dashArray: "6, 6"
      }).addTo(map);
      drawingRectRef.current = rect;

      // Draggable nodes
      const handleIcon = L.divIcon({
        className: "edit-handle-icon",
        iconSize: [14, 14],
        iconAnchor: [7, 7]
      });

      const getCorners = () => [
        rect.getBounds().getNorthWest(),
        rect.getBounds().getNorthEast(),
        rect.getBounds().getSouthEast(),
        rect.getBounds().getSouthWest()
      ];

      const syncMarkers = (newCorners: L.LatLng[]) => {
        newCorners.forEach((c, i) => cornerMarkersRef.current[i]?.setLatLng(c));
      };

      getCorners().forEach((c, idx) => {
        const marker = L.marker(c, { draggable: true, icon: handleIcon }).addTo(map);
        
        marker.on('drag', (e) => {
          const movedPos = e.target.getLatLng();
          const oldBounds = rect.getBounds();
          
          // Identify the perfectly opposite anchor point
          const anchors = [
            oldBounds.getSouthEast(), // Opposite of NW
            oldBounds.getSouthWest(), // Opposite of NE
            oldBounds.getNorthWest(), // Opposite of SE
            oldBounds.getNorthEast()  // Opposite of SW
          ];
          const anchor = anchors[idx];
          
          const newBounds = L.latLngBounds(movedPos, anchor);
          rect.setBounds(newBounds);
          tempBoundsRef.current = newBounds;
          
          syncMarkers([
            newBounds.getNorthWest(),
            newBounds.getNorthEast(),
            newBounds.getSouthEast(),
            newBounds.getSouthWest()
          ]);
        });
        
        cornerMarkersRef.current.push(marker);
      });

      return () => {}; // Rect layer removed next rerender loop top
    }

  }, [interactionState, bbox]);


  /* ── Zoom Controls ── */
  const zoomIn = () => leafletMapRef.current?.zoomIn();
  const zoomOut = () => leafletMapRef.current?.zoomOut();

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
        // ...
      }
    }, 3000);
  }, []);

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

          {/* ── Premium Control Nav Bar (Bottom Overlay) ── */}
          <motion.div
            className="absolute bottom-8 left-1/2 -translate-x-1/2 z-[500] glass px-3 py-2.5 rounded-2xl flex items-center gap-2 shadow-2xl border-primary/20 bg-card/60"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <button onClick={zoomIn} title="Zoom In" className="p-2.5 hover:bg-white/10 rounded-[12px] transition-colors text-foreground/80 hover:text-foreground active:scale-95">
              <ZoomIn className="w-[22px] h-[22px]" />
            </button>
            <button onClick={zoomOut} title="Zoom Out" className="p-2.5 hover:bg-white/10 rounded-[12px] transition-colors text-foreground/80 hover:text-foreground active:scale-95">
              <ZoomOut className="w-[22px] h-[22px]" />
            </button>
            
            <div className="w-px h-8 bg-border/60 mx-1.5" />

            {interactionState === "IDLE" && !bbox && (
              <button onClick={() => setInteractionState("DRAWING")} title="Draw Region" className="p-2.5 hover:bg-primary/20 rounded-[12px] transition-colors text-primary active:scale-95">
                <MousePointerSquareDashed className="w-[22px] h-[22px]" />
              </button>
            )}

            {interactionState === "IDLE" && bbox && (
              <button onClick={enterEditMode} title="Edit Region" className="p-2.5 hover:bg-yellow-500/20 rounded-[12px] transition-colors text-yellow-500 active:scale-95">
                <Pencil className="w-[22px] h-[22px]" />
              </button>
            )}

            {interactionState === "EDITING" && (
              <>
                <button onClick={confirmEdit} title="Confirm (Enter)" className="p-2.5 bg-primary/20 hover:bg-primary/40 rounded-[12px] transition-colors text-primary active:scale-95 shadow-[0_0_15px_hsl(166,72%,51%,0.2)]">
                  <Check className="w-[22px] h-[22px]" />
                </button>
                <button onClick={cancelEdit} title="Cancel (Esc)" className="p-2.5 hover:bg-destructive/20 rounded-[12px] transition-colors text-destructive active:scale-95">
                  <X className="w-[22px] h-[22px]" />
                </button>
              </>
            )}

            {(bbox || interactionState !== "IDLE") && (
              <>
                <div className="w-px h-8 bg-border/60 mx-1.5" />
                <button onClick={clearSelection} title="Delete Region" className="p-2.5 hover:bg-destructive/20 rounded-[12px] transition-colors text-destructive active:scale-95">
                  <Trash2 className="w-[22px] h-[22px]" />
                </button>
              </>
            )}
          </motion.div>
          {/* ── End Control Nav Bar ── */}
        </div>

        {/* ─── Side Panel ─── */}
        <motion.aside
          className="w-full lg:w-[380px] border-l border-border/40 bg-card/80 backdrop-blur-md overflow-y-auto z-[450]"
          initial={{ x: 40, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="p-6 space-y-6">
            <div>
              <h1 className="font-heading font-bold text-xl text-foreground">Pipeline Control</h1>
              <p className="text-xs text-muted-foreground mt-1">
                Select your region of interest, set the timeline, and launch the analysis pipeline.
              </p>
            </div>

            <div className="h-px bg-border/40" />

            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground font-mono uppercase tracking-wider">Status</span>
              <StatusBadge status={runStatus} state={interactionState} />
            </div>

            <AnimatePresence mode="wait">
              {bbox && interactionState === "IDLE" && (
                <motion.div
                  key="bbox-card"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="glass rounded-lg p-4 space-y-2 border border-primary/20 bg-primary/5"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-primary uppercase tracking-wider">Confirmed Region</span>
                  </div>
                  <p className="text-xs font-mono text-foreground/80 break-all leading-relaxed">
                    {formatBBox(bbox)}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="h-px bg-border/40" />

            <div className="space-y-4">
              <h2 className="text-xs font-mono text-muted-foreground uppercase tracking-wider">Parameters</h2>
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

            <motion.button
              onClick={runPipeline}
              disabled={!bbox || interactionState !== "IDLE" || runStatus === "RUNNING" || runStatus === "PENDING"}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-primary text-primary-foreground font-heading font-semibold text-sm
                         hover:shadow-[0_0_30px_hsl(166,72%,51%,0.4)] disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300"
              whileHover={{ scale: bbox && interactionState === "IDLE" && runStatus !== "RUNNING" ? 1.02 : 1 }}
              whileTap={{ scale: bbox && interactionState === "IDLE" && runStatus !== "RUNNING" ? 0.98 : 1 }}
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

            {runId && (
              <p className="text-[10px] font-mono text-muted-foreground text-center break-all">
                Run ID: {runId}
              </p>
            )}

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

            <AnimatePresence>
              {summary && (
               <motion.div
                 initial={{ opacity: 0, y: 6 }}
                 animate={{ opacity: 1, y: 0 }}
                 exit={{ opacity: 0 }}
                 className="glass rounded-lg p-4 space-y-2 border border-primary/20"
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
