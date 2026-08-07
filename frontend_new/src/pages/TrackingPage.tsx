import React, { useState, useCallback, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { MapContainer, TileLayer, Rectangle, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import {
  MapPin,
  ChevronRight,
  Layers,
  RotateCcw,
  Info,
} from "lucide-react";

const TILE_LAYERS = {
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Esri",
    label: "Satellite",
  },
  streets: {
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "&copy; OpenStreetMap",
    label: "Streets",
  },
};

interface BBox {
  north: string;
  south: string;
  east: string;
  west: string;
}

// Component to handle bbox drawing by click-drag
function BBoxDrawer({
  onBBoxChange,
  bbox,
}: {
  onBBoxChange: (b: BBox) => void;
  bbox: BBox | null;
}) {
  const [start, setStart] = useState<L.LatLng | null>(null);
  const [end, setEnd] = useState<L.LatLng | null>(null);
  const [drawing, setDrawing] = useState(false);

  useMapEvents({
    mousedown(e) {
      if (e.originalEvent.shiftKey) {
        setStart(e.latlng);
        setEnd(null);
        setDrawing(true);
      }
    },
    mousemove(e) {
      if (drawing && start) {
        setEnd(e.latlng);
      }
    },
    mouseup(e) {
      if (drawing && start) {
        const endPt = e.latlng;
        setEnd(endPt);
        setDrawing(false);
        const north = Math.max(start.lat, endPt.lat);
        const south = Math.min(start.lat, endPt.lat);
        const east = Math.max(start.lng, endPt.lng);
        const west = Math.min(start.lng, endPt.lng);
        onBBoxChange({
          north: north.toFixed(6),
          south: south.toFixed(6),
          east: east.toFixed(6),
          west: west.toFixed(6),
        });
      }
    },
  });

  const bounds =
    start && end
      ? L.latLngBounds(start, end)
      : bbox
      ? L.latLngBounds(
          [parseFloat(bbox.south), parseFloat(bbox.west)],
          [parseFloat(bbox.north), parseFloat(bbox.east)]
        )
      : null;

  return bounds && bounds.isValid() ? (
    <Rectangle
      bounds={bounds}
      pathOptions={{
        color: "#00C9A7",
        weight: 2,
        fillColor: "#00C9A7",
        fillOpacity: 0.15,
        dashArray: "6 4",
      }}
    />
  ) : null;
}

// Fly to bbox when it changes
function FlyToBBox({ bbox }: { bbox: BBox | null }) {
  const map = useMap();
  useEffect(() => {
    if (bbox) {
      const n = parseFloat(bbox.north);
      const s = parseFloat(bbox.south);
      const e = parseFloat(bbox.east);
      const w = parseFloat(bbox.west);
      if (!isNaN(n) && !isNaN(s) && !isNaN(e) && !isNaN(w) && n !== s && e !== w) {
        map.flyToBounds(
          [
            [s, w],
            [n, e],
          ],
          { padding: [60, 60], duration: 1 }
        );
      }
    }
  }, [bbox, map]);
  return null;
}

const TrackingPage: React.FC = () => {
  const [bbox, setBBox] = useState<BBox | null>(null);
  const [tileKey, setTileKey] = useState<"satellite" | "streets">("satellite");
  const [areaInfo, setAreaInfo] = useState<{ width: number; height: number; area: number } | null>(null);
  const [rawInput, setRawInput] = useState<string>("");
  const [inputError, setInputError] = useState<string | null>(null);
  const tile = TILE_LAYERS[tileKey];

  const applyCoordinates = (text: string) => {
    setInputError(null);
    if (!text.trim()) return;
    const parts = text.split(",").map((p) => p.trim());
    if (parts.length !== 4) {
      setInputError("Please enter 4 comma-separated values (West,South,East,North)");
      return;
    }
    const [w, s, e, n] = parts.map((p) => parseFloat(p));
    if (isNaN(w) || isNaN(s) || isNaN(e) || isNaN(n)) {
      setInputError("Invalid coordinate numbers");
      return;
    }
    const newBBox: BBox = {
      west: w.toString(),
      south: s.toString(),
      east: e.toString(),
      north: n.toString(),
    };
    handleBBoxChange(newBBox);
  };

  const handleBBoxChange = useCallback((b: BBox) => {
    setBBox(b);
    setRawInput(`${b.west},${b.south},${b.east},${b.north}`);
    setInputError(null);

    // Calculate approximate area
    const n = parseFloat(b.north);
    const s = parseFloat(b.south);
    const e = parseFloat(b.east);
    const w = parseFloat(b.west);
    const latDiff = Math.abs(n - s);
    const lonDiff = Math.abs(e - w);
    const avgLat = (n + s) / 2;
    const heightKm = latDiff * 111.32;
    const widthKm = lonDiff * 111.32 * Math.cos((avgLat * Math.PI) / 180);
    setAreaInfo({
      width: parseFloat(widthKm.toFixed(2)),
      height: parseFloat(heightKm.toFixed(2)),
      area: parseFloat((widthKm * heightKm).toFixed(2)),
    });
  }, []);

  const handleReset = () => {
    setBBox(null);
    setRawInput("");
    setInputError(null);
    setAreaInfo(null);
  };

  const canProceed =
    bbox &&
    bbox.north &&
    bbox.south &&
    bbox.east &&
    bbox.west &&
    parseFloat(bbox.north) !== parseFloat(bbox.south);

  return (
    <div className="min-h-screen bg-background pt-14">
      <div className="flex flex-col lg:flex-row h-[calc(100vh-56px)]">
        {/* Left panel */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-full lg:w-[400px] xl:w-[440px] flex-shrink-0 glass-card lg:rounded-none border-r border-border/30 p-6 overflow-y-auto"
        >
          <h1 className="font-heading text-2xl font-bold mb-1">Define Search Area</h1>
          <p className="text-sm text-muted-foreground mb-6">
            Hold <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs font-mono">Shift</kbd> + drag on the map to draw a bounding box, or paste coordinates below.
          </p>

          {/* Single Bounding Box Input */}
          <div className="mb-6 space-y-2">
            <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider block">
              Bounding Box Coordinates (West, South, East, North)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={rawInput}
                onChange={(e) => setRawInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    applyCoordinates(rawInput);
                  }
                }}
                placeholder="76.15723,10.73618,77.73926,11.49079"
                className="flex-1 px-3 py-2 bg-muted/50 border border-border/50 rounded-lg text-sm text-foreground placeholder-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
              />
              <button
                type="button"
                onClick={() => applyCoordinates(rawInput)}
                className="px-4 py-2 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary/90 transition-colors whitespace-nowrap"
              >
                Apply
              </button>
            </div>
            {inputError && (
              <p className="text-xs text-destructive mt-1">{inputError}</p>
            )}
            <p className="text-[11px] text-muted-foreground">
              Format: <span className="font-mono text-foreground">West, South, East, North</span>
            </p>
          </div>

          {/* Coordinate Display Table */}
          <div className="mb-6 glass-card overflow-hidden">
            <div className="px-4 py-2.5 bg-muted/30 border-b border-border/30 flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Parsed Bounding Box</span>
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/20 text-muted-foreground bg-muted/10">
                  <th className="text-left px-4 py-2 font-medium">Direction</th>
                  <th className="text-left px-4 py-2 font-medium">Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/10 font-mono">
                <tr>
                  <td className="px-4 py-2 text-foreground font-sans">North</td>
                  <td className="px-4 py-2 text-primary">{bbox?.north || "—"}</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-foreground font-sans">South</td>
                  <td className="px-4 py-2 text-primary">{bbox?.south || "—"}</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-foreground font-sans">East</td>
                  <td className="px-4 py-2 text-primary">{bbox?.east || "—"}</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 text-foreground font-sans">West</td>
                  <td className="px-4 py-2 text-primary">{bbox?.west || "—"}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Area info */}
          {areaInfo && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-4 mb-6 space-y-2"
            >
              <h3 className="text-sm font-semibold text-foreground mb-2">Area Info</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-muted-foreground">Width:</span>
                  <span className="ml-2 text-foreground font-medium">{areaInfo.width} km</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Height:</span>
                  <span className="ml-2 text-foreground font-medium">{areaInfo.height} km</span>
                </div>
                <div className="col-span-2">
                  <span className="text-muted-foreground">Area:</span>
                  <span className="ml-2 text-foreground font-medium">~{areaInfo.area} km²</span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-4 py-2.5 glass rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
            <Link
              to={canProceed ? "/dashboard" : "#"}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                canProceed
                  ? "bg-primary text-primary-foreground hover:bg-primary/90 hover:shadow-[0_0_20px_hsl(var(--primary)/0.3)]"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              }`}
            >
              Proceed to Dashboard
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>

          {/* Tip */}
          <div className="mt-6 flex items-start gap-2 text-xs text-muted-foreground/70">
            <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>
              Tip: Press Enter or click Apply to render the bounding box on the map.
            </span>
          </div>
        </motion.div>

        {/* Map */}
        <div className="flex-1 relative">
          <MapContainer
            center={[20, 0]}
            zoom={3}
            className="w-full h-full"
            style={{ background: "#0A1628" }}
            zoomControl={false}
          >
            <TileLayer url={tile.url} attribution={tile.attribution} />
            <BBoxDrawer onBBoxChange={handleBBoxChange} bbox={bbox} />
            <FlyToBBox bbox={bbox} />
          </MapContainer>

          {/* Tile layer toggle */}
          <div className="absolute top-4 right-4 z-[1000]">
            <button
              onClick={() => setTileKey(tileKey === "satellite" ? "streets" : "satellite")}
              className="flex items-center gap-2 px-3 py-2 glass rounded-lg text-sm font-medium hover:bg-muted/60 transition-colors"
            >
              <Layers className="w-4 h-4" />
              {tileKey === "satellite" ? "Streets" : "Satellite"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrackingPage;
