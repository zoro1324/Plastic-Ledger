import React, { useEffect, useState, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import L from "leaflet";
import { loadFinalReport } from "@/services/dataService";
import { DetectionFeatureCollection, DetectionFeature, POLYMER_COLORS } from "@/types";
import {
  Layers,
  Filter,
  X,
  Eye,
  EyeOff,
  ChevronRight,
  MapPin,
} from "lucide-react";

// Fit map to GeoJSON bounds
function FitBounds({ data }: { data: DetectionFeatureCollection | null }) {
  const map = useMap();
  useEffect(() => {
    if (data && data.features.length > 0) {
      const layer = L.geoJSON(data as any);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.flyToBounds(bounds, { padding: [40, 40], duration: 1.2 });
      }
    }
  }, [data, map]);
  return null;
}

const DetectionMapPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get("id") ?? "run_001";
  const [data, setData] = useState<DetectionFeatureCollection | null>(null);
  const [loading, setLoading] = useState(true);
  const [showFP, setShowFP] = useState(true);
  const [minConf, setMinConf] = useState(0);
  const [minArea, setMinArea] = useState(0);
  const [typeFilter, setTypeFilter] = useState("all");
  const [tileKey, setTileKey] = useState<"satellite" | "dark">("satellite");
  const [selected, setSelected] = useState<DetectionFeature | null>(null);
  const [showFilters, setShowFilters] = useState(true);

  useEffect(() => {
    loadFinalReport(runId).then((d) => { setData(d); setLoading(false); }).catch(() => setLoading(false));
  }, [runId]);

  const filtered = useMemo(() => {
    if (!data) return null;
    const features = data.features.filter((f) => {
      const p = f.properties;
      if (!showFP && p.is_false_positive) return false;
      if (p.mean_confidence < minConf) return false;
      if (p.area_m2 < minArea) return false;
      if (typeFilter !== "all" && p.polymer_type !== typeFilter) return false;
      return true;
    });
    return { ...data, features };
  }, [data, showFP, minConf, minArea, typeFilter]);

  const polymerTypes = useMemo(() => {
    if (!data) return [];
    const types = new Set(data.features.map((f) => f.properties.polymer_type));
    return Array.from(types).sort();
  }, [data]);

  const geoStyle = (feature: any) => {
    const p = feature.properties;
    const color = POLYMER_COLORS[p.polymer_type] || "#6B7280";
    return {
      color,
      weight: 1.5,
      fillColor: color,
      fillOpacity: Math.max(0.15, p.mean_confidence * 0.5),
    };
  };

  const onEachFeature = (feature: any, layer: L.Layer) => {
    layer.on("click", () => setSelected(feature));
  };

  const tiles = tileKey === "satellite"
    ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    : "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

  const plasticCount = filtered?.features.filter((f) => !f.properties.is_false_positive).length || 0;
  const totalCount = filtered?.features.length || 0;

  if (loading) {
    return (
      <div className="min-h-screen bg-background pt-14 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background pt-14 flex">
      {/* Filters sidebar */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="flex-shrink-0 glass-card lg:rounded-none border-r border-border/30 overflow-y-auto"
          >
            <div className="p-5 space-y-5">
              <div className="flex items-center justify-between">
                <h2 className="font-heading font-semibold">Filters</h2>
                <button onClick={() => setShowFilters(false)} className="text-muted-foreground hover:text-foreground">
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Show FP toggle */}
              <div>
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm text-muted-foreground">Show False Positives</span>
                  <button
                    onClick={() => setShowFP(!showFP)}
                    className={`w-10 h-5 rounded-full transition-colors relative ${showFP ? "bg-primary" : "bg-muted"}`}
                  >
                    <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${showFP ? "left-5" : "left-0.5"}`} />
                  </button>
                </label>
              </div>

              {/* Polymer type filter */}
              <div>
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5 block">
                  Filter by Polymer Type
                </label>
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                  className="w-full px-3 py-2 bg-muted/50 border border-border/50 rounded-lg text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
                >
                  <option value="all">All Types</option>
                  {polymerTypes.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              {/* Confidence slider */}
              <div>
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5 block">
                  Min Confidence: {minConf.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={minConf}
                  onChange={(e) => setMinConf(parseFloat(e.target.value))}
                  className="w-full accent-primary"
                />
              </div>

              {/* Area slider */}
              <div>
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1.5 block">
                  Min Area: {minArea} m²
                </label>
                <input
                  type="range"
                  min={0}
                  max={10000}
                  step={100}
                  value={minArea}
                  onChange={(e) => setMinArea(parseInt(e.target.value))}
                  className="w-full accent-primary"
                />
              </div>

              {/* Stats */}
              <div className="glass-card p-3 space-y-1.5 text-xs">
                <div className="flex justify-between"><span className="text-muted-foreground">Showing:</span><span className="font-semibold">{totalCount} clusters</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Plastic:</span><span className="font-semibold text-destructive">{plasticCount}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">False Positive:</span><span className="font-semibold">{totalCount - plasticCount}</span></div>
              </div>

              {/* Legend */}
              <div>
                <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Legend</h4>
                <div className="space-y-1">
                  {Object.entries(POLYMER_COLORS).map(([name, color]) => (
                    <div key={name} className="flex items-center gap-2 text-xs">
                      <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: color }} />
                      <span className="text-muted-foreground truncate">{name}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Layers */}
              <div>
                <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Layers</h4>
                <div className="space-y-1.5">
                  {[
                    { key: "satellite", label: "Satellite View" },
                    { key: "dark", label: "Dark Map" },
                  ].map((layer) => (
                    <button
                      key={layer.key}
                      onClick={() => setTileKey(layer.key as any)}
                      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                        tileKey === layer.key ? "bg-primary/15 text-primary border border-primary/30" : "text-muted-foreground hover:bg-muted/30"
                      }`}
                    >
                      {tileKey === layer.key ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                      {layer.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Map */}
      <div className="flex-1 relative">
        {!showFilters && (
          <button
            onClick={() => setShowFilters(true)}
            className="absolute top-4 left-4 z-[1000] flex items-center gap-1.5 px-3 py-2 glass rounded-lg text-sm font-medium hover:bg-muted/60 transition-colors"
          >
            <Filter className="w-4 h-4" />
            Filters
          </button>
        )}

        <MapContainer
          center={[16.1, -88.4]}
          zoom={10}
          className="w-full h-[calc(100vh-56px)]"
          style={{ background: "#0A1628" }}
          zoomControl={false}
        >
          <TileLayer url={tiles} attribution="Esri | OSM" />
          {filtered && (
            <GeoJSON
              key={`${showFP}-${minConf}-${minArea}-${typeFilter}`}
              data={filtered as any}
              style={geoStyle}
              onEachFeature={onEachFeature}
            />
          )}
          <FitBounds data={filtered} />
        </MapContainer>

        {/* Tile toggle */}
        <div className="absolute top-4 right-4 z-[1000]">
          <button
            onClick={() => setTileKey(tileKey === "satellite" ? "dark" : "satellite")}
            className="flex items-center gap-2 px-3 py-2 glass rounded-lg text-sm font-medium hover:bg-muted/60 transition-colors"
          >
            <Layers className="w-4 h-4" />
            {tileKey === "satellite" ? "Dark" : "Satellite"}
          </button>
        </div>

        {/* Selected cluster panel */}
        <AnimatePresence>
          {selected && (
            <motion.div
              initial={{ x: 300, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 300, opacity: 0 }}
              className="absolute top-4 right-4 bottom-4 w-[320px] z-[1000] glass-card overflow-y-auto"
            >
              <div className="p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-heading font-semibold">Cluster #{selected.properties.cluster_id}</h3>
                  <button onClick={() => setSelected(null)} className="text-muted-foreground hover:text-foreground">
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Type badge */}
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: POLYMER_COLORS[selected.properties.polymer_type] }}
                  />
                  <span className="text-sm font-medium">{selected.properties.polymer_type}</span>
                  {selected.properties.is_false_positive && (
                    <span className="px-2 py-0.5 bg-yellow-500/15 text-yellow-400 rounded text-xs font-medium">FP</span>
                  )}
                </div>

                {/* Details */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-muted-foreground text-xs block">Area</span>
                    <span className="font-semibold">{selected.properties.area_m2.toLocaleString()} m²</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground text-xs block">Confidence</span>
                    <span className="font-semibold">{(selected.properties.mean_confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground text-xs block">Latitude</span>
                    <span className="font-mono text-xs">{selected.properties.centroid_lat.toFixed(6)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground text-xs block">Longitude</span>
                    <span className="font-mono text-xs">{selected.properties.centroid_lon.toFixed(6)}</span>
                  </div>
                </div>

                {/* Spectral indices */}
                <div>
                  <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Spectral Indices</h4>
                  {[
                    { label: "PI", value: selected.properties.pi_value, color: "#3B82F6" },
                    { label: "SR", value: selected.properties.sr_value, color: "#8B5CF6" },
                    { label: "NSI", value: selected.properties.nsi_value, color: "#10B981" },
                    { label: "FDI", value: selected.properties.fdi_value, color: "#F59E0B" },
                  ].map((idx) => (
                    <div key={idx.label} className="flex items-center gap-2 mb-1.5">
                      <span className="w-8 text-xs text-muted-foreground font-mono">{idx.label}</span>
                      <div className="flex-1 h-2 bg-muted/50 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${Math.min(100, Math.abs(idx.value) * 100)}%`,
                            backgroundColor: idx.color,
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono w-12 text-right">{idx.value.toFixed(4)}</span>
                    </div>
                  ))}
                </div>

                {/* Attribution */}
                {selected.properties.source_type && selected.properties.source_type !== "" && (
                  <div className="glass-card p-3 space-y-1 text-xs">
                    <h4 className="font-medium text-muted-foreground uppercase tracking-wider">Attribution</h4>
                    <div className="flex justify-between"><span className="text-muted-foreground">Source:</span><span className="font-semibold">{selected.properties.source_type}</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Score:</span><span className="font-semibold">{selected.properties.attribution_score}</span></div>
                    {selected.properties.country && (
                      <div className="flex justify-between"><span className="text-muted-foreground">Country:</span><span>{selected.properties.country}</span></div>
                    )}
                    {selected.properties.explanation && (
                      <p className="text-muted-foreground mt-2 leading-relaxed">{selected.properties.explanation}</p>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default DetectionMapPage;
