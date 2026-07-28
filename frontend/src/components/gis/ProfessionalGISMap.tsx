import React, { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useDashboard } from '@/context/DashboardContext';
import { PlasticCluster, SelectedRegion } from '@/types';
import { MOCK_FACTORIES, RegionCalculationResult, calculateGeodesicStats } from '@/services/apiService';

// Fix Leaflet Default Icon Urls
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

interface ProfessionalGISMapProps {
  clusters: PlasticCluster[];
  currentRegion: SelectedRegion | null;
  onClusterClick: (cluster: PlasticCluster) => void;
  onRegionSelected: (coords: Array<[number, number]>, stats: RegionCalculationResult) => void;
  driftDays: number;
}

export const ProfessionalGISMap: React.FC<ProfessionalGISMapProps> = ({
  clusters,
  currentRegion,
  onClusterClick,
  onRegionSelected,
  driftDays,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);

  const { gisToolbar, mapLayers } = useDashboard();

  // Mouse coordinate live display state
  const [mouseCoords, setMouseCoords] = useState<{ lat: number; lng: number; zoom: number; x: number; y: number }>({
    lat: 12.9213,
    lng: 80.2488,
    zoom: 11,
    x: 0,
    y: 0,
  });

  // Layer groups references
  const drawnItemsRef = useRef<L.FeatureGroup | null>(null);
  const clusterLayerRef = useRef<L.LayerGroup | null>(null);
  const factoryLayerRef = useRef<L.LayerGroup | null>(null);
  const particleLayerRef = useRef<L.LayerGroup | null>(null);
  const currentsLayerRef = useRef<L.LayerGroup | null>(null);
  const tileLayersRef = useRef<{ [key: string]: L.TileLayer }>({});

  // Drawing state
  const isDrawingRef = useRef(false);
  const drawStartRef = useRef<L.LatLng | null>(null);
  const drawPolygonPointsRef = useRef<L.LatLng[]>([]);
  const tempShapeRef = useRef<L.Layer | null>(null);

  // Initialize Map cleanly
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [12.9213, 80.2488],
      zoom: 11,
      zoomControl: false,
    });
    mapRef.current = map;

    // Base Tile Layers
    const satelliteTile = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { attribution: '&copy; Esri World Imagery', maxZoom: 19 }
    );

    const terrainTile = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
      { attribution: '&copy; Esri Topo Map', maxZoom: 19 }
    );

    const oceanTile = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
      { attribution: '&copy; Esri Ocean Basemap', maxZoom: 13 }
    );

    satelliteTile.addTo(map);
    tileLayersRef.current = {
      satellite: satelliteTile,
      terrain: terrainTile,
      ocean: oceanTile,
    };

    // Feature / Layer Groups
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    drawnItemsRef.current = drawnItems;

    const clusterGroup = L.layerGroup().addTo(map);
    clusterLayerRef.current = clusterGroup;

    const factoryGroup = L.layerGroup().addTo(map);
    factoryLayerRef.current = factoryGroup;

    const particleGroup = L.layerGroup().addTo(map);
    particleLayerRef.current = particleGroup;

    const currentsGroup = L.layerGroup().addTo(map);
    currentsLayerRef.current = currentsGroup;

    // Add Zoom Control at bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Mousemove tracking
    map.on('mousemove', (e: L.LeafletMouseEvent) => {
      setMouseCoords({
        lat: Number(e.latlng.lat.toFixed(4)),
        lng: Number(e.latlng.lng.toFixed(4)),
        zoom: map.getZoom(),
        x: Math.round(e.containerPoint.x),
        y: Math.round(e.containerPoint.y),
      });

      if (isDrawingRef.current && drawStartRef.current && gisToolbar.activeTool === 'rectangle') {
        if (tempShapeRef.current) map.removeLayer(tempShapeRef.current);
        const bounds = L.latLngBounds(drawStartRef.current, e.latlng);
        tempShapeRef.current = L.rectangle(bounds, {
          color: '#00F5D4',
          weight: 2,
          fillColor: '#00F5D4',
          fillOpacity: 0.2,
          dashArray: '4, 4',
        }).addTo(map);
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Clear drawn shapes when currentRegion is reset to null
  useEffect(() => {
    if (!currentRegion && drawnItemsRef.current) {
      drawnItemsRef.current.clearLayers();
      if (tempShapeRef.current && mapRef.current) {
        mapRef.current.removeLayer(tempShapeRef.current);
        tempShapeRef.current = null;
      }
      drawPolygonPointsRef.current = [];
      isDrawingRef.current = false;
      drawStartRef.current = null;
    }
  }, [currentRegion]);

  // Base Layer & Overlay Visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const terrainVisible = mapLayers.find((l) => l.id === 'terrain')?.visible;
    const oceanVisible = mapLayers.find((l) => l.id === 'ocean')?.visible;

    Object.values(tileLayersRef.current).forEach((tile) => {
      if (map.hasLayer(tile)) map.removeLayer(tile);
    });

    if (terrainVisible && tileLayersRef.current.terrain) {
      tileLayersRef.current.terrain.addTo(map);
    } else if (oceanVisible && tileLayersRef.current.ocean) {
      tileLayersRef.current.ocean.addTo(map);
    } else if (tileLayersRef.current.satellite) {
      tileLayersRef.current.satellite.addTo(map);
    }

    // Factory Layer
    const factoryLayerConf = mapLayers.find((l) => l.id === 'factory');
    if (factoryLayerRef.current) {
      factoryLayerRef.current.clearLayers();
      if (factoryLayerConf?.visible) {
        MOCK_FACTORIES.forEach((fac) => {
          const icon = L.divIcon({
            className: 'custom-factory-icon',
            html: `<div class="w-8 h-8 bg-slate-900/90 border-2 border-cyan-400 rounded-full flex items-center justify-center text-cyan-300 shadow-[0_0_15px_rgba(0,245,212,0.6)] font-mono text-[10px] font-bold">🏭</div>`,
            iconSize: [32, 32],
            iconAnchor: [16, 16],
          });
          L.marker([fac.latitude, fac.longitude], { icon })
            .bindTooltip(`<div class="font-mono text-xs font-bold text-cyan-300">${fac.name}</div>`, {
              direction: 'top',
            })
            .addTo(factoryLayerRef.current!);
        });
      }
    }

    // Ocean Currents Layer
    const currentLayerConf = mapLayers.find((l) => l.id === 'current');
    if (currentsLayerRef.current) {
      currentsLayerRef.current.clearLayers();
      if (currentLayerConf?.visible) {
        const gridPoints = [
          [12.85, 80.20], [12.88, 80.25], [12.92, 80.30], [12.96, 80.32],
          [13.00, 80.28], [13.05, 80.35], [12.80, 80.22], [13.10, 80.38],
        ];
        gridPoints.forEach(([lat, lng]) => {
          L.polyline([[lat, lng], [lat + 0.025, lng + 0.035]], {
            color: '#00F5D4',
            weight: 2,
            opacity: 0.6,
            dashArray: '6, 6',
          }).addTo(currentsLayerRef.current!);
        });
      }
    }
  }, [mapLayers]);

  // Plastic Clusters & Particle Path Animation
  useEffect(() => {
    const map = mapRef.current;
    const group = clusterLayerRef.current;
    if (!map || !group) return;

    group.clearLayers();

    const detectionLayerConf = mapLayers.find((l) => l.id === 'detection');
    if (!detectionLayerConf?.visible) return;

    clusters.forEach((cluster) => {
      const latDrift = cluster.latitude + (driftDays * 0.0008);
      const lngDrift = cluster.longitude + (driftDays * 0.0012);

      let color = '#22C55E';
      if (cluster.riskLevel === 'critical' || cluster.riskLevel === 'high') color = '#EF4444';
      else if (cluster.riskLevel === 'medium') color = '#F59E0B';
      else if (cluster.riskLevel === 'low') color = '#EAB308';

      const circle = L.circleMarker([latDrift, lngDrift], {
        radius: Math.max(6, Math.min(16, cluster.estimatedArea / 50)),
        color,
        fillColor: color,
        fillOpacity: 0.7,
        weight: 2,
        className: 'animate-pulse cursor-pointer',
      });

      circle.on('click', () => onClusterClick(cluster));
      circle.bindTooltip(
        `<div class="font-mono text-xs">
          <span class="font-bold text-cyan-300">${cluster.id.toUpperCase()}</span> (${cluster.polymerType})<br/>
          Risk: <span style="color:${color}">${cluster.riskLevel.toUpperCase()}</span> | Waste: ${cluster.estimatedQuantity} T
         </div>`,
        { direction: 'top' }
      );

      circle.addTo(group);
    });

    if (clusters.length > 0 && particleLayerRef.current) {
      particleLayerRef.current.clearLayers();
      const fac = MOCK_FACTORIES[0];
      const targetCluster = clusters[0];

      const polyline = L.polyline(
        [
          [fac.latitude, fac.longitude],
          [fac.latitude - 0.015, fac.longitude + 0.02],
          [fac.latitude - 0.03, fac.longitude + 0.045],
          [(fac.latitude + targetCluster.latitude) / 2, (fac.longitude + targetCluster.longitude) / 2],
          [targetCluster.latitude, targetCluster.longitude],
        ],
        { color: '#0A84FF', weight: 3, opacity: 0.8, dashArray: '8, 8' }
      ).addTo(particleLayerRef.current);

      polyline.bindTooltip(
        `<div class="font-mono text-xs text-cyan-300">Pollution Transport Pathway (Factory → Ocean Current)</div>`,
        { sticky: true }
      );
    }
  }, [clusters, driftDays, mapLayers, onClusterClick]);

  // Click handler for drawing tools
  const handleMapClick = useCallback(
    (e: L.LeafletMouseEvent) => {
      const map = mapRef.current;
      const tool = gisToolbar.activeTool;
      if (!map || !tool) return;

      if (tool === 'rectangle' || tool === 'select') {
        if (!isDrawingRef.current) {
          isDrawingRef.current = true;
          drawStartRef.current = e.latlng;
        } else {
          isDrawingRef.current = false;
          const bounds = L.latLngBounds(drawStartRef.current!, e.latlng);
          if (tempShapeRef.current) map.removeLayer(tempShapeRef.current);

          const coords: Array<[number, number]> = [
            [bounds.getNorthWest().lat, bounds.getNorthWest().lng],
            [bounds.getNorthEast().lat, bounds.getNorthEast().lng],
            [bounds.getSouthEast().lat, bounds.getSouthEast().lng],
            [bounds.getSouthWest().lat, bounds.getSouthWest().lng],
          ];

          if (drawnItemsRef.current) {
            drawnItemsRef.current.clearLayers();
            L.polygon(coords, {
              color: '#0A84FF',
              weight: 2,
              fillColor: '#0A84FF',
              fillOpacity: 0.15,
              dashArray: '5, 5',
            }).addTo(drawnItemsRef.current);
          }

          const stats = calculateGeodesicStats(coords);
          onRegionSelected(coords, stats);
        }
      } else if (tool === 'polygon' || tool === 'area') {
        drawPolygonPointsRef.current.push(e.latlng);
        if (tempShapeRef.current) map.removeLayer(tempShapeRef.current);
        const pts = drawPolygonPointsRef.current.map((p) => [p.lat, p.lng] as [number, number]);

        tempShapeRef.current = L.polyline(pts, {
          color: '#00F5D4',
          weight: 2,
          dashArray: '4, 4',
        }).addTo(map);

        if (pts.length >= 3) {
          if (drawnItemsRef.current) {
            drawnItemsRef.current.clearLayers();
            L.polygon(pts, {
              color: '#0A84FF',
              weight: 2,
              fillColor: '#0A84FF',
              fillOpacity: 0.15,
            }).addTo(drawnItemsRef.current);
          }
          const stats = calculateGeodesicStats(pts);
          onRegionSelected(pts, stats);
        }
      } else if (tool === 'marker') {
        if (drawnItemsRef.current) {
          L.marker(e.latlng)
            .bindTooltip(`Coordinates: ${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`)
            .addTo(drawnItemsRef.current);
        }
        const coords: Array<[number, number]> = [
          [e.latlng.lat + 0.02, e.latlng.lng - 0.02],
          [e.latlng.lat + 0.02, e.latlng.lng + 0.02],
          [e.latlng.lat - 0.02, e.latlng.lng + 0.02],
          [e.latlng.lat - 0.02, e.latlng.lng - 0.02],
        ];
        const stats = calculateGeodesicStats(coords);
        onRegionSelected(coords, stats);
      }
    },
    [gisToolbar.activeTool, onRegionSelected]
  );

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    map.on('click', handleMapClick);
    return () => {
      map.off('click', handleMapClick);
    };
  }, [handleMapClick]);

  return (
    <div className="relative w-full h-full bg-[#07172A] overflow-hidden select-none">
      <div ref={mapContainerRef} className="w-full h-full z-0" />

      {/* Clean Live Coordinate Panel placed in Bottom Right */}
      <div className="absolute right-14 bottom-6 z-[800] bg-[#07172A]/90 border border-[#0A84FF]/30 backdrop-blur-xl px-3 py-1.5 rounded-xl text-white font-mono text-[11px] shadow-lg flex items-center space-x-3">
        <div className="flex items-center space-x-1">
          <span className="text-gray-400">LAT:</span>
          <span className="text-cyan-300 font-bold">{mouseCoords.lat}°</span>
        </div>
        <div className="h-3 w-px bg-white/10" />
        <div className="flex items-center space-x-1">
          <span className="text-gray-400">LNG:</span>
          <span className="text-cyan-300 font-bold">{mouseCoords.lng}°</span>
        </div>
        <div className="h-3 w-px bg-white/10" />
        <div className="flex items-center space-x-1">
          <span className="text-gray-400">ZOOM:</span>
          <span className="text-white font-bold">{mouseCoords.zoom}</span>
        </div>
      </div>
    </div>
  );
};
