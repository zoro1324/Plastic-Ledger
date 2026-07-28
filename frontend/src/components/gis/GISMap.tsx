import React, { useRef, useEffect } from 'react';
import { MapContainer, TileLayer, FeatureGroup, CircleMarker, Popup } from 'react-leaflet';
import { FeatureGroup as LeafletFeatureGroup } from 'leaflet';
import { useGeoLocation } from '@/hooks/useGeoLocation';
import { useDashboard } from '@/context/DashboardContext';
import { PlasticCluster } from '@/types';
import 'leaflet/dist/leaflet.css';
import '@react-leaflet/core';

interface GISMapProps {
  clusters?: PlasticCluster[];
  onClusterClick?: (cluster: PlasticCluster) => void;
  onRegionDraw?: (coordinates: any) => void;
}

export const GISMap: React.FC<GISMapProps> = ({ clusters = [], onClusterClick, onRegionDraw }) => {
  const mapRef = useRef<any>(null);
  const { gisToolbar } = useDashboard();
  const { latitude, longitude } = useGeoLocation();

  const initialCenter: [number, number] = [latitude || 20, longitude || 0];

  // Color mapping for risk levels
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return '#22C55E';
      case 'medium':
        return '#FBBF24';
      case 'high':
        return '#F97316';
      case 'critical':
        return '#DC2626';
      default:
        return '#9CA3AF';
    }
  };

  return (
    <div className="w-full h-full bg-[#071A2E] rounded-xl overflow-hidden border border-[#0A84FF]/20">
      <MapContainer
        center={initialCenter}
        zoom={6}
        style={{ width: '100%', height: '100%' }}
        ref={mapRef}
        className="rounded-xl"
      >
        {/* Base Satellite Layer */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="&copy; Esri"
          maxZoom={19}
        />

        {/* Feature Group for drawing */}
        <FeatureGroup ref={mapRef}>
          {/* Plastic Clusters */}
          {clusters.map((cluster) => (
            <CircleMarker
              key={cluster.id}
              center={[cluster.latitude, cluster.longitude]}
              radius={Math.log(cluster.estimatedArea + 1) * 5}
              fillColor={getRiskColor(cluster.riskLevel)}
              color={getRiskColor(cluster.riskLevel)}
              weight={2}
              opacity={0.7}
              fillOpacity={0.6}
              eventHandlers={{
                click: () => onClusterClick?.(cluster),
              }}
            >
              <Popup>
                <div className="text-xs">
                  <p className="font-bold">{cluster.polymerType}</p>
                  <p>Risk: {cluster.riskLevel}</p>
                  <p>Area: {cluster.estimatedArea.toFixed(2)} km²</p>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </FeatureGroup>
      </MapContainer>
    </div>
  );
};
