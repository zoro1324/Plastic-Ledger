import React, { createContext, useContext, ReactNode, useState, useCallback } from 'react';
import {
  SelectedRegion,
  DetectionResult,
  MapLayer,
  GISToolbarState,
  PlasticCluster,
} from '@/types';

interface DashboardContextType {
  selectedRegion: SelectedRegion | null;
  setSelectedRegion: (region: SelectedRegion | null) => void;
  detectionResult: DetectionResult | null;
  setDetectionResult: (result: DetectionResult | null) => void;
  mapLayers: MapLayer[];
  toggleLayer: (layerId: string) => void;
  setLayerOpacity: (layerId: string, opacity: number) => void;
  gisToolbar: GISToolbarState;
  setGisToolbar: (state: GISToolbarState) => void;
  selectedCluster: PlasticCluster | null;
  setSelectedCluster: (cluster: PlasticCluster | null) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
  analysisProgress: number;
  setAnalysisProgress: (progress: number) => void;
}

const defaultMapLayers: MapLayer[] = [
  { id: 'satellite', name: 'Satellite View', type: 'satellite', visible: true, opacity: 1 },
  { id: 'terrain', name: 'Terrain', type: 'terrain', visible: false, opacity: 0.7 },
  { id: 'ocean', name: 'Ocean Layer', type: 'heatmap', visible: false, opacity: 0.6 },
  { id: 'detection', name: 'Plastic Detection', type: 'detection', visible: true, opacity: 0.8 },
  { id: 'factory', name: 'Industrial Sources', type: 'factory', visible: false, opacity: 0.7 },
  { id: 'current', name: 'Ocean Currents', type: 'current', visible: false, opacity: 0.6 },
  { id: 'wind', name: 'Wind Patterns', type: 'wind', visible: false, opacity: 0.5 },
  { id: 'river', name: 'River Systems', type: 'river', visible: false, opacity: 0.7 },
];

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [selectedRegion, setSelectedRegion] = useState<SelectedRegion | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [mapLayers, setMapLayers] = useState<MapLayer[]>(defaultMapLayers);
  const [gisToolbar, setGisToolbar] = useState<GISToolbarState>({
    activeTool: null,
    isDrawing: false,
    isVisible: true,
  });
  const [selectedCluster, setSelectedCluster] = useState<PlasticCluster | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);

  const toggleLayer = useCallback((layerId: string) => {
    setMapLayers((prev) =>
      prev.map((layer) =>
        layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
      )
    );
  }, []);

  const setLayerOpacity = useCallback((layerId: string, opacity: number) => {
    setMapLayers((prev) =>
      prev.map((layer) =>
        layer.id === layerId ? { ...layer, opacity: Math.min(1, Math.max(0, opacity)) } : layer
      )
    );
  }, []);

  return (
    <DashboardContext.Provider
      value={{
        selectedRegion,
        setSelectedRegion,
        detectionResult,
        setDetectionResult,
        mapLayers,
        toggleLayer,
        setLayerOpacity,
        gisToolbar,
        setGisToolbar,
        selectedCluster,
        setSelectedCluster,
        isAnalyzing,
        setIsAnalyzing,
        analysisProgress,
        setAnalysisProgress,
      }}
    >
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (context === undefined) {
    throw new Error('useDashboard must be used within DashboardProvider');
  }
  return context;
}
