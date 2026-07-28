// Core domain types for PlasticLedger AI

export interface Coordinate {
  latitude: number;
  longitude: number;
}

export interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface SelectedRegion {
  id: string;
  name?: string;
  type: 'rectangle' | 'polygon' | 'circle';
  coordinates: Coordinate[];
  boundingBox: BoundingBox;
  area: number; // km²
  width?: number; // km
  height?: number; // km
  perimeter?: number; // km
  center: Coordinate;
  timestamp: Date;
}

export type PolymerType = 'PET' | 'HDPE' | 'LDPE' | 'PVC' | 'PP' | 'PS' | 'Other';

export interface PlasticCluster {
  id: string;
  latitude: number;
  longitude: number;
  confidence: number; // 0-1
  estimatedArea: number; // km²
  estimatedQuantity: number; // metric tons
  polymerType: PolymerType;
  estimatedAge: number; // days
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  possibleSourceFactory?: string;
  nearestRiver?: string;
  nearestCoast?: string;
  timestamp: Date;
}

export interface DetectionResult {
  regionId: string;
  timestamp: Date;
  totalClusters: number;
  clusters: PlasticCluster[];
  overallConfidence: number;
  estimatedTotalPlastic: number; // metric tons
  dominantPolymer: PolymerType;
  processingTimeSeconds: number;
}

export interface Factory {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  type: string;
  industryRisk: number; // 0-1
  productionRate: number;
}

export interface OceanCurrent {
  latitude: number;
  longitude: number;
  velocityU: number; // m/s
  velocityV: number; // m/s
  speed: number; // m/s
  direction: number; // degrees
}

export interface SourceAttribution {
  sourceType: 'factory' | 'river' | 'shipping' | 'unknown';
  sourceName: string;
  latitude: number;
  longitude: number;
  confidence: number; // 0-1
  distanceKm: number;
  estimatedTravelDays: number;
  trajectory: Coordinate[];
}

export interface BacktrackingResult {
  clusterId: string;
  possibleSources: SourceAttribution[];
  mostLikelySource: SourceAttribution;
  trajectoryAnimation: Coordinate[];
  timeToReachCluster: number; // days
}

export interface EnvironmentalReport {
  title: string;
  description: string;
  selectedRegion: SelectedRegion;
  plasticClusters: number;
  dominantPolymer: PolymerType;
  estimatedPlasticWaste: number;
  detectionAccuracy: number;
  likelySource: string;
  cleanupPriority: 'low' | 'medium' | 'high' | 'critical';
  riskScore: number; // 0-100
  recommendation: string;
  timestamp: Date;
}

export interface DashboardStats {
  totalClusters: number;
  detectionAccuracy: number;
  activeRegions: number;
  estimatedPlasticWaste: number;
  highRiskAreas: number;
  averageProcessingTime: number; // seconds
}

export interface AnalyticsData {
  monthlyTrend: Array<{
    month: string;
    clusters: number;
    weight: number;
  }>;
  polymerDistribution: Array<{
    polymer: PolymerType;
    percentage: number;
  }>;
  highRiskRegions: Array<{
    region: string;
    riskScore: number;
  }>;
  detectionAccuracyTrend: Array<{
    date: string;
    accuracy: number;
  }>;
  sourceRanking: Array<{
    source: string;
    contribution: number;
  }>;
  cleanupProgress: Array<{
    month: string;
    progress: number;
  }>;
}

export interface Report {
  id: string;
  type: 'detection' | 'environmental' | 'attribution';
  title: string;
  description: string;
  regionId: string;
  timestamp: Date;
  data: unknown;
}

export interface AIWorkflowStep {
  stepNumber: number;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
}

export interface MapLayer {
  id: string;
  name: string;
  type: 'satellite' | 'terrain' | 'heatmap' | 'detection' | 'factory' | 'current' | 'wind' | 'river';
  visible: boolean;
  opacity: number; // 0-1
}

export interface GISToolbarState {
  activeTool: 'select' | 'rectangle' | 'polygon' | 'marker' | 'distance' | 'area' | 'zoom' | null;
  isDrawing: boolean;
  isVisible: boolean;
}
