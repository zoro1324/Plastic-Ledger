import axios from 'axios';
import {
  SelectedRegion,
  DetectionResult,
  PlasticCluster,
  DashboardStats,
  AnalyticsData,
  PolymerType,
  Coordinate,
  BacktrackingResult,
  SourceAttribution,
  EnvironmentalReport,
} from '@/types';

// Mock base URL
const API_BASE_URL = '/api';

// Simulated delay for realistic UX
const SIMULATED_DELAY = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Mock plastic clusters database
const generateMockClusters = (region: SelectedRegion, count: number = 15): PlasticCluster[] => {
  const clusters: PlasticCluster[] = [];
  const { center, boundingBox } = region;
  const polymers: PolymerType[] = ['PET', 'HDPE', 'LDPE', 'PVC', 'PP', 'PS'];

  for (let i = 0; i < count; i++) {
    const latOffset = (Math.random() - 0.5) * (boundingBox.north - boundingBox.south);
    const lonOffset = (Math.random() - 0.5) * (boundingBox.east - boundingBox.west);

    clusters.push({
      id: `cluster_${region.id}_${i}`,
      latitude: center.latitude + latOffset,
      longitude: center.longitude + lonOffset,
      confidence: 0.7 + Math.random() * 0.3,
      estimatedArea: Math.random() * 2 + 0.1,
      estimatedQuantity: Math.random() * 50 + 5,
      polymerType: polymers[Math.floor(Math.random() * polymers.length)],
      estimatedAge: Math.floor(Math.random() * 30) + 1,
      riskLevel: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any,
      possibleSourceFactory: `Factory ${Math.floor(Math.random() * 20) + 1}`,
      nearestRiver: `River ${String.fromCharCode(65 + (Math.random() * 5))}`,
      nearestCoast: `Coast ${Math.floor(Math.random() * 10) + 1}`,
      timestamp: new Date(),
    });
  }

  return clusters;
};

// API Calls

export const selectRegionAPI = async (region: SelectedRegion): Promise<{ success: boolean }> => {
  await SIMULATED_DELAY(300);
  return { success: true };
};

export const calculateAreaAPI = async (region: SelectedRegion): Promise<{
  area: number;
  width: number;
  height: number;
  perimeter: number;
}> => {
  await SIMULATED_DELAY(200);
  return {
    area: region.area,
    width: region.width || 0,
    height: region.height || 0,
    perimeter: region.perimeter || 0,
  };
};

export const plasticsDetectionAPI = async (
  regionId: string,
  region: SelectedRegion
): Promise<DetectionResult> => {
  // Simulate multi-step pipeline
  await SIMULATED_DELAY(2000);
  const clusters = generateMockClusters(region, 20);
  const polymers: PolymerType[] = ['PET', 'HDPE', 'LDPE', 'PVC', 'PP', 'PS'];

  return {
    regionId,
    timestamp: new Date(),
    totalClusters: clusters.length,
    clusters,
    overallConfidence: 0.82,
    estimatedTotalPlastic: clusters.reduce((sum, c) => sum + c.estimatedQuantity, 0),
    dominantPolymer: polymers[Math.floor(Math.random() * polymers.length)],
    processingTimeSeconds: Math.random() * 120 + 80,
  };
};

export const polymerAnalysisAPI = async (
  clusterId: string
): Promise<{
  clusterId: string;
  detailedAnalysis: Record<string, number>;
}> => {
  await SIMULATED_DELAY(800);
  return {
    clusterId,
    detailedAnalysis: {
      spectralIndex1: Math.random(),
      spectralIndex2: Math.random(),
      polymerConfidence: 0.85 + Math.random() * 0.1,
      degradationLevel: Math.random(),
    },
  };
};

export const sourceAttributionAPI = async (
  clusterId: string,
  cluster: PlasticCluster
): Promise<BacktrackingResult> => {
  await SIMULATED_DELAY(1500);

  const sources: SourceAttribution[] = [];
  const sourceTypes: Array<'factory' | 'river' | 'shipping'> = ['factory', 'river', 'shipping'];

  for (let i = 0; i < 3; i++) {
    const latOffset = (Math.random() - 0.5) * 5;
    const lonOffset = (Math.random() - 0.5) * 5;

    sources.push({
      sourceType: sourceTypes[i],
      sourceName: `Source ${i + 1}`,
      latitude: cluster.latitude + latOffset,
      longitude: cluster.longitude + lonOffset,
      confidence: 0.6 + Math.random() * 0.35,
      distanceKm: Math.random() * 100 + 5,
      estimatedTravelDays: Math.random() * 30 + 5,
      trajectory: [],
    });
  }

  sources.sort((a, b) => b.confidence - a.confidence);

  return {
    clusterId,
    possibleSources: sources,
    mostLikelySource: sources[0],
    trajectoryAnimation: [cluster, sources[0]],
    timeToReachCluster: Math.random() * 30 + 5,
  };
};

export const dashboardStatsAPI = async (): Promise<DashboardStats> => {
  await SIMULATED_DELAY(500);
  return {
    totalClusters: Math.floor(Math.random() * 5000) + 1000,
    detectionAccuracy: 0.82 + Math.random() * 0.15,
    activeRegions: Math.floor(Math.random() * 50) + 20,
    estimatedPlasticWaste: Math.random() * 10000 + 5000,
    highRiskAreas: Math.floor(Math.random() * 30) + 5,
    averageProcessingTime: Math.random() * 120 + 60,
  };
};

export const analyticsAPI = async (): Promise<AnalyticsData> => {
  await SIMULATED_DELAY(800);

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
  const polymers: PolymerType[] = ['PET', 'HDPE', 'LDPE', 'PVC', 'PP', 'PS'];

  return {
    monthlyTrend: months.map((month, i) => ({
      month,
      clusters: Math.floor(Math.random() * 1000) + 200,
      weight: Math.random() * 5000 + 1000,
    })),
    polymerDistribution: polymers.map((polymer) => ({
      polymer,
      percentage: Math.random() * 25 + 5,
    })),
    highRiskRegions: [
      { region: 'Bay of Bengal', riskScore: 95 },
      { region: 'South China Sea', riskScore: 88 },
      { region: 'Mediterranean', riskScore: 82 },
      { region: 'North Atlantic', riskScore: 75 },
      { region: 'East China Sea', riskScore: 70 },
    ],
    detectionAccuracyTrend: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 86400000).toLocaleDateString(),
      accuracy: 0.75 + Math.random() * 0.2,
    })),
    sourceRanking: [
      { source: 'Industrial Discharge', contribution: 35 },
      { source: 'Fishing Equipment', contribution: 28 },
      { source: 'Shipping Routes', contribution: 22 },
      { source: 'River Pollution', contribution: 12 },
      { source: 'Coastal Waste', contribution: 3 },
    ],
    cleanupProgress: months.map((month, i) => ({
      month,
      progress: (i + 1) * 15,
    })),
  };
};

export const generateReportAPI = async (
  region: SelectedRegion,
  detectionResult: DetectionResult
): Promise<EnvironmentalReport> => {
  await SIMULATED_DELAY(1000);

  return {
    title: `Environmental Report - ${region.name || 'Selected Region'}`,
    description: 'Comprehensive plastic pollution analysis and attribution assessment',
    selectedRegion: region,
    plasticClusters: detectionResult.totalClusters,
    dominantPolymer: detectionResult.dominantPolymer,
    estimatedPlasticWaste: detectionResult.estimatedTotalPlastic,
    detectionAccuracy: detectionResult.overallConfidence,
    likelySource: 'Industrial Discharge - Bay of Bengal Region',
    cleanupPriority: 'high',
    riskScore: 78,
    recommendation:
      'Immediate intervention recommended. Coordinate with local authorities for cleanup operations.',
    timestamp: new Date(),
  };
};

// Export report as PDF/CSV (mock)
export const exportReportAPI = async (
  reportId: string,
  format: 'pdf' | 'csv' | 'geojson'
): Promise<Blob> => {
  await SIMULATED_DELAY(1000);
  const content =
    format === 'csv'
      ? 'cluster_id,latitude,longitude,confidence,polymer_type,risk_level\n'
      : JSON.stringify(
          { reportId, format, data: 'mock_data' },
          null,
          2
        );

  return new Blob([content], {
    type:
      format === 'pdf'
        ? 'application/pdf'
        : format === 'csv'
          ? 'text/csv'
          : 'application/geo+json',
  });
};

// Axios interceptors for error handling
export const setupAPIClient = () => {
  axios.interceptors.response.use(
    (response) => response,
    (error) => {
      console.error('API Error:', error);
      return Promise.reject(error);
    }
  );
};
