import * as turf from '@turf/turf';
import { SelectedRegion, PlasticCluster, SourceAttribution, EnvironmentalReport, BoundingBox, Coordinate, PolymerType } from '@/types';

export interface RegionCalculationResult {
  areaKm2: number;
  widthKm: number;
  heightKm: number;
  perimeterKm: number;
  center: Coordinate;
  boundingBox: BoundingBox;
  corners: {
    northWest: Coordinate;
    northEast: Coordinate;
    southEast: Coordinate;
    southWest: Coordinate;
  };
}

/**
 * Perform exact Turf.js geodesic calculations for any polygon/rectangle coordinates.
 * Coordinates passed as Array<[latitude, longitude]>
 */
export function calculateGeodesicStats(coords: Array<[number, number]>): RegionCalculationResult {
  if (!coords || coords.length < 3) {
    // Default fallback if insufficient points
    return {
      areaKm2: 0,
      widthKm: 0,
      heightKm: 0,
      perimeterKm: 0,
      center: { latitude: 0, longitude: 0 },
      boundingBox: { north: 0, south: 0, east: 0, west: 0 },
      corners: {
        northWest: { latitude: 0, longitude: 0 },
        northEast: { latitude: 0, longitude: 0 },
        southEast: { latitude: 0, longitude: 0 },
        southWest: { latitude: 0, longitude: 0 },
      },
    };
  }

  // Turf expects [lng, lat] format
  const turfCoords = coords.map(([lat, lng]) => [lng, lat]);
  // Ensure closed polygon
  if (
    turfCoords[0][0] !== turfCoords[turfCoords.length - 1][0] ||
    turfCoords[0][1] !== turfCoords[turfCoords.length - 1][1]
  ) {
    turfCoords.push(turfCoords[0]);
  }

  const poly = turf.polygon([turfCoords]);
  const areaM2 = turf.area(poly);
  const areaKm2 = Number((areaM2 / 1_000_000).toFixed(2));

  const bbox = turf.bbox(poly); // [west, south, east, north]
  const west = bbox[0];
  const south = bbox[1];
  const east = bbox[2];
  const north = bbox[3];

  const centerPoint = turf.center(poly);
  const centerLat = Number(centerPoint.geometry.coordinates[1].toFixed(4));
  const centerLng = Number(centerPoint.geometry.coordinates[0].toFixed(4));

  // Geodesic distances for width and height
  const nw = turf.point([west, north]);
  const ne = turf.point([east, north]);
  const sw = turf.point([west, south]);

  const widthKm = Number(turf.distance(nw, ne, { units: 'kilometers' }).toFixed(2));
  const heightKm = Number(turf.distance(nw, sw, { units: 'kilometers' }).toFixed(2));
  const line = turf.polygonToLine(poly);
  const perimeterKm = Number(turf.length(line, { units: 'kilometers' }).toFixed(2));

  return {
    areaKm2,
    widthKm,
    heightKm,
    perimeterKm,
    center: { latitude: centerLat, longitude: centerLng },
    boundingBox: { north, south, east, west },
    corners: {
      northWest: { latitude: north, longitude: west },
      northEast: { latitude: north, longitude: east },
      southEast: { latitude: south, longitude: east },
      southWest: { latitude: south, longitude: west },
    },
  };
}

/**
 * Mock Factory sources nearby ocean regions (e.g. Chennai / Bay of Bengal / Coastal hubs)
 */
export const MOCK_FACTORIES = [
  {
    id: 'fac-1',
    name: 'Industrial Zone Alpha (Petrochemicals)',
    latitude: 13.0827,
    longitude: 80.2707,
    river: 'Adyar River',
    distanceToCoastKm: 2.4,
  },
  {
    id: 'fac-2',
    name: 'Coastal Packaging Plant B',
    latitude: 12.8387,
    longitude: 80.2045,
    river: 'Kovalam Estuary',
    distanceToCoastKm: 1.2,
  },
  {
    id: 'fac-3',
    name: 'North Port Logistics Hub',
    latitude: 13.2500,
    longitude: 80.3200,
    river: 'Kosasthalaiyar River',
    distanceToCoastKm: 0.8,
  },
];

/**
 * Mock APIs for Dashboard
 */
export const mockApiService = {
  // POST /api/select-region
  async selectRegion(regionData: Partial<SelectedRegion>): Promise<{ success: boolean; region: SelectedRegion }> {
    await new Promise((res) => setTimeout(res, 300));
    const coords = regionData.coordinates?.map((c) => [c.latitude, c.longitude] as [number, number]) || [];
    const stats = calculateGeodesicStats(coords);

    const region: SelectedRegion = {
      id: regionData.id || `reg-${Date.now()}`,
      name: regionData.name || 'Custom Selected Area',
      type: regionData.type || 'rectangle',
      coordinates: regionData.coordinates || [],
      boundingBox: stats.boundingBox,
      area: stats.areaKm2,
      width: stats.widthKm,
      height: stats.heightKm,
      perimeter: stats.perimeterKm,
      center: stats.center,
      timestamp: new Date(),
    };
    return { success: true, region };
  },

  // POST /api/calculate-area
  async calculateArea(coords: Array<[number, number]>): Promise<RegionCalculationResult> {
    await new Promise((res) => setTimeout(res, 150));
    return calculateGeodesicStats(coords);
  },

  // POST /api/plastic-detection
  async detectPlastics(region: SelectedRegion): Promise<PlasticCluster[]> {
    await new Promise((res) => setTimeout(res, 800));
    const { center, area } = region;
    const count = Math.max(8, Math.min(60, Math.floor(area * 3.5)));
    const clusters: PlasticCluster[] = [];

    const polymers: PolymerType[] = ['PET', 'HDPE', 'LDPE', 'PP', 'PS', 'PVC'];
    const risks: Array<'low' | 'medium' | 'high' | 'critical'> = ['low', 'medium', 'high', 'critical'];

    for (let i = 1; i <= count; i++) {
      const latOffset = (Math.random() - 0.5) * 0.08;
      const lngOffset = (Math.random() - 0.5) * 0.08;
      const lat = Number((center.latitude + latOffset).toFixed(4));
      const lng = Number((center.longitude + lngOffset).toFixed(4));

      const factory = MOCK_FACTORIES[i % MOCK_FACTORIES.length];
      const pt1 = turf.point([lng, lat]);
      const ptFac = turf.point([factory.longitude, factory.latitude]);
      const distKm = Number(turf.distance(pt1, ptFac, { units: 'kilometers' }).toFixed(1));

      const riskLevel = risks[Math.floor(Math.random() * risks.length)];

      clusters.push({
        id: `cluster-${i}`,
        latitude: lat,
        longitude: lng,
        confidence: Number((0.82 + Math.random() * 0.16).toFixed(2)),
        estimatedArea: Number((120 + Math.random() * 850).toFixed(0)), // m²
        estimatedQuantity: Number((0.05 + Math.random() * 0.45).toFixed(2)), // Metric Tons
        polymerType: polymers[Math.floor(Math.random() * polymers.length)],
        estimatedAge: Math.floor(20 + Math.random() * 180), // Days
        riskLevel,
        possibleSourceFactory: factory.name,
        nearestRiver: factory.river,
        nearestCoast: `${distKm} km`,
        timestamp: new Date(),
      });
    }

    return clusters;
  },

  // POST /api/source-attribution
  async getSourceAttribution(cluster: PlasticCluster): Promise<{
    factoryDistanceKm: number;
    riverDistanceKm: number;
    travelTimeDays: number;
    trajectory: Coordinate[];
    factory: string;
    river: string;
  }> {
    await new Promise((res) => setTimeout(res, 200));

    const factory = MOCK_FACTORIES[0];
    const ptCluster = turf.point([cluster.longitude, cluster.latitude]);
    const ptFac = turf.point([factory.longitude, factory.latitude]);
    const distKm = Number(turf.distance(ptCluster, ptFac, { units: 'kilometers' }).toFixed(1));
    const riverDist = Number((distKm * 0.22).toFixed(1));
    const travelDays = Math.max(3, Math.round(distKm / 1.6));

    // Generate animated flow points (Factory -> River -> Coast -> Cluster)
    const trajectory: Coordinate[] = [
      { latitude: factory.latitude, longitude: factory.longitude },
      { latitude: factory.latitude - 0.015, longitude: factory.longitude + 0.02 },
      { latitude: factory.latitude - 0.03, longitude: factory.longitude + 0.045 },
      { latitude: (factory.latitude + cluster.latitude) / 2, longitude: (factory.longitude + cluster.longitude) / 2 },
      { latitude: cluster.latitude, longitude: cluster.longitude },
    ];

    return {
      factoryDistanceKm: distKm,
      riverDistanceKm: riverDist,
      travelTimeDays: travelDays,
      trajectory,
      factory: factory.name,
      river: factory.river,
    };
  },

  // GET /api/plastic-clusters
  async getClusters(): Promise<PlasticCluster[]> {
    return this.detectPlastics({
      id: 'default',
      type: 'rectangle',
      coordinates: [],
      boundingBox: { north: 13.1, south: 12.8, east: 80.35, west: 80.15 },
      area: 12.84,
      center: { latitude: 12.9213, longitude: 80.2488 },
      timestamp: new Date(),
    });
  },

  // GET /api/region-summary
  async getRegionSummary(clusters: PlasticCluster[], regionArea: number): Promise<EnvironmentalReport> {
    const totalPlastic = clusters.reduce((acc, c) => acc + c.estimatedQuantity, 0);
    const highRiskCount = clusters.filter((c) => c.riskLevel === 'high' || c.riskLevel === 'critical').length;
    const dominantPolymer: PolymerType = 'PET';

    return {
      title: 'Ocean Synthetic Waste Analysis Report',
      description: 'Satellite hyperspectral ocean surface scan & hydro-dynamic trajectory tracking.',
      selectedRegion: {
        id: 'reg-summary',
        type: 'rectangle',
        coordinates: [],
        boundingBox: { north: 13.0, south: 12.8, east: 80.3, west: 80.1 },
        area: regionArea,
        center: { latitude: 12.9213, longitude: 80.2488 },
        timestamp: new Date(),
      },
      plasticClusters: clusters.length,
      dominantPolymer,
      estimatedPlasticWaste: Number(totalPlastic.toFixed(2)),
      detectionAccuracy: 96,
      likelySource: MOCK_FACTORIES[0].name,
      cleanupPriority: highRiskCount > 10 ? 'high' : 'medium',
      riskScore: 87,
      recommendation: 'Deploy coastal barrier traps at Adyar Estuary mouth to mitigate drift into Bay of Bengal.',
      timestamp: new Date(),
    };
  },
};
