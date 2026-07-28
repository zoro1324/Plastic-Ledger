import { describe, it, expect } from 'vitest';
import { calculateGeodesicStats, mockApiService } from '../services/apiService';

describe('GIS Geodesic Spatial Calculations & API Service', () => {
  const sampleCoords: Array<[number, number]> = [
    [12.95, 80.20],
    [12.95, 80.30],
    [12.89, 80.30],
    [12.89, 80.20],
  ];

  it('should calculate geodesic stats correctly via Turf.js', () => {
    const stats = calculateGeodesicStats(sampleCoords);

    expect(stats.areaKm2).toBeGreaterThan(0);
    expect(stats.widthKm).toBeGreaterThan(0);
    expect(stats.heightKm).toBeGreaterThan(0);
    expect(stats.perimeterKm).toBeGreaterThan(0);

    expect(stats.center.latitude).toBeCloseTo(12.92, 1);
    expect(stats.center.longitude).toBeCloseTo(80.25, 1);

    expect(stats.boundingBox.north).toBe(12.95);
    expect(stats.boundingBox.south).toBe(12.89);
    expect(stats.boundingBox.east).toBe(80.30);
    expect(stats.boundingBox.west).toBe(80.20);

    expect(stats.corners.northWest.latitude).toBe(12.95);
    expect(stats.corners.northWest.longitude).toBe(80.20);
    expect(stats.corners.southEast.latitude).toBe(12.89);
    expect(stats.corners.southEast.longitude).toBe(80.30);
  });

  it('should return empty stats gracefully for invalid input', () => {
    const stats = calculateGeodesicStats([]);
    expect(stats.areaKm2).toBe(0);
    expect(stats.widthKm).toBe(0);
  });

  it('should simulate region selection API endpoint', async () => {
    const res = await mockApiService.selectRegion({
      coordinates: [
        { latitude: 12.95, longitude: 80.20 },
        { latitude: 12.95, longitude: 80.30 },
        { latitude: 12.89, longitude: 80.30 },
        { latitude: 12.89, longitude: 80.20 },
      ],
      type: 'rectangle',
    });

    expect(res.success).toBe(true);
    expect(res.region.area).toBeGreaterThan(0);
    expect(res.region.center.latitude).toBeDefined();
  });

  it('should run plastic detection and return risk-classified clusters', async () => {
    const clusters = await mockApiService.detectPlastics({
      id: 'test-region',
      type: 'rectangle',
      coordinates: [],
      boundingBox: { north: 12.95, south: 12.89, east: 80.3, west: 80.2 },
      area: 12.84,
      center: { latitude: 12.92, longitude: 80.25 },
      timestamp: new Date(),
    });

    expect(clusters.length).toBeGreaterThan(0);
    expect(clusters[0].polymerType).toBeDefined();
    expect(clusters[0].confidence).toBeGreaterThan(0.5);
    expect(['low', 'medium', 'high', 'critical']).toContain(clusters[0].riskLevel);
  });

  it('should calculate source attribution trajectory', async () => {
    const cluster = {
      id: 'cluster-1',
      latitude: 12.92,
      longitude: 80.25,
      confidence: 0.95,
      estimatedArea: 420,
      estimatedQuantity: 2.4,
      polymerType: 'PET' as const,
      estimatedAge: 148,
      riskLevel: 'high' as const,
      timestamp: new Date(),
    };

    const attribution = await mockApiService.getSourceAttribution(cluster);
    expect(attribution.factoryDistanceKm).toBeGreaterThan(0);
    expect(attribution.riverDistanceKm).toBeGreaterThan(0);
    expect(attribution.travelTimeDays).toBeGreaterThan(0);
    expect(attribution.trajectory.length).toBeGreaterThan(2);
  });

  it('should generate environmental summary report', async () => {
    const clusters = await mockApiService.getClusters();
    const summary = await mockApiService.getRegionSummary(clusters, 12.84);

    expect(summary.plasticClusters).toBe(clusters.length);
    expect(summary.dominantPolymer).toBeDefined();
    expect(summary.riskScore).toBeGreaterThan(0);
    expect(summary.likelySource).toBeDefined();
  });
});
