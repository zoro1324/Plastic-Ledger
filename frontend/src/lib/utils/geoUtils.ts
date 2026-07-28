import * as turf from '@turf/turf';
import { Coordinate, SelectedRegion, BoundingBox } from '@/types';

// Geospatial Calculations using Turf.js

export const calculateDistance = (from: Coordinate, to: Coordinate): number => {
  const from_point = turf.point([from.longitude, from.latitude]);
  const to_point = turf.point([to.longitude, to.latitude]);
  return turf.distance(from_point, to_point, { units: 'kilometers' });
};

export const calculateBearing = (from: Coordinate, to: Coordinate): number => {
  const from_point = turf.point([from.longitude, from.latitude]);
  const to_point = turf.point([to.longitude, to.latitude]);
  return turf.bearing(from_point, to_point);
};

export const calculatePolygonArea = (coordinates: Coordinate[]): number => {
  if (coordinates.length < 3) return 0;
  const polygon = turf.polygon([[...coordinates.map((c) => [c.longitude, c.latitude]), [coordinates[0].longitude, coordinates[0].latitude]]]);
  return turf.area(polygon) / 1000000; // Convert to km²
};

export const calculatePolygonPerimeter = (coordinates: Coordinate[]): number => {
  if (coordinates.length < 2) return 0;
  let perimeter = 0;
  for (let i = 0; i < coordinates.length; i++) {
    const from = coordinates[i];
    const to = coordinates[(i + 1) % coordinates.length];
    perimeter += calculateDistance(from, to);
  }
  return perimeter;
};

export const calculateRectangleBoundingBox = (
  center: Coordinate,
  width: number,
  height: number
): BoundingBox => {
  const center_point = turf.point([center.longitude, center.latitude]);
  
  // Calculate corners
  const north_point = turf.destination(center_point, height / 2, 0, { units: 'kilometers' });
  const south_point = turf.destination(center_point, height / 2, 180, { units: 'kilometers' });
  const east_point = turf.destination(center_point, width / 2, 90, { units: 'kilometers' });
  const west_point = turf.destination(center_point, width / 2, 270, { units: 'kilometers' });

  return {
    north: (north_point.geometry.coordinates[1] + south_point.geometry.coordinates[1]) / 2 + height / 111,
    south: (north_point.geometry.coordinates[1] + south_point.geometry.coordinates[1]) / 2 - height / 111,
    east: (east_point.geometry.coordinates[0] + west_point.geometry.coordinates[0]) / 2 + width / (111 * Math.cos(turf.degreesToRadians(center.latitude))),
    west: (east_point.geometry.coordinates[0] + west_point.geometry.coordinates[0]) / 2 - width / (111 * Math.cos(turf.degreesToRadians(center.latitude))),
  };
};

export const getBoundingBoxCenter = (bbox: BoundingBox): Coordinate => {
  return {
    latitude: (bbox.north + bbox.south) / 2,
    longitude: (bbox.east + bbox.west) / 2,
  };
};

export const getCoordinatesFromBoundingBox = (bbox: BoundingBox): Coordinate[] => {
  return [
    { latitude: bbox.north, longitude: bbox.west },
    { latitude: bbox.north, longitude: bbox.east },
    { latitude: bbox.south, longitude: bbox.east },
    { latitude: bbox.south, longitude: bbox.west },
  ];
};

export const calculateCoordinatesDimensions = (coordinates: Coordinate[]): { width: number; height: number } => {
  if (coordinates.length === 0) return { width: 0, height: 0 };

  const latitudes = coordinates.map((c) => c.latitude);
  const longitudes = coordinates.map((c) => c.longitude);

  const minLat = Math.min(...latitudes);
  const maxLat = Math.max(...latitudes);
  const minLon = Math.min(...longitudes);
  const maxLon = Math.max(...longitudes);

  const height = calculateDistance(
    { latitude: maxLat, longitude: (minLon + maxLon) / 2 },
    { latitude: minLat, longitude: (minLon + maxLon) / 2 }
  );

  const width = calculateDistance(
    { latitude: (minLat + maxLat) / 2, longitude: maxLon },
    { latitude: (minLat + maxLat) / 2, longitude: minLon }
  );

  return { width, height };
};

// Animation and interpolation utilities

export const easeOutQuart = (t: number): number => 1 - Math.pow(1 - t, 4);

export const easeInOutQuad = (t: number): number =>
  t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

export const interpolateCoordinate = (
  from: Coordinate,
  to: Coordinate,
  progress: number
): Coordinate => {
  return {
    latitude: from.latitude + (to.latitude - from.latitude) * progress,
    longitude: from.longitude + (to.longitude - from.longitude) * progress,
  };
};

export const generateTrajectoryPoints = (
  from: Coordinate,
  to: Coordinate,
  steps: number = 20
): Coordinate[] => {
  const points: Coordinate[] = [];
  for (let i = 0; i <= steps; i++) {
    const progress = i / steps;
    points.push(interpolateCoordinate(from, to, progress));
  }
  return points;
};

// Heatmap color mapping
export const getColorForDensity = (density: number): string => {
  // density from 0 to 1
  if (density < 0.25) return '#22C55E'; // Green
  if (density < 0.5) return '#FBBF24'; // Amber
  if (density < 0.75) return '#F97316'; // Orange
  return '#DC2626'; // Red
};

// Number formatting
export const formatNumber = (num: number, decimals: number = 2): string => {
  return num.toFixed(decimals);
};

export const formatKilometers = (km: number): string => {
  if (km > 1000) return `${(km / 1000).toFixed(2)} Mm`;
  return `${km.toFixed(2)} km`;
};

export const formatArea = (areaKm2: number): string => {
  return `${areaKm2.toFixed(2)} km²`;
};

export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`;
};

// Date formatting
export const formatDate = (date: Date): string => {
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

export const formatTime = (date: Date): string => {
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

export const formatDateTime = (date: Date): string => {
  return `${formatDate(date)} ${formatTime(date)}`;
};

// Risk level color
export const getRiskLevelColor = (riskLevel: string): string => {
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

export const getRiskLevelBgColor = (riskLevel: string): string => {
  switch (riskLevel) {
    case 'low':
      return 'bg-green-500/20';
    case 'medium':
      return 'bg-amber-500/20';
    case 'high':
      return 'bg-orange-500/20';
    case 'critical':
      return 'bg-red-500/20';
    default:
      return 'bg-gray-500/20';
  }
};
