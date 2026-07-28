// Application configuration and constants

export const CONFIG = {
  // API
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || '/api',
  
  // Maps
  MAP_DEFAULT_ZOOM: 6,
  MAP_MIN_ZOOM: 2,
  MAP_MAX_ZOOM: 19,
  MAP_DEFAULT_CENTER: [20, 0] as [number, number],
  
  // Tiles
  TILE_SIZE: 256,
  TILES_PER_SCENE: 4,
  
  // UI
  ANIMATION_DURATION: 0.5,
  ANIMATION_DELAY: 0.1,
  
  // Analysis
  MIN_CLUSTER_AREA_KM2: 0.01,
  DEFAULT_CLOUD_COVER_THRESHOLD: 20,
  DEFAULT_BACKTRACK_DAYS: 7,
  
  // Colors
  RISK_COLORS: {
    low: '#22C55E',
    medium: '#FBBF24',
    high: '#F97316',
    critical: '#DC2626',
  } as const,
  
  // Theme
  THEME: {
    primary: '#0A84FF',
    secondary: '#00C2A8',
    accent: '#4CC9F0',
    background: '#071A2E',
    surface: '#0F2D4A',
  } as const,
};

// Feature flags
export const FEATURES = {
  ENABLE_3D_GLOBE: false,
  ENABLE_REAL_API: false,
  ENABLE_ANIMATIONS: true,
  ENABLE_NOTIFICATIONS: true,
};

// Polymer types
export const POLYMER_TYPES = ['PET', 'HDPE', 'LDPE', 'PVC', 'PP', 'PS', 'Other'] as const;

// Report types
export const REPORT_TYPES = {
  DETECTION: 'detection',
  ENVIRONMENTAL: 'environmental',
  ATTRIBUTION: 'attribution',
} as const;

// Export formats
export const EXPORT_FORMATS = {
  PDF: 'pdf',
  CSV: 'csv',
  GEOJSON: 'geojson',
} as const;
