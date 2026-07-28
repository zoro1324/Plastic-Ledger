// Export all components for easier imports
export { Navigation } from '@/components/layout/Navigation';
export { Footer } from '@/components/layout/Footer';
export { KPICard, KPIGrid } from '@/components/dashboard/KPICard';
export { AIWorkflow } from '@/components/dashboard/AIWorkflow';
export { AIInsightsPanel } from '@/components/dashboard/AIInsightsPanel';
export { GISMap } from '@/components/gis/GISMap';
export { GISToolbar } from '@/components/gis/GISToolbar';
export { ClusterPopup } from '@/components/gis/ClusterPopup';
export { RegionInfoPanel } from '@/components/gis/RegionInfoPanel';

// Export all hooks
export * from '@/hooks/useAnimations';
export * from '@/hooks/useGeoLocation';

// Export all utilities
export * from '@/lib/utils/geoUtils';
export { CONFIG, FEATURES } from '@/lib/config';

// Export context
export { DashboardProvider, useDashboard } from '@/context/DashboardContext';

// Export API
export * from '@/lib/api/mockAPI';

// Export types
export type * from '@/types';
