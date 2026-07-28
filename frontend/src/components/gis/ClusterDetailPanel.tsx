import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PlasticCluster } from '@/types';
import { mockApiService } from '@/services/apiService';
import {
  X,
  MapPin,
  Flame,
  Clock,
  Factory,
  ShieldAlert,
  Dna,
  Zap,
  TrendingUp,
} from 'lucide-react';

interface ClusterDetailPanelProps {
  cluster: PlasticCluster | null;
  onClose: () => void;
}

export const ClusterDetailPanel: React.FC<ClusterDetailPanelProps> = ({ cluster, onClose }) => {
  const [attribution, setAttribution] = useState<{
    factoryDistanceKm: number;
    riverDistanceKm: number;
    travelTimeDays: number;
    factory: string;
    river: string;
  } | null>(null);

  useEffect(() => {
    if (cluster) {
      mockApiService.getSourceAttribution(cluster).then(setAttribution);
    } else {
      setAttribution(null);
    }
  }, [cluster]);

  if (!cluster) return null;

  const getRiskBadge = (risk: PlasticCluster['riskLevel']) => {
    switch (risk) {
      case 'critical':
      case 'high':
        return {
          bg: 'bg-red-500/20 text-red-400 border-red-500/40',
          label: 'HIGH RISK',
          dot: 'bg-red-500',
        };
      case 'medium':
        return {
          bg: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
          label: 'MEDIUM RISK',
          dot: 'bg-amber-500',
        };
      case 'low':
        return {
          bg: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40',
          label: 'LOW RISK',
          dot: 'bg-yellow-400',
        };
      default:
        return {
          bg: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
          label: 'CLEAN / MINIMAL',
          dot: 'bg-emerald-400',
        };
    }
  };

  const riskBadge = getRiskBadge(cluster.riskLevel);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: 50, scale: 0.95 }}
        animate={{ opacity: 1, x: 0, scale: 1 }}
        exit={{ opacity: 0, x: 50, scale: 0.95 }}
        transition={{ duration: 0.2 }}
        className="absolute right-6 top-32 z-[1000] w-88 max-h-[calc(100vh-200px)] overflow-y-auto bg-[#07172A]/95 border border-[#0A84FF]/40 backdrop-blur-2xl p-4 rounded-2xl shadow-2xl text-white font-sans scrollbar-thin scrollbar-thumb-white/10"
      >
        {/* Glow Header Accent */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 via-amber-400 to-[#0A84FF]" />

        {/* Top bar */}
        <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-3">
          <div className="flex items-center space-x-2">
            <div className="p-1 bg-red-500/20 rounded-lg text-red-400 border border-red-500/30">
              <Flame className="w-4 h-4 animate-pulse" />
            </div>
            <div>
              <h3 className="font-bold text-xs text-white font-mono uppercase tracking-wider">
                {cluster.id.toUpperCase()}
              </h3>
              <p className="text-[10px] text-cyan-400/90 font-mono">Ocean Hotspot Inspection</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Risk Badge & Confidence */}
        <div className="flex items-center justify-between mb-3">
          <span
            className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold border flex items-center space-x-1.5 ${riskBadge.bg}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full animate-ping ${riskBadge.dot}`} />
            <span>{riskBadge.label}</span>
          </span>

          <div className="flex items-center space-x-1 text-[10px] font-mono bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 px-2.5 py-0.5 rounded-full">
            <Zap className="w-3 h-3" />
            <span>{(cluster.confidence * 100).toFixed(0)}% Confidence</span>
          </div>
        </div>

        {/* Primary Cluster Metrics Grid */}
        <div className="grid grid-cols-2 gap-1.5 mb-3">
          <div className="bg-white/5 p-2 rounded-xl border border-white/5">
            <span className="text-[9px] text-gray-400 uppercase font-mono block">Polymer Type</span>
            <div className="flex items-center space-x-1 mt-0.5">
              <Dna className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-sm font-extrabold text-cyan-300 font-mono">
                {cluster.polymerType}
              </span>
            </div>
          </div>

          <div className="bg-white/5 p-2 rounded-xl border border-white/5">
            <span className="text-[9px] text-gray-400 uppercase font-mono block">Estimated Area</span>
            <div className="flex items-center space-x-1 mt-0.5">
              <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-sm font-extrabold text-white font-mono">
                {cluster.estimatedArea} <span className="text-[9px] text-gray-400">m²</span>
              </span>
            </div>
          </div>
        </div>

        {/* Detail Rows */}
        <div className="space-y-1.5 text-xs font-mono mb-3">
          <div className="flex justify-between items-center bg-white/5 px-2.5 py-1.5 rounded-lg border border-white/5">
            <span className="text-gray-400 text-[11px] flex items-center gap-1">
              <MapPin className="w-3 h-3 text-cyan-400" /> Coordinates:
            </span>
            <span className="text-gray-200 text-[11px] font-bold">
              {cluster.latitude.toFixed(4)}°, {cluster.longitude.toFixed(4)}°
            </span>
          </div>

          <div className="flex justify-between items-center bg-white/5 px-2.5 py-1.5 rounded-lg border border-white/5">
            <span className="text-gray-400 text-[11px] flex items-center gap-1">
              <Clock className="w-3 h-3 text-indigo-400" /> Estimated Age:
            </span>
            <span className="text-gray-200 text-[11px] font-bold">{cluster.estimatedAge} Days</span>
          </div>

          <div className="flex justify-between items-center bg-white/5 px-2.5 py-1.5 rounded-lg border border-white/5">
            <span className="text-gray-400 text-[11px] flex items-center gap-1">
              <ShieldAlert className="w-3 h-3 text-amber-400" /> Estimated Waste:
            </span>
            <span className="text-amber-300 text-[11px] font-bold">{cluster.estimatedQuantity} Metric Tons</span>
          </div>
        </div>

        {/* Source Attribution Section */}
        <div className="bg-gradient-to-br from-white/5 to-white/[0.02] p-3 rounded-xl border border-white/10 space-y-1.5">
          <div className="flex items-center justify-between border-b border-white/10 pb-1">
            <span className="text-[11px] font-bold font-mono text-cyan-300 uppercase flex items-center gap-1">
              <Factory className="w-3.5 h-3.5 text-[#0A84FF]" /> Source Attribution
            </span>
            <span className="text-[9px] text-gray-400 font-mono">Particle Path</span>
          </div>

          <div className="space-y-1 text-[11px] font-mono">
            <div className="flex justify-between">
              <span className="text-gray-400">Likely Source:</span>
              <span className="text-cyan-200 font-semibold truncate max-w-[150px]" title={attribution?.factory}>
                {attribution?.factory || cluster.possibleSourceFactory || 'Industrial Zone'}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-gray-400">Nearest Factory:</span>
              <span className="text-gray-200 font-bold">
                {attribution ? `${attribution.factoryDistanceKm} km` : '18.6 km'}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-gray-400">Nearest River:</span>
              <span className="text-gray-200 font-bold">
                {attribution ? `${attribution.riverDistanceKm} km` : '4.1 km'}
              </span>
            </div>

            <div className="flex justify-between">
              <span className="text-gray-400">Drift Time:</span>
              <span className="text-amber-400 font-bold">
                {attribution ? `${attribution.travelTimeDays} Days` : '11 Days'}
              </span>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};
