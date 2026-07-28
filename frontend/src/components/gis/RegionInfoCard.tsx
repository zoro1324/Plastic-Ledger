import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SelectedRegion } from '@/types';
import { RegionCalculationResult } from '@/services/apiService';
import { Sparkles, MapPin, Compass, Crosshair, ChevronRight } from 'lucide-react';

interface RegionInfoCardProps {
  region: SelectedRegion | null;
  stats: RegionCalculationResult | null;
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

export const RegionInfoCard: React.FC<RegionInfoCardProps> = ({
  region,
  stats,
  onAnalyze,
  isAnalyzing,
}) => {
  if (!region || !stats) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: -20, scale: 0.95 }}
        animate={{ opacity: 1, x: 0, scale: 1 }}
        exit={{ opacity: 0, x: -20, scale: 0.95 }}
        className="absolute left-[185px] top-20 z-[990] w-88 max-h-[calc(100vh-180px)] overflow-y-auto bg-[#07172A]/95 border border-[#0A84FF]/40 backdrop-blur-2xl p-4 rounded-2xl shadow-2xl text-white font-sans scrollbar-thin scrollbar-thumb-white/10"
      >
        {/* Glow Header Accent */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-[#0A84FF] via-cyan-400 to-[#00F5D4]" />

        <div className="flex items-center justify-between border-b border-white/10 pb-2.5 mb-3">
          <div className="flex items-center space-x-2">
            <div className="p-1 bg-[#0A84FF]/20 rounded-lg text-cyan-400">
              <Crosshair className="w-4 h-4 animate-pulse" />
            </div>
            <div>
              <h3 className="font-bold text-xs text-white tracking-wide">Selected Region</h3>
              <p className="text-[10px] text-cyan-400/80 font-mono capitalize">
                {region.type} Geodesic Area
              </p>
            </div>
          </div>
          <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 rounded-full text-[9px] font-mono">
            Active Target
          </span>
        </div>

        {/* Primary Metrics Grid */}
        <div className="grid grid-cols-3 gap-1.5 mb-3 bg-white/5 p-2.5 rounded-xl border border-white/5">
          <div className="text-center">
            <span className="text-[9px] text-gray-400 uppercase font-mono block">Area</span>
            <span className="text-sm font-extrabold text-cyan-300 font-mono">
              {stats.areaKm2} <span className="text-[9px] text-gray-400">km²</span>
            </span>
          </div>
          <div className="text-center border-x border-white/10">
            <span className="text-[9px] text-gray-400 uppercase font-mono block">Width</span>
            <span className="text-sm font-extrabold text-white font-mono">
              {stats.widthKm} <span className="text-[9px] text-gray-400">km</span>
            </span>
          </div>
          <div className="text-center">
            <span className="text-[9px] text-gray-400 uppercase font-mono block">Height</span>
            <span className="text-sm font-extrabold text-white font-mono">
              {stats.heightKm} <span className="text-[9px] text-gray-400">km</span>
            </span>
          </div>
        </div>

        {/* Center & Bounding Details */}
        <div className="space-y-1.5 text-xs font-mono mb-3">
          <div className="flex justify-between items-center bg-white/5 px-2.5 py-1 rounded-lg">
            <span className="text-gray-400 text-[11px] flex items-center gap-1">
              <MapPin className="w-3 h-3 text-cyan-400" /> Center:
            </span>
            <span className="text-cyan-300 text-[11px] font-bold">
              {stats.center.latitude.toFixed(4)}°, {stats.center.longitude.toFixed(4)}°
            </span>
          </div>

          <div className="flex justify-between items-center bg-white/5 px-2.5 py-1 rounded-lg">
            <span className="text-gray-400 text-[11px] flex items-center gap-1">
              <Compass className="w-3 h-3 text-indigo-400" /> Perimeter:
            </span>
            <span className="text-gray-200 text-[11px] font-bold">{stats.perimeterKm} km</span>
          </div>

          {/* Bounding Box */}
          <div className="bg-white/5 p-2 rounded-lg space-y-1 border border-white/5">
            <div className="text-[9px] text-gray-400 uppercase font-mono tracking-wider font-semibold border-b border-white/5 pb-0.5 flex justify-between">
              <span>Bounding Coordinates</span>
              <span className="text-cyan-400">WGS84</span>
            </div>
            <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px]">
              <div className="flex justify-between">
                <span className="text-gray-400">N:</span>
                <span className="text-gray-200 font-semibold">{stats.boundingBox.north.toFixed(4)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">S:</span>
                <span className="text-gray-200 font-semibold">{stats.boundingBox.south.toFixed(4)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">E:</span>
                <span className="text-gray-200 font-semibold">{stats.boundingBox.east.toFixed(4)}°</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">W:</span>
                <span className="text-gray-200 font-semibold">{stats.boundingBox.west.toFixed(4)}°</span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <motion.button
          onClick={onAnalyze}
          disabled={isAnalyzing}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="w-full py-2.5 px-3 bg-gradient-to-r from-[#0A84FF] via-cyan-500 to-[#00F5D4] hover:from-[#0066CC] hover:to-cyan-400 text-slate-950 font-extrabold text-xs rounded-xl shadow-[0_0_20px_rgba(10,132,255,0.4)] transition-all duration-200 flex items-center justify-center space-x-1.5 disabled:opacity-50 cursor-pointer"
        >
          <Sparkles className="w-3.5 h-3.5 text-slate-950 fill-current" />
          <span>Analyze Selected Region</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </motion.button>
      </motion.div>
    </AnimatePresence>
  );
};
