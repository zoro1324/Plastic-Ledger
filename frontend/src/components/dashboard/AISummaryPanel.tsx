import React from 'react';
import { motion } from 'framer-motion';
import { EnvironmentalReport } from '@/types';
import { Sparkles, Factory } from 'lucide-react';

interface AISummaryPanelProps {
  summary: EnvironmentalReport | null;
}

export const AISummaryPanel: React.FC<AISummaryPanelProps> = ({ summary }) => {
  if (!summary) return null;

  const priorityColor =
    summary.cleanupPriority === 'high' || summary.cleanupPriority === 'critical'
      ? 'bg-red-500/20 text-red-400 border-red-500/40'
      : summary.cleanupPriority === 'medium'
      ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
      : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-[#07172A]/95 border border-[#0A84FF]/40 backdrop-blur-2xl p-4 rounded-2xl shadow-2xl text-white font-sans overflow-hidden"
    >
      {/* Glow Header Accent */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-[#0A84FF] via-cyan-400 to-[#00F5D4]" />

      <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1 bg-[#0A84FF]/20 text-cyan-400 rounded-lg">
            <Sparkles className="w-4 h-4" />
          </div>
          <h3 className="font-extrabold text-xs text-white font-mono tracking-wide">AI Environmental Summary</h3>
        </div>
        <span className="text-[9px] font-mono bg-cyan-500/10 text-cyan-300 px-2 py-0.5 rounded border border-cyan-500/30">
          Neural Scan Output
        </span>
      </div>

      <div className="grid grid-cols-2 gap-1.5 text-xs font-mono mb-3">
        <div className="bg-white/5 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400 text-[10px]">Selected Region:</span>
          <span className="text-cyan-300 font-bold text-[11px]">{summary.selectedRegion.area} km²</span>
        </div>

        <div className="bg-white/5 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400 text-[10px]">Plastic Clusters:</span>
          <span className="text-white font-bold text-[11px]">{summary.plasticClusters}</span>
        </div>

        <div className="bg-white/5 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400 text-[10px]">Dominant Polymer:</span>
          <span className="text-cyan-400 font-bold text-[11px]">{summary.dominantPolymer}</span>
        </div>

        <div className="bg-white/5 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400 text-[10px]">Est. Plastic:</span>
          <span className="text-amber-300 font-bold text-[11px]">{summary.estimatedPlasticWaste} T</span>
        </div>
      </div>

      <div className="space-y-1.5 text-[11px] font-mono">
        <div className="bg-white/5 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400 flex items-center gap-1">
            <Factory className="w-3.5 h-3.5 text-indigo-400" /> Likely Source:
          </span>
          <span className="text-gray-200 font-semibold truncate max-w-[140px]" title={summary.likelySource}>
            {summary.likelySource}
          </span>
        </div>

        <div className="flex items-center justify-between bg-white/5 p-2 rounded-lg border border-white/5">
          <span className="text-gray-400">Cleanup Priority:</span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${priorityColor}`}>
            {summary.cleanupPriority}
          </span>
        </div>

        <div className="flex items-center justify-between bg-white/5 p-2 rounded-lg border border-white/5">
          <span className="text-gray-400">Environmental Risk:</span>
          <span className="text-red-400 font-bold text-xs">{summary.riskScore}%</span>
        </div>
      </div>
    </motion.div>
  );
};
