import React from 'react';
import { motion } from 'framer-motion';
import { Brain, AlertTriangle, TrendingUp, MapPin } from 'lucide-react';
import { DetectionResult, SelectedRegion } from '@/types';

interface AIInsightsPanelProps {
  detectionResult: DetectionResult | null;
  selectedRegion: SelectedRegion | null;
  onClose?: () => void;
}

export const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({
  detectionResult,
  selectedRegion,
  onClose,
}) => {
  if (!detectionResult || !selectedRegion) return null;

  const insights = [
    {
      label: 'Plastic Clusters Detected',
      value: detectionResult.totalClusters,
      icon: AlertTriangle,
      color: '#EF4444',
    },
    {
      label: 'Detection Accuracy',
      value: `${(detectionResult.overallConfidence * 100).toFixed(1)}%`,
      icon: TrendingUp,
      color: '#22C55E',
    },
    {
      label: 'Estimated Plastic Waste',
      value: `${(detectionResult.estimatedTotalPlastic / 1000).toFixed(1)}K MT`,
      icon: Brain,
      color: '#F59E0B',
    },
    {
      label: 'Dominant Polymer',
      value: detectionResult.dominantPolymer,
      icon: MapPin,
      color: '#0A84FF',
    },
  ];

  return (
    <motion.div
      className="fixed top-24 right-6 max-w-sm bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/30 rounded-xl p-6 shadow-2xl backdrop-blur-md z-40"
      initial={{ opacity: 0, x: 100 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 100 }}
      transition={{ type: 'spring', damping: 25 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <motion.div className="flex items-center space-x-2">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity }}
            className="text-[#0A84FF]"
          >
            <Brain size={20} />
          </motion.div>
          <h3 className="text-lg font-bold text-white">AI Insights</h3>
        </motion.div>
        {onClose && (
          <motion.button
            className="text-gray-400 hover:text-white transition-colors"
            whileHover={{ rotate: 90 }}
            onClick={onClose}
          >
            ✕
          </motion.button>
        )}
      </div>

      {/* Region Info */}
      <motion.div
        className="bg-[#0A84FF]/10 border border-[#0A84FF]/20 rounded-lg p-3 mb-4"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <p className="text-xs text-gray-400 mb-1">Analyzed Region</p>
        <p className="font-semibold text-white">{selectedRegion.name || 'Selected Area'}</p>
        <p className="text-xs text-gray-400">{selectedRegion.area.toFixed(2)} km²</p>
      </motion.div>

      {/* Insights Grid */}
      <motion.div
        className="space-y-3 mb-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ staggerChildren: 0.1, delayChildren: 0.2 }}
      >
        {insights.map((insight, idx) => {
          const Icon = insight.icon;
          return (
            <motion.div
              key={idx}
              className="bg-[#0F2D4A]/50 rounded-lg p-3 border-l-2"
              style={{ borderColor: insight.color }}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <div className="flex items-start space-x-3">
                <Icon size={16} style={{ color: insight.color }} className="mt-1 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-400">{insight.label}</p>
                  <p className="font-bold text-white text-sm">{insight.value}</p>
                </div>
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Recommendation */}
      <motion.div
        className="bg-gradient-to-br from-[#F59E0B]/20 to-[#EF4444]/20 border border-[#F59E0B]/20 rounded-lg p-3"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <p className="text-xs font-semibold text-[#F59E0B] mb-1">AI Recommendation</p>
        <p className="text-xs text-gray-300">
          High priority cleanup required. Initiate coordination with local authorities for debris recovery operations.
        </p>
      </motion.div>

      {/* Environmental Risk */}
      <motion.div
        className="mt-4 pt-4 border-t border-[#0A84FF]/20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <p className="text-xs text-gray-400 mb-2">Environmental Risk Score</p>
        <div className="flex items-center space-x-2">
          <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-[#F59E0B] to-[#EF4444]"
              initial={{ width: 0 }}
              animate={{ width: '75%' }}
              transition={{ duration: 1.5 }}
            />
          </div>
          <span className="text-sm font-bold text-[#EF4444]">75/100</span>
        </div>
      </motion.div>
    </motion.div>
  );
};
