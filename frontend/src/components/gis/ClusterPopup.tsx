import React from 'react';
import { motion } from 'framer-motion';
import { MapPin, AlertCircle, Droplet, Calendar, TrendingUp, Wind } from 'lucide-react';
import { PlasticCluster } from '@/types';
import { getRiskLevelColor, getRiskLevelBgColor, formatDate } from '@/lib/utils/geoUtils';

interface ClusterPopupProps {
  cluster: PlasticCluster;
  onClose: () => void;
  onAnalyze?: () => void;
}

export const ClusterPopup: React.FC<ClusterPopupProps> = ({ cluster, onClose, onAnalyze }) => {
  return (
    <motion.div
      className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/30 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl backdrop-blur-md pointer-events-auto"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        transition={{ type: 'spring', damping: 20 }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-bold text-white">Plastic Cluster</h3>
            <p className="text-xs text-gray-400">{cluster.id}</p>
          </div>
          <motion.button
            className="text-gray-400 hover:text-white transition-colors"
            whileHover={{ rotate: 90 }}
            onClick={onClose}
          >
            ✕
          </motion.button>
        </div>

        {/* Location */}
        <motion.div
          className="bg-[#0A84FF]/10 border border-[#0A84FF]/20 rounded-lg p-3 mb-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-start space-x-2">
            <MapPin size={16} className="text-[#4CC9F0] mt-1 flex-shrink-0" />
            <div className="flex-1 text-sm">
              <p className="text-gray-300">
                <span className="font-semibold">{cluster.latitude.toFixed(4)}</span>°,{' '}
                <span className="font-semibold">{cluster.longitude.toFixed(4)}</span>°
              </p>
            </div>
          </div>
        </motion.div>

        {/* Confidence & Risk */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <motion.div
            className="bg-[#0A84FF]/10 border border-[#0A84FF]/20 rounded-lg p-3"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            <p className="text-xs text-gray-400 mb-1">Confidence</p>
            <div className="flex items-baseline space-x-1">
              <p className="text-lg font-bold text-[#4CC9F0]">
                {(cluster.confidence * 100).toFixed(0)}%
              </p>
            </div>
            <div className="w-full h-1 bg-gray-700 rounded-full mt-2">
              <motion.div
                className="h-full bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0] rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${cluster.confidence * 100}%` }}
                transition={{ duration: 1 }}
              />
            </div>
          </motion.div>

          <motion.div
            className={`${getRiskLevelBgColor(cluster.riskLevel)} border border-[#0A84FF]/20 rounded-lg p-3`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <p className="text-xs text-gray-400 mb-1">Risk Level</p>
            <p className="text-lg font-bold capitalize" style={{ color: getRiskLevelColor(cluster.riskLevel) }}>
              {cluster.riskLevel}
            </p>
          </motion.div>
        </div>

        {/* Details Grid */}
        <motion.div
          className="space-y-2 mb-4 max-h-48 overflow-y-auto"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.25 }}
        >
          <div className="bg-[#0F2D4A]/50 rounded-lg p-3 text-sm">
            <div className="flex items-center space-x-2 mb-1">
              <Droplet size={14} className="text-[#00C2A8]" />
              <span className="text-gray-400">Polymer Type</span>
            </div>
            <p className="font-semibold text-white ml-6">{cluster.polymerType}</p>
          </div>

          <div className="bg-[#0F2D4A]/50 rounded-lg p-3 text-sm">
            <div className="flex items-center space-x-2 mb-1">
              <AlertCircle size={14} className="text-[#F59E0B]" />
              <span className="text-gray-400">Estimated Area</span>
            </div>
            <p className="font-semibold text-white ml-6">{cluster.estimatedArea.toFixed(2)} km²</p>
          </div>

          <div className="bg-[#0F2D4A]/50 rounded-lg p-3 text-sm">
            <div className="flex items-center space-x-2 mb-1">
              <TrendingUp size={14} className="text-[#22C55E]" />
              <span className="text-gray-400">Quantity (est.)</span>
            </div>
            <p className="font-semibold text-white ml-6">{cluster.estimatedQuantity.toFixed(2)} MT</p>
          </div>

          <div className="bg-[#0F2D4A]/50 rounded-lg p-3 text-sm">
            <div className="flex items-center space-x-2 mb-1">
              <Calendar size={14} className="text-[#4CC9F0]" />
              <span className="text-gray-400">Age Estimate</span>
            </div>
            <p className="font-semibold text-white ml-6">{cluster.estimatedAge} days</p>
          </div>

          {cluster.possibleSourceFactory && (
            <div className="bg-[#0F2D4A]/50 rounded-lg p-3 text-sm">
              <div className="flex items-center space-x-2 mb-1">
                <Wind size={14} className="text-[#F97316]" />
                <span className="text-gray-400">Likely Source</span>
              </div>
              <p className="font-semibold text-white ml-6">{cluster.possibleSourceFactory}</p>
            </div>
          )}
        </motion.div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <motion.button
            className="flex-1 bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0] text-white font-semibold py-2 rounded-lg hover:shadow-lg transition-all"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onAnalyze}
          >
            Analyze
          </motion.button>
          <motion.button
            className="flex-1 bg-[#0A84FF]/10 text-[#4CC9F0] font-semibold py-2 rounded-lg hover:bg-[#0A84FF]/20 transition-all"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onClose}
          >
            Close
          </motion.button>
        </div>

        {/* Updated */}
        <p className="text-xs text-gray-500 text-center mt-3">
          Detected: {formatDate(cluster.timestamp)}
        </p>
      </motion.div>
    </motion.div>
  );
};
