import React from 'react';
import { motion } from 'framer-motion';
import { MapPin, Maximize2, CheckCircle } from 'lucide-react';
import { SelectedRegion } from '@/types';
import { formatArea, formatKilometers } from '@/lib/utils/geoUtils';

interface RegionInfoPanelProps {
  region: SelectedRegion | null;
  onClose: () => void;
}

export const RegionInfoPanel: React.FC<RegionInfoPanelProps> = ({ region, onClose }) => {
  if (!region) return null;

  return (
    <motion.div
      className="fixed bottom-6 right-6 max-w-sm bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/30 rounded-xl p-6 shadow-2xl backdrop-blur-md z-40"
      initial={{ opacity: 0, y: 100 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 100 }}
      transition={{ type: 'spring', damping: 25 }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-white">Selected Region</h3>
          <p className="text-xs text-gray-400">{region.type}</p>
        </div>
        <motion.button
          className="text-gray-400 hover:text-white transition-colors"
          whileHover={{ rotate: 90 }}
          onClick={onClose}
        >
          ✕
        </motion.button>
      </div>

      {/* Region Info */}
      <motion.div
        className="space-y-3 mb-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ staggerChildren: 0.1 }}
      >
        {/* Area */}
        <motion.div
          className="bg-[#0A84FF]/10 border border-[#0A84FF]/20 rounded-lg p-3"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <p className="text-xs text-gray-400 mb-1">Area</p>
          <p className="text-xl font-bold text-white">{formatArea(region.area)}</p>
        </motion.div>

        {/* Dimensions */}
        {region.width && region.height && (
          <motion.div
            className="grid grid-cols-2 gap-3"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            <div className="bg-[#00C2A8]/10 border border-[#00C2A8]/20 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Width</p>
              <p className="font-bold text-white">{region.width.toFixed(2)} km</p>
            </div>
            <div className="bg-[#00C2A8]/10 border border-[#00C2A8]/20 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Height</p>
              <p className="font-bold text-white">{region.height.toFixed(2)} km</p>
            </div>
          </motion.div>
        )}

        {/* Perimeter */}
        {region.perimeter && (
          <motion.div
            className="bg-[#4CC9F0]/10 border border-[#4CC9F0]/20 rounded-lg p-3"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15 }}
          >
            <p className="text-xs text-gray-400 mb-1">Perimeter</p>
            <p className="font-bold text-white">{region.perimeter.toFixed(2)} km</p>
          </motion.div>
        )}

        {/* Center Coordinates */}
        <motion.div
          className="bg-[#F59E0B]/10 border border-[#F59E0B]/20 rounded-lg p-3"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <p className="text-xs text-gray-400 mb-2 flex items-center space-x-1">
            <MapPin size={12} />
            <span>Center</span>
          </p>
          <div className="text-sm font-mono text-white">
            <p>{region.center.latitude.toFixed(6)}°</p>
            <p>{region.center.longitude.toFixed(6)}°</p>
          </div>
        </motion.div>

        {/* Bounding Box */}
        <motion.div
          className="bg-[#EF4444]/10 border border-[#EF4444]/20 rounded-lg p-3 text-xs"
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.25 }}
        >
          <p className="text-gray-400 mb-2 font-semibold">Bounding Box</p>
          <div className="space-y-1 font-mono text-white grid grid-cols-2 gap-2">
            <div>
              <p className="text-gray-500 text-xs">North</p>
              <p>{region.boundingBox.north.toFixed(4)}°</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">South</p>
              <p>{region.boundingBox.south.toFixed(4)}°</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">East</p>
              <p>{region.boundingBox.east.toFixed(4)}°</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">West</p>
              <p>{region.boundingBox.west.toFixed(4)}°</p>
            </div>
          </div>
        </motion.div>
      </motion.div>

      {/* Status */}
      <motion.div
        className="flex items-center space-x-2 text-xs text-[#22C55E] pt-3 border-t border-[#0A84FF]/20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <CheckCircle size={14} />
        <span>Region ready for analysis</span>
      </motion.div>
    </motion.div>
  );
};
