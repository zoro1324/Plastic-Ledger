import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Zap,
  Plus,
  Square,
  Hexagon,
  Map,
  Ruler,
  Circle,
  RotateCcw,
  Trash2,
  Maximize2,
} from 'lucide-react';
import { useDashboard } from '@/context/DashboardContext';

export const GISToolbar: React.FC = () => {
  const { gisToolbar, setGisToolbar } = useDashboard();
  const [isExpanded, setIsExpanded] = useState(false);

  const tools = [
    {
      id: 'select',
      label: 'Select',
      icon: Square,
      description: 'Select or drag on map',
    },
    {
      id: 'rectangle',
      label: 'Rectangle',
      icon: Square,
      description: 'Draw rectangular region',
    },
    {
      id: 'polygon',
      label: 'Polygon',
      icon: Hexagon,
      description: 'Draw polygon region',
    },
    {
      id: 'marker',
      label: 'Marker',
      icon: Plus,
      description: 'Add marker point',
    },
    {
      id: 'distance',
      label: 'Distance',
      icon: Ruler,
      description: 'Measure distance',
    },
    {
      id: 'area',
      label: 'Area',
      icon: Square,
      description: 'Measure area',
    },
    {
      id: 'zoom',
      label: 'Zoom',
      icon: Maximize2,
      description: 'Zoom tool',
    },
  ];

  const handleToolClick = (toolId: string) => {
    setGisToolbar({
      ...gisToolbar,
      activeTool: gisToolbar.activeTool === toolId ? null : (toolId as any),
      isDrawing: true,
    });
  };

  const handleReset = () => {
    setGisToolbar({
      activeTool: null,
      isDrawing: false,
      isVisible: true,
    });
  };

  return (
    <motion.div
      className="fixed left-6 top-24 z-40 flex flex-col space-y-2"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      {/* Main Toolbar */}
      <motion.div
        className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/30 rounded-xl backdrop-blur-md shadow-xl"
        whileHover={{ boxShadow: '0 0 30px rgba(10, 132, 255, 0.3)' }}
      >
        {/* Tool Buttons */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              className="p-2 space-y-2"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              {tools.map((tool, idx) => {
                const Icon = tool.icon;
                const isActive = gisToolbar.activeTool === tool.id;

                return (
                  <motion.button
                    key={tool.id}
                    className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-all group ${
                      isActive
                        ? 'bg-[#0A84FF]/30 text-[#4CC9F0]'
                        : 'text-gray-300 hover:bg-[#0A84FF]/10 hover:text-[#4CC9F0]'
                    }`}
                    onClick={() => handleToolClick(tool.id)}
                    whileHover={{ x: 5 }}
                    whileTap={{ scale: 0.95 }}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                  >
                    <Icon size={18} />
                    <span className="text-xs font-medium">{tool.label}</span>
                    <span className="text-xs text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity ml-auto whitespace-nowrap">
                      {tool.description}
                    </span>
                  </motion.button>
                );
              })}

              <div className="border-t border-[#0A84FF]/20 pt-2 space-y-2">
                <motion.button
                  className="w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-gray-300 hover:bg-[#F59E0B]/10 hover:text-[#F59E0B] transition-all"
                  onClick={handleReset}
                  whileHover={{ x: 5 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <RotateCcw size={18} />
                  <span className="text-xs font-medium">Reset</span>
                </motion.button>
                <motion.button
                  className="w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-gray-300 hover:bg-[#EF4444]/10 hover:text-[#EF4444] transition-all"
                  onClick={() => setGisToolbar({ ...gisToolbar, activeTool: null })}
                  whileHover={{ x: 5 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Trash2 size={18} />
                  <span className="text-xs font-medium">Clear</span>
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Toggle Button */}
        <motion.button
          className={`w-full p-3 rounded-lg flex items-center justify-center transition-all ${
            isExpanded
              ? 'bg-[#0A84FF]/20 text-[#4CC9F0] border-b border-[#0A84FF]/20'
              : 'hover:bg-[#0A84FF]/10 text-gray-300 hover:text-[#4CC9F0]'
          }`}
          onClick={() => setIsExpanded(!isExpanded)}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          title="GIS Tools"
        >
          <Zap size={20} />
        </motion.button>
      </motion.div>

      {/* Floating Info */}
      {gisToolbar.activeTool && (
        <motion.div
          className="bg-[#0A84FF]/20 border border-[#0A84FF]/30 rounded-lg px-3 py-2 text-xs text-gray-300 backdrop-blur-md"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
        >
          <p className="font-semibold text-[#4CC9F0]">
            {tools.find((t) => t.id === gisToolbar.activeTool)?.label} Active
          </p>
          <p className="text-gray-400">
            {tools.find((t) => t.id === gisToolbar.activeTool)?.description}
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};
