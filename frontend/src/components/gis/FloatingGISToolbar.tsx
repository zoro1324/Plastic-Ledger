import React from 'react';
import { motion } from 'framer-motion';
import {
  MousePointer,
  Square,
  Hexagon,
  MapPin,
  Ruler,
  Maximize,
  RotateCcw,
} from 'lucide-react';
import { useDashboard } from '@/context/DashboardContext';

export interface GISToolItem {
  id: 'select' | 'rectangle' | 'polygon' | 'marker' | 'distance' | 'area';
  name: string;
  icon: React.ElementType;
  shortcut: string;
  tooltip: string;
}

export const GIS_TOOLS: GISToolItem[] = [
  {
    id: 'select',
    name: 'Select Tool',
    icon: MousePointer,
    shortcut: 'V',
    tooltip: 'Select or inspect map elements',
  },
  {
    id: 'rectangle',
    name: 'Rectangle Selection',
    icon: Square,
    shortcut: 'R',
    tooltip: 'Click & drag to draw a bounding rectangle region',
  },
  {
    id: 'polygon',
    name: 'Polygon Selection',
    icon: Hexagon,
    shortcut: 'P',
    tooltip: 'Click multiple points to draw custom polygon region',
  },
  {
    id: 'marker',
    name: 'Marker Tool',
    icon: MapPin,
    shortcut: 'M',
    tooltip: 'Place custom GIS pin coordinates',
  },
  {
    id: 'distance',
    name: 'Distance Measure',
    icon: Ruler,
    shortcut: 'D',
    tooltip: 'Measure geodesic distance line on ocean map',
  },
  {
    id: 'area',
    name: 'Area Measurement',
    icon: Maximize,
    shortcut: 'A',
    tooltip: 'Measure exact polygon surface area',
  },
];

interface FloatingGISToolbarProps {
  onReset: () => void;
}

export const FloatingGISToolbar: React.FC<FloatingGISToolbarProps> = ({ onReset }) => {
  const { gisToolbar, setGisToolbar } = useDashboard();

  const handleSelectTool = (toolId: GISToolItem['id']) => {
    const isSame = gisToolbar.activeTool === toolId;
    setGisToolbar({
      ...gisToolbar,
      activeTool: isSame ? null : toolId,
      isDrawing: !isSame,
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -30 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className="absolute left-5 top-24 z-[1000] flex flex-col items-center bg-[#07172A]/90 border border-[#0A84FF]/30 backdrop-blur-xl p-2 rounded-2xl shadow-2xl space-y-2 select-none"
    >
      <div className="px-2 py-1 text-[10px] font-mono tracking-widest text-[#4CC9F0] uppercase border-b border-white/10 w-full text-center">
        GIS Tools
      </div>

      <div className="flex flex-col space-y-1.5 w-full">
        {GIS_TOOLS.map((tool) => {
          const Icon = tool.icon;
          const isActive = gisToolbar.activeTool === tool.id;

          return (
            <div key={tool.id} className="relative group flex items-center">
              <button
                onClick={() => handleSelectTool(tool.id)}
                className={`p-2.5 rounded-xl transition-all duration-200 flex items-center justify-center w-11 h-11 ${
                  isActive
                    ? 'bg-gradient-to-r from-[#0A84FF] to-[#0066CC] text-white shadow-[0_0_15px_rgba(10,132,255,0.6)] scale-105'
                    : 'text-gray-300 hover:bg-white/10 hover:text-cyan-400'
                }`}
                aria-label={tool.name}
              >
                <Icon className="w-5 h-5" />
              </button>

              {/* Tooltip */}
              <div className="absolute left-14 hidden group-hover:flex flex-col bg-[#07172A] border border-[#0A84FF]/40 px-3 py-1.5 rounded-lg shadow-xl whitespace-nowrap z-[1100] text-xs pointer-events-none">
                <div className="font-semibold text-white flex items-center gap-2">
                  <span>{tool.name}</span>
                  <span className="text-[10px] bg-white/10 px-1.5 py-0.5 rounded text-cyan-300 font-mono">
                    {tool.shortcut}
                  </span>
                </div>
                <span className="text-[11px] text-gray-400 font-normal">{tool.tooltip}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="w-full border-t border-white/10 pt-1.5">
        <div className="relative group flex items-center">
          <button
            onClick={onReset}
            className="p-2.5 rounded-xl w-11 h-11 flex items-center justify-center text-gray-400 hover:text-amber-400 hover:bg-amber-500/10 transition-all duration-200"
            title="Reset Selection"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
          <div className="absolute left-14 hidden group-hover:flex flex-col bg-[#07172A] border border-[#0A84FF]/40 px-3 py-1.5 rounded-lg shadow-xl whitespace-nowrap z-[1100] text-xs pointer-events-none">
            <span className="font-semibold text-amber-400">Reset Selection</span>
            <span className="text-[11px] text-gray-400">Clear drawn region & analysis</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
