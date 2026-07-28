import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, RotateCcw, Waves } from 'lucide-react';

interface TimelineSliderProps {
  onTimeChange: (days: number) => void;
  selectedDays: number;
}

export const TIMELINE_STEPS = [
  { label: 'Today', days: 0 },
  { label: '7 Days', days: 7 },
  { label: '14 Days', days: 14 },
  { label: '30 Days', days: 30 },
  { label: '60 Days', days: 60 },
];

export const TimelineSlider: React.FC<TimelineSliderProps> = ({ onTimeChange, selectedDays }) => {
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        onTimeChange(
          (() => {
            const currentIdx = TIMELINE_STEPS.findIndex((s) => s.days === selectedDays);
            const nextIdx = (currentIdx + 1) % TIMELINE_STEPS.length;
            return TIMELINE_STEPS[nextIdx].days;
          })()
        );
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isPlaying, selectedDays, onTimeChange]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 30 }}
      className="absolute bottom-6 left-[180px] z-[900] w-[450px] max-w-[calc(100vw-450px)] bg-[#07172A]/90 border border-[#0A84FF]/40 backdrop-blur-xl px-4 py-2.5 rounded-2xl shadow-2xl text-white font-sans select-none"
    >
      <div className="flex items-center justify-between mb-1.5 text-xs font-mono">
        <div className="flex items-center space-x-2">
          <Waves className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span className="font-bold text-cyan-300 text-xs">Hydro-Dynamic Drift Simulation</span>
        </div>
        <span className="text-[9px] text-gray-400 bg-white/5 px-2 py-0.5 rounded border border-white/5 font-mono">
          +{selectedDays} Days
        </span>
      </div>

      <div className="flex items-center space-x-3">
        {/* Play/Pause Button */}
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="p-2 rounded-xl bg-[#0A84FF] hover:bg-cyan-400 text-slate-950 font-bold shadow-[0_0_15px_rgba(10,132,255,0.5)] transition-all cursor-pointer flex items-center justify-center"
          title={isPlaying ? 'Pause Simulation' : 'Play Simulation'}
        >
          {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current ml-0.5" />}
        </button>

        {/* Step Buttons / Track */}
        <div className="flex-1 flex items-center justify-between relative bg-white/5 p-1 rounded-xl border border-white/5">
          {TIMELINE_STEPS.map((step) => {
            const isActive = selectedDays === step.days;
            return (
              <button
                key={step.days}
                onClick={() => {
                  setIsPlaying(false);
                  onTimeChange(step.days);
                }}
                className={`px-2.5 py-0.5 rounded-lg text-[11px] font-mono transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-r from-[#0A84FF] to-cyan-400 text-slate-950 font-extrabold shadow-md scale-105'
                    : 'text-gray-400 hover:text-white hover:bg-white/10'
                }`}
              >
                {step.label}
              </button>
            );
          })}
        </div>

        {/* Reset button */}
        <button
          onClick={() => {
            setIsPlaying(false);
            onTimeChange(0);
          }}
          className="p-1.5 rounded-xl text-gray-400 hover:text-amber-400 hover:bg-amber-500/10 transition-colors"
          title="Reset to Today"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>
    </motion.div>
  );
};
