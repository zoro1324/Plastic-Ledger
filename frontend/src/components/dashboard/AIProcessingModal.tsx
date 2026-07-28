import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Download,
  Cpu,
  Search,
  Dna,
  GitMerge,
  FileCheck,
  CheckCircle2,
  Loader2,
  Sparkles,
} from 'lucide-react';

interface AIProcessingModalProps {
  isOpen: boolean;
  onComplete: () => void;
}

export const AI_STEPS = [
  { id: 1, text: 'Downloading Satellite Tile...', icon: Download, duration: 600 },
  { id: 2, text: 'Processing Image...', icon: Cpu, duration: 700 },
  { id: 3, text: 'Running Plastic Detection...', icon: Search, duration: 800 },
  { id: 4, text: 'Classifying Polymer...', icon: Dna, duration: 600 },
  { id: 5, text: 'Running Source Attribution...', icon: GitMerge, duration: 700 },
  { id: 6, text: 'Generating Report...', icon: FileCheck, duration: 600 },
];

export const AIProcessingModal: React.FC<AIProcessingModalProps> = ({ isOpen, onComplete }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  useEffect(() => {
    if (!isOpen) {
      setCurrentStepIndex(0);
      return;
    }

    let timeoutId: NodeJS.Timeout;

    const runStep = (index: number) => {
      if (index >= AI_STEPS.length) {
        // All steps done
        timeoutId = setTimeout(() => {
          onComplete();
        }, 500);
        return;
      }

      setCurrentStepIndex(index);
      const step = AI_STEPS[index];
      timeoutId = setTimeout(() => {
        runStep(index + 1);
      }, step.duration);
    };

    runStep(0);

    return () => clearTimeout(timeoutId);
  }, [isOpen, onComplete]);

  if (!isOpen) return null;

  const currentStep = AI_STEPS[Math.min(currentStepIndex, AI_STEPS.length - 1)];
  const progressPercent = Math.min(
    100,
    Math.round(((currentStepIndex + 1) / AI_STEPS.length) * 100)
  );

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          className="w-full max-w-lg bg-[#07172A] border border-[#0A84FF]/40 rounded-3xl p-6 shadow-[0_0_50px_rgba(10,132,255,0.25)] relative overflow-hidden text-white"
        >
          {/* Top glowing beam */}
          <div className="absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r from-blue-600 via-cyan-400 to-teal-400 animate-pulse" />

          {/* Header */}
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-3 bg-[#0A84FF]/20 rounded-2xl border border-[#0A84FF]/40 text-cyan-400">
              <Sparkles className="w-6 h-6 animate-spin" style={{ animationDuration: '4s' }} />
            </div>
            <div>
              <h2 className="text-lg font-extrabold text-white flex items-center gap-2">
                PlasticLedger Neural Pipeline
                <span className="text-[10px] bg-cyan-500/20 text-cyan-300 font-mono px-2 py-0.5 rounded-full border border-cyan-500/40">
                  AI v2.4
                </span>
              </h2>
              <p className="text-xs text-gray-400">Analyzing hyperspectral satellite imagery & drift models</p>
            </div>
          </div>

          {/* Progress Meter */}
          <div className="mb-6">
            <div className="flex justify-between items-center text-xs font-mono text-cyan-300 mb-2">
              <span>Overall Progress</span>
              <span className="font-bold">{progressPercent}%</span>
            </div>
            <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden p-0.5 border border-white/10">
              <motion.div
                className="h-full bg-gradient-to-r from-[#0A84FF] via-cyan-400 to-[#00F5D4] rounded-full"
                initial={{ width: '0%' }}
                animate={{ width: `${progressPercent}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>

          {/* Steps List */}
          <div className="space-y-3 mb-6">
            {AI_STEPS.map((step, idx) => {
              const Icon = step.icon;
              const isDone = idx < currentStepIndex;
              const isCurrent = idx === currentStepIndex;

              return (
                <div
                  key={step.id}
                  className={`flex items-center justify-between p-3 rounded-xl border transition-all duration-300 ${
                    isCurrent
                      ? 'bg-[#0A84FF]/20 border-[#0A84FF]/60 shadow-[0_0_15px_rgba(10,132,255,0.3)]'
                      : isDone
                      ? 'bg-white/5 border-emerald-500/30 text-gray-300'
                      : 'bg-white/[0.02] border-white/5 text-gray-500'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className={`p-2 rounded-lg ${
                        isCurrent
                          ? 'bg-[#0A84FF] text-white'
                          : isDone
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'bg-white/5 text-gray-500'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className={`text-xs font-mono font-medium ${isCurrent ? 'text-white font-bold' : ''}`}>
                      {step.text}
                    </span>
                  </div>

                  {isDone ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : isCurrent ? (
                    <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
                  ) : (
                    <span className="text-[10px] font-mono text-gray-600">Queued</span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Status footer */}
          <div className="text-center font-mono text-[11px] text-gray-400 bg-slate-900/80 p-2.5 rounded-xl border border-white/5 flex items-center justify-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span>Active Step: {currentStep?.text}</span>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
