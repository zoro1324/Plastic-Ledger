import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Zap, Check, AlertCircle, Loader } from 'lucide-react';
import { AIWorkflowStep } from '@/types';

interface AIWorkflowProps {
  isActive: boolean;
  progress?: number;
  onComplete?: () => void;
}

export const AIWorkflow: React.FC<AIWorkflowProps> = ({ isActive, progress = 0, onComplete }) => {
  const [steps, setSteps] = useState<AIWorkflowStep[]>([
    {
      stepNumber: 1,
      name: 'Downloading Sentinel-2 Tile',
      description: 'Fetching satellite imagery',
      status: 'pending',
      progress: 0,
    },
    {
      stepNumber: 2,
      name: 'Preprocessing Satellite Image',
      description: 'Normalizing and tiling',
      status: 'pending',
      progress: 0,
    },
    {
      stepNumber: 3,
      name: 'Running Plastic Detection',
      description: 'U-Net inference + TTA',
      status: 'pending',
      progress: 0,
    },
    {
      stepNumber: 4,
      name: 'Polymer Classification',
      description: 'Spectral analysis',
      status: 'pending',
      progress: 0,
    },
    {
      stepNumber: 5,
      name: 'PINN Source Attribution',
      description: 'Backtracking simulation',
      status: 'pending',
      progress: 0,
    },
    {
      stepNumber: 6,
      name: 'Generating Environmental Report',
      description: 'Compiling results',
      status: 'pending',
      progress: 0,
    },
  ]);

  // Simulate progress
  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      setSteps((prev) => {
        const updated = [...prev];
        let totalProgress = 0;

        // Update steps based on overall progress
        for (let i = 0; i < updated.length; i++) {
          const stepProgress = Math.max(0, progress - i * 16.67);
          if (stepProgress > 0) {
            updated[i].status = stepProgress >= 100 ? 'completed' : 'running';
            updated[i].progress = Math.min(100, stepProgress);
          }
          totalProgress += updated[i].progress;
        }

        if (progress >= 100 && onComplete) {
          onComplete();
        }

        return updated;
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isActive, progress, onComplete]);

  if (!isActive) return null;

  return (
    <motion.div
      className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/30 rounded-2xl p-8 max-w-2xl w-full mx-4 shadow-2xl backdrop-blur-md pointer-events-auto"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', damping: 20 }}
      >
        {/* Header */}
        <motion.div
          className="flex items-center space-x-3 mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity }}
            className="text-[#0A84FF]"
          >
            <Zap size={24} />
          </motion.div>
          <div>
            <h2 className="text-2xl font-bold text-white">AI Analysis Pipeline</h2>
            <p className="text-gray-400 text-sm">Processing selected region...</p>
          </div>
        </motion.div>

        {/* Progress Bar */}
        <motion.div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-400">Overall Progress</span>
            <span className="text-sm font-bold text-[#4CC9F0]">{Math.round(progress)}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0]"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </motion.div>

        {/* Steps */}
        <motion.div className="space-y-4">
          {steps.map((step, idx) => (
            <motion.div
              key={step.stepNumber}
              className="bg-[#0A84FF]/10 border border-[#0A84FF]/20 rounded-lg p-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <div className="flex items-start space-x-4">
                {/* Status Icon */}
                <motion.div
                  className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center"
                  animate={
                    step.status === 'completed'
                      ? {}
                      : step.status === 'running'
                        ? { scale: [1, 1.2, 1] }
                        : {}
                  }
                  transition={{ duration: 1, repeat: step.status === 'running' ? Infinity : 0 }}
                >
                  {step.status === 'completed' && (
                    <motion.div
                      className="text-[#22C55E]"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                    >
                      <Check size={20} />
                    </motion.div>
                  )}
                  {step.status === 'running' && (
                    <motion.div
                      className="text-[#0A84FF]"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      <Loader size={20} />
                    </motion.div>
                  )}
                  {step.status === 'pending' && (
                    <div className="w-2 h-2 bg-gray-600 rounded-full" />
                  )}
                </motion.div>

                {/* Content */}
                <div className="flex-1">
                  <h4 className="font-semibold text-white text-sm">{step.name}</h4>
                  <p className="text-xs text-gray-400">{step.description}</p>

                  {step.status !== 'pending' && (
                    <motion.div className="mt-2 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0]"
                        initial={{ width: 0 }}
                        animate={{ width: `${step.progress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </motion.div>
                  )}
                </div>

                {/* Progress % */}
                {step.status !== 'pending' && (
                  <motion.span className="text-xs font-bold text-[#4CC9F0]">
                    {Math.round(step.progress)}%
                  </motion.span>
                )}
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Footer Info */}
        <motion.div
          className="mt-8 pt-6 border-t border-[#0A84FF]/20 text-center text-xs text-gray-400"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <p>Processing time: ~{Math.round(progress * 2)} seconds</p>
          <p className="mt-1">Results will be displayed on completion</p>
        </motion.div>
      </motion.div>
    </motion.div>
  );
};
