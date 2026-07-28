import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, Activity } from 'lucide-react';
import { useAnimatedCounter } from '@/hooks/useAnimations';
import { formatNumber, formatKilometers, formatPercentage, formatArea } from '@/lib/utils/geoUtils';

interface KPICardProps {
  title: string;
  value: number;
  unit: string;
  icon?: React.ReactNode;
  trend?: number;
  trendLabel?: string;
  status?: 'good' | 'warning' | 'alert';
  delay?: number;
  showMiniChart?: boolean;
}

export const KPICard: React.FC<KPICardProps> = ({
  title,
  value,
  unit,
  icon,
  trend,
  trendLabel,
  status = 'good',
  delay = 0,
  showMiniChart = false,
}) => {
  const animatedValue = useAnimatedCounter(value, 1000, 1);

  const statusColors = {
    good: 'from-[#22C55E]/20 to-[#00C2A8]/10 border-[#22C55E]/20',
    warning: 'from-[#F59E0B]/20 to-[#F97316]/10 border-[#F59E0B]/20',
    alert: 'from-[#EF4444]/20 to-[#DC2626]/10 border-[#EF4444]/20',
  };

  const trendColors = {
    good: 'text-[#22C55E]',
    warning: 'text-[#F59E0B]',
    alert: 'text-[#EF4444]',
  };

  return (
    <motion.div
      className={`bg-gradient-to-br ${statusColors[status]} border rounded-xl p-6 backdrop-blur-sm hover:shadow-lg transition-shadow`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -5 }}
    >
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center space-x-3">
          {icon && (
            <div className="p-2 bg-[#0A84FF]/20 rounded-lg text-[#4CC9F0]">
              {icon}
            </div>
          )}
          <div>
            <p className="text-sm text-gray-400">{title}</p>
          </div>
        </div>
        {trend !== undefined && (
          <motion.div
            className={`flex items-center space-x-1 ${trendColors[status]}`}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: delay + 0.3 }}
          >
            <TrendingUp size={16} />
            <span className="text-xs font-semibold">{trend > 0 ? '+' : ''}{trend}%</span>
          </motion.div>
        )}
      </div>

      <motion.div
        className="mb-2"
        initial={{ scale: 0.5, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: delay + 0.2 }}
      >
        <p className="text-3xl font-bold text-white">
          {formatNumber(animatedValue, 1)}
        </p>
        <p className="text-xs text-gray-400">{unit}</p>
      </motion.div>

      {trendLabel && (
        <p className="text-xs text-gray-400 mt-2">{trendLabel}</p>
      )}

      {showMiniChart && (
        <div className="mt-4 pt-4 border-t border-[#0A84FF]/10 flex items-end space-x-1 h-8">
          {Array.from({ length: 8 }).map((_, i) => (
            <motion.div
              key={i}
              className="flex-1 bg-gradient-to-t from-[#0A84FF] to-[#4CC9F0] rounded-t"
              initial={{ height: 0 }}
              animate={{ height: `${Math.random() * 100}%` }}
              transition={{ delay: delay + 0.1 + i * 0.05 }}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
};

interface KPIGridProps {
  stats: KPICardProps[];
}

export const KPIGrid: React.FC<KPIGridProps> = ({ stats }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {stats.map((stat, idx) => (
        <KPICard key={idx} {...stat} delay={idx * 0.1} />
      ))}
    </div>
  );
};
