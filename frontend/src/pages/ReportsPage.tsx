import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Download, Share2, Printer, FileText } from 'lucide-react';

const ReportsPage: React.FC = () => {
  const [reports] = useState([
    {
      id: 1,
      type: 'Detection Report',
      title: 'Bay of Bengal - January 2024',
      date: '2024-01-15',
      clusters: 234,
      accuracy: 82,
    },
    {
      id: 2,
      type: 'Environmental Report',
      title: 'South China Sea Comprehensive Analysis',
      date: '2024-01-10',
      clusters: 456,
      accuracy: 88,
    },
    {
      id: 3,
      type: 'Attribution Report',
      title: 'Source Tracing - Mediterranean Region',
      date: '2024-01-05',
      clusters: 189,
      accuracy: 85,
    },
  ]);

  return (
    <div className="min-h-screen bg-[#071A2E] pt-24 pb-12">
      <div className="max-w-7xl mx-auto px-6">
        <motion.h1
          className="text-4xl font-bold text-white mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          Reports & Documentation
        </motion.h1>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {reports.map((report, idx) => (
            <motion.div
              key={report.id}
              className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/20 rounded-xl p-6 hover:border-[#0A84FF]/50 transition-all"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              whileHover={{ y: -5 }}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-[#0A84FF]/20 rounded-lg text-[#4CC9F0]">
                    <FileText size={20} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">{report.type}</p>
                    <p className="font-bold text-white text-lg">{report.title}</p>
                  </div>
                </div>
              </div>

              <div className="space-y-2 mb-4 text-sm">
                <p className="text-gray-400">
                  Generated: <span className="text-white font-semibold">{report.date}</span>
                </p>
                <p className="text-gray-400">
                  Clusters: <span className="text-white font-semibold">{report.clusters}</span>
                </p>
                <p className="text-gray-400">
                  Accuracy: <span className="text-white font-semibold">{report.accuracy}%</span>
                </p>
              </div>

              <div className="flex gap-2 pt-4 border-t border-[#0A84FF]/20">
                <motion.button
                  className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-[#0A84FF]/20 text-[#4CC9F0] rounded-lg hover:bg-[#0A84FF]/30 transition-all text-sm font-semibold"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Download size={16} />
                  <span>PDF</span>
                </motion.button>
                <motion.button
                  className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-[#00C2A8]/20 text-[#00C2A8] rounded-lg hover:bg-[#00C2A8]/30 transition-all text-sm font-semibold"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Download size={16} />
                  <span>CSV</span>
                </motion.button>
                <motion.button
                  className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-[#F59E0B]/20 text-[#F59E0B] rounded-lg hover:bg-[#F59E0B]/30 transition-all text-sm font-semibold"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Share2 size={16} />
                  <span>Share</span>
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ReportsPage;
