import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Zap, Globe, TrendingUp } from 'lucide-react';
import { Footer } from '@/components/layout/Footer';

const AboutPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#071A2E] text-white">
      <div className="pt-24 pb-12 px-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <motion.div
            className="mb-12"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0] bg-clip-text text-transparent">
              About PlasticLedger AI
            </h1>
            <p className="text-xl text-gray-300">
              Revolutionizing environmental monitoring through AI and satellite intelligence
            </p>
          </motion.div>

          {/* Mission & Vision */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            <motion.div
              className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/20 rounded-xl p-6"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <h2 className="text-2xl font-bold mb-4 text-[#4CC9F0]">Our Mission</h2>
              <p className="text-gray-300">
                To detect, identify, and trace marine plastic pollution using cutting-edge AI and
                multispectral satellite imagery, enabling rapid response and effective cleanup initiatives.
              </p>
            </motion.div>

            <motion.div
              className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#00C2A8]/20 rounded-xl p-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <h2 className="text-2xl font-bold mb-4 text-[#00C2A8]">Our Vision</h2>
              <p className="text-gray-300">
                A world where ocean plastic pollution is monitored continuously, sources are identified
                precisely, and cleanup efforts are guided by real-time environmental intelligence.
              </p>
            </motion.div>
          </div>

          {/* Technology Stack */}
          <motion.section
            className="mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className="text-3xl font-bold mb-6 text-white">Technology Stack</h2>
            <div className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/20 rounded-xl p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-bold text-[#4CC9F0] mb-3">Satellite Imagery</h3>
                  <ul className="text-gray-300 space-y-2">
                    <li>• Sentinel-2 Multispectral Imagery</li>
                    <li>• Google Earth Engine</li>
                    <li>• Copernicus Data Space</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-bold text-[#4CC9F0] mb-3">AI & Deep Learning</h3>
                  <ul className="text-gray-300 space-y-2">
                    <li>• U-Net CNN Architecture</li>
                    <li>• Physics-Informed Neural Networks (PINNs)</li>
                    <li>• Test-Time Augmentation</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-bold text-[#4CC9F0] mb-3">Environmental Data</h3>
                  <ul className="text-gray-300 space-y-2">
                    <li>• CMEMS Ocean Currents</li>
                    <li>• ERA5 Climate Data</li>
                    <li>• Wind & Atmospheric Models</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-bold text-[#4CC9F0] mb-3">Data Integration</h3>
                  <ul className="text-gray-300 space-y-2">
                    <li>• Lagrangian Particle Tracking (RK4)</li>
                    <li>• Multi-source Attribution</li>
                    <li>• Geospatial Analysis (Turf.js)</li>
                  </ul>
                </div>
              </div>
            </div>
          </motion.section>

          {/* SDG Goals */}
          <motion.section
            className="mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h2 className="text-3xl font-bold mb-6 text-white">Sustainable Development Goals</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                { goal: 'SDG 6', title: 'Clean Water & Sanitation', icon: '💧' },
                { goal: 'SDG 12', title: 'Responsible Consumption', icon: '♻️' },
                { goal: 'SDG 13', title: 'Climate Action', icon: '🌍' },
                { goal: 'SDG 14', title: 'Life Below Water', icon: '🌊' },
                { goal: 'SDG 15', title: 'Life On Land', icon: '🌿' },
                { goal: 'SDG 17', title: 'Partnerships', icon: '🤝' },
              ].map((item, idx) => (
                <motion.div
                  key={idx}
                  className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/20 rounded-lg p-4 text-center"
                  whileHover={{ y: -5 }}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: idx * 0.05 }}
                >
                  <p className="text-3xl mb-2">{item.icon}</p>
                  <p className="font-bold text-[#4CC9F0]">{item.goal}</p>
                  <p className="text-xs text-gray-400">{item.title}</p>
                </motion.div>
              ))}
            </div>
          </motion.section>

          {/* Research Methodology */}
          <motion.section
            className="mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <h2 className="text-3xl font-bold mb-6 text-white">Research Methodology</h2>
            <div className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/20 rounded-xl p-8">
              <ol className="space-y-4 text-gray-300">
                <li className="flex items-start space-x-4">
                  <span className="flex-shrink-0 w-8 h-8 bg-[#0A84FF]/20 text-[#4CC9F0] rounded-full flex items-center justify-center font-bold">
                    1
                  </span>
                  <p>
                    <strong className="text-white">Data Acquisition:</strong> Multi-temporal Sentinel-2 satellite
                    imagery at 10m resolution, combined with ocean current and wind data from operational models.
                  </p>
                </li>
                <li className="flex items-start space-x-4">
                  <span className="flex-shrink-0 w-8 h-8 bg-[#0A84FF]/20 text-[#4CC9F0] rounded-full flex items-center justify-center font-bold">
                    2
                  </span>
                  <p>
                    <strong className="text-white">Preprocessing:</strong> Atmospheric correction, cloud masking, and
                    spectral normalization using established protocols from the MARIDA dataset.
                  </p>
                </li>
                <li className="flex items-start space-x-4">
                  <span className="flex-shrink-0 w-8 h-8 bg-[#0A84FF]/20 text-[#4CC9F0] rounded-full flex items-center justify-center font-bold">
                    3
                  </span>
                  <p>
                    <strong className="text-white">Detection:</strong> U-Net CNN trained on MARIDA with test-time
                    augmentation, followed by spatial clustering (DBSCAN) for debris aggregation.
                  </p>
                </li>
                <li className="flex items-start space-x-4">
                  <span className="flex-shrink-0 w-8 h-8 bg-[#0A84FF]/20 text-[#4CC9F0] rounded-full flex items-center justify-center font-bold">
                    4
                  </span>
                  <p>
                    <strong className="text-white">Classification:</strong> Spectral index calculations (NDVI, FDI,
                    SAVI) with decision rules for polymer type identification.
                  </p>
                </li>
                <li className="flex items-start space-x-4">
                  <span className="flex-shrink-0 w-8 h-8 bg-[#0A84FF]/20 text-[#4CC9F0] rounded-full flex items-center justify-center font-bold">
                    5
                  </span>
                  <p>
                    <strong className="text-white">Backtracking:</strong> Lagrangian particle tracking using RK4
                    integration of ocean velocity fields to estimate pollution source.
                  </p>
                </li>
                <li className="flex items-start space-x-4">
                  <span className="flex-shrink-0 w-8 h-8 bg-[#0A84FF]/20 text-[#4CC9F0] rounded-full flex items-center justify-center font-bold">
                    6
                  </span>
                  <p>
                    <strong className="text-white">Attribution:</strong> Multi-source scoring combining proximity,
                    industrial activity, and simulated trajectories to identify likely emission source.
                  </p>
                </li>
              </ol>
            </div>
          </motion.section>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default AboutPage;
