import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ArrowRight, Zap, Globe, Satellite, TrendingUp } from 'lucide-react';
import { Navigation } from '@/components/layout/Navigation';
import { Footer } from '@/components/layout/Footer';

const LandingPage: React.FC = () => {
  const features = [
    {
      icon: Satellite,
      title: 'Multispectral Imagery',
      description: 'Advanced Sentinel-2 satellite imagery analysis',
      color: 'from-[#0A84FF]',
    },
    {
      icon: Zap,
      title: 'AI Detection',
      description: 'U-Net powered plastic pollution detection',
      color: 'from-[#00C2A8]',
    },
    {
      icon: Globe,
      title: 'Source Attribution',
      description: 'Physics-informed neural networks for tracking',
      color: 'from-[#F59E0B]',
    },
    {
      icon: TrendingUp,
      title: 'Predictive Analytics',
      description: 'Forecasting pollution trends and impacts',
      color: 'from-[#22C55E]',
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: 'easeOut' },
    },
  };

  return (
    <div className="min-h-screen bg-[#071A2E] text-white overflow-hidden">
      <Navigation />

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 z-0">
          <div className="absolute top-0 right-0 w-96 h-96 bg-[#0A84FF] rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" />
          <div className="absolute bottom-0 left-0 w-96 h-96 bg-[#00C2A8] rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse delay-2000" />
        </div>

        <div className="max-w-7xl mx-auto relative z-10">
          <motion.div
            className="text-center mb-12"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {/* Main Title */}
            <motion.h1
              className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-[#0A84FF] via-[#4CC9F0] to-[#00C2A8] bg-clip-text text-transparent"
              variants={itemVariants}
            >
              PlasticLedger AI
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              className="text-2xl md:text-3xl text-[#4CC9F0] mb-8"
              variants={itemVariants}
            >
              Detect • Identify • Trace • Protect
            </motion.p>

            {/* Description */}
            <motion.p
              className="text-lg text-gray-300 max-w-3xl mx-auto mb-12"
              variants={itemVariants}
            >
              An AI-powered environmental intelligence platform that detects floating plastic pollution
              using multispectral satellite imagery, identifies polymer types, predicts degradation, and
              traces pollution back to its source using Physics-Informed AI.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              className="flex flex-col md:flex-row gap-6 justify-center mb-12"
              variants={itemVariants}
            >
              <Link to="/dashboard">
                <motion.button
                  className="px-8 py-4 bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0] rounded-lg font-bold text-white flex items-center space-x-2 hover:shadow-2xl transition-all"
                  whileHover={{ scale: 1.05, boxShadow: '0 0 40px rgba(10, 132, 255, 0.5)' }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>Launch Dashboard</span>
                  <ArrowRight size={20} />
                </motion.button>
              </Link>

              <motion.button
                className="px-8 py-4 border-2 border-[#0A84FF] text-[#0A84FF] rounded-lg font-bold hover:bg-[#0A84FF]/10 transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Learn More
              </motion.button>
            </motion.div>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-20"
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            {features.map((feature, idx) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={idx}
                  className="bg-gradient-to-br from-[#0F2D4A] to-[#071A2E] border border-[#0A84FF]/20 rounded-xl p-6 hover:border-[#0A84FF]/50 transition-all"
                  variants={itemVariants}
                  whileHover={{ y: -10, boxShadow: '0 20px 40px rgba(10, 132, 255, 0.2)' }}
                >
                  <motion.div
                    className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} to-[#4CC9F0] flex items-center justify-center mb-4`}
                    whileHover={{ rotate: 10, scale: 1.1 }}
                  >
                    <Icon size={24} className="text-white" />
                  </motion.div>
                  <h3 className="text-lg font-bold mb-2">{feature.title}</h3>
                  <p className="text-gray-400 text-sm">{feature.description}</p>
                </motion.div>
              );
            })}
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4 bg-gradient-to-r from-[#0A84FF]/10 to-[#00C2A8]/10 border-y border-[#0A84FF]/20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center"
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
          >
            {[
              { number: '5000+', label: 'Detections' },
              { number: '98%', label: 'Accuracy' },
              { number: '50+', label: 'Regions' },
              { number: '24/7', label: 'Monitoring' },
            ].map((stat, idx) => (
              <motion.div key={idx} variants={itemVariants}>
                <motion.p className="text-4xl font-bold bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0] bg-clip-text text-transparent mb-2">
                  {stat.number}
                </motion.p>
                <p className="text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.h2
            className="text-4xl font-bold mb-6"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Ready to Protect Our Oceans?
          </motion.h2>
          <motion.p
            className="text-gray-400 mb-8 text-lg"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            Start monitoring marine plastic pollution today and contribute to environmental protection.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <Link to="/dashboard">
              <motion.button
                className="px-10 py-4 bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0] rounded-lg font-bold text-white hover:shadow-2xl transition-all"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Get Started Now
              </motion.button>
            </Link>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default LandingPage;
