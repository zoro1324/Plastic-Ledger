import React from 'react';
import { motion } from 'framer-motion';
import { Menu, X } from 'lucide-react';
import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

export const Navigation: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { label: 'Home', path: '/' },
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Analytics', path: '/analytics' },
    { label: 'Reports', path: '/reports' },
    { label: 'About', path: '/about' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <motion.nav
      className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-[#071A2E] via-[#0F2D4A] to-[#071A2E] border-b border-[#0A84FF]/20 backdrop-blur-md"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 group">
            <div className="w-8 h-8 bg-gradient-to-br from-[#0A84FF] to-[#00C2A8] rounded-lg flex items-center justify-center text-white font-bold text-lg">
              ◆
            </div>
            <motion.span
              className="text-xl font-bold bg-gradient-to-r from-[#0A84FF] to-[#4CC9F0] bg-clip-text text-transparent"
              whileHover={{ scale: 1.05 }}
            >
              PlasticLedger
            </motion.span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex space-x-1">
            {navItems.map((item) => (
              <Link key={item.path} to={item.path}>
                <motion.button
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    isActive(item.path)
                      ? 'bg-[#0A84FF]/20 text-[#4CC9F0]'
                      : 'text-gray-300 hover:text-white hover:bg-[#0A84FF]/10'
                  }`}
                  whileHover={{ y: -2 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {item.label}
                </motion.button>
              </Link>
            ))}
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden text-white"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <motion.div
            className="md:hidden pb-4 space-y-2"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            {navItems.map((item) => (
              <Link key={item.path} to={item.path}>
                <motion.button
                  className={`w-full text-left px-4 py-2 rounded-lg font-medium ${
                    isActive(item.path)
                      ? 'bg-[#0A84FF]/20 text-[#4CC9F0]'
                      : 'text-gray-300 hover:text-white hover:bg-[#0A84FF]/10'
                  }`}
                  onClick={() => setIsOpen(false)}
                >
                  {item.label}
                </motion.button>
              </Link>
            ))}
          </motion.div>
        )}
      </div>
    </motion.nav>
  );
};
